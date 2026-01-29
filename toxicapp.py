# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import MT5Manager
from datetime import datetime, timedelta, timezone
import plotly.express as px

import json
from pathlib import Path


# =========================================================
# HELPERS – MANAGER CREDENTIAL STORAGE
# =========================================================
MANAGER_STORE = Path("managers.json")

def load_managers():
    if MANAGER_STORE.exists():
        return json.loads(MANAGER_STORE.read_text())
    return {}

def save_managers(data):
    MANAGER_STORE.write_text(json.dumps(data, indent=2))


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="MT5 Live Trade Analyzer", layout="wide")
st.title("📊 MT5 Live Trade Analyzer")


# =========================================================
# SIDEBAR – MANAGER CONNECTION (PERSISTENT)
# =========================================================
st.sidebar.header("🔌 Manager Connection")

managers = load_managers()
manager_names = ["➕ Add new manager"] + list(managers.keys())

selected_manager = st.sidebar.selectbox("Login as", manager_names)

if selected_manager != "➕ Add new manager":
    saved = managers[selected_manager]
    server = saved["server"]
    mgr_login = saved["login"]
    mgr_pass = saved["password"]
    st.sidebar.success(f"Loaded: {selected_manager}")
else:
    server = st.sidebar.text_input("Server", "127.0.0.1:443")
    mgr_login = st.sidebar.number_input("Manager Login", min_value=1, step=1)
    mgr_pass = st.sidebar.text_input("Manager Password", type="password")

    save_name = st.sidebar.text_input("Save as name")
    if st.sidebar.button("💾 Save Manager"):
        if save_name:
            managers[save_name] = {
                "server": server,
                "login": int(mgr_login),
                "password": mgr_pass,
            }
            save_managers(managers)
            st.sidebar.success("Saved. Reload page.")


# =========================================================
# CONNECT TO MT5 MANAGER
# =========================================================
if "api" not in st.session_state:
    st.session_state.api = None

if st.sidebar.button("Connect"):
    api = MT5Manager.ManagerAPI()
    ok = api.Connect(
        server,
        int(mgr_login),
        mgr_pass,
        MT5Manager.ManagerAPI.EnPumpModes.PUMP_MODE_USERS
        | MT5Manager.ManagerAPI.EnPumpModes.PUMP_MODE_POSITIONS
        | MT5Manager.ManagerAPI.EnPumpModes.PUMP_MODE_ORDERS,
        30000,
    )



    if not ok:
        st.error(MT5Manager.LastError())
        st.stop()

    st.session_state.api = api
    st.sidebar.success("✅ Connected")

if not st.session_state.api:
    st.stop()

api = st.session_state.api


# =========================================================
# CLIENT SELECTION
# =========================================================
st.header("👤 Client")

login = int(st.number_input("Client Login", min_value=1, step=1))
user = api.UserGet(login)
if not user:
    st.error("User not found")
    st.stop()

# =========================================================
# DATE RANGE
# =========================================================
st.subheader("📅 History Range")

range_opt = st.selectbox(
    "Range",
    ["Today", "Last 3 days", "Last week", "Last month", "All history", "Custom"]
)

now = datetime.now(timezone.utc)
to_ts = int(now.timestamp())

if range_opt == "Today":
    from_ts = int(datetime.combine(
        now.date(), datetime.min.time(), tzinfo=timezone.utc
    ).timestamp())

elif range_opt == "Last 3 days":
    from_ts = int((now - timedelta(days=3)).timestamp())

elif range_opt == "Last week":
    from_ts = int((now - timedelta(days=7)).timestamp())

elif range_opt == "Last month":
    from_ts = int((now - timedelta(days=30)).timestamp())

elif range_opt == "All history":
    from_ts = 0

else:
    c1, c2 = st.columns(2)
    with c1:
        fd = st.date_input("From", now.date())
    with c2:
        td = st.date_input("To", now.date())
    from_ts = int(datetime.combine(fd, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    to_ts   = int(datetime.combine(td, datetime.max.time(), tzinfo=timezone.utc).timestamp())

# =========================================================
# FETCH DEALS → BUILD POSITIONS DF
# =========================================================
def build_positions_df(api, login, from_ts, to_ts):
    deals = api.DealRequestByLogins([login], from_ts, to_ts) or []

    rows = []
    pos_map = {}

    for d in deals:
        if d.Action not in (0, 1):  # BUY / SELL only
            continue
        pos_map.setdefault(d.PositionID, []).append(d)

    for pos_id, dlist in pos_map.items():
        if len(dlist) < 2:
            continue  # not a closed position

        dlist.sort(key=lambda x: x.Time)
        open_deal  = dlist[0]
        close_deal = dlist[-1]

        rows.append({
            "PositionID": pos_id,
            "Symbol": open_deal.Symbol,
            "Type": "buy" if open_deal.Action == 0 else "sell",
            "Open Time": datetime.fromtimestamp(open_deal.Time, tz=timezone.utc),
            "Close Time": datetime.fromtimestamp(close_deal.Time, tz=timezone.utc),
            "Open Price": open_deal.Price,
            "Close Price": close_deal.Price,
            "Volume": open_deal.Volume / 10000,
            "Profit": sum(d.Profit for d in dlist),
        })

    df = pd.DataFrame(
        rows,
        columns=[
            "PositionID",
            "Symbol",
            "Type",
            "Open Time",
            "Close Time",
            "Open Price",
            "Close Price",
            "Volume",
            "Profit",
        ],
    )

    if df.empty:
        return df

    df["Hold_Time"] = df["Close Time"] - df["Open Time"]
    df.sort_values("Open Time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def build_ticket_position_map(api, login, from_ts, to_ts):
    """
    Maps DEAL ID -> POSITION ID
    Journal prints deal numbers (#2553), NOT position IDs.
    """
    deals = api.DealRequestByLogins([login], from_ts, to_ts) or []
    return {d.Deal: d.PositionID for d in deals}

def detect_martingale(df):
    """
    Returns martingale_score (0–100) and details
    """
    if df.empty:
        return 0, {}

    hits = 0
    checks = 0

    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Open Time")

        for i in range(1, len(g)):
            prev, curr = g.iloc[i - 1], g.iloc[i]

            if prev["Profit"] < 0:
                checks += 1
                if curr["Volume"] > prev["Volume"]:
                    hits += 1

    score = min(100, int((hits / checks) * 100)) if checks else 0
    return score, {
        "martingale_hits": hits,
        "checks": checks,
    }

def detect_grid_trading(df, price_tolerance=0.002, time_window_sec=300):
    """
    Returns grid_score (0–100) and details
    """
    if df.empty:
        return 0, {}

    grid_hits = 0
    checks = 0

    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Open Time")

        for i in range(1, len(g)):
            p1, p2 = g.iloc[i - 1], g.iloc[i]

            time_diff = (p2["Open Time"] - p1["Open Time"]).total_seconds()
            price_diff = abs(p2["Open Price"] - p1["Open Price"])
            vol_diff   = abs(p2["Volume"] - p1["Volume"])

            checks += 1

            if (
                time_diff <= time_window_sec
                and price_diff <= price_tolerance * p1["Open Price"]
                and vol_diff <= 0.01
            ):
                grid_hits += 1

    score = min(100, int((grid_hits / checks) * 100)) if checks else 0
    return score, {
        "grid_hits": grid_hits,
        "checks": checks,
    }

# =========================================================
# JOURNAL → EXECUTION IP EXTRACTION (MT5 TRUTH SOURCE)
# =========================================================

import re

IP_REGEX  = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
POS_REGEX = re.compile(r"#(\d+)")

def fetch_journal_logs(api, from_ts, to_ts):
    try:
        return api.JournalRequest(from_ts, to_ts) or []
    except Exception:
        return []

def parse_journal_ips(journal_logs, ticket_pos_map):
    rows = []

    for j in journal_logs:
        msg = str(getattr(j, "Message", ""))
        jid = str(getattr(j, "ID", ""))
        t   = getattr(j, "Time", None)

        if not t:
            continue

        ticket_match = POS_REGEX.search(msg)   # #2553
        ip_match = IP_REGEX.search(jid) or IP_REGEX.search(msg)    # 152.58.x.x

        if not ticket_match or not ip_match:
            continue

        ticket = int(ticket_match.group(1))
        pos_id = ticket_pos_map.get(ticket)

        if not pos_id:
            continue

        rows.append({
            "PositionID": pos_id,
            "IP": ip_match.group(),
            "Time": datetime.fromtimestamp(t, tz=timezone.utc),
        })

    return rows

def attach_execution_ips(positions_df, journal_rows):
    """
    Attach FIRST execution IP (entry-time IP) per position.
    MT5 journal logs may contain multiple entries per position;
    we intentionally pick the earliest one.
    """
    ip_map = {}

    for r in journal_rows:
        ip_map.setdefault(r["PositionID"], []).append(
            (r["Time"], r["IP"])
        )

    def pick_execution_ip(pid):
        entries = ip_map.get(pid, [])
        if not entries:
            return "Unavailable"

        # earliest journal entry = execution IP
        entries.sort(key=lambda x: x[0])
        return entries[0][1]

    positions_df["Execution_IPs"] = positions_df["PositionID"].apply(pick_execution_ip)
    return positions_df

def infer_execution_ip_from_login(positions_df, user_ips):
    """
    Fallback inference:
    If no journal IP exists, assume execution came from
    the most recent known login IP.
    """
    def infer(ip):
        return ip if ip != "Unavailable" else (
            user_ips[0] if user_ips else "Inferred (Login IP)"
        )

    positions_df["Execution_IPs"] = positions_df["Execution_IPs"].apply(infer)
    return positions_df


# =========================================================
# BUILD POSITIONS DATAFRAME (MISSING STEP ❗)
# =========================================================

positions_df = build_positions_df(api, login, from_ts, to_ts)

ticket_pos_map = build_ticket_position_map(
    api,
    login,
    from_ts,
    to_ts
)

st.write("Ticket→Position map size:", len(ticket_pos_map))




if positions_df.empty:
    st.info("No closed trades in selected range")
    st.stop()

# =========================================================
# FETCH EXECUTION IPS FROM MT5 JOURNAL (CRITICAL)
# =========================================================

journal_logs = fetch_journal_logs(api, from_ts, to_ts)
st.write("Journal entries:", len(journal_logs))  # ✅ correct place




# =========================================================
# HIGH PROFIT TRADES + USER IP CORRELATION (MT5 SAFE)
# =========================================================

# ---------- CONFIG ----------
PROFIT_THRESHOLD = st.number_input(
    "High Profit Threshold",
    min_value=0.0,
    value=500.0,
    step=100.0
)

# ---------- HELPER: FETCH MULTIPLE USER IPS (MT5 SAFE) ----------
def get_user_ips(api, login, max_ips=5):
    ips = []
    try:
        user = api.UserGet(login)
        if not user:
            return []

        possible_fields = [
            "LastIP",
            "LastLoginIP",
            "RegistrationIP",
            "CurrentIP",
            "PrevIP",
        ]

        for field in possible_fields:
            if hasattr(user, field):
                val = getattr(user, field)
                if val and val not in ips:
                    ips.append(str(val))
            if len(ips) >= max_ips:
                break
    except Exception:
        pass

    return ips


# ---------- FILTER HIGH PROFIT TRADES ----------
# parse journal
journal_rows = parse_journal_ips(journal_logs, ticket_pos_map)

# attach journal-based execution IPs
positions_df = attach_execution_ips(positions_df, journal_rows)

# fetch user login IPs
user_ips = get_user_ips(api, login, max_ips=6)

# infer execution IP when journal missing
positions_df = infer_execution_ip_from_login(positions_df, user_ips)
# =========================================================
# FINAL IP DISPLAY (SINGLE SOURCE OF TRUTH)
# =========================================================

# Prefer journal-derived execution IPs
execution_ips = sorted(
    {
        ip
        for ip in positions_df["Execution_IPs"].dropna().astype(str)
        if ip not in ("Unavailable", "Inferred (No IP Available)")
    }
)

if execution_ips:
    ip_display = ", ".join(execution_ips)
elif user_ips:
    ip_display = ", ".join(user_ips)
else:
    ip_display = "Unavailable"


# NOW filter high profit trades
high_profit_df = positions_df[
    positions_df["Profit"] >= PROFIT_THRESHOLD
].copy()

if "Execution_IPs" not in high_profit_df.columns:
    high_profit_df["Execution_IPs"] = "Unavailable"


# ---------------------------------------------------------------------------------------------------------------
st.subheader("🌐 Execution IPs (Journal Derived)")

unique_ips = sorted(
    ip for ip in positions_df["Execution_IPs"].unique()
    if ip not in ("Unavailable", "Inferred (No IP Available)")
)


for ip in unique_ips[:6]:
    st.write(f"- {ip}")

# ---------- DISPLAY IP LIST ----------
st.subheader("🌐 Detected Account IPs")

if user_ips:
    for ip in user_ips:
        st.write(f"- {ip}")
else:
    st.write("Unavailable")


# ---------- DISPLAY ----------
st.subheader("💰 High Profitable Trades (IP Correlated)")

if high_profit_df.empty:
    st.info("No trades exceeded the selected profit threshold.")
else:
# 🔑 CRITICAL: define schema even if empty
    st.dataframe(
        high_profit_df[
            [
                "Open Time",
                "Close Time",
                "Symbol",
                "Type",
                "Volume",
                "Profit",
                "Execution_IPs",
            ]
        ],
        use_container_width=True
    )






# ---------- OPTIONAL: SUMMARY METRICS ----------
st.subheader("🧠 High Profit + IP Summary")

c1, c2, c3 = st.columns(3)
c1.metric("High Profit Trades", len(high_profit_df))
c2.metric(
    "High Profit Total",
    f"${high_profit_df['Profit'].sum():.2f}"
    if not high_profit_df.empty else "$0.00"
)
c3.metric(
    "Execution IPs",
    ip_display if execution_ips else "Fallback to Login IP"
)


# # ---------- REPORT INJECTION ----------
# ip_line = user_ip if user_ip else "Unavailable"

# report += f"""

# --- High Profit IP Correlation ---

# High Profit Threshold: ${PROFIT_THRESHOLD:.2f}
# High Profit Trades: {len(high_profit_df)}
# Last Known Account IP: {ip_line}

# NOTE:
# IP shown is the last known login IP of the account.
# MT5 does NOT expose IP per individual trade.
# """


# =========================================================
# FILTERS
# =========================================================
symbols = sorted(positions_df["Symbol"].unique().tolist())
selected_symbols = st.multiselect("Filter Symbols", symbols, default=[])

scalping_limit = st.slider("Scalping Time (minutes)", 1, 5, 3)

if selected_symbols:
    positions_df = positions_df[
        positions_df["Symbol"].isin(selected_symbols)
    ].reset_index(drop=True)

# =========================================================
# ANALYSIS LOGIC
# =========================================================
# positions_df["Hold_Time"] = positions_df["Close Time"] - positions_df["Open Time"]

scalping_df = positions_df[
    positions_df["Hold_Time"] <= pd.Timedelta(minutes=scalping_limit)
]

# REVERSAL
positions_df["Reversal"] = False
for i in range(1, len(positions_df)):
    prev = positions_df.iloc[i - 1]
    curr = positions_df.iloc[i]
    if (
        prev["Symbol"] == curr["Symbol"]
        and abs((curr["Open Time"] - prev["Close Time"]).total_seconds()) <= 20
        and prev["Type"] != curr["Type"]
    ):
        positions_df.loc[i, "Reversal"] = True

reversal_df = positions_df[positions_df["Reversal"]]

# BURST
positions_df["Burst"] = False
for i in range(1, len(positions_df)):
    if abs(
        (positions_df.iloc[i]["Open Time"]
         - positions_df.iloc[i - 1]["Open Time"]).total_seconds()
    ) <= 2:
        positions_df.loc[i, "Burst"] = True
        positions_df.loc[i - 1, "Burst"] = True

burst_df = positions_df[positions_df["Burst"]]

# =========================================================
# STRATEGY FINGERPRINTING (GRID / MARTINGALE)
# =========================================================

grid_score, grid_details = detect_grid_trading(positions_df)
martingale_score, martingale_details = detect_martingale(positions_df)


# =========================================================
# DERIVED STATISTICS (SAFE – AFTER ALL DFS EXIST)
# =========================================================

# CORE COUNTS (REQUIRED FOR TOXICITY)
# =========================================================
# total_trades = len(positions_df)
# total_profit = positions_df["Profit"].sum()


# =========================================================
# TOXICITY SCORE (SELF-CONTAINED & SAFE)
# =========================================================
trade_count = len(positions_df)

scalp_component = (
    len(scalping_df) / trade_count * 100
) if trade_count else 0

toxic_pct = min(
    100,
    round(
        0.4 * scalp_component
        + 0.3 * grid_score
        + 0.3 * martingale_score,
        1
    )
)


# EQUITY CURVE
equity_df = positions_df.sort_values("Close Time").copy()
equity_df["Cumulative_Profit"] = equity_df["Profit"].cumsum()

# =========================================================
# METRICS
# =========================================================
st.subheader("📊 Summary")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Trades", len(positions_df))
m2.metric("Total Profit", f"{positions_df['Profit'].sum():.2f}")
m3.metric("Scalping Trades", len(scalping_df))
m4.metric("Reversal Trades", len(reversal_df))

st.subheader("🧠 Strategy Fingerprinting")

c1, c2, c3 = st.columns(3)
c1.metric("Grid Trading Score", f"{grid_score}/100")
c2.metric("Martingale Score", f"{martingale_score}/100")
c3.metric("Overall Toxicity", f"{toxic_pct}%")


# =========================================================
# VISUALS
# =========================================================
st.subheader("📈 Equity Curve")

fig = px.line(
    equity_df,
    x="Close Time",
    y="Cumulative_Profit",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Profit by Symbol")

profit_by_symbol = positions_df.groupby("Symbol")["Profit"].sum().sort_values()

fig2 = px.bar(
    profit_by_symbol,
    orientation="h",
    labels={"value": "Profit", "index": "Symbol"}
)
st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TABLE
# =========================================================
st.subheader("📜 Trades")

st.dataframe(
    positions_df[
        ["Open Time", "Symbol", "Type", "Volume", "Profit", "Hold_Time"]
    ],
    use_container_width=True
)


# =========================================================
# DERIVED STATISTICS (FOR REPORT)
# =========================================================

total_trades = len(positions_df)
total_profit = positions_df["Profit"].sum()

def win_rate(df):
    return (df["Profit"] > 0).mean() * 100 if len(df) > 0 else 0

avg_hold_time = positions_df["Hold_Time"].mean()

scalp_pct = (len(scalping_df) / total_trades * 100) if total_trades else 0
rev_pct   = (len(reversal_df) / total_trades * 100) if total_trades else 0
burst_pct = (len(burst_df) / total_trades * 100) if total_trades else 0

scalp_profit_pct = (scalping_df["Profit"].sum() / total_profit * 100) if total_profit else 0
rev_profit_pct   = (reversal_df["Profit"].sum() / total_profit * 100) if total_profit else 0
burst_profit_pct = (burst_df["Profit"].sum() / total_profit * 100) if total_profit else 0

st.subheader("📊 Overall Statistics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades", total_trades)
c2.metric("Total Profit", f"${total_profit:.2f}")
c3.metric("Avg Hold Time", str(avg_hold_time).split(".")[0])
c4.metric(
    "Profit / Trade",
    f"${(total_profit / total_trades):.2f}" if total_trades else "$0.00"
)

st.subheader("⚡ Scalping Statistics (<3 min holds)")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Scalping Trades",
    len(scalping_df),
    delta=f"{scalp_pct:.1f}% of total"
)
c2.metric(
    "Scalping Profit",
    f"${scalping_df['Profit'].sum():.2f}",
    delta=f"{scalp_profit_pct:.1f}% of total profit"
)
c3.metric(
    "Scalping Win Rate",
    f"{win_rate(scalping_df):.1f}%"
)
c4.metric(
    "Avg Scalp Time",
    str(scalping_df["Hold_Time"].mean()).split(".")[0]
    if len(scalping_df) else "N/A"
)

st.subheader("🔁 Reversal Trade Statistics (Opposite within 20s)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reversal Trades", len(reversal_df), delta=f"{rev_pct:.1f}% of total")
c2.metric(
    "Reversal Profit",
    f"${reversal_df['Profit'].sum():.2f}",
    delta=f"{rev_profit_pct:.1f}% of total profit"
)
c3.metric(
    "Reversal Win Rate",
    f"{win_rate(reversal_df):.1f}%" if len(reversal_df) else "N/A"
)
c4.metric(
    "Avg Reversal Profit",
    f"${reversal_df['Profit'].mean():.2f}" if len(reversal_df) else "N/A"
)

st.subheader("🚀 Burst Trade Statistics (≥2 trades within 2s)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Burst Trades", len(burst_df), delta=f"{burst_pct:.1f}% of total")
c2.metric(
    "Burst Profit",
    f"${burst_df['Profit'].sum():.2f}",
    delta=f"{burst_profit_pct:.1f}% of total profit"
)
c3.metric(
    "Burst Win Rate",
    f"{win_rate(burst_df):.1f}%" if len(burst_df) else "N/A"
)
c4.metric(
    "Avg Burst Profit",
    f"${burst_df['Profit'].mean():.2f}" if len(burst_df) else "N/A"
)


st.subheader("🧾 Client Trade Analysis Report")
# ip_line = user_ip if user_ip else "Unavailable"

report = f"""
Account: {login}
Detected IPs: {ip_display}

Total Trades: {total_trades}
Total Profit: ${total_profit:.2f}

Scalping Trades: {len(scalping_df)} ({scalp_pct:.1f}%)
Scalping Profit: ${scalping_df['Profit'].sum():.2f}

Grid Trading Score: {grid_score}/100
Martingale Score: {martingale_score}/100

Overall Toxic Trading %: {toxic_pct}%

High Profit Threshold: ${PROFIT_THRESHOLD:.2f}
High Profit Trades: {len(high_profit_df)}

Detected Pattern:
- {"Grid" if grid_score > 60 else "No Grid"}
- {"Martingale" if martingale_score > 60 else "No Martingale"}

NOTE:
Execution IPs are derived from MT5 server Journal logs
and mapped to positions via deal tickets.
If Journal data is unavailable, login IPs are used as fallback.
"""


st.code(report, language="text")
