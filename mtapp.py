# =========================================================
# MT5 MANAGER – UNIFIED CONTROL CENTER (PRODUCTION)
# =========================================================

import streamlit as st
import time
import re
from datetime import datetime, timezone, timedelta
import pandas as pd
import MT5Manager

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(page_title="MT5 Manager Control Center", layout="wide")
st.title("🛠️ MT5 Manager – Unified Control Center")

RIGHTS = MT5Manager.MTUser.EnUsersRights

# =========================================================
# SIDEBAR – MANAGER CONNECTION
# =========================================================

st.sidebar.header("🔌 Manager Connection")

server = st.sidebar.text_input("Server", "127.0.0.1:443")
mgr_login = st.sidebar.number_input("Manager Login", min_value=1, step=1)
mgr_pass = st.sidebar.text_input("Manager Password", type="password")

live_mode = st.sidebar.toggle("🔴 Live Refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh Interval (sec)", 1, 5, 1)

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
    st.success("✅ Connected to MT5 Manager")

if not st.session_state.api:
    st.stop()

api = st.session_state.api

# =========================================================
# HELPERS
# =========================================================

def update_user(user):
    if not api.UserUpdate(user):
        raise RuntimeError(MT5Manager.LastError())

def validate_password(pwd):
    if len(pwd) < 8 or len(pwd) > 16:
        return "8–16 chars required"
    if not re.search(r"[A-Z]", pwd): return "Uppercase missing"
    if not re.search(r"[a-z]", pwd): return "Lowercase missing"
    if not re.search(r"\d", pwd): return "Digit missing"
    if not re.search(r"[#@!$%^&*]", pwd): return "Special char missing"
    return None

def calculate_margin_mt5(api, positions, user):
    total = 0.0
    leverage = user.Leverage or 1
    for p in positions:
        s = api.SymbolGet(p.Symbol)
        if not s:
            continue
        lots = p.Volume / 10000
        total += (s.ContractSize * lots * p.PriceOpen) / leverage
    return round(total, 2)
# =========================================================
# ALL USERS
# =========================================================

st.divider()
st.subheader("👥 All Users")

users = api.UserRequestByGroup("*") or []
rows = []

for u in users:
    acc = api.UserAccountGet(int(u.Login))
    rows.append({
        "Login": u.Login,
        "Name": f"{u.FirstName} {u.LastName}".strip(),
        "Group": u.Group,
        "Leverage": u.Leverage,
        "Balance": acc.Balance if acc else 0,
        "Equity": acc.Equity if acc else 0,
        "Enabled": bool(u.Rights & RIGHTS.USER_RIGHT_ENABLED),
    })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No users visible")

# =========================================================
# CLIENT SELECTION
# =========================================================

st.header("👤 Client Context")

login = int(st.number_input("Client Login", min_value=1, step=1))
if login <= 0:
    st.stop()

user = api.UserGet(login)
if not user:
    st.error("User not found")
    st.stop()

positions = api.PositionGetByLogins([login]) or []
orders    = api.OrderGetByLogins([login]) or []

# =========================================================
# ACCOUNT METRICS (MT5-CORRECT)
# =========================================================

balance = user.Balance
floating = sum(p.Profit for p in positions)
equity = balance + floating
margin = calculate_margin_mt5(api, positions, user)
free_margin = equity - margin
margin_level = (equity / margin * 100) if margin > 0 else 0

st.divider()
st.subheader("🏦 Account Metrics (Live)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Balance", f"{balance:.2f}")
m2.metric("Equity", f"{equity:.2f}")
m3.metric("Margin", f"{margin:.2f}")
m4.metric("Free Margin", f"{free_margin:.2f}")
m5.metric("Margin Level", f"{margin_level:.2f} %")

# =========================================================
# OPEN POSITIONS
# =========================================================

st.divider()
st.subheader("📈 Open Positions")

st.metric("Open Positions", len(positions))
st.metric("Floating P/L", f"{floating:.2f}")

pos_rows = []
for p in positions:
    pos_rows.append({
        "Symbol": p.Symbol,
        "Ticket": p.Position,
        "Type": "BUY" if p.Action == MT5Manager.MTPosition.EnPositionAction.POSITION_BUY else "SELL",
        "Volume (Lots)": round(p.Volume / 10000, 2),
        "Open Price": p.PriceOpen,
        "Current Price": p.PriceCurrent,
        "Swap": round(p.Storage, 2),
        "Profit": round(p.Profit, 2),
        "Comment": p.Comment,
    })

if pos_rows:
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)
else:
    st.info("No positions")


# =========================================================
# ORDERS
# =========================================================

st.divider()
st.subheader("📄 Orders")

ord_rows = [{
    "Order": o.Order,
    "Symbol": o.Symbol,
    "Type": o.Type,
    "Volume": o.VolumeInitial / 10000,
    "Price": o.PriceOrder,
    "State": o.State,
    "Time": datetime.fromtimestamp(o.TimeSetup, tz=timezone.utc)
} for o in orders]

if ord_rows:
    st.dataframe(pd.DataFrame(ord_rows), use_container_width=True)
else:
    st.info("No orders")


# =========================================================
# CLIENT SETTINGS (GROUP & LEVERAGE)
# =========================================================

st.divider()
st.subheader("👤 Client Settings")

# Client name (safe)
client_name = f"{user.FirstName} {user.LastName}".strip() or "—"

# Fetch all groups
groups = []
for i in range(api.GroupTotal()):
    g = api.GroupNext(i)
    if g:
        groups.append(g.Group)

groups = sorted(set(groups))

c1, c2, c3 = st.columns([2, 3, 2])

# ---------------- LOGIN + NAME ----------------
with c1:
    st.markdown("**Login**")
    st.markdown(f"### {login}")
    st.caption(client_name)

# ---------------- GROUP ----------------
with c2:
    st.markdown("**Group**")
    new_group = st.selectbox(
        label="Group",
        options=groups,
        index=groups.index(user.Group) if user.Group in groups else 0,
        label_visibility="collapsed"
    )

# ---------------- LEVERAGE ----------------
with c3:
    st.markdown("**Leverage**")
    new_leverage = st.number_input(
        label="Leverage",
        min_value=1,
        step=1,
        value=int(user.Leverage),
        label_visibility="collapsed"
    )

# ---------------- APPLY BUTTON ----------------
if st.button("Apply Changes", type="primary"):
    changed = False

    if new_group != user.Group:
        user.Group = new_group
        changed = True

    if int(new_leverage) != int(user.Leverage):
        user.Leverage = int(new_leverage)
        changed = True

    if not changed:
        st.info("No changes to apply")
    else:
        if api.UserUpdate(user):
            st.success("✅ Client settings updated successfully")
        else:
            st.error(f"❌ Update failed: {MT5Manager.LastError()}")
# =========================================================
# DEALS HISTORY (MT5 – DATE-AWARE & 100% ACCURATE)
# =========================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

st.divider()
st.subheader("📜 Deals History")

# ---------------------------------------------------------
# RANGE SELECTION
# ---------------------------------------------------------
range_opt = st.selectbox(
    "Range",
    [
        "Today",
        "Last 3 days",
        "Last week",
        "Last month",
        "Last 3 months",
        "Last 6 months",
        "All history",
        "Custom",
    ],
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
    # MT5 "Last week" = previous calendar week (Mon → Sun)
    today = now.date()
    start_of_this_week = today - timedelta(days=today.weekday())  # Monday
    start_of_last_week = start_of_this_week - timedelta(days=7)
    end_of_last_week   = start_of_this_week - timedelta(seconds=1)

    from_ts = int(datetime.combine(
        start_of_last_week,
        datetime.min.time(),
        tzinfo=timezone.utc
    ).timestamp())

    to_ts = int(datetime.combine(
        end_of_last_week,
        datetime.max.time(),
        tzinfo=timezone.utc
    ).timestamp())


elif range_opt == "Last month":
    from_ts = int((now - timedelta(days=30)).timestamp())

elif range_opt == "Last 3 months":
    from_ts = int((now - timedelta(days=90)).timestamp())

elif range_opt == "Last 6 months":
    from_ts = int((now - timedelta(days=180)).timestamp())

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

# ---------------------------------------------------------
# REQUEST DEALS FOR RANGE
# ---------------------------------------------------------
deals = api.DealRequestByLogins([login], from_ts, to_ts) or []

rows = []
for d in deals:
    rows.append({
        "Time": datetime.fromtimestamp(d.Time, tz=timezone.utc),
        "Ticket": d.Deal,
        "Type": d.Action,                     # 0=BUY,1=SELL,2=BALANCE,3=CREDIT
        "Volume": round(d.Volume / 10000, 2), # MT5 internal → lots
        "Symbol": d.Symbol or "",
        "Price": round(d.Price, 5),
        "S/L": round(getattr(d, "PriceSL", 0.0), 5),
        "T/P": round(getattr(d, "PriceTP", 0.0), 5),
        "Close Time": datetime.fromtimestamp(
            d.Time, tz=timezone.utc
        ) if d.Price != 0 else "",
        "Close Price": round(d.Price, 5),
        "Reason": d.Reason,
        "Commission": round(d.Commission, 2),
        "Swap": round(d.Storage, 2),
        "Profit": round(d.Profit, 2),
        "Comment": d.Comment or "",
    })

# ---------------------------------------------------------
# DISPLAY TABLE
# ---------------------------------------------------------
if not rows:
    st.info("No deals in selected range")

else:
    df = pd.DataFrame(rows)
    df.sort_values("Time", ascending=False, inplace=True)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
BUY, SELL, BALANCE, CREDIT = 0, 1, 2, 3
DEPOSIT_REASON    = 6
WITHDRAWAL_REASON = 7

# --------------------------------------------------
# OPENING BALANCE = ALL balance + credit BEFORE from_ts
# --------------------------------------------------
# opening_balance = 0.0
# prev_deals = api.DealRequestByLogins([login], 0, from_ts - 1) or []

# for d in prev_deals:
#     if d.Action in (BALANCE, CREDIT):
#         opening_balance += d.Profit

# --------------------------------------------------
# CURRENT RANGE DATA
# --------------------------------------------------
trade_df   = df[df["Type"].isin([BUY, SELL])]
balance_df = df[df["Type"] == BALANCE]
credit_df  = df[df["Type"] == CREDIT]

# Trade totals
gross_profit = trade_df["Profit"].sum()
commission   = trade_df["Commission"].sum()
swap         = trade_df["Swap"].sum()

profit = gross_profit + commission + swap

# Deposits & withdrawals (MT5 reason-aware)
deposit = balance_df[
    (balance_df["Reason"] == DEPOSIT_REASON) &
    (balance_df["Profit"] > 0)
]["Profit"].sum()

withdrawal = abs(balance_df[
    (balance_df["Reason"] == WITHDRAWAL_REASON) &
    (balance_df["Profit"] < 0)
]["Profit"].sum())

# Credit
credit = credit_df["Profit"].sum()

# --------------------------------------------------
# CLOSING BALANCE (MT5 LOGIC)
# --------------------------------------------------
# closing_balance = (
#     opening_balance
#     + profit        # already net of commission & swap
#     + deposit
#     - withdrawal
#     + credit
# )
closing_balance = user.Balance

# --------------------------------------------------
# DISPLAY (MATCHES MT5 MANAGER)
# --------------------------------------------------
st.caption(
    f"**Profit:** {profit:.2f}    "
    f"**Deposit:** {deposit:.2f}    "
    f"**Withdrawal:** {withdrawal:.2f}    "
    f"**Credit:** {credit:.2f}    "
    f"**Commission:** {commission:.2f}    "
    f"**Swap:** {swap:.2f}    "
    f"**Balance:** {closing_balance:.2f}"
)




# =========================================================
# ACCOUNT ACCESS
# =========================================================

st.divider()
st.subheader("🔐 Account Access")

# Current rights state
login_enabled  = bool(user.Rights & RIGHTS.USER_RIGHT_ENABLED)
trading_enabled = not bool(user.Rights & RIGHTS.USER_RIGHT_TRADE_DISABLED)

c1, c2 = st.columns(2)

with c1:
    new_login_enabled = st.toggle(
        "Login Enabled",
        value=login_enabled
    )

with c2:
    new_trading_enabled = st.toggle(
        "Trading Enabled",
        value=trading_enabled
    )

rights_changed = False

# Login enable / disable
if new_login_enabled != login_enabled:
    if new_login_enabled:
        user.Rights |= RIGHTS.USER_RIGHT_ENABLED
    else:
        user.Rights &= ~RIGHTS.USER_RIGHT_ENABLED
    rights_changed = True

# Trading enable / disable
if new_trading_enabled != trading_enabled:
    if new_trading_enabled:
        user.Rights &= ~RIGHTS.USER_RIGHT_TRADE_DISABLED
    else:
        user.Rights |= RIGHTS.USER_RIGHT_TRADE_DISABLED
    rights_changed = True

if rights_changed:
    if api.UserUpdate(user):
        st.success("✅ Account access updated")
    else:
        st.error(MT5Manager.LastError())

# =========================================================
# TRADING OPTIONS
# =========================================================

st.subheader("⚙️ Trading Options")

algo_enabled = bool(user.Rights & RIGHTS.USER_RIGHT_EXPERT)
trailing_enabled = bool(user.Rights & RIGHTS.USER_RIGHT_TRAILING)

c1, c2 = st.columns(2)

with c1:
    new_algo = st.checkbox(
        "Algo Trading",
        value=algo_enabled
    )

with c2:
    new_trailing = st.checkbox(
        "Trailing Stops",
        value=trailing_enabled
    )

options_changed = False

# Algo trading
if new_algo != algo_enabled:
    if new_algo:
        user.Rights |= RIGHTS.USER_RIGHT_EXPERT
    else:
        user.Rights &= ~RIGHTS.USER_RIGHT_EXPERT
    options_changed = True

# Trailing stop
if new_trailing != trailing_enabled:
    if new_trailing:
        user.Rights |= RIGHTS.USER_RIGHT_TRAILING
    else:
        user.Rights &= ~RIGHTS.USER_RIGHT_TRAILING
    options_changed = True

if options_changed:
    if api.UserUpdate(user):
        st.success("✅ Trading options updated")
    else:
        st.error(MT5Manager.LastError())


# =========================================================
# BALANCE / CREDIT OPERATIONS
# =========================================================

st.divider()
st.subheader("💰 Balance / Credit Operations")

amount = st.number_input("Amount", min_value=0.0, step=1.0)
comment = st.text_input("Comment", "Manager operation")

OPS = {
    "Balance": MT5Manager.MTDeal.EnDealAction.DEAL_BALANCE,
    "Credit": MT5Manager.MTDeal.EnDealAction.DEAL_CREDIT,
    "Bonus": MT5Manager.MTDeal.EnDealAction.DEAL_BONUS,
    "Charge": MT5Manager.MTDeal.EnDealAction.DEAL_CHARGE,
    "Correction": MT5Manager.MTDeal.EnDealAction.DEAL_CORRECTION,
}

op = st.selectbox("Operation", OPS.keys())
direction = st.radio("Action", ["In", "Out"], horizontal=True)
signed_amount = amount if direction == "In" else -amount

if st.button("Execute", disabled=amount <= 0):
    if api.DealerBalance(login, signed_amount, OPS[op], comment):
        st.success("✅ Operation completed")
    else:
        st.error(MT5Manager.LastError())



# =========================================================
# PASSWORD CHANGE
# =========================================================

st.divider()
st.subheader("🔑 Password Change")

new_password = st.text_input(
    "New Password",
    type="password",
    help="8–16 chars, upper, lower, digit, special char required"
)

password_type = st.radio(
    "Password Type",
    ["Main", "Investor"],
    horizontal=True
)

def validate_password(pwd: str):
    if len(pwd) < 8 or len(pwd) > 16:
        return "Password must be 8–16 characters long"
    if not re.search(r"[A-Z]", pwd):
        return "At least one uppercase letter required"
    if not re.search(r"[a-z]", pwd):
        return "At least one lowercase letter required"
    if not re.search(r"\d", pwd):
        return "At least one digit required"
    if not re.search(r"[#@!$%^&*]", pwd):
        return "At least one special character required (#@!$%^&*)"
    return None

if st.button("Change Password"):
    error = validate_password(new_password)

    if error:
        st.error(error)
    else:
        pwd_type = (
            MT5Manager.MTUser.EnUsersPasswords.USER_PASS_MAIN
            if password_type == "Main"
            else MT5Manager.MTUser.EnUsersPasswords.USER_PASS_INVESTOR
        )

        ok = api.UserPasswordChange(
            pwd_type,
            int(login),
            new_password
        )

        if not ok:
            st.error(f"❌ Password change failed: {MT5Manager.LastError()}")
        else:
            st.success(f"✅ {password_type} password updated successfully")


# =========================================================
# TOXIC RISK ENGINE
# =========================================================

st.divider()
st.subheader("☠️ Toxic Trading Risk")

score = 0
if len(positions) >= 5: score += 40
if abs(floating) > balance * 0.7: score += 40
if margin > 0 and margin_level < 100: score += 20

if score >= 70:
    st.error(f"🚨 HIGH RISK ({score}/100)")
elif score >= 40:
    st.warning(f"⚠️ MEDIUM RISK ({score}/100)")
else:
    st.success(f"✅ LOW RISK ({score}/100)")

# =========================================================
# LIVE REFRESH
# =========================================================

if live_mode:
    time.sleep(refresh_seconds)
    st.rerun()
