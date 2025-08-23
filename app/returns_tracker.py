import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob
import numpy as np
import datetime

# --- CONFIGURATION ---
MAPPINGS_CSV = "ticker_mappings.csv"
TRADE_REPORTS_DIR = "trade_reports"
PORTFOLIO_HISTORY_CSV = "portfolio_history.csv"

# --- HELPERS: IO / MAPPINGS / TRADE READ ---

def load_user_mappings():
    if os.path.exists(MAPPINGS_CSV):
        df = pd.read_csv(MAPPINGS_CSV, dtype=str)
        df = df.dropna(subset=["scrip_code", "yahoo_ticker"])
        return dict(zip(df['scrip_code'].str.strip(), df['yahoo_ticker'].str.strip()))
    return {}

def save_user_mappings(mapping_dict):
    df = pd.DataFrame([(code, ticker) for code, ticker in mapping_dict.items()],
                      columns=["scrip_code", "yahoo_ticker"])
    df.to_csv(MAPPINGS_CSV, index=False)

def default_bse_mapping(scrip_code):
    if pd.isnull(scrip_code):
        return None
    s = str(scrip_code).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return f"{s.zfill(6)}.BO"
    return None

def yahoo_ticker_valid(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return not data.empty
    except Exception:
        return False

def map_ticker_for_row(row, user_mappings):
    code = str(row.get('scrip_code', '')).strip()
    if code in user_mappings:
        return user_mappings[code]
    default_map = default_bse_mapping(code)
    if default_map and yahoo_ticker_valid(default_map):
        return default_map
    return ""

def read_trade_report(file):
    df = pd.read_excel(file, dtype=str)
    colmap = {
        "Scrip Code": "scrip_code",
        "Scrip_Code": "scrip_code",
        "scrip_code": "scrip_code",
        "scrip code": "scrip_code",
        "Company": "company_name",
        "company_name": "company_name",
        "company": "company_name",
        "Quantity": "quantity",
        "Price": "price",
        "Side": "side",
        "Buy/Sell": "side",
        "Date": "date",
        "date": "date"
    }
    df = df.rename(columns={c: colmap[c] for c in df.columns if c in colmap})
    if 'scrip_code' in df.columns:
        df['scrip_code'] = df['scrip_code'].astype(str).str.strip()
    if 'company_name' in df.columns:
        df['company_name'] = df['company_name'].astype(str).str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    # Normalize numeric columns
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    return df

def get_repository_trade_reports():
    if not os.path.exists(TRADE_REPORTS_DIR):
        os.makedirs(TRADE_REPORTS_DIR)
    files = glob(os.path.join(TRADE_REPORTS_DIR, "*.xlsx"))
    reports = []
    for file in files:
        try:
            df = read_trade_report(file)
            reports.append((os.path.basename(file), df))
        except Exception as e:
            st.warning(f"Could not read {file}: {e}")
    return reports

# --- FORMATTING HELPERS ---

def inr_format(amount):
    try:
        amount = int(round(float(amount)))
    except:
        return "₹0"
    sign = "-" if amount < 0 else ""
    s = str(abs(amount))
    if len(s) <= 3:
        base = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        base = ','.join(reversed(groups)) + ',' + last3
    return f"{sign}₹{base}"

def color_pct_html(pct):
    try:
        pct = float(pct)
    except:
        return "<span style='color:gray;'>N/A</span>"
    color = "green" if pct >= 0 else "red"
    return f"<span style='color:{color}; font-weight:bold'>{round(pct,2)}%</span>"

# --- PRICE LOOKUP (current price only) ---

_price_cache = {}

def get_current_price(ticker):
    """Get latest available close price for ticker; cache during session."""
    if not isinstance(ticker, str) or ticker.strip() == "":
        return None
    key = (ticker, "latest")
    if key in _price_cache:
        return _price_cache[key]
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist.iloc[-1]["Close"])
            _price_cache[key] = price
            return price
        # fallback to info
        info = ticker_obj.info
        price = info.get("regularMarketPrice", None)
        _price_cache[key] = price
        return price
    except Exception:
        _price_cache[key] = None
        return None

# --- XIRR IMPLEMENTATION (Newton method) ---

def xirr(cashflows, dates, guess=0.1, tol=1e-6, maxiter=200):
    """
    Compute XIRR (annualized) using Newton-Raphson.
    cashflows: list of floats (positive = inflow to investor, negative = outflow)
    dates: list of dates (pd.Timestamp / datetime)
    returns decimal rate (e.g., 0.12 for 12%) or raises ArithmeticError on no convergence
    """
    if len(cashflows) != len(dates) or len(cashflows) < 2:
        raise ValueError("Need at least two cashflows with dates")
    d0 = pd.to_datetime(dates[0])
    times = np.array([(pd.to_datetime(d) - d0).days / 365.0 for d in dates], dtype=float)
    amounts = np.array(cashflows, dtype=float)

    def f(r):
        return np.sum(amounts / ((1.0 + r) ** times))

    def fprime(r):
        return np.sum(-amounts * times / ((1.0 + r) ** (times + 1.0)))

    r = guess
    for i in range(maxiter):
        try:
            val = f(r)
            der = fprime(r)
            if der == 0:
                break
            step = val / der
            r -= step
            if abs(step) < tol:
                return r
        except Exception:
            break
    raise ArithmeticError("XIRR did not converge")

# --- REALIZED/UNREALIZED PER-TICKER (average-costing) ---

def calc_realized_unrealized_avgcost(df):
    result_realized = []
    result_unrealized = []

    for ticker, trades in df.groupby("yahoo_ticker"):
        trades = trades.sort_values("date")
        buys = trades[trades["side"].str.lower() == "buy"].copy()
        sells = trades[trades["side"].str.lower() == "sell"].copy()

        total_buy_qty = buys["quantity"].astype(float).sum()
        total_buy_amt = (buys["quantity"].astype(float) * buys["price"].astype(float)).sum()
        avg_buy_price = total_buy_amt / total_buy_qty if total_buy_qty else 0

        total_sell_qty = sells["quantity"].astype(float).sum()
        total_sell_amt = (sells["quantity"].astype(float) * sells["price"].astype(float)).sum()
        avg_sell_price = total_sell_amt / total_sell_qty if total_sell_qty else 0

        # Realized portion: total sell qty at avg buy price
        realized_qty = min(total_sell_qty, total_buy_qty)
        realized_buy_amt = realized_qty * avg_buy_price
        realized_sell_amt = realized_qty * avg_sell_price
        realized_pl_value = realized_sell_amt - realized_buy_amt
        realized_pl_pct = ((avg_sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price else 0

        if realized_qty > 0:
            result_realized.append({
                "Ticker": ticker,
                "Quantity": int(realized_qty),
                "Average Buy Price": round(avg_buy_price, 2),
                "Average Sell Price": round(avg_sell_price, 2),
                "Profit/Loss Value": realized_pl_value,
                "Profit/Loss %": realized_pl_pct
            })

        # Unrealized portion: remaining holding at avg buy price
        unrealized_qty = total_buy_qty - total_sell_qty
        if unrealized_qty > 0:
            current_price = get_current_price(ticker)
            unrealized_buy_amt = unrealized_qty * avg_buy_price
            unrealized_curr_amt = unrealized_qty * current_price if current_price is not None else None
            unrealized_pl_value = (unrealized_curr_amt - unrealized_buy_amt) if (current_price is not None) else None
            unrealized_pl_pct = ((current_price - avg_buy_price) / avg_buy_price * 100) if (current_price is not None and avg_buy_price) else None
            result_unrealized.append({
                "Ticker": ticker,
                "Quantity": int(unrealized_qty),
                "Average Buy Price": round(avg_buy_price, 2),
                "Current Price": round(current_price, 2) if current_price is not None else "N/A",
                "Profit/Loss Value": unrealized_pl_value if unrealized_pl_value is not None else "N/A",
                "Profit/Loss %": unrealized_pl_pct if unrealized_pl_pct is not None else "N/A"
            })

    realized_df = pd.DataFrame(result_realized)
    unrealized_df = pd.DataFrame(result_unrealized)
    total_realized = realized_df["Profit/Loss Value"].sum() if not realized_df.empty else 0
    total_unrealized = unrealized_df["Profit/Loss Value"].replace("N/A", 0).astype(float).sum() if not unrealized_df.empty else 0

    return realized_df, unrealized_df, total_realized, total_unrealized

# --- Nifty 50 CAGR (using ^NSEI) between first trade date and today ---

def get_nifty50_cagr(start_dt, end_dt):
    try:
        ticker = "^NSEI"
        nifty = yf.Ticker(ticker)
        hist = nifty.history(start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        if hist is None or hist.empty:
            return None
        start_val = hist.iloc[0]["Close"]
        end_val = hist.iloc[-1]["Close"]
        years = (end_dt - start_dt).days / 365.25
        if years <= 0:
            return None
        cagr = (end_val / start_val) ** (1.0 / years) - 1.0
        return cagr * 100.0
    except Exception:
        return None

# --- CASHFLOW / TOTAL INVESTED BASED METRICS ---

def build_cashflows_from_history(history_df):
    """
    Build cashflow list for XIRR:
    - buys are negative (investor outflow)
    - sells are positive (investor inflow)
    """
    cfs = []
    for _, r in history_df.sort_values("date").iterrows():
        qty = float(r["quantity"])
        price = float(r["price"])
        dt = pd.to_datetime(r["date"])
        amt = qty * price
        if str(r["side"]).strip().lower().startswith("buy"):
            cfs.append((-amt, dt))  # outflow from investor
        else:
            cfs.append((amt, dt))   # inflow to investor (sell)
    return cfs

def compute_current_portfolio_value(history_df):
    """Compute current portfolio value using current prices for tickers in history."""
    holdings = {}
    for _, r in history_df.iterrows():
        t = r.get("yahoo_ticker", "")
        if not t or pd.isna(t):
            continue
        side = str(r["side"]).strip().lower()
        qty = float(r["quantity"])
        holdings.setdefault(t, 0.0)
        if side == "buy":
            holdings[t] += qty
        else:
            holdings[t] -= qty
    total = 0.0
    missing_prices = []
    for t, q in holdings.items():
        if q == 0:
            continue
        p = get_current_price(t)
        if p is None:
            missing_prices.append(t)
            continue
        total += q * p
    return total, holdings, missing_prices

# --- MAIN APP UI ---

st.set_page_config(page_title="Portfolio Returns Tracker", layout="wide")
st.title("Portfolio Returns Tracker")

st.markdown("""
This version avoids relying on historical stock prices for beginning/end valuations (which can be missing on Yahoo).
Instead it:
- Derives cashflows from your trade history (buys/sells) and uses XIRR on those flows + current portfolio value to compute an annualized return.
- Shows simple total-return metrics based on total net money invested vs current portfolio value.
- Keeps sortable, scrollable tables for realized and unrealized P/L (20-row visible window).
""")

# Load mappings
user_mappings = load_user_mappings()

tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
    st.markdown("""
Upload or select trade reports, map scrip codes to Yahoo tickers and save the portfolio history.
""")
    uploaded_files = st.file_uploader("Upload your trade reports (Excel)", type=["xlsx"], accept_multiple_files=True)
    repo_reports = get_repository_trade_reports()
    repo_files = [fname for fname, _ in repo_reports]
    selected_repo_files = st.multiselect("Select repo trade reports to include", repo_files, default=[])

    all_trades = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = read_trade_report(file)
                all_trades.append(df)
            except Exception as e:
                st.error(f"Could not read {file.name}: {e}")
    for fname, df in repo_reports:
        if fname in selected_repo_files:
            all_trades.append(df)

    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        st.success(f"Loaded {len(trades)} trades from {len(all_trades)} report(s).")
        st.dataframe(trades.head(10))

        trades['scrip_code'] = trades['scrip_code'].astype(str).str.strip()
        trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)

        unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
        st.subheader("Unmapped Scrip Codes")
        if not unmapped.empty:
            st.write("Enter Yahoo tickers for unmapped scrip codes.")
            edited = st.data_editor(unmapped.assign(yahoo_ticker=""), key="edit_tickers", num_rows="dynamic")
            if st.button("Update Mapping"):
                new_mappings = {}
                failed_codes = []
                for _, row in edited.iterrows():
                    code = str(row['scrip_code']).strip()
                    ticker = str(row['yahoo_ticker']).strip().upper()
                    if ticker:
                        if yahoo_ticker_valid(ticker):
                            new_mappings[code] = ticker
                        else:
                            failed_codes.append((code, ticker))
                if new_mappings:
                    user_mappings.update(new_mappings)
                    save_user_mappings(user_mappings)
                    st.success("Saved new mappings to ticker_mappings.csv. Reload to apply.")
                if failed_codes:
                    for code, ticker in failed_codes:
                        st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo and was not saved.")
        else:
            st.success("All scrip codes mapped.")

        mapped = trades[trades['yahoo_ticker'] != ""].copy()
        st.subheader("Mapped Trades (first 20 rows)")
        st.dataframe(mapped.head(20))

        st.markdown("---")
        st.subheader("Save Portfolio History")
        overwrite = st.checkbox("Overwrite existing portfolio_history.csv", value=False)
        if st.button("Save Portfolio History to CSV"):
            if os.path.exists(PORTFOLIO_HISTORY_CSV) and not overwrite:
                st.warning("portfolio_history.csv already exists. Check the overwrite box to overwrite.")
            else:
                mapped.to_csv(PORTFOLIO_HISTORY_CSV, index=False)
                st.success("Saved portfolio_history.csv in app folder. Returns tab will use this file.")
    else:
        st.info("Upload trades or select repo trade reports to start mapping.")

    st.markdown("---")
    st.subheader("Current Mappings")
    mapping_df = pd.DataFrame([{"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()])
    st.dataframe(mapping_df)

with tabs[1]:
    st.header("Portfolio Returns")

    if not os.path.exists(PORTFOLIO_HISTORY_CSV):
        st.warning("No portfolio_history.csv found. Save mapped trades from Trade Mapping tab first.")
    else:
        history_df = pd.read_csv(PORTFOLIO_HISTORY_CSV)
        if "date" not in history_df.columns:
            st.error("portfolio_history.csv missing 'date' column.")
            st.stop()
        history_df["date"] = pd.to_datetime(history_df["date"])
        required_cols = {"yahoo_ticker", "side", "quantity", "price", "date"}
        if not required_cols.issubset(set(history_df.columns)):
            st.error(f"portfolio_history.csv missing columns: {required_cols - set(history_df.columns)}")
            st.stop()

        # Compute current portfolio value and holdings
        current_value, holdings, missing_price_tickers = compute_current_portfolio_value(history_df)
        current_value_str = inr_format(current_value)

        # Compute total buys and sells
        buys_df = history_df[history_df['side'].str.lower().str.startswith("buy")]
        sells_df = history_df[history_df['side'].str.lower().str.startswith("sell")]
        total_buys = (buys_df['quantity'].astype(float) * buys_df['price'].astype(float)).sum()
        total_sells = (sells_df['quantity'].astype(float) * sells_df['price'].astype(float)).sum()
        net_invested = total_buys - total_sells  # positive means net money put in
        total_buys_str = inr_format(total_buys)
        total_sells_str = inr_format(total_sells)
        net_invested_str = inr_format(net_invested)

        # Build cashflows for XIRR: buy = -amt, sell = +amt, final value as +current_value at today
        cashflows = build_cashflows_from_history(history_df)
        if cashflows:
            amounts = [cf for cf, dt in cashflows]
            dates = [dt for cf, dt in cashflows]
            # append final value as inflow on today
            today = pd.Timestamp(datetime.date.today())
            amounts.append(current_value)
            dates.append(today)
            try:
                irr = xirr(amounts, dates) * 100.0
                irr_str = color_pct_html(irr)
            except Exception:
                irr = None
                irr_str = "<span style='color:gray;'>N/A</span>"
        else:
            irr = None
            irr_str = "<span style='color:gray;'>N/A</span>"

        # Simple total return % based on net invested (ignores timing)
        if net_invested > 0:
            total_return_amt = current_value - net_invested
            total_return_pct = (total_return_amt / net_invested) * 100.0
            # For a simple CAGR approximation: use earliest buy date as start
            first_buy = buys_df['date'].min() if not buys_df.empty else history_df['date'].min()
            years = (pd.Timestamp(datetime.date.today()) - pd.to_datetime(first_buy)).days / 365.25 if first_buy is not None else None
            if years and years > 0:
                cagr_simple = (current_value / net_invested) ** (1.0 / years) - 1.0
                cagr_simple_pct = cagr_simple * 100.0
            else:
                cagr_simple_pct = None
        else:
            total_return_amt = None
            total_return_pct = None
            cagr_simple_pct = None

        # Nifty 50 comparison (between first trade and today)
        first_trade = history_df['date'].min()
        nifty_cagr_val = get_nifty50_cagr(first_trade, pd.Timestamp(datetime.date.today())) if first_trade is not None else None
        nifty_cagr_str = color_pct_html(nifty_cagr_val) if nifty_cagr_val is not None else "<span style='color:gray;'>N/A</span>"

        # Headline (three columns)
        st.subheader("Headline Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Annualized (XIRR)**")
            st.markdown(irr_str, unsafe_allow_html=True)
        with c2:
            st.markdown("**Current Portfolio Value**")
            st.markdown(f"**{current_value_str}**")
        with c3:
            st.markdown("**Nifty 50 CAGR (since first trade)**")
            st.markdown(nifty_cagr_str, unsafe_allow_html=True)

        # Secondary headline: total-return based metrics
        st.markdown("### Total invested-based metrics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("Total Money Invested (Buys)")
            st.markdown(f"**{total_buys_str}**")
        with col_b:
            st.markdown("Total Money Withdrawn (Sells)")
            st.markdown(f"**{total_sells_str}**")
        with col_c:
            st.markdown("Net Money Put In")
            st.markdown(f"**{net_invested_str}**")

        if total_return_pct is not None:
            st.markdown(f"Total Return: <b>{inr_format(total_return_amt)}</b> ({color_pct_html(total_return_pct)})", unsafe_allow_html=True)
            if cagr_simple_pct is not None:
                st.markdown(f"Approx. CAGR on net invested (using first buy as start): {color_pct_html(cagr_simple_pct)}", unsafe_allow_html=True)
            else:
                st.markdown("Approx. CAGR: N/A (insufficient time data)", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:gray;'>No net invested capital found (buys minus sells <= 0) — cannot compute total return/CAGR.</span>", unsafe_allow_html=True)

        if missing_price_tickers:
            st.warning(f"Current prices unavailable for these tickers (they are excluded from current value): {sorted(set(missing_price_tickers))[:10]}")

        st.markdown("---")

        # Realized / Unrealized tables
        realized_df, unrealized_df, total_realized, total_unrealized = calc_realized_unrealized_avgcost(history_df)

        st.subheader("Realized Returns (Average Costing)")
        if not realized_df.empty:
            # keep numeric columns so st.dataframe can sort
            rf = realized_df.copy()
            rf["Profit/Loss Value INR"] = rf["Profit/Loss Value"].apply(lambda x: inr_format(x))
            rf["Profit/Loss % (num)"] = rf["Profit/Loss %"].astype(float).round(2)
            rf["Average Buy Price INR"] = rf["Average Buy Price"].apply(inr_format)
            rf["Average Sell Price INR"] = rf["Average Sell Price"].apply(inr_format)
            rf_display = rf[["Ticker", "Quantity", "Average Buy Price INR", "Average Sell Price INR", "Profit/Loss Value INR", "Profit/Loss % (num)"]]
            st.dataframe(rf_display, height=20 * 28)
            # styled expander for colored % column
            with st.expander("Styled Realized Table (colored %)", expanded=False):
                styled = rf.copy()
                styled["Profit/Loss Value INR"] = styled["Profit/Loss Value"].apply(inr_format)
                styled["Profit/Loss %"] = styled["Profit/Loss %"].apply(lambda x: color_pct_html(x))
                styled = styled.rename(columns={
                    "Average Buy Price": "Average Buy Price (₹)",
                    "Average Sell Price": "Average Sell Price (₹)"
                })
                st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown(f"**Total Realized Profit/Loss:** {inr_format(total_realized)}")
        else:
            st.info("No realized trades or profit/loss yet.")

        st.subheader("Unrealized Returns (Average Costing)")
        if not unrealized_df.empty:
            uf = unrealized_df.copy()
            uf["Profit/Loss Value INR"] = uf["Profit/Loss Value"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
            # Profit/Loss % numeric for sorting
            def to_num(x):
                try:
                    return float(x)
                except:
                    return np.nan
            uf["Profit/Loss % (num)"] = uf["Profit/Loss %"].apply(to_num)
            uf["Average Buy Price INR"] = uf["Average Buy Price"].apply(inr_format)
            uf["Current Price INR"] = uf["Current Price"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
            uf_display = uf[["Ticker", "Quantity", "Average Buy Price INR", "Current Price INR", "Profit/Loss Value INR", "Profit/Loss % (num)"]]
            st.dataframe(uf_display, height=20 * 28)
            with st.expander("Styled Unrealized Table (colored %)", expanded=False):
                styled = uf.copy()
                styled["Profit/Loss Value INR"] = styled["Profit/Loss Value"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
                styled["Profit/Loss %"] = styled["Profit/Loss %"].apply(lambda x: color_pct_html(x) if x != "N/A" else "<span style='color:gray;'>N/A</span>")
                st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown(f"**Total Unrealized Profit/Loss:** {inr_format(total_unrealized)}")
        else:
            st.info("No unrealized holdings.")

        st.markdown("---")
        st.success("Finished computing returns. Use the styled expanders to see colored % views. XIRR uses the cashflow timing + current portfolio value, and is the recommended annualized metric here.")
