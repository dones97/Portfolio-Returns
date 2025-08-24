import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob
import numpy as np
import datetime
import altair as alt
from functools import lru_cache

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

# --- PRICE LOOKUP with caching (historical & current) ---

_price_cache = {}

def get_prev_trading_price(ticker, target_date):
    """
    Try to fetch close price on or before target_date from Yahoo.
    Caches results. Returns None if not available.
    """
    if not isinstance(target_date, (pd.Timestamp, datetime.date, datetime.datetime)):
        target_date = pd.to_datetime(target_date)
    target_day = pd.Timestamp(target_date).normalize()
    key = ("hist", ticker, target_day.strftime("%Y-%m-%d"))
    if key in _price_cache:
        return _price_cache[key]

    start = (target_day - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (target_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist is None or hist.empty:
            _price_cache[key] = None
            return None
        hist = hist[hist.index <= target_day]
        if hist.empty:
            _price_cache[key] = None
            return None
        price = float(hist.iloc[-1]["Close"])
        _price_cache[key] = price
        return price
    except Exception:
        _price_cache[key] = None
        return None

def get_current_price(ticker):
    key = ("cur", ticker)
    if key in _price_cache:
        return _price_cache[key]
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist.iloc[-1]["Close"])
            _price_cache[key] = price
            return price
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice", None)
        _price_cache[key] = price
        return price
    except Exception:
        _price_cache[key] = None
        return None

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

# --- Build cashflows for XIRR / invested-based approach ---

def build_cashflows_from_history(history_df):
    cfs = []
    for _, r in history_df.sort_values("date").iterrows():
        qty = float(r["quantity"])
        price = float(r["price"])
        dt = pd.to_datetime(r["date"])
        amt = qty * price
        if str(r["side"]).strip().lower().startswith("buy"):
            cfs.append((-amt, dt))  # investor outflow
        else:
            cfs.append((amt, dt))   # inflow on sell
    return cfs

# --- Per-year realized P/L (by simulating average-cost) ---

def compute_realized_pl_by_year(history_df):
    df = history_df.copy().sort_values("date")
    realized_by_year = {}
    pos = {}
    for _, r in df.iterrows():
        ticker = r.get('yahoo_ticker', None)
        if pd.isna(ticker) or ticker is None or ticker == "":
            continue
        side = str(r['side']).strip().lower()
        qty = float(r['quantity'])
        price = float(r['price'])
        year = pd.to_datetime(r['date']).year
        if ticker not in pos:
            pos[ticker] = {"qty": 0.0, "cost": 0.0}
        if side == "buy":
            prev_qty = pos[ticker]["qty"]
            prev_cost = pos[ticker]["cost"]
            pos[ticker]["qty"] = prev_qty + qty
            pos[ticker]["cost"] = prev_cost + qty * price
        else:
            avg_cost = (pos[ticker]["cost"] / pos[ticker]["qty"]) if pos[ticker]["qty"] > 0 else 0.0
            realized = qty * (price - avg_cost)
            realized_by_year[year] = realized_by_year.get(year, 0.0) + realized
            remaining_qty = pos[ticker]["qty"] - qty
            if remaining_qty <= 0:
                pos[ticker]["qty"] = 0.0
                pos[ticker]["cost"] = 0.0
            else:
                pos[ticker]["qty"] = remaining_qty
                pos[ticker]["cost"] = avg_cost * remaining_qty
    return realized_by_year

# --- Fallback price resolution: Yahoo historical -> last trade price <= date in history_df ---

def get_price_for_date_with_fallback(ticker, target_date, history_df):
    """
    Resolve a reasonable price for ticker on target_date:
      1) Try Yahoo prev trading close
      2) Fallback to the last trade price in history_df at or before that date
      3) If none, return None
    """
    price = get_prev_trading_price(ticker, target_date)
    if price is not None:
        return price, "yahoo"
    # fallback: find last trade price for that ticker on or before target_date
    if history_df is not None:
        mask = (history_df['yahoo_ticker'] == ticker) & (history_df['date'] <= pd.to_datetime(target_date))
        trades = history_df.loc[mask].sort_values('date')
        if not trades.empty:
            last_price = float(trades.iloc[-1]['price'])
            return last_price, "last_trade"
    return None, None

# --- Per-year metrics computed by simulating trades and using fallback prices ---

def compute_yearly_metrics_from_trades(history_df):
    """
    For each calendar year present in history_df:
      - compute BMV (holdings before year start) using fallback prices
      - compute EMV (holdings at year end) using fallback prices
      - compute net_flows (buys positive, sells negative)
      - compute total_return_amt = EMV - BMV - net_flows
      - compute realized_by_year via average-cost simulation
      - compute unrealized = total_return_amt - realized
      - compute return_pct = total_return_amt / avg_portfolio_val if reasonable
    Returns list of dicts per year and list of missing_price tuples
    """
    if history_df.empty:
        return [], []

    history_df = history_df.copy().sort_values('date')
    history_df['date'] = pd.to_datetime(history_df['date'])
    first_year = history_df['date'].dt.year.min()
    last_year = history_df['date'].dt.year.max()
    years = list(range(first_year, last_year + 1))

    realized_by_year = compute_realized_pl_by_year(history_df)

    missing_prices = []
    results = []

    # helper to compute holdings as of a date
    def holdings_as_of(dt, inclusive=False):
        if inclusive:
            mask = history_df['date'] <= pd.to_datetime(dt)
        else:
            mask = history_df['date'] < pd.to_datetime(dt)
        sub = history_df.loc[mask].dropna(subset=['yahoo_ticker'])
        net = {}
        for t, grp in sub.groupby('yahoo_ticker'):
            buy_qty = grp[grp['side'].str.lower() == 'buy']['quantity'].astype(float).sum()
            sell_qty = grp[grp['side'].str.lower() == 'sell']['quantity'].astype(float).sum()
            net[t] = buy_qty - sell_qty
        return net

    for y in years:
        start_dt = pd.Timestamp(datetime.date(y, 1, 1))
        end_dt = pd.Timestamp(datetime.date(y, 12, 31))
        b_hold = holdings_as_of(start_dt, inclusive=False)
        e_hold = holdings_as_of(end_dt, inclusive=True)

        BMV = 0.0
        EMV = 0.0

        # value beginning holdings using fallback price
        for t, qty in b_hold.items():
            if qty == 0:
                continue
            price, src = get_price_for_date_with_fallback(t, start_dt, history_df)
            if price is None:
                missing_prices.append((t, start_dt.date()))
                continue
            BMV += qty * price

        # value ending holdings using fallback price
        for t, qty in e_hold.items():
            if qty == 0:
                continue
            price, src = get_price_for_date_with_fallback(t, end_dt, history_df)
            if price is None:
                missing_prices.append((t, end_dt.date()))
                continue
            EMV += qty * price

        # cashflows during year: buys positive invested, sells negative withdrawn
        mask_year = (history_df['date'] >= start_dt) & (history_df['date'] <= (end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
        df_year = history_df.loc[mask_year]
        net_flows = 0.0
        for _, r in df_year.iterrows():
            amt = float(r['quantity']) * float(r['price'])
            if str(r['side']).strip().lower().startswith("buy"):
                net_flows += amt
            else:
                net_flows -= amt

        total_return_amt = EMV - BMV - net_flows
        avg_portfolio_val = (BMV + EMV) / 2.0 if (BMV + EMV) != 0 else None

        # Compute percent: prefer dividing by avg_portfolio_val; if not available, fall back to net_flows if net_flows>0
        if avg_portfolio_val and avg_portfolio_val > 0:
            return_pct = (total_return_amt / avg_portfolio_val) * 100.0
        elif net_flows > 0:
            return_pct = (total_return_amt / net_flows) * 100.0
        else:
            return_pct = None

        realized_amt = realized_by_year.get(y, 0.0)
        unrealized_amt = total_return_amt - realized_amt

        results.append({
            "year": y,
            "BMV": BMV,
            "EMV": EMV,
            "net_flows": net_flows,
            "total_return_amt": total_return_amt,
            "return_pct": return_pct,
            "realized_amt": realized_amt,
            "unrealized_amt": unrealized_amt,
            "avg_portfolio_val": avg_portfolio_val
        })

    return results, missing_prices

# --- Yearly Nifty returns (per calendar year) ---
@lru_cache(maxsize=64)
def nifty_year_return(year):
    start_dt = pd.Timestamp(datetime.date(year, 1, 1))
    end_dt = pd.Timestamp(datetime.date(year, 12, 31))
    try:
        ticker = "^NSEI"
        hist = yf.Ticker(ticker).history(start=(start_dt - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
                                         end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if hist is None or hist.empty:
            return None
        hist_start = hist[hist.index <= start_dt]
        hist_end = hist[hist.index <= end_dt]
        if hist_start.empty or hist_end.empty:
            return None
        start_val = hist_start.iloc[0]["Close"]
        end_val = hist_end.iloc[-1]["Close"]
        return (end_val / start_val - 1.0) * 100.0
    except Exception:
        return None

# --- Realized profit time series (cumulative) ---

def realized_profit_time_series(history_df):
    df = history_df.copy().sort_values("date")
    rows = []
    pos = {}
    cum = 0.0
    for _, r in df.iterrows():
        ticker = r.get('yahoo_ticker', None)
        if pd.isna(ticker) or ticker is None or ticker == "":
            continue
        side = str(r['side']).strip().lower()
        qty = float(r['quantity'])
        price = float(r['price'])
        dt = pd.to_datetime(r['date'])
        if ticker not in pos:
            pos[ticker] = {"qty": 0.0, "cost": 0.0}
        if side == "buy":
            pos[ticker]["qty"] += qty
            pos[ticker]["cost"] += qty * price
            rows.append({"date": dt, "realized_on_date": 0.0, "cumulative_realized": cum})
        else:
            avg_cost = (pos[ticker]["cost"] / pos[ticker]["qty"]) if pos[ticker]["qty"] > 0 else 0.0
            realized = qty * (price - avg_cost)
            cum += realized
            remaining_qty = pos[ticker]["qty"] - qty
            if remaining_qty <= 0:
                pos[ticker]["qty"] = 0.0
                pos[ticker]["cost"] = 0.0
            else:
                pos[ticker]["qty"] = remaining_qty
                pos[ticker]["cost"] = avg_cost * remaining_qty
            rows.append({"date": dt, "realized_on_date": realized, "cumulative_realized": cum})
    if not rows:
        return pd.DataFrame(columns=["date", "realized_on_date", "cumulative_realized"])
    ts = pd.DataFrame(rows)
    ts = ts.groupby("date", as_index=False).agg({"realized_on_date": "sum", "cumulative_realized": "max"})
    ts = ts.sort_values("date").reset_index(drop=True)
    return ts

# --- MAIN APP UI AND FLOW ---

st.set_page_config(page_title="Portfolio Returns Tracker", layout="wide")
st.title("Portfolio Returns")

st.markdown("""
This version avoids relying solely on Yahoo historical prices for beginning/end-of-year valuations.
When Yahoo prices are missing we fall back to the last trade price on or before the target date from your trade history.
Per-year metrics are computed by simulating the trade history year-by-year.
""")

tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
    st.markdown("""
Upload and map trade reports. Save mapped trades to portfolio_history.csv for the returns tab to use.
""", unsafe_allow_html=True)

    user_mappings = load_user_mappings()

    uploaded_files = st.file_uploader("Upload your trade reports (Excel)", type=["xlsx"], accept_multiple_files=True)
    repo_reports = get_repository_trade_reports()
    repo_file_names = [fname for fname, _ in repo_reports] if repo_reports else []
    selected_repo_files = st.multiselect("Select repo trade reports to include", repo_file_names, default=[])

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
        st.subheader("Mapped Trades (preview)")
        st.dataframe(mapped.head(20))

        st.markdown("---")
        st.subheader("Save Portfolio History")
        overwrite = st.checkbox("Overwrite existing portfolio_history.csv", value=False)
        if st.button("Save Portfolio History to CSV"):
            if os.path.exists(PORTFOLIO_HISTORY_CSV) and not overwrite:
                st.warning("portfolio_history.csv already exists. Check the overwrite box to overwrite.")
            else:
                mapped.to_csv(PORTFOLIO_HISTORY_CSV, index=False)
                st.success("Saved portfolio_history.csv. Returns tab will use this file.")
    else:
        st.info("Upload trades or select repo trade reports to start mapping.")

    st.markdown("---")
    st.subheader("Current Mappings")
    mapping_df = pd.DataFrame([{"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()])
    st.dataframe(mapping_df)

with tabs[1]:
    st.header("Portfolio Returns")
    st.info("Calculations use portfolio_history.csv saved in the app folder.")

    if not os.path.exists(PORTFOLIO_HISTORY_CSV):
        st.warning("No portfolio_history.csv found. Save mapped trades from Trade Mapping tab first.")
        st.stop()

    history_df = pd.read_csv(PORTFOLIO_HISTORY_CSV)
    if 'date' not in history_df.columns:
        st.error("portfolio_history.csv missing 'date' column.")
        st.stop()
    history_df['date'] = pd.to_datetime(history_df['date'])
    required_cols = {"yahoo_ticker", "side", "quantity", "price", "date"}
    if not required_cols.issubset(set(history_df.columns)):
        st.error(f"portfolio_history.csv missing columns: {required_cols - set(history_df.columns)}")
        st.stop()

    # Compute realized/unrealized (per-ticker) using average-cost method (current prices for unrealized)
    realized_df, unrealized_df, total_realized, total_unrealized = calc_realized_unrealized_avgcost(history_df)

    # Current portfolio value derived from unrealized DF
    curr_value = unrealized_df.apply(
        lambda row: float(row["Quantity"]) * (float(row["Current Price"]) if row["Current Price"] != "N/A" else 0), axis=1
    ).sum() if not unrealized_df.empty else 0.0

    # Simple net invested: total buys - total sells (for percent reference)
    buys_df = history_df[history_df['side'].str.lower().str.startswith("buy")]
    sells_df = history_df[history_df['side'].str.lower().str.startswith("sell")]
    total_buys = (buys_df['quantity'].astype(float) * buys_df['price'].astype(float)).sum()
    total_sells = (sells_df['quantity'].astype(float) * sells_df['price'].astype(float)).sum()
    net_invested = total_buys - total_sells

    # Total return amount is realized + unrealized
    total_return_amt = total_realized + total_unrealized
    total_return_pct = (total_return_amt / net_invested * 100.0) if net_invested > 0 else None
    realized_pct = (total_realized / net_invested * 100.0) if net_invested > 0 else None
    unrealized_pct = (total_unrealized / net_invested * 100.0) if net_invested > 0 else None

    # XIRR (investor perspective): buys negative, sells positive, final value as positive today
    cashflows = build_cashflows_from_history(history_df)
    irr_str = "<span style='color:gray;'>N/A</span>"
    irr_val = None
    if cashflows:
        amounts = [amt for amt, dt in cashflows]
        dates = [dt for amt, dt in cashflows]
        # append final current portfolio value inflow on today's date
        today = pd.Timestamp(datetime.date.today())
        amounts.append(curr_value)
        dates.append(today)
        try:
            try:
                import numpy_financial as npf
                irr_val = npf.xirr(amounts, dates) * 100.0
            except Exception:
                # fallback Newton implementation
                def simple_xirr(amounts, dates):
                    d0 = pd.to_datetime(dates[0])
                    times = np.array([(pd.to_datetime(d) - d0).days / 365.0 for d in dates], dtype=float)
                    amounts_np = np.array(amounts, dtype=float)
                    def f(r):
                        return np.sum(amounts_np / ((1.0 + r) ** times))
                    def fprime(r):
                        return np.sum(-amounts_np * times / ((1.0 + r) ** (times + 1.0)))
                    r = 0.1
                    for i in range(200):
                        val = f(r)
                        der = fprime(r)
                        if der == 0:
                            break
                        step = val / der
                        r -= step
                        if abs(step) < 1e-6:
                            return r
                    raise ArithmeticError("xirr did not converge")
                irr_val = simple_xirr(amounts, dates) * 100.0
            irr_str = color_pct_html(irr_val) if irr_val is not None else irr_str
        except Exception:
            irr_val = None
            irr_str = "<span style='color:gray;'>N/A</span>"

    # Nifty 50 CAGR for full period (first trade to last trade)
    first_trade = history_df['date'].min()
    last_trade = history_df['date'].max()
    nifty_cagr_val = None
    try:
        if pd.notnull(first_trade) and pd.notnull(last_trade) and (last_trade > first_trade):
            try:
                ticker = "^NSEI"
                hist = yf.Ticker(ticker).history(start=first_trade.strftime("%Y-%m-%d"),
                                                 end=(last_trade + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                if hist is not None and not hist.empty:
                    start_val = hist.iloc[0]["Close"]
                    end_val = hist.iloc[-1]["Close"]
                    years = (last_trade - first_trade).days / 365.25
                    if years > 0:
                        nifty_cagr_val = (end_val / start_val) ** (1.0 / years) - 1.0
                        nifty_cagr_val = nifty_cagr_val * 100.0
            except Exception:
                nifty_cagr_val = None
    except Exception:
        nifty_cagr_val = None

    # --- UI: Headline metrics with realized/unrealized split ---
    st.subheader("Portfolio Headline Performance")

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.markdown("**Annualized (XIRR)**")
        st.markdown(irr_str, unsafe_allow_html=True)
    with col2:
        st.markdown("**Current Portfolio Value**")
        st.markdown(f"**{inr_format(curr_value)}**")
    with col3:
        st.markdown("**Nifty 50 CAGR (period)**")
        st.markdown(color_pct_html(nifty_cagr_val) if nifty_cagr_val is not None else "<span style='color:gray;'>N/A</span>", unsafe_allow_html=True)

    # Show total return with split realized/unrealized next to it
    st.markdown("---")
    st.subheader("Total Return & Split")

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("Total Return (₹)")
        st.markdown(f"**{inr_format(total_return_amt)}**")
        if total_return_pct is not None:
            st.markdown(f"({color_pct_html(total_return_pct)})", unsafe_allow_html=True)
    with colB:
        st.markdown("Realized (₹)")
        st.markdown(f"**{inr_format(total_realized)}**")
        if realized_pct is not None:
            st.markdown(f"({color_pct_html(realized_pct)})", unsafe_allow_html=True)
    with colC:
        st.markdown("Unrealized (₹)")
        st.markdown(f"**{inr_format(total_unrealized)}**")
        if unrealized_pct is not None:
            st.markdown(f"({color_pct_html(unrealized_pct)})", unsafe_allow_html=True)

    st.markdown("---")

    # --- Per-year comparison: portfolio vs Nifty (clustered column)
    st.subheader("Portfolio vs Nifty 50: Yearly Comparison (calendar years)")

    per_year_results, missing_prices = compute_yearly_metrics_from_trades(history_df)

    if not per_year_results:
        st.info("No per-year results could be computed from trade history.")
    else:
        per_year_df = pd.DataFrame(per_year_results)
        # Add nifty year returns
        per_year_df['nifty_pct'] = per_year_df['year'].apply(lambda y: nifty_year_return(y))
        # build chart data
        chart_rows = []
        for _, r in per_year_df.iterrows():
            chart_rows.append({"year": int(r['year']), "series": "Portfolio", "return_pct": (r['return_pct'] if r['return_pct'] is not None else np.nan)})
            chart_rows.append({"year": int(r['year']), "series": "Nifty 50", "return_pct": (r['nifty_pct'] if r['nifty_pct'] is not None else np.nan)})
        chart_df = pd.DataFrame(chart_rows).dropna(subset=['return_pct'])
        if not chart_df.empty:
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('return_pct:Q', title='Return %'),
                color=alt.Color('series:N', title='Series'),
                tooltip=[alt.Tooltip('year:O', title='Year'), alt.Tooltip('series:N', title='Series'), alt.Tooltip('return_pct:Q', title='Return %')]
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No yearly returns available to plot (all NaN).")

        if missing_prices:
            st.warning("Some tickers were missing Yahoo historical prices; we used the last trade price from your history as a fallback for those dates. Examples (ticker, date):")
            st.write(sorted(list(set(missing_prices)))[:20])

        # Show per-year table with realized/unrealized split
        display_rows = []
        for r in per_year_results:
            display_rows.append({
                "Year": r["year"],
                "BMV": inr_format(r["BMV"]),
                "EMV": inr_format(r["EMV"]),
                "Net Flows (₹)": inr_format(r["net_flows"]),
                "Total Return (₹)": inr_format(r["total_return_amt"]),
                "Return (%)": f"{round(r['return_pct'],2)}%" if r["return_pct"] is not None else "N/A",
                "Realized (₹)": inr_format(r["realized_amt"]),
                "Unrealized (₹)": inr_format(r["unrealized_amt"]),
                "Avg Portfolio Value (₹)": inr_format(r["avg_portfolio_val"]) if r["avg_portfolio_val"] else "N/A"
            })
        year_table_df = pd.DataFrame(display_rows)
        st.dataframe(year_table_df, height=20 * 28)

    st.markdown("---")

    # --- Area chart: cumulative realized profit over time ---
    st.subheader("Cumulative Realized Profit Over Time (INR)")

    realized_ts = realized_profit_time_series(history_df)
    if realized_ts.empty:
        st.info("No realized sells found to build profit timeline.")
    else:
        realized_ts_sorted = realized_ts.sort_values("date").reset_index(drop=True)
        chart = alt.Chart(realized_ts_sorted).mark_area(opacity=0.35).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('cumulative_realized:Q', title='Cumulative Realized Profit (₹)'),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('cumulative_realized:Q', title='Cumulative Realized (₹)')]
        ).properties(width='100%', height=300)
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Show realized profit timeline table"):
            rt = realized_ts_sorted.copy()
            rt["date"] = rt["date"].dt.date
            rt["realized_on_date_INR"] = rt["realized_on_date"].apply(lambda x: inr_format(x))
            rt["cumulative_realized_INR"] = rt["cumulative_realized"].apply(lambda x: inr_format(x))
            st.dataframe(rt[["date", "realized_on_date_INR", "cumulative_realized_INR"]], height=300)

    st.markdown("---")

    # --- Realized / Unrealized tables (scrollable & sortable) ---
    st.subheader("Realized Returns (Average Costing)")
    if not realized_df.empty:
        rf = realized_df.copy()
        rf["Profit/Loss Value INR"] = rf["Profit/Loss Value"].apply(lambda x: inr_format(x))
        rf["Profit/Loss % (num)"] = rf["Profit/Loss %"].astype(float).round(2)
        rf["Average Buy Price INR"] = rf["Average Buy Price"].apply(inr_format)
        rf["Average Sell Price INR"] = rf["Average Sell Price"].apply(inr_format)
        rf_display = rf[["Ticker", "Quantity", "Average Buy Price INR", "Average Sell Price INR", "Profit/Loss Value INR", "Profit/Loss % (num)"]]
        st.dataframe(rf_display, height=20 * 28)
        with st.expander("Styled Realized Table (colored %)"):
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
        with st.expander("Styled Unrealized Table (colored %)"):
            styled = uf.copy()
            styled["Profit/Loss Value INR"] = styled["Profit/Loss Value"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
            styled["Profit/Loss %"] = styled["Profit/Loss %"].apply(lambda x: color_pct_html(x) if x != "N/A" else "<span style='color:gray;'>N/A</span>")
            st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown(f"**Total Unrealized Profit/Loss:** {inr_format(total_unrealized)}")
    else:
        st.info("No unrealized holdings.")

    st.markdown("---")
    st.success("Done. Per-year returns now use trade-history-first fallback pricing; this avoids missing-Yahoo issues and yields sensible realized/unrealized splits.")
