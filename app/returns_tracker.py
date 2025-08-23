import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob
import numpy as np
import datetime
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
    code = str(row['scrip_code']).strip()
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
    sign = "-" if float(amount) < 0 else ""
    return f"{sign}₹{base}"

def color_pct_html(pct):
    try:
        pct = float(pct)
    except:
        return "<span style='color:gray;'>N/A</span>"
    color = "green" if pct >= 0 else "red"
    return f"<span style='color:{color}; font-weight:bold'>{round(pct,2)}%</span>"

# --- PRICE LOOKUP with caching ---

_price_cache = {}

def get_prev_trading_price(ticker, target_date):
    """
    Get closing price for ticker on or before target_date.
    Uses simple caching. If not available, returns None.
    """
    # target_date expected as pd.Timestamp or datetime.date/datetime
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    target_day = pd.Timestamp(target_date).normalize()
    cache_key = (ticker, target_day.strftime("%Y-%m-%d"))
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    # Query a small window before target_day (e.g., 10 calendar days) to find last close.
    start = (target_day - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (target_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist is None or hist.empty:
            _price_cache[cache_key] = None
            return None
        # take last available <= target_day
        hist = hist[hist.index <= target_day]
        if hist.empty:
            _price_cache[cache_key] = None
            return None
        price = float(hist.iloc[-1]["Close"])
        _price_cache[cache_key] = price
        return price
    except Exception:
        _price_cache[cache_key] = None
        return None

# --- XIRR IMPLEMENTATION (simple Newton) ---
def xirr(cashflows, dates, guess=0.10, tol=1e-6, maxiter=200):
    """
    cashflows: list of amounts (positive inflow to portfolio, negative outflow)
    dates: list of dates (pd.Timestamp or datetime)
    returns annualized rate as decimal (e.g., 0.12 for 12%)
    """
    if len(cashflows) != len(dates) or len(cashflows) < 2:
        raise ValueError("Need at least two cashflows with dates")
    # convert to float days from first date
    d0 = pd.to_datetime(dates[0])
    times = np.array([(pd.to_datetime(d) - d0).days / 365.0 for d in dates], dtype=float)
    amounts = np.array(cashflows, dtype=float)

    def f(r):
        return np.sum(amounts / ((1.0 + r) ** times))

    def fprime(r):
        # derivative wrt r
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

# --- CORE RETURNS LOGIC (Modified Dietz per year + realized P/L average costing) ---

def compute_holdings_as_of(history_df, as_of_date, inclusive=False):
    """
    Returns dict ticker -> net quantity as of given datetime.
    inclusive: if True, include trades on as_of_date (<=), else strictly before (<)
    """
    if 'date' not in history_df.columns:
        return {}
    # Ensure datetime
    hist = history_df.copy()
    hist['date'] = pd.to_datetime(hist['date'])
    if inclusive:
        mask = hist['date'] <= pd.to_datetime(as_of_date)
    else:
        mask = hist['date'] < pd.to_datetime(as_of_date)
    sub = hist[mask]
    # compute net qty per yahoo_ticker
    sub = sub.dropna(subset=['yahoo_ticker'])
    net = {}
    for t, grp in sub.groupby('yahoo_ticker'):
        buy_qty = grp[grp['side'].str.lower() == 'buy']['quantity'].astype(float).sum()
        sell_qty = grp[grp['side'].str.lower() == 'sell']['quantity'].astype(float).sum()
        net_qty = buy_qty - sell_qty
        net[t] = net_qty
    return net

def compute_modified_dietz_for_year(history_df, year, price_cache, realized_by_year_lookup):
    """
    Compute Modified Dietz return for calendar year.
    Returns a dict with BMV, EMV, net_flows, total_return_amt, return_pct, realized_amt, unrealized_amt, avg_portfolio_val
    """
    # define period start and end
    start_dt = pd.Timestamp(datetime.date(year, 1, 1))
    end_dt = pd.Timestamp(datetime.date(year, 12, 31))
    # Beginning holdings (as of 00:00 Jan 1)
    b_holdings = compute_holdings_as_of(history_df, start_dt, inclusive=False)
    # Ending holdings (as of end of Dec 31 inclusive)
    e_holdings = compute_holdings_as_of(history_df, end_dt, inclusive=True)
    # BMV and EMV: multiply by prices at nearest trading day on/before start and end
    BMV = 0.0
    EMV = 0.0
    missing_prices = []
    for ticker, qty in b_holdings.items():
        if qty == 0:
            continue
        price = get_prev_trading_price(ticker, start_dt)
        if price is None:
            missing_prices.append((ticker, start_dt.date()))
            continue
        BMV += qty * price
    for ticker, qty in e_holdings.items():
        if qty == 0:
            continue
        price = get_prev_trading_price(ticker, end_dt)
        if price is None:
            missing_prices.append((ticker, end_dt.date()))
            continue
        EMV += qty * price

    # Cash flows during the year: buys positive (invested), sells negative (withdrawn)
    mask_year = (history_df['date'] >= start_dt) & (history_df['date'] <= (end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
    df_year = history_df[mask_year].copy()
    cashflows = []
    cashflow_dates = []
    for _, r in df_year.iterrows():
        amt = float(r['quantity']) * float(r['price'])
        if str(r['side']).strip().lower().startswith("buy"):
            cf = amt  # investor put money in => positive inflow to portfolio
        else:
            cf = -amt  # sell => investor took money out (negative)
        cashflows.append(cf)
        cashflow_dates.append(pd.to_datetime(r['date']))

    # Modified Dietz calculation
    period_days = (end_dt - start_dt).days + 1
    # numerator = EMV - BMV - sum(CF)
    numerator = EMV - BMV - sum(cashflows)
    # denominator = BMV + sum(w_i * CF_i) where w_i = (end_date - t_i).days / period_days
    weighted_flows = 0.0
    for cf, dt in zip(cashflows, cashflow_dates):
        days_remaining = (end_dt - pd.to_datetime(dt)).days
        w = days_remaining / period_days
        weighted_flows += w * cf
    denominator = BMV + weighted_flows

    if abs(denominator) < 1e-9:
        return_pct = None
    else:
        return_pct = numerator / denominator  # decimal

    # realized P/L for the year from precomputed lookup
    realized_amt = realized_by_year_lookup.get(year, 0.0)

    # unrealized contribution = total return (numerator) - realized (since numerator is total return in ₹)
    unrealized_amt = numerator - realized_amt

    avg_portfolio_val = (BMV + EMV) / 2.0

    return {
        "year": year,
        "BMV": BMV,
        "EMV": EMV,
        "net_flows": sum(cashflows),
        "total_return_amt": numerator,
        "return_pct": return_pct,
        "realized_amt": realized_amt,
        "unrealized_amt": unrealized_amt,
        "avg_portfolio_val": avg_portfolio_val,
        "missing_prices": missing_prices
    }

def compute_realized_pl_by_year(history_df):
    """
    Simulate average-costing per ticker over all trades chronologically and record realized P/L per sell.
    Returns dict year -> realized_pl_amount
    """
    df = history_df.copy().sort_values("date")
    realized_by_year = {}
    # per-ticker running average cost structure: {ticker: {"qty": q, "cost": total_cost}}
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
            # increase avg cost
            prev_qty = pos[ticker]["qty"]
            prev_cost = pos[ticker]["cost"]
            new_qty = prev_qty + qty
            new_cost = prev_cost + qty * price
            pos[ticker]["qty"] = new_qty
            pos[ticker]["cost"] = new_cost
        else:
            # sell: realized = qty * (sell_price - avg_cost)
            avg_cost = (pos[ticker]["cost"] / pos[ticker]["qty"]) if pos[ticker]["qty"] > 0 else 0.0
            realized = qty * (price - avg_cost)
            realized_by_year[year] = realized_by_year.get(year, 0.0) + realized
            # reduce position quantity and cost by proportion
            remaining_qty = pos[ticker]["qty"] - qty
            if remaining_qty <= 0:
                # sold more than held: set to zero (we assume no short accounting)
                pos[ticker]["qty"] = 0.0
                pos[ticker]["cost"] = 0.0
            else:
                # reduce cost proportionally
                pos[ticker]["qty"] = remaining_qty
                pos[ticker]["cost"] = avg_cost * remaining_qty
    return realized_by_year

# --- NIFTY 50 CAGR for period ---
def get_nifty50_cagr(start_dt, end_dt):
    try:
        ticker = "^NSEI"
        nifty = yf.Ticker(ticker)
        hist = nifty.history(start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
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

# --- MAIN APP UI AND FLOW ---

st.set_page_config(page_title="Portfolio Returns Tracker", layout="wide")
st.title("Portfolio Ticker Mapping & Portfolio Returns")

st.markdown("""
This app:
- Ingests Excel trade reports and/or reads `portfolio_history.csv` in the repo.
- Lets you map scrip codes to Yahoo tickers and save mappings to `ticker_mappings.csv`.
- Saves mapped portfolio history to `portfolio_history.csv` and computes returns.
""")

# Load mappings
user_mappings = load_user_mappings()

# File upload and repo trade reports selection (Trade Mapping tab)
tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
    st.header("Trade Mapping & Portfolio History Save")
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
        st.write("Preview (first 5 rows):")
        st.dataframe(trades.head())

        trades['scrip_code'] = trades['scrip_code'].astype(str).str.strip()
        trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)

        unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
        st.subheader("Unmapped Scrip Codes")
        if not unmapped.empty:
            st.write("Enter Yahoo tickers for unmapped scrip codes (e.g., RELIANCE.NS or 543306.BO).")
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
                    st.success("Saved new mappings to ticker_mappings.csv. Please reload or re-upload for them to take effect in this session.")
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
        st.write("Save the mapped trades to portfolio_history.csv (overwrites existing file). Check to confirm.")
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
    st.write("This tab reads portfolio_history.csv in the repository and computes returns. You can overwrite the CSV from the Trade Mapping tab if needed.")

    if not os.path.exists(PORTFOLIO_HISTORY_CSV):
        st.warning("No portfolio_history.csv found. Go to Trade Mapping tab to save mapped trades.")
    else:
        history_df = pd.read_csv(PORTFOLIO_HISTORY_CSV)
        # normalize columns
        if 'date' in history_df.columns:
            history_df['date'] = pd.to_datetime(history_df['date'])
        else:
            st.error("portfolio_history.csv missing 'date' column. Please ensure the saved CSV includes a date column.")
            st.stop()

        required_cols = {"yahoo_ticker", "side", "quantity", "price", "date"}
        if not required_cols.issubset(set(history_df.columns)):
            st.error(f"portfolio_history.csv missing columns: {required_cols - set(history_df.columns)}")
            st.stop()

        # compute realized by year
        realized_by_year = compute_realized_pl_by_year(history_df)

        # build list of years present
        first_date = history_df['date'].min()
        last_date = history_df['date'].max()
        years = list(range(first_date.year, last_date.year + 1))

        # compute per-year metrics
        per_year = []
        any_missing_prices = []
        for y in years:
            yr_metrics = compute_modified_dietz_for_year(history_df, y, _price_cache, realized_by_year)
            per_year.append(yr_metrics)
            if yr_metrics.get("missing_prices"):
                any_missing_prices.extend(yr_metrics["missing_prices"])

        per_year_df = pd.DataFrame(per_year)
        # compute weighted annualized headline: weight by avg_portfolio_val
        valid_mask = per_year_df['return_pct'].notnull() & (per_year_df['avg_portfolio_val'] > 0)
        if valid_mask.any():
            weighted_num = (per_year_df.loc[valid_mask, 'return_pct'] * per_year_df.loc[valid_mask, 'avg_portfolio_val']).sum()
            total_weight = per_year_df.loc[valid_mask, 'avg_portfolio_val'].sum()
            weighted_annualized_return = (weighted_num / total_weight) * 100.0  # convert to percent
        else:
            weighted_annualized_return = None

        # XIRR calculation across entire set
        # Build cashflows for xirr: buys positive, sells negative; final value as positive on last_date
        cfs = []
        cfs_dates = []
        for _, r in history_df.sort_values("date").iterrows():
            amt = float(r['quantity']) * float(r['price'])
            if str(r['side']).strip().lower().startswith("buy"):
                cf = -amt  # for XIRR, convention often: investments are negative outflows from investor; we use investor perspective
            else:
                cf = amt
            cfs.append(cf)
            cfs_dates.append(pd.to_datetime(r['date']))
        # append final portfolio value as inflow on last_date (so investor could theoretically exit)
        # compute current portfolio value (unrealized)
        realized_df_dummy, unrealized_df, total_realized, total_unrealized = calc_realized_unrealized_avgcost(history_df) if 'calc_realized_unrealized_avgcost' in globals() else (pd.DataFrame(), pd.DataFrame(), 0, 0)
        # If our calc_realized_unrealized_avgcost is not available (older contexts), compute current by holdings * current price:
        if unrealized_df is None or unrealized_df.empty:
            # compute current value by holdings now
            holdings_now = compute_holdings_as_of(history_df, pd.Timestamp(datetime.date.today()), inclusive=True)
            curr_val = 0.0
            for t, q in holdings_now.items():
                p = get_prev_trading_price(t, pd.Timestamp(datetime.date.today()))
                if p:
                    curr_val += q * p
        else:
            # sum unrealized current prices * qty
            curr_val = unrealized_df.apply(lambda row: float(row["Quantity"]) * (float(row["Current Price"]) if row["Current Price"] != "N/A" else 0), axis=1).sum() if not unrealized_df.empty else 0.0

        # For investor perspective XIRR: treat buys negative, sells positive, and add final value as positive on last_date
        if cfs and len(cfs) >= 2:
            cfs_with_final = cfs.copy()
            cfs_dates_with_final = cfs_dates.copy()
            cfs_with_final.append(curr_val)
            cfs_dates_with_final.append(pd.to_datetime(last_date))
            try:
                # our xirr function expects inflows/outflows with investor convention
                irr = xirr(cfs_with_final, cfs_dates_with_final) * 100.0
                irr_str = color_pct_html(irr)
            except Exception:
                irr = None
                irr_str = "<span style='color:gray;'>N/A</span>"
        else:
            irr = None
            irr_str = "<span style='color:gray;'>N/A</span>"

        # Nifty 50 CAGR between first and last trade date
        nifty_cagr_val = get_nifty50_cagr(first_date, last_date)
        nifty_cagr_str = color_pct_html(nifty_cagr_val) if nifty_cagr_val is not None else "<span style='color:gray;'>N/A</span>"

        # Current portfolio value formatting
        # compute current holdings and current value
        holdings_now = compute_holdings_as_of(history_df, pd.Timestamp(datetime.date.today()), inclusive=True)
        current_value = 0.0
        for t, q in holdings_now.items():
            p = get_prev_trading_price(t, pd.Timestamp(datetime.date.today()))
            if p:
                current_value += q * p
        current_value_str = inr_format(current_value)

        # Top headline: show XIRR, Current Portfolio Value, Nifty 50 CAGR
        st.markdown("### Portfolio Headline Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**XIRR (annualized)**")
            st.markdown(irr_str, unsafe_allow_html=True)
        with col2:
            st.markdown("**Current Portfolio Value**")
            st.markdown(f"**{current_value_str}**")
        with col3:
            st.markdown("**Nifty 50 CAGR (period)**")
            st.markdown(nifty_cagr_str, unsafe_allow_html=True)

        # Annualized weighted (calendar-year) headline right below
        st.markdown("### Calendar-year Weighted Annualized Return")
        if weighted_annualized_return is not None:
            st.markdown(f"<span style='font-size:1.3em;font-weight:bold'>{color_pct_html(weighted_annualized_return)}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:gray;'>N/A</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Per-year table: prepare DF with formatted columns
        display_rows = []
        for r in per_year:
            row = {
                "Year": r["year"],
                "BMV": inr_format(r["BMV"]),
                "EMV": inr_format(r["EMV"]),
                "Net Flows": inr_format(r["net_flows"]),
                "Total Return (₹)": inr_format(r["total_return_amt"]),
                "Return (%)": round(r["return_pct"] * 100, 2) if r["return_pct"] is not None else None,
                "Realized (₹)": inr_format(r["realized_amt"]),
                "Unrealized (₹)": inr_format(r["unrealized_amt"]),
                "Avg Portfolio Value": inr_format(r["avg_portfolio_val"])
            }
            display_rows.append(row)
        year_table_df = pd.DataFrame(display_rows)

        st.subheader("Calendar Year Returns (Modified Dietz)")
        # Show sortable/scrollable table using st.dataframe (sortable). Height approximates 20 rows visible.
        st.dataframe(year_table_df, height=20 * 28)  # ~20 rows * row height

        # Also provide a styled colored view (optional) for percent coloring (user can expand)
        with st.expander("Styled year table (colored %)", expanded=False):
            styled = year_table_df.copy()
            # convert Return (%) to numeric for styling
            styled["Return (%)"] = styled["Return (%)"].apply(lambda x: x if x is not None else np.nan)
            sty = styled.style.format({
                "BMV": lambda x: x,
                "EMV": lambda x: x,
                "Net Flows": lambda x: x,
                "Total Return (₹)": lambda x: x,
                "Return (%)": "{:.2f}%",
                "Realized (₹)": lambda x: x,
                "Unrealized (₹)": lambda x: x,
                "Avg Portfolio Value": lambda x: x
            }).applymap(lambda v: 'color:green; font-weight:bold' if (isinstance(v, (int, float, np.floating, np.integer)) and v >= 0) else 'color:red; font-weight:bold',
                        subset=["Return (%)"])
            st.write(sty.to_html(escape=False), unsafe_allow_html=True)

        st.markdown("---")

        # Realized per-ticker table (from earlier function calc_realized_unrealized_avgcost used previously)
        st.subheader("Realized Returns (Per Ticker)")
        # We'll compute realized/unrealized per ticker similar to earlier implementation but return numeric DF for sorting
        realized_rows = []
        unrealized_rows = []
        # Recompute per-ticker average-cost based realized/unrealized
        # Iterate groups by ticker
        for ticker, grp in history_df.groupby("yahoo_ticker"):
            grp = grp.sort_values("date")
            buys = grp[grp['side'].str.lower() == 'buy']
            sells = grp[grp['side'].str.lower() == 'sell']
            total_buy_qty = buys['quantity'].astype(float).sum()
            total_buy_amt = (buys['quantity'].astype(float) * buys['price'].astype(float)).sum()
            avg_buy_price = total_buy_amt / total_buy_qty if total_buy_qty else 0.0
            total_sell_qty = sells['quantity'].astype(float).sum()
            total_sell_amt = (sells['quantity'].astype(float) * sells['price'].astype(float)).sum()
            avg_sell_price = total_sell_amt / total_sell_qty if total_sell_qty else 0.0

            realized_qty = min(total_sell_qty, total_buy_qty)
            if realized_qty > 0:
                realized_buy_amt = realized_qty * avg_buy_price
                realized_sell_amt = realized_qty * avg_sell_price
                pl_value = realized_sell_amt - realized_buy_amt
                pl_pct = ((avg_sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price else None
                realized_rows.append({
                    "Ticker": ticker,
                    "Quantity": int(realized_qty),
                    "Avg Buy Price (₹)": inr_format(avg_buy_price),
                    "Avg Sell Price (₹)": inr_format(avg_sell_price),
                    "Profit/Loss (₹)": pl_value,
                    "Profit/Loss (%)": pl_pct
                })

            unrealized_qty = total_buy_qty - total_sell_qty
            if unrealized_qty > 0:
                current_price = get_prev_trading_price(ticker, pd.Timestamp(datetime.date.today()))
                unrealized_buy_amt = unrealized_qty * avg_buy_price
                unrealized_curr_amt = unrealized_qty * current_price if current_price is not None else None
                unrealized_pl = unrealized_curr_amt - unrealized_buy_amt if unrealized_curr_amt is not None else None
                unrealized_pct = ((current_price - avg_buy_price) / avg_buy_price * 100) if (current_price is not None and avg_buy_price) else None
                unrealized_rows.append({
                    "Ticker": ticker,
                    "Quantity": int(unrealized_qty),
                    "Avg Buy Price (₹)": inr_format(avg_buy_price),
                    "Current Price (₹)": inr_format(current_price) if current_price is not None else "N/A",
                    "Profit/Loss (₹)": unrealized_pl if unrealized_pl is not None else None,
                    "Profit/Loss (%)": unrealized_pct
                })

        realized_df_display = pd.DataFrame(realized_rows)
        unrealized_df_display = pd.DataFrame(unrealized_rows)

        # Show sortable & scrollable primary tables (st.dataframe) - height set for ~20 rows
        if not realized_df_display.empty:
            # sortability preserved
            st.dataframe(realized_df_display, height=20 * 28)
            # show styled colored variant in expander
            with st.expander("Styled Realized Table (colored %)", expanded=False):
                rsty = realized_df_display.copy()
                # format Profit/Loss (₹) as INR
                rsty["Profit/Loss (₹)"] = rsty["Profit/Loss (₹)"].apply(lambda x: inr_format(x) if pd.notnull(x) else "N/A")
                rsty["Profit/Loss (%)"] = rsty["Profit/Loss (%)"].apply(lambda x: color_pct_html(x) if pd.notnull(x) else "<span style='color:gray;'>N/A</span>")
                st.write(rsty.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No realized positions to show.")

        if not unrealized_df_display.empty:
            st.dataframe(unrealized_df_display, height=20 * 28)
            with st.expander("Styled Unrealized Table (colored %)", expanded=False):
                usty = unrealized_df_display.copy()
                usty["Profit/Loss (₹)"] = usty["Profit/Loss (₹)"].apply(lambda x: inr_format(x) if pd.notnull(x) else "N/A")
                usty["Profit/Loss (%)"] = usty["Profit/Loss (%)"].apply(lambda x: color_pct_html(x) if pd.notnull(x) else "<span style='color:gray;'>N/A</span>")
                st.write(usty.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No unrealized holdings to show.")

        # Show any missing price diagnostics
        if any_missing_prices:
            st.markdown("---")
            st.warning("Some tickers were missing historical prices for BMV/EMV calculations. Those tickers are excluded from the relevant year's market value. Example missing entries (ticker, date):")
            sample = any_missing_prices[:10]
            st.write(sample)

# --- Utility compatibility: ensure calc_realized_unrealized_avgcost exists for compatibility with earlier code blocks ---
def calc_realized_unrealized_avgcost(df):
    """
    Compatibility function used earlier - computes per-ticker realized/unrealized using average cost.
    Returns realized_df, unrealized_df, total_realized, total_unrealized
    """
    result_realized = []
    result_unrealized = []
    for ticker, trades in df.groupby("yahoo_ticker"):
        buys = trades[trades["side"].str.lower() == "buy"].copy()
        sells = trades[trades["side"].str.lower() == "sell"].copy()

        total_buy_qty = buys["quantity"].astype(float).sum()
        total_buy_amt = (buys["quantity"].astype(float) * buys["price"].astype(float)).sum()
        avg_buy_price = total_buy_amt / total_buy_qty if total_buy_qty else 0

        total_sell_qty = sells["quantity"].astype(float).sum()
        total_sell_amt = (sells["quantity"].astype(float) * sells["price"].astype(float)).sum()
        avg_sell_price = total_sell_amt / total_sell_qty if total_sell_qty else 0

        realized_qty = min(total_sell_qty, total_buy_qty)
        realized_buy_amt = realized_qty * avg_buy_price
        realized_sell_amt = realized_qty * avg_sell_price
        realized_pl_value = realized_sell_amt - realized_buy_amt
        realized_pl_pct = ((avg_sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price else 0

        if realized_qty > 0:
            result_realized.append({
                "Ticker": ticker,
                "Quantity": int(realized_qty),
                "Average Buy Price": avg_buy_price,
                "Average Sell Price": avg_sell_price,
                "Profit/Loss Value": realized_pl_value,
                "Profit/Loss %": realized_pl_pct
            })

        unrealized_qty = total_buy_qty - total_sell_qty
        if unrealized_qty > 0:
            current_price = get_prev_trading_price(ticker, pd.Timestamp(datetime.date.today()))
            unrealized_buy_amt = unrealized_qty * avg_buy_price
            unrealized_curr_amt = unrealized_qty * current_price if current_price is not None else None
            unrealized_pl_value = (unrealized_curr_amt - unrealized_buy_amt) if (current_price is not None) else None
            unrealized_pl_pct = ((current_price - avg_buy_price) / avg_buy_price * 100) if (current_price is not None and avg_buy_price) else None
            result_unrealized.append({
                "Ticker": ticker,
                "Quantity": int(unrealized_qty),
                "Average Buy Price": avg_buy_price,
                "Current Price": current_price if current_price is not None else "N/A",
                "Profit/Loss Value": unrealized_pl_value if unrealized_pl_value is not None else "N/A",
                "Profit/Loss %": unrealized_pl_pct if unrealized_pl_pct is not None else "N/A"
            })

    realized_df = pd.DataFrame(result_realized)
    unrealized_df = pd.DataFrame(result_unrealized)
    total_realized = realized_df["Profit/Loss Value"].sum() if not realized_df.empty else 0
    total_unrealized = unrealized_df["Profit/Loss Value"].replace("N/A", 0).astype(float).sum() if not unrealized_df.empty else 0

    return realized_df, unrealized_df, total_realized, total_unrealized
