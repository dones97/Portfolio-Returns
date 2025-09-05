import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob
import numpy as np
import datetime
import altair as alt

# --- CONFIG ---
MAPPINGS_CSV = "ticker_mappings.csv"
TRADE_REPORTS_DIR = "trade_reports"
PORTFOLIO_HISTORY_CSV = "portfolio_history.csv"

# --- TICKER MAPPING / IO ---

def load_user_mappings():
    if os.path.exists(MAPPINGS_CSV):
        df = pd.read_csv(MAPPINGS_CSV, dtype=str)
        df = df.dropna(subset=["scrip_code", "yahoo_ticker"])
        return dict(zip(df["scrip_code"].str.strip(), df["yahoo_ticker"].str.strip()))
    return {}

def save_user_mappings(mapping_dict):
    df = pd.DataFrame([(code, ticker) for code, ticker in mapping_dict.items()],
                      columns=["scrip_code", "yahoo_ticker"])
    df.to_csv(MAPPINGS_CSV, index=False)

def default_bse_mapping(scrip_code):
    if pd.isnull(scrip_code):
        return None
    s = str(scrip_code).strip()
    if s.endswith(".0"):
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
    code = str(row.get("scrip_code", "")).strip()
    if code in user_mappings:
        return user_mappings[code]
    default_map = default_bse_mapping(code)
    if default_map and yahoo_ticker_valid(default_map):
        return default_map
    return ""

def read_trade_report(file):
    df = pd.read_excel(file, dtype=str)
    colmap = {
        "Scrip Code": "scrip_code", "Scrip_Code": "scrip_code", "scrip_code": "scrip_code", "scrip code": "scrip_code",
        "Company": "company_name", "company_name": "company_name", "company": "company_name",
        "Quantity": "quantity", "Price": "price", "Side": "side", "Buy/Sell": "side", "Date": "date", "date": "date"
    }
    df = df.rename(columns={c: colmap[c] for c in df.columns if c in colmap})
    if "scrip_code" in df.columns:
        df["scrip_code"] = df["scrip_code"].astype(str).str.strip()
    if "company_name" in df.columns:
        df["company_name"] = df["company_name"].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
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

# --- FORMATTING ---

def inr_format(amount):
    try:
        amount = int(round(float(amount)))
    except:
        return "₹0"
    neg = amount < 0
    s = str(abs(amount))
    if len(s) <= 3:
        base = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        base = ",".join(reversed(parts)) + "," + last3
    return f"{'-' if neg else ''}₹{base}"

def color_pct_html(pct):
    try:
        pct = float(pct)
    except:
        return "<span style='color:gray;'>N/A</span>"
    color = "green" if pct >= 0 else "red"
    return f"<span style='color:{color}; font-weight:bold'>{round(pct,2)}%</span>"

# --- PRICE LOOKUP (cached) ---

_price_cache = {}

def get_prev_trading_price(ticker, target_date):
    if not isinstance(target_date, (pd.Timestamp, datetime.date, datetime.datetime)):
        target_date = pd.to_datetime(target_date)
    target_day = pd.Timestamp(target_date).normalize()
    key = ("hist", ticker, target_day.strftime("%Y-%m-%d"))
    if key in _price_cache:
        return _price_cache[key]
    # wider window for safety
    start = (target_day - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
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

# --- REALIZED / UNREALIZED (average-cost) ---

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

        unrealized_qty = total_buy_qty - total_sell_qty
        if unrealized_qty > 0:
            current_price = get_current_price(ticker)
            unrealized_buy_amt = unrealized_qty * avg_buy_price
            unrealized_curr_amt = unrealized_qty * current_price if current_price is not None else None
            unrealized_pl_value = (unrealized_curr_amt - unrealized_buy_amt) if current_price is not None else None
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

# --- XIRR cashflows ---

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

# --- Realized by year (avg-cost sim) ---

def compute_realized_pl_by_year(history_df):
    df = history_df.copy().sort_values("date")
    realized_by_year = {}
    pos = {}
    for _, r in df.iterrows():
        ticker = r.get("yahoo_ticker", "")
        if not ticker or pd.isna(ticker):
            continue
        side = str(r["side"]).strip().lower()
        qty = float(r["quantity"])
        price = float(r["price"])
        year = pd.to_datetime(r["date"]).year
        if ticker not in pos:
            pos[ticker] = {"qty": 0.0, "cost": 0.0}
        if side == "buy":
            pos[ticker]["qty"] += qty
            pos[ticker]["cost"] += qty * price
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

# --- Fallback price on a date: Yahoo -> last trade price ---

def get_price_for_date_with_fallback(ticker, target_date, history_df):
    price = get_prev_trading_price(ticker, target_date)
    if price is not None:
        return price, "yahoo"
    if history_df is not None:
        mask = (history_df["yahoo_ticker"] == ticker) & (history_df["date"] <= pd.to_datetime(target_date))
        trades = history_df.loc[mask].sort_values("date")
        if not trades.empty:
            last_price = float(trades.iloc[-1]["price"])
            return last_price, "last_trade"
    return None, None

# --- Calendar-year metrics from trades (robust) ---

def compute_yearly_metrics_from_trades(history_df):
    if history_df.empty:
        return [], []

    history_df = history_df.copy().sort_values("date")
    history_df["date"] = pd.to_datetime(history_df["date"])
    first_year = history_df["date"].dt.year.min()
    last_year = history_df["date"].dt.year.max()
    years = list(range(first_year, last_year + 1))

    realized_by_year = compute_realized_pl_by_year(history_df)
    missing_prices = []
    results = []
    today = pd.Timestamp(datetime.date.today())

    def holdings_as_of(dt, inclusive=False):
        if inclusive:
            mask = history_df["date"] <= pd.to_datetime(dt)
        else:
            mask = history_df["date"] < pd.to_datetime(dt)
        sub = history_df.loc[mask].dropna(subset=["yahoo_ticker"])
        net = {}
        for t, grp in sub.groupby("yahoo_ticker"):
            buy_qty = grp[grp["side"].str.lower() == "buy"]["quantity"].astype(float).sum()
            sell_qty = grp[grp["side"].str.lower() == "sell"]["quantity"].astype(float).sum()
            net[t] = buy_qty - sell_qty
        return net

    for y in years:
        start_dt = pd.Timestamp(datetime.date(y, 1, 1))
        end_dt = pd.Timestamp(datetime.date(y, 12, 31))
        end_eff = min(end_dt, today)

        b_hold = holdings_as_of(start_dt, inclusive=False)
        e_hold = holdings_as_of(end_eff, inclusive=True)

        BMV = 0.0
        EMV = 0.0

        for t, qty in b_hold.items():
            if qty == 0:
                continue
            price, _ = get_price_for_date_with_fallback(t, start_dt, history_df)
            if price is None:
                missing_prices.append((t, start_dt.date()))
                continue
            BMV += qty * price

        for t, qty in e_hold.items():
            if qty == 0:
                continue
            price, _ = get_price_for_date_with_fallback(t, end_eff, history_df)
            if price is None:
                missing_prices.append((t, end_eff.date()))
                continue
            EMV += qty * price

        mask_year = (history_df["date"] >= start_dt) & (history_df["date"] <= end_eff)
        df_year = history_df.loc[mask_year]
        net_flows = 0.0
        for _, r in df_year.iterrows():
            amt = float(r["quantity"]) * float(r["price"])
            if str(r["side"]).strip().lower().startswith("buy"):
                net_flows += amt
            else:
                net_flows -= amt

        if abs(net_flows) < 1e-9 and abs(BMV) < 1e-9 and abs(EMV) < 1e-9:
            continue

        total_return_amt = EMV - BMV - net_flows
        avg_portfolio_val = (BMV + EMV) / 2.0 if (BMV + EMV) != 0 else None

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

# --- Nifty 50 yearly returns for given years (single fetch; same index as headline: ^NSEI) ---

def compute_nifty_yearly_returns(years, first_trade=None):
    today = pd.Timestamp(datetime.date.today())
    min_year = min(years)
    max_year = max(years)
    start_all = pd.Timestamp(datetime.date(min_year, 1, 1)) - pd.Timedelta(days=10)
    end_all = min(pd.Timestamp(datetime.date(max_year, 12, 31)), today) + pd.Timedelta(days=1)
    hist = yf.Ticker("^NSEI").history(start=start_all.strftime("%Y-%m-%d"), end=end_all.strftime("%Y-%m-%d"))
    if hist is None or hist.empty:
        return {}
    closes = hist["Close"]
    # Make closes index timezone-naive
    closes.index = closes.index.tz_localize(None)
    out = {}
    for i, year in enumerate(sorted(years)):
        # For the first year, start with first_trade date
        if i == 0 and first_trade is not None:
            start_dt = pd.Timestamp(first_trade).tz_localize(None)
        else:
            start_dt = pd.Timestamp(datetime.date(year, 1, 1)).tz_localize(None)
        if year == today.year:
            end_dt = today.tz_localize(None)
        else:
            end_dt = pd.Timestamp(datetime.date(year, 12, 31)).tz_localize(None)
        # Get last close on or before start
        start_prices = closes[closes.index <= start_dt]
        end_prices = closes[closes.index <= end_dt]
        if not start_prices.empty and not end_prices.empty:
            start_val = start_prices.iloc[-1]
            end_val = end_prices.iloc[-1]
            out[year] = (end_val / start_val - 1.0) * 100.0
        else:
            out[year] = None
    return out
    
# --- P&L timeline (cumulative total = realized + unrealized estimate using last trade prices) ---

def pnl_over_time(history_df):
    df = history_df.copy().sort_values("date")
    pos = {}  # ticker -> dict(qty, cost_sum, last_price)
    realized_cum = 0.0
    rows = []

    for _, r in df.iterrows():
        t = r.get("yahoo_ticker", "")
        if not t or pd.isna(t):
            continue
        side = str(r["side"]).strip().lower()
        qty = float(r["quantity"])
        price = float(r["price"])
        dt = pd.to_datetime(r["date"])

        if t not in pos:
            pos[t] = {"qty": 0.0, "cost": 0.0, "last_price": None}

        if side == "buy":
            pos[t]["qty"] += qty
            pos[t]["cost"] += qty * price
            pos[t]["last_price"] = price
        else:
            avg_cost = (pos[t]["cost"] / pos[t]["qty"]) if pos[t]["qty"] > 0 else 0.0
            realized = qty * (price - avg_cost)
            realized_cum += realized
            remaining_qty = pos[t]["qty"] - qty
            if remaining_qty <= 0:
                pos[t]["qty"] = 0.0
                pos[t]["cost"] = 0.0
            else:
                pos[t]["qty"] = remaining_qty
                pos[t]["cost"] = avg_cost * remaining_qty
            pos[t]["last_price"] = price

        unrealized = 0.0
        for k, p in pos.items():
            if p["qty"] > 0 and p["last_price"] is not None:
                avg_c = (p["cost"] / p["qty"]) if p["qty"] > 0 else 0.0
                unrealized += p["qty"] * (p["last_price"] - avg_c)

        total = realized_cum + unrealized
        rows.append({"date": dt, "realized_cum": realized_cum, "unrealized_est": unrealized, "total_pnl": total})

    if not rows:
        return pd.DataFrame(columns=["date", "realized_cum", "unrealized_est", "total_pnl"])

    out = pd.DataFrame(rows)
    out = out.groupby("date", as_index=False).agg(
        realized_cum=("realized_cum", "max"),
        unrealized_est=("unrealized_est", "last"),
        total_pnl=("total_pnl", "last"),
    ).sort_values("date").reset_index(drop=True)
    return out

# --- STREAMLIT UI ---

st.set_page_config(page_title="Portfolio Returns Tracker", layout="wide")
st.title("Portfolio Returns")

tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
    st.markdown("Upload or select trade reports, map scrip codes to Yahoo tickers, and save the portfolio history.", unsafe_allow_html=True)

    user_mappings = load_user_mappings()
    uploaded_files = st.file_uploader("Upload trade reports (Excel)", type=["xlsx"], accept_multiple_files=True)

    repo_reports = get_repository_trade_reports()
    repo_file_names = [fname for fname, _ in repo_reports] if repo_reports else []
    selected_repo_files = st.multiselect("Select repository trade reports to include", repo_file_names, default=[])

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

        trades["scrip_code"] = trades["scrip_code"].astype(str).str.strip()
        trades["yahoo_ticker"] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)

        unmapped = trades[trades["yahoo_ticker"] == ""][["scrip_code", "company_name"]].drop_duplicates()
        st.subheader("Unmapped Scrip Codes")
        if not unmapped.empty:
            edited = st.data_editor(unmapped.assign(yahoo_ticker=""), key="edit_tickers", num_rows="dynamic")
            if st.button("Update Mapping"):
                new_mappings = {}
                failed = []
                for _, row in edited.iterrows():
                    code = str(row["scrip_code"]).strip()
                    ticker = str(row["yahoo_ticker"]).strip().upper()
                    if ticker:
                        if yahoo_ticker_valid(ticker):
                            new_mappings[code] = ticker
                        else:
                            failed.append((code, ticker))
                if new_mappings:
                    user_mappings.update(new_mappings)
                    save_user_mappings(user_mappings)
                    st.success("Saved new mappings to ticker_mappings.csv. Reload to apply.")
                if failed:
                    for code, ticker in failed:
                        st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo and was not saved.")
        else:
            st.success("All scrip codes mapped.")

        mapped = trades[trades["yahoo_ticker"] != ""].copy()
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
                st.success("Saved portfolio_history.csv.")
    else:
        st.info("Upload trades or pick from repository to begin.")

    st.markdown("---")
    st.subheader("Current Mappings")
    mapping_df = pd.DataFrame([{"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()])
    st.dataframe(mapping_df)

with tabs[1]:
    st.header("Portfolio Returns")

    if not os.path.exists(PORTFOLIO_HISTORY_CSV):
        st.warning("No portfolio_history.csv found. Save mapped trades first.")
        st.stop()

    history_df = pd.read_csv(PORTFOLIO_HISTORY_CSV)
    if "date" not in history_df.columns:
        st.error("portfolio_history.csv missing 'date' column.")
        st.stop()
    history_df["date"] = pd.to_datetime(history_df["date"])
    required_cols = {"yahoo_ticker", "side", "quantity", "price", "date"}
    if not required_cols.issubset(set(history_df.columns)):
        st.error(f"portfolio_history.csv missing columns: {required_cols - set(history_df.columns)}")
        st.stop()

    # Headline metrics
    realized_df, unrealized_df, total_realized, total_unrealized = calc_realized_unrealized_avgcost(history_df)

    curr_value = unrealized_df.apply(
        lambda row: float(row["Quantity"]) * (float(row["Current Price"]) if row["Current Price"] != "N/A" else 0), axis=1
    ).sum() if not unrealized_df.empty else 0.0

    buys_df = history_df[history_df["side"].str.lower().str.startswith("buy")]
    sells_df = history_df[history_df["side"].str.lower().str.startswith("sell")]
    total_buys = (buys_df["quantity"].astype(float) * buys_df["price"].astype(float)).sum()
    total_sells = (sells_df["quantity"].astype(float) * sells_df["price"].astype(float)).sum()
    net_invested = total_buys - total_sells

    total_return_amt = total_realized + total_unrealized
    total_return_pct = (total_return_amt / net_invested * 100.0) if net_invested > 0 else None
    realized_pct = (total_realized / net_invested * 100.0) if net_invested > 0 else None
    unrealized_pct = (total_unrealized / net_invested * 100.0) if net_invested > 0 else None

    # XIRR
    cashflows = build_cashflows_from_history(history_df)
    irr_str = "<span style='color:gray;'>N/A</span>"
    if cashflows:
        amounts = [amt for amt, _ in cashflows]
        dates = [dt for _, dt in cashflows]
        today = pd.Timestamp(datetime.date.today())
        amounts.append(curr_value)
        dates.append(today)
        try:
            try:
                import numpy_financial as npf
                irr_val = npf.xirr(amounts, dates) * 100.0
            except Exception:
                # simple Newton fallback
                d0 = pd.to_datetime(dates[0])
                times = np.array([(pd.to_datetime(d) - d0).days / 365.0 for d in dates], dtype=float)
                amts = np.array(amounts, dtype=float)
                def f(r): return np.sum(amts / ((1.0 + r) ** times))
                def fp(r): return np.sum(-amts * times / ((1.0 + r) ** (times + 1.0)))
                r = 0.1
                for _ in range(200):
                    val = f(r); der = fp(r)
                    if der == 0: break
                    step = val / der
                    r -= step
                    if abs(step) < 1e-6: 
                        irr_val = r * 100.0
                        break
            irr_str = color_pct_html(irr_val) if 'irr_val' in locals() and irr_val is not None else irr_str
        except Exception:
            irr_str = "<span style='color:gray;'>N/A</span>"

    # Nifty 50 CAGR headline (same index we'll use for yearly)
    first_trade = history_df["date"].min()
    last_trade = history_df["date"].max()
    nifty_cagr_val = None
    try:
        if pd.notnull(first_trade) and pd.notnull(last_trade) and (last_trade > first_trade):
            hist = yf.Ticker("^NSEI").history(start=first_trade.strftime("%Y-%m-%d"),
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

    st.subheader("Portfolio Headline Performance")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown("**Annualized (XIRR)**")
        st.markdown(irr_str, unsafe_allow_html=True)
    with c2:
        st.markdown("**Current Portfolio Value**")
        st.markdown(f"**{inr_format(curr_value)}**")
    with c3:
        st.markdown("**Nifty 50 CAGR (period)**")
        st.markdown(color_pct_html(nifty_cagr_val) if nifty_cagr_val is not None else "<span style='color:gray;'>N/A</span>", unsafe_allow_html=True)

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

    # --- Calendar-year comparison: grouped bars (Portfolio vs Nifty 50) ---
    st.subheader("Portfolio vs Nifty 50: Yearly Comparison (calendar years)")

    per_year_results, missing_prices = compute_yearly_metrics_from_trades(history_df)
    if not per_year_results:
        st.info("No per-year results could be computed from trade history.")
    else:
        per_year_df = pd.DataFrame(per_year_results)

        years_list = per_year_df["year"].astype(int).tolist()
        nifty_map = compute_nifty_yearly_returns(years_list, first_trade=history_df["date"].min())
        
        chart_rows = []
        for yr in years_list:
            port_row = per_year_df[per_year_df["year"] == yr]
            port_val = port_row.iloc[0]["return_pct"] if not port_row.empty else None
            nifty_val = nifty_map.get(yr, None)
            chart_rows.append({
                "year": yr, "series": "Portfolio",
                "return_pct": float(port_val) if port_val is not None and not pd.isna(port_val) else np.nan
            })
            chart_rows.append({
                "year": yr, "series": "Nifty 50 (^NSEI)",
                "return_pct": float(nifty_val) if nifty_val is not None and not pd.isna(nifty_val) else np.nan
            })

        chart_df = pd.DataFrame(chart_rows)
        chart_df = chart_df.dropna(subset=["return_pct"])
        chart_df["label"] = chart_df["return_pct"].round(1).astype(str) + "%"

        if not chart_df.empty:
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("year:O", title="Year", sort=sorted(chart_df["year"].unique())),
                y=alt.Y("return_pct:Q", title="Return %"),
                color=alt.Color("series:N", title="Series"),
                xOffset=alt.XOffset("series:N"),
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("return_pct:Q", title="Return %", format=".2f"),
                ],
            )
        
            # Overlay positive labels in green
            text_pos = alt.Chart(chart_df[chart_df["return_pct"] >= 0]).mark_text(
                dy=-10, fontSize=13, color="green"
            ).encode(
                x=alt.X("year:O", sort=sorted(chart_df["year"].unique())),
                y=alt.Y("return_pct:Q"),
                text=alt.Text("label:N"),
                xOffset=alt.XOffset("series:N"),
            )
        
            # Overlay negative labels in red
            text_neg = alt.Chart(chart_df[chart_df["return_pct"] < 0]).mark_text(
                dy=12, fontSize=13, color="red"
            ).encode(
                x=alt.X("year:O", sort=sorted(chart_df["year"].unique())),
                y=alt.Y("return_pct:Q"),
                text=alt.Text("label:N"),
                xOffset=alt.XOffset("series:N"),
            )
        
            chart = (bars + text_pos + text_neg).properties(height=420)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No yearly returns available to plot.")
    
        # Data table under the chart: includes both Portfolio and Nifty returns for each year
        display_rows = []
        for yr in years_list:
            port_row = per_year_df[per_year_df["year"] == yr]
            nifty_val = nifty_map.get(yr, None)
            display_rows.append({
                "Year": int(yr),
                "BMV": inr_format(port_row.iloc[0]["BMV"]) if not port_row.empty else "N/A",
                "EMV": inr_format(port_row.iloc[0]["EMV"]) if not port_row.empty else "N/A",
                "Net Flows (₹)": inr_format(port_row.iloc[0]["net_flows"]) if not port_row.empty else "N/A",
                "Total Return (₹)": inr_format(port_row.iloc[0]["total_return_amt"]) if not port_row.empty else "N/A",
                "Portfolio Return (%)": f"{round(port_row.iloc[0]['return_pct'],2)}%" if not port_row.empty and port_row.iloc[0]["return_pct"] is not None and not pd.isna(port_row.iloc[0]["return_pct"]) else "N/A",
                "Nifty 50 Return (%)": f"{round(nifty_val,2)}%" if nifty_val is not None and not pd.isna(nifty_val) else "N/A",
                "Realized (₹)": inr_format(port_row.iloc[0]["realized_amt"]) if not port_row.empty else "N/A",
                "Unrealized (₹)": inr_format(port_row.iloc[0]["unrealized_amt"]) if not port_row.empty else "N/A",
                "Avg Portfolio Value (₹)": inr_format(port_row.iloc[0]["avg_portfolio_val"]) if not port_row.empty and port_row.iloc[0]["avg_portfolio_val"] else "N/A",
            })
        year_table_df = pd.DataFrame(display_rows)
        st.dataframe(year_table_df)

        if missing_prices:
            st.warning("Used last trade price fallback when Yahoo historical prices were missing for some tickers/dates.")
            st.write(sorted(list(set(missing_prices)))[:20])

    st.markdown("---")
    
    # --- Cumulative portfolio profit over time (Total = realized + unrealized estimate) ---
    st.subheader("Cumulative Portfolio Profit Over Time (INR)")

    pnl_ts = pnl_over_time(history_df)
    if pnl_ts.empty:
        st.info("No trades found to build profit timeline.")
    else:
        axis_indian = alt.Axis(
            title="Cumulative Profit (₹)",
            labelExpr=(
                "datum.value >= 1e7 ? format(datum.value/1e7, ',.1f') + ' Cr' : "
                "datum.value >= 1e5 ? format(datum.value/1e5, ',.1f') + ' L' : "
                "datum.value >= 1e3 ? format(datum.value/1e3, ',.0f') + ' K' : format(datum.value, ',')"
            )
        )

        stacked_df = pnl_ts.copy()
        stacked_df["date"] = pd.to_datetime(stacked_df["date"])
        stacked_df["Realized"] = stacked_df["realized_cum"]
        stacked_df["Unrealized"] = stacked_df["unrealized_est"]

        area_data = stacked_df.melt(
            id_vars=["date"],
            value_vars=["Realized", "Unrealized"],
            var_name="series",
            value_name="value"
        )
        area_data["series"] = pd.Categorical(area_data["series"], categories=["Realized", "Unrealized"], ordered=True)

        color_scale = alt.Scale(domain=["Realized", "Unrealized"], range=["#d62728", "#1f77b4"])

        stacked_area = alt.Chart(area_data).mark_area().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", axis=axis_indian, stack="zero"),
            color=alt.Color("series:N", title="Series", scale=color_scale, sort=["Realized", "Unrealized"]),
            order=alt.Order('series', sort='ascending'),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Cumulative (₹)", format=",.0f"),
            ]
        )

        # Nifty 50 comparison line WITH LEGEND
        trade_dates = pnl_ts["date"].sort_values().unique()
        first_trade_dt = pd.to_datetime(history_df["date"].min())
        last_trade_dt = pd.to_datetime(history_df["date"].max())
        nifty_hist = yf.Ticker("^NSEI").history(
            start=first_trade_dt.strftime("%Y-%m-%d"),
            end=(last_trade_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        nifty_hist.index = nifty_hist.index.tz_localize(None)
        nifty_closes = nifty_hist["Close"]

        cashflows = build_cashflows_from_history(history_df)
        cf_by_date = {}
        for amt, dt in cashflows:
            d = pd.Timestamp(dt).tz_localize(None)
            cf_by_date.setdefault(d, 0)
            cf_by_date[d] += amt

        invested = 0.0
        units = 0.0
        nifty_pnl_rows = []
        for dt in trade_dates:
            dt_naive = pd.Timestamp(dt).tz_localize(None)
            price_series = nifty_closes[nifty_closes.index <= dt_naive]
            if price_series.empty:
                continue
            price = price_series.iloc[-1]
            cf = cf_by_date.get(dt_naive, 0)
            if cf < 0:  # buy
                units += (-cf) / price
                invested += -cf
            elif cf > 0:  # sell
                units -= (cf / price)
                invested -= cf
            curr_value = units * price
            cumulative_pnl = curr_value - invested
            nifty_pnl_rows.append({
                "date": dt_naive,
                "Nifty_Cum_PNL": cumulative_pnl,
                "series": "Nifty 50 Simulated"
            })

        # Combine for legend
        nifty_pnl_df = pd.DataFrame(nifty_pnl_rows)
        if not nifty_pnl_df.empty:
            # Use color for legend, but keep line distinct
            all_series_df = pd.concat([
                area_data,
                nifty_pnl_df.rename(columns={"Nifty_Cum_PNL": "value"})[["date", "series", "value"]]
            ], ignore_index=True)

            chart_area = alt.Chart(area_data).mark_area().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", axis=axis_indian, stack="zero"),
                color=alt.Color("series:N", title="Series", scale=alt.Scale(
                    domain=["Realized", "Unrealized", "Nifty 50 Simulated"],
                    range=["#d62728", "#1f77b4", "green"]
                )),
                order=alt.Order('series', sort='ascending'),
                opacity=alt.condition(
                    alt.datum.series == "Nifty 50 Simulated",
                    alt.value(0),  # Don't show area for Nifty
                    alt.value(0.7)
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Cumulative (₹)", format=",.0f"),
                ]
            )

            chart_nifty = alt.Chart(nifty_pnl_df).mark_line(
                strokeDash=[7,3], color="green", strokeWidth=2
            ).encode(
                x=alt.X("date:T"),
                y=alt.Y("Nifty_Cum_PNL:Q", axis=axis_indian),
                color=alt.Color("series:N", title="Series", scale=alt.Scale(
                    domain=["Realized", "Unrealized", "Nifty 50 Simulated"],
                    range=["#d62728", "#1f77b4", "green"]
                )),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Nifty_Cum_PNL:Q", title="Nifty Cumulative P&L (₹)", format=",.0f"),
                    alt.Tooltip("series:N", title="Series"),
                ]
            )

            chart_pnl = (chart_area + chart_nifty).properties(height=320)
        else:
            chart_pnl = stacked_area.properties(height=320)

        st.altair_chart(chart_pnl, use_container_width=True)

        with st.expander("Profit timeline table"):
            rt = pnl_ts.copy()
            rt["date"] = rt["date"].dt.date
            rt["total_pnl_INR"] = rt["total_pnl"].apply(inr_format)
            rt["realized_cum_INR"] = rt["realized_cum"].apply(inr_format)
            rt["unrealized_est_INR"] = rt["unrealized_est"].apply(inr_format)
            st.dataframe(rt[["date", "total_pnl_INR", "realized_cum_INR", "unrealized_est_INR"]])
   
    st.markdown("---")

        # --- Portfolio Value Over Time (Stacked: Invested + Profit), vs Nifty 50 ---
    st.subheader("Portfolio Value Over Time (INR)")

    if pnl_ts.empty:
        st.info("No trades found to build portfolio value timeline.")
    else:
        # Calculate invested amount over time
        history_df_sorted = history_df.sort_values("date")
        cumsum_cf = []
        net_invested = 0.0
        cf_dates = []
        for _, r in history_df_sorted.iterrows():
            amt = float(r["quantity"]) * float(r["price"])
            if str(r["side"]).strip().lower().startswith("buy"):
                net_invested += amt
            else:
                net_invested -= amt
            cf_dates.append(r["date"])
            cumsum_cf.append(net_invested)
        # Map to unique dates for merge with pnl_ts
        invested_ts = pd.DataFrame({"date": cf_dates, "invested": cumsum_cf}).drop_duplicates("date")
        # Merge with pnl_ts to align dates
        value_ts = pd.merge(pnl_ts, invested_ts, on="date", how="left").sort_values("date")
        value_ts["invested"] = value_ts["invested"].ffill().fillna(0.0)
        value_ts["portfolio_value"] = value_ts["invested"] + value_ts["total_pnl"]
        value_ts["profit"] = value_ts["total_pnl"]
        # Prepare stacked area data
        area_value = value_ts.melt(
            id_vars=["date"], 
            value_vars=["invested", "profit"],
            var_name="series", 
            value_name="amount"
        )
        area_value["series"] = pd.Categorical(area_value["series"], categories=["invested", "profit"], ordered=True)

        color_scale_val = alt.Scale(domain=["invested", "profit"], range=["#d3d3d3", "#86c06c"])
        axis_indian_val = alt.Axis(
            title="Portfolio Value (₹)",
            labelExpr=(
                "datum.value >= 1e7 ? format(datum.value/1e7, ',.1f') + ' Cr' : "
                "datum.value >= 1e5 ? format(datum.value/1e5, ',.1f') + ' L' : "
                "datum.value >= 1e3 ? format(datum.value/1e3, ',.0f') + ' K' : format(datum.value, ',')"
            )
        )

        stacked_val = alt.Chart(area_value).mark_area().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount:Q", stack="zero", axis=axis_indian_val),
            color=alt.Color("series:N", title="Component", scale=color_scale_val, sort=["invested", "profit"]),
            order=alt.Order('series', sort='ascending'),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Component"),
                alt.Tooltip("amount:Q", title="Amount (₹)", format=",.0f"),
            ]
        )

        # Nifty 50 "all invested" simulated value line (use same logic as in profit plot)
        nifty_val_rows = []
        invested_nifty = 0.0
        units_nifty = 0.0
        for dt in value_ts["date"]:
            dt_naive = pd.Timestamp(dt).tz_localize(None)
            price_series = nifty_closes[nifty_closes.index <= dt_naive]
            if price_series.empty:
                continue
            price = price_series.iloc[-1]
            cf = cf_by_date.get(dt_naive, 0)
            if cf < 0:  # Buy
                units_nifty += (-cf) / price
                invested_nifty += -cf
            elif cf > 0:  # Sell
                units_nifty -= (cf / price)
                invested_nifty -= cf
            curr_value = units_nifty * price
            nifty_val_rows.append({
                "date": dt_naive,
                "Nifty_Portfolio_Value": curr_value,
                "series": "Nifty 50 Simulated"
            })
        nifty_val_df = pd.DataFrame(nifty_val_rows)

        # Compose final chart with legend
        if not nifty_val_df.empty:
            chart_area_val = alt.Chart(area_value).mark_area().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("amount:Q", axis=axis_indian_val, stack="zero"),
                color=alt.Color("series:N", title="Component", scale=alt.Scale(
                    domain=["invested", "profit", "Nifty 50 Simulated"],
                    range=["#d3d3d3", "#86c06c", "green"]
                )),
                order=alt.Order('series', sort='ascending'),
                opacity=alt.condition(
                    alt.datum.series == "Nifty 50 Simulated",
                    alt.value(0),  # Don't show area for Nifty
                    alt.value(0.7)
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Component"),
                    alt.Tooltip("amount:Q", title="Amount (₹)", format=",.0f"),
                ]
            )
            chart_nifty_val = alt.Chart(nifty_val_df).mark_line(
                strokeDash=[7,3], color="green", strokeWidth=2
            ).encode(
                x=alt.X("date:T"),
                y=alt.Y("Nifty_Portfolio_Value:Q", axis=axis_indian_val),
                color=alt.Color("series:N", title="Series", scale=alt.Scale(
                    domain=["invested", "profit", "Nifty 50 Simulated"],
                    range=["#d3d3d3", "#86c06c", "green"]
                )),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Nifty_Portfolio_Value:Q", title="Nifty Portfolio Value (₹)", format=",.0f"),
                    alt.Tooltip("series:N", title="Series"),
                ]
            )
            chart_val = (chart_area_val + chart_nifty_val).properties(height=320)
        else:
            chart_val = stacked_val.properties(height=320)
        st.altair_chart(chart_val, use_container_width=True)

        with st.expander("Portfolio value timeline table"):
            vt = value_ts.copy()
            vt["date"] = vt["date"].dt.date
            vt["portfolio_value_INR"] = vt["portfolio_value"].apply(inr_format)
            vt["invested_INR"] = vt["invested"].apply(inr_format)
            vt["profit_INR"] = vt["profit"].apply(inr_format)
            st.dataframe(vt[["date", "portfolio_value_INR", "invested_INR", "profit_INR"]])

    # --- Realized / Unrealized tables (no styled variants) ---
    st.subheader("Realized Returns (Average Costing)")
    if not realized_df.empty:
        rf = realized_df.copy()
        rf["Profit/Loss Value INR"] = rf["Profit/Loss Value"].apply(inr_format)
        rf["Average Buy Price INR"] = rf["Average Buy Price"].apply(inr_format)
        rf["Average Sell Price INR"] = rf["Average Sell Price"].apply(inr_format)
        rf["Profit/Loss %"] = rf["Profit/Loss %"].round(2)
        rf_display = rf[["Ticker", "Quantity", "Average Buy Price INR", "Average Sell Price INR",
                         "Profit/Loss Value INR", "Profit/Loss %"]]
        st.dataframe(rf_display)
        st.markdown(f"**Total Realized Profit/Loss:** {inr_format(total_realized)}")
    else:
        st.info("No realized trades or profit/loss yet.")

    st.subheader("Unrealized Returns (Average Costing)")
    if not unrealized_df.empty:
        uf = unrealized_df.copy()
        uf["Profit/Loss Value INR"] = uf["Profit/Loss Value"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
        uf["Average Buy Price INR"] = uf["Average Buy Price"].apply(inr_format)
        uf["Current Price INR"] = uf["Current Price"].apply(lambda x: inr_format(x) if x != "N/A" else "N/A")
        # numeric for sorting
        def to_num(x):
            try:
                return float(x)
            except:
                return np.nan
        uf["Profit/Loss %"] = uf["Profit/Loss %"].apply(lambda x: round(float(x), 2) if x != "N/A" else np.nan)
        uf_display = uf[["Ticker", "Quantity", "Average Buy Price INR", "Current Price INR",
                         "Profit/Loss Value INR", "Profit/Loss %"]]
        st.dataframe(uf_display)
        st.markdown(f"**Total Unrealized Profit/Loss:** {inr_format(total_unrealized)}")
    else:
        st.info("No unrealized holdings.")
