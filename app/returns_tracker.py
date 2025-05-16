import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress

st.set_page_config(page_title="Returns Tracker", layout="wide")

# --- SETTINGS ---
TRADE_REPORTS_DIR = Path(__file__).parent.parent / "trade_reports"
INDEX_TICKERS = {
    "NSE500": "^CRSLDX",      # Nifty 500 (may require adjustment)
    "NSE50": "^NSEI",         # Nifty 50
    "NSEMidcap": "^NSEMDCP50", # Nifty Midcap 50 (or best available)
    "NSESmallcap": "^NSESMCP", # Nifty Smallcap (or best available)
}
INDEX_LABELS = {
    "^CRSLDX": "NSE 500",
    "^NSEI": "Nifty 50",
    "^NSEMDCP50": "Nifty Midcap 50",
    "^NSESMCP": "Nifty Smallcap"
}

# --- FUNCTIONS ---

def read_trade_report(file):
    if str(file).endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    # Standardize columns
    df = df.rename(columns=lambda x: x.strip().lower())
    for col in df.columns:
        if "scrip" in col and "code" in col:
            df.rename(columns={col: "scrip_code"}, inplace=True)
        if "company" in col and "name" in col:
            df.rename(columns={col: "company_name"}, inplace=True)
        if "isin" in col:
            df.rename(columns={col: "isin"}, inplace=True)
    return df

def try_yahoo_bse_ticker(scrip_code):
    """
    Given a scrip code, returns its Yahoo Finance ticker if it exists (with .BO suffix).
    Handles numeric and alphanumeric scrip codes.
    """
    s = str(scrip_code).strip()
    if s.isdigit():
        ticker = f"{s.zfill(6)}.BO"
    else:
        ticker = f"{s}.BO"
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return ticker
    except Exception:
        pass
    return None

def try_yahoo_nse_ticker(nse_ticker):
    ticker = f"{nse_ticker.upper()}.NS"
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return ticker
    except Exception:
        pass
    return None

def load_all_trade_data(file_objs):
    trades = []
    for file in file_objs:
        df = read_trade_report(file)
        trades.append(df)
    if trades:
        return pd.concat(trades, ignore_index=True)
    return pd.DataFrame()

def realized_unrealized(trades, price_map, today):
    # Assumes trades DataFrame has columns: scrip_code, company_name, buy/sell, quantity, price, date
    trades = trades.copy()
    trades['date'] = pd.to_datetime(trades['date'], errors='coerce')
    trades = trades.dropna(subset=['date'])
    trades = trades.sort_values('date')
    summary = []
    for ticker in trades['yahoo_ticker'].dropna().unique():
        stk = trades[trades['yahoo_ticker'] == ticker]
        qty = 0
        realized = 0
        avg_cost = 0
        for _, row in stk.iterrows():
            if row['buy/sell'].lower().startswith('b'):
                total_cost = avg_cost * qty + row['quantity'] * row['price']
                qty += row['quantity']
                avg_cost = total_cost / qty if qty else 0
            else:
                realized += (row['price'] - avg_cost) * row['quantity']
                qty -= row['quantity']
        # Unrealized on remaining qty
        if ticker in price_map:
            last_price = price_map[ticker]
            unrealized = (last_price - avg_cost) * qty
        else:
            unrealized = np.nan
        summary.append({
            "yahoo_ticker": ticker,
            "company_name": stk['company_name'].iloc[0],
            "realized": realized,
            "unrealized": unrealized,
            "quantity": qty,
            "current_price": price_map.get(ticker, np.nan),
        })
    return pd.DataFrame(summary)

def calc_time_series_returns(trades, price_histories, freq="M"):
    """Returns a DataFrame with datetime index and portfolio value for each period."""
    # 1. Build a date range from first trade to now
    all_dates = pd.date_range(trades['date'].min(), datetime.today(), freq='D')
    portfolio = pd.DataFrame(index=all_dates)
    # 2. For each stock, build cumulative position and value
    for ticker, trades_stk in trades.groupby('yahoo_ticker'):
        if ticker not in price_histories:
            continue
        price_df = price_histories[ticker]
        # Fwd fill for missing dates
        price_df = price_df.reindex(all_dates).ffill()
        position = pd.Series(0, index=all_dates)
        for _, row in trades_stk.iterrows():
            sign = 1 if row['buy/sell'].lower().startswith('b') else -1
            position.loc[row['date']:] += sign * row['quantity']
        portfolio[ticker] = position * price_df['Close']
    # 3. Sum across stocks
    portfolio['portfolio_value'] = portfolio.sum(axis=1)
    returns = portfolio['portfolio_value'].resample(freq).last().pct_change().dropna()
    value = portfolio['portfolio_value'].resample(freq).last()
    return value, returns

def get_index_history(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def calc_alpha_beta(portfolio_returns, benchmark_returns):
    # Align
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    df = df.dropna()
    if df.shape[0] < 2:
        return np.nan, np.nan
    slope, intercept, r_value, p_value, std_err = linregress(df.iloc[:,1], df.iloc[:,0])
    alpha = intercept * 12  # annualized
    beta = slope
    return alpha, beta

def calc_var(returns, level=0.05):
    """Parametric VaR (95%)"""
    if len(returns) == 0:
        return np.nan
    return np.percentile(returns, 100 * level)

# --- BEGIN STREAMLIT APP ---

st.title("Indian Portfolio Returns Tracker")

# --- 1. Load repo trade reports ---
repo_files = list(TRADE_REPORTS_DIR.glob("*.xlsx")) + list(TRADE_REPORTS_DIR.glob("*.csv"))
use_repo = st.toggle("Reference prior trade reports from repo", value=True)
if use_repo and repo_files:
    st.write("Using trade reports from repository:", [f.name for f in repo_files])
else:
    repo_files = []

# --- 2. Upload new trade reports ---
uploaded_files = st.file_uploader("Upload trade report(s)", type=["xlsx", "csv"], accept_multiple_files=True)

# --- 3. Load and combine reports ---
files_to_use = repo_files + (uploaded_files if uploaded_files else [])
if not files_to_use:
    st.warning("Please upload a trade report or enable repo reference.")
    st.stop()

trades = load_all_trade_data(files_to_use)
if trades.empty:
    st.warning("No trades found.")
    st.stop()

# --- 4. Auto-mapping: scrip_code.BO ---
trades['scrip_code'] = trades['scrip_code'].astype(str).str.zfill(6)
trades['yahoo_ticker'] = trades['scrip_code'].apply(try_yahoo_bse_ticker)

# --- 5. Editable mapping for unmapped stocks ---
unmapped = trades[trades['yahoo_ticker'].isnull()][['scrip_code', 'company_name']].drop_duplicates()
if not unmapped.empty:
    st.info("Manual mapping required for the following stocks (enter NSE ticker without .NS):")
    unmapped['nse_ticker'] = ""
    edited = st.data_editor(unmapped, num_rows="dynamic", key="map")
    confirmed = st.button("Confirm Manual Mapping")
    if confirmed:
        for idx, row in edited.iterrows():
            if row['nse_ticker']:
                yahoo_ticker = try_yahoo_nse_ticker(row['nse_ticker'])
                if yahoo_ticker:
                    trades.loc[(trades['scrip_code'] == row['scrip_code']), 'yahoo_ticker'] = yahoo_ticker
                else:
                    st.error(f"Ticker {row['nse_ticker']}.NS not found.")
        st.experimental_rerun()
    else:
        st.stop()

# --- 6. Fetch latest prices ---
st.subheader("Fetching latest prices ...")
tickers = trades['yahoo_ticker'].dropna().unique().tolist()
price_map = {}
for ticker in tickers:
    data = yf.Ticker(ticker).history(period="1d")
    if not data.empty:
        price_map[ticker] = data['Close'].iloc[-1]
    else:
        price_map[ticker] = np.nan

# --- 7. Calculate realized/unrealized and summary ---
today = datetime.today()
summary = realized_unrealized(trades, price_map, today)
st.subheader("Portfolio Summary")
st.dataframe(summary[['company_name', 'yahoo_ticker', 'quantity', 'current_price', 'realized', 'unrealized']])

st.metric("Total Realized Gain/Loss", f"₹{summary['realized'].sum():,.0f}")
st.metric("Total Unrealized Gain/Loss", f"₹{summary['unrealized'].sum():,.0f}")

# --- 8. Monthly/Yearly Returns ---
st.subheader("Monthly and Yearly Returns")
# Fetch historical prices for all tickers
price_histories = {}
start_date = trades['date'].min()
for ticker in tickers:
    hist = yf.download(ticker, start=start_date, end=datetime.today())
    price_histories[ticker] = hist
# Monthly
value_monthly, returns_monthly = calc_time_series_returns(trades, price_histories, freq="M")
st.line_chart(value_monthly.rename("Portfolio Value (Monthly)"))
st.bar_chart(returns_monthly.rename("Monthly Return"))
# Yearly
value_yearly, returns_yearly = calc_time_series_returns(trades, price_histories, freq="Y")
st.line_chart(value_yearly.rename("Portfolio Value (Yearly)"))
st.bar_chart(returns_yearly.rename("Yearly Return"))

# --- 9. Benchmark Comparison ---
st.subheader("Compare with Index")
benchmarks = {}
for idx, ticker in INDEX_TICKERS.items():
    h = get_index_history(ticker, start=start_date, end=datetime.today())
    benchmarks[ticker] = h
for label, bench in benchmarks.items():
    if 'Close' in bench:
        st.line_chart(bench['Close'].rename(INDEX_LABELS.get(label,label)))

# --- 10. Alpha, Beta vs NSE500 (Nifty 500) ---
st.subheader("Portfolio Alpha/Beta vs NSE500")
bench500 = benchmarks.get(INDEX_TICKERS['NSE500'])
if bench500 is not None and 'Close' in bench500:
    # Benchmark returns
    bench_val = bench500['Close'].resample('M').last().pct_change().dropna()
    alpha, beta = calc_alpha_beta(returns_monthly, bench_val)
    st.metric("Alpha (annualized)", f"{alpha:.2%}" if not np.isnan(alpha) else "N/A")
    st.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A")

# --- 11. Value at Risk (VaR) ---
st.subheader("Portfolio Value at Risk (95%)")
var95 = calc_var(returns_monthly, 0.05)
st.metric("Monthly VaR (95%)", f"{var95:.2%}" if not np.isnan(var95) else "N/A")

st.success("Portfolio return analytics are ready!")
