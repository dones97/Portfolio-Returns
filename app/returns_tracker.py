import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob

# --- CONFIGURATION ---

MAPPINGS_CSV = "ticker_mappings.csv"
TRADE_REPORTS_DIR = "trade_reports"

# --- HELPER FUNCTIONS ---

def load_user_mappings():
    if os.path.exists(MAPPINGS_CSV):
        df = pd.read_csv(MAPPINGS_CSV, dtype=str)
        df = df.dropna(subset=["scrip_code", "yahoo_ticker"])
        return dict(zip(df['scrip_code'].str.strip(), df['yahoo_ticker'].str.strip()))
    else:
        return {}

def save_user_mappings(mapping_dict):
    df = pd.DataFrame(
        [(code, ticker) for code, ticker in mapping_dict.items()],
        columns=["scrip_code", "yahoo_ticker"]
    )
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

def yahoo_ticker_valid(ticker, timeout=5):
    """
    Improved validation:
      1) Try yf.Ticker(ticker).info (fast) and look for useful fields.
      2) Try history(period='5d') or history(period='1mo') to catch thin tickers.
      3) As a final fallback try yf.download for the last 30 days (start/end).
    Returns True if we see non-empty info or non-empty history/data.
    """
    if not ticker or not isinstance(ticker, str):
        return False
    tstr = ticker.strip().upper()
    try:
        t = yf.Ticker(tstr)
        # 1) info check (many tickers return useful fields here)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        if info and (info.get("shortName") or info.get("regularMarketPrice") is not None or info.get("symbol")):
            return True

        # 2) try short history (5d) then 1mo
        for p in ("5d", "1mo"):
            try:
                hist = t.history(period=p, timeout=timeout)
                if hist is not None and not hist.empty:
                    return True
            except Exception:
                # ignore and try next
                pass

        # 3) fallback: try yf.download for last 30 days with explicit start/end
        try:
            end = pd.Timestamp.today()
            start = end - pd.Timedelta(days=30)
            df = yf.download(tstr, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=False, timeout=timeout)
            if df is not None and not df.empty:
                return True
        except Exception:
            pass

        return False
    except Exception:
        return False

def map_ticker_for_row(row, user_mappings):
    """
    Map a single trade report row (expects 'scrip_code' and optional 'company_name').
    Priority:
      1) user_mappings (persistent csv)
      2) default_bse_mapping (numeric -> zero-padded 6-digit .BO)
      3) if default_map fails, try raw code + .BO, code as-is, and a couple common variants
    Returns ticker string or "" if not resolved.
    """
    code = str(row.get('scrip_code', '')).strip()
    if not code:
        return ""
    # 1) explicit user mapping
    if code in user_mappings:
        return user_mappings[code]

    # 2) default BSE numeric mapping (existing behavior)
    default_map = default_bse_mapping(code)
    if default_map and yahoo_ticker_valid(default_map):
        return default_map

    # 3) try other common variants (explicit attempts) - lowers false negatives
    cand = code.upper()
    candidates = []
    # if code looks numeric try unpadded and padded variants
    if code.replace('.', '', 1).isdigit():
        if cand.endswith('.0'):
            cand = cand[:-2]
        candidates.extend([f"{cand.zfill(6)}.BO", f"{cand}.BO", cand])
    else:
        # alpha codes: try with common exchange suffixes
        candidates.extend([cand, f"{cand}.NS", f"{cand}.BO"])

    # try candidates
    for c in candidates:
        if c and yahoo_ticker_valid(c):
            return c

    # unresolved: return empty so UI can ask user
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
        "Buy/Sell": "side"
    }
    df = df.rename(columns={c: colmap[c] for c in df.columns if c in colmap})
    df['scrip_code'] = df['scrip_code'].astype(str).str.strip()
    df['company_name'] = df['company_name'].astype(str).str.strip()
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

# --- MAIN APP UI ---

st.title("Portfolio Ticker Mapping & Storage Demo (Excel + Historical Reports)")

st.markdown("""
This app maps your scrip codes to Yahoo Finance tickers.<br>
- **Upload one or more Excel trade reports below.**
- **Or select from trade reports available in the repository.**
- **Numeric codes** are mapped to `.BO` tickers (e.g., `543306` → `543306.BO`).
- **If mapping fails**, you can edit unmapped tickers below and update your custom mapping in one go.
- **Your mappings are saved in `ticker_mappings.csv`** for reuse and manual editing.
""", unsafe_allow_html=True)

user_mappings = load_user_mappings()

uploaded_files = st.file_uploader("Upload your trade reports (Excel)", type=["xlsx"], accept_multiple_files=True)

st.subheader("Trade Reports from Previous Years (in repository)")
repo_reports = get_repository_trade_reports()
selected_repo_files = []
if repo_reports:
    repo_file_names = [fname for fname, _ in repo_reports]
    selected_repo_files = st.multiselect(
        "Select previous trade reports to include:",
        repo_file_names,
        default=[]
    )

all_trades = []

if uploaded_files:
    for file in uploaded_files:
        try:
            df = read_trade_report(file)
            all_trades.append(df)
        except Exception as e:
            st.error(f"Could not read uploaded file {file.name}: {e}")

for fname, df in repo_reports:
    if fname in selected_repo_files:
        all_trades.append(df)

if all_trades:
    trades = pd.concat(all_trades, ignore_index=True)
    st.success(f"Loaded {len(trades)} trades from {len(all_trades)} report(s).")
    st.write("First 5 rows of combined trades:", trades.head())

    trades['scrip_code'] = trades['scrip_code'].astype(str).str.strip()
    trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)

    unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
    st.subheader("Unmapped Scrip Codes")
    # Debug: show first 20 unmapped codes for inspection
    if not unmapped.empty:
        st.write("Sample unmapped (first 20) for inspection:", unmapped.head(20))
        st.write("For these codes, enter the correct Yahoo ticker (e.g., 'RELIANCE.NS' or '543306.BO').")
        edited = st.data_editor(
            unmapped.assign(yahoo_ticker=""),
            key="edit_tickers",
            num_rows="dynamic"
        )
        if st.button("Update Mapping"):
            # Process all at once when button is pressed
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
                st.success("Saved new mappings. Please reload or re-upload your files.")
            if failed_codes:
                for code, ticker in failed_codes:
                    st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo Finance and was not saved.")
    else:
        st.success("All scrip codes mapped successfully!")

    mapped = trades[trades['yahoo_ticker'] != ""]
    st.subheader("Mapped Trades")
    st.write(mapped)

    st.markdown("---")
    st.subheader("Current User Ticker Mappings")
    mapping_df = pd.DataFrame([
        {"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()
    ])
    st.dataframe(mapping_df)
    st.info(f"Mappings are stored in `{MAPPINGS_CSV}`. You can edit this file by hand for persistent custom mappings.")

else:
    st.info("Upload at least one Excel file or select trade reports from the repository to begin.")

st.markdown("---")
st.info(f"To add new trade reports for previous years, place `.xlsx` files in the `{TRADE_REPORTS_DIR}` folder in this repository. The app will pick them up automatically.")
