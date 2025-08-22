import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob
import requests
from base64 import b64encode

# --- CONFIGURATION ---
MAPPINGS_CSV = "ticker_mappings.csv"
TRADE_REPORTS_DIR = "trade_reports"
GITHUB_REPO = "dones97/Portfolio-Returns"
GITHUB_BRANCH = "main"

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

def push_mapping_to_github(filepath, repo, branch, token):
    with open(filepath, "rb") as f:
        content = f.read()
    b64_content = b64encode(content).decode()
    url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"token {token}"}
    r = requests.get(url, headers=headers)
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha", None)
    data = {
        "message": "Update ticker_mappings.csv via Streamlit app",
        "content": b64_content,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        st.success("Mapping file updated and pushed to GitHub!")
        return True
    else:
        st.error(f"Error pushing file to GitHub: {r.text}")
        return False

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

def get_yahoo_stock_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or info.get("name")
        if name:
            return name
        return info.get("symbol", "")
    except Exception:
        return ""

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

# --- SESSION STATE KEYS ---
UNMAPPED_EDITOR_KEY = "unmapped_editor_data"
DELETE_LIST_KEY = "delete_ticker_list"

# --- MAIN APP UI ---

st.title("Portfolio Ticker Mapping & Storage Demo (Excel + Historical Reports)")

st.markdown("""
This app maps your scrip codes to Yahoo Finance tickers.<br>
- **Upload one or more Excel trade reports below.**
- **Or select from trade reports available in the repository.**
- **Numeric codes** are mapped to `.BO` tickers (e.g., `543306` → `543306.BO`).
- **If mapping fails**, you can edit unmapped tickers below and update your custom mapping in one go.
- **You can also delete trades for unmappable/delisted tickers.**
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

    # --- MAPPING LOGIC ---
    # Only update mappings/table when user hits button
    # Prepare unmapped and mapped tables via session state
    if UNMAPPED_EDITOR_KEY not in st.session_state:
        # Initial mapping
        trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)
        unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
        unmapped['yahoo_ticker'] = ""
        st.session_state[UNMAPPED_EDITOR_KEY] = unmapped.reset_index(drop=True)
        st.session_state[DELETE_LIST_KEY] = []
    else:
        # Use the session state unmapped editor data
        unmapped = st.session_state[UNMAPPED_EDITOR_KEY]
    
    # Show unmapped table (no recalculation when editing)
    st.subheader("Unmapped Scrip Codes")
    st.write("Enter Yahoo tickers for unmapped codes, or select tickers to delete all trades for them (e.g., delisted stocks).")
    edited = st.data_editor(
        unmapped,
        key="unmapped_editor",
        num_rows="dynamic"
    )

    # Option to select tickers to delete
    delete_candidates = edited['scrip_code'].tolist()
    delete_selection = st.multiselect(
        "Select tickers to delete all trades for (delisted, not needed):",
        delete_candidates,
        default=st.session_state.get(DELETE_LIST_KEY, [])
    )
    st.session_state[DELETE_LIST_KEY] = delete_selection

    # Update mapping button
    if st.button("Update Mapping and Delete Selected Tickers"):
        # Update unmapped editor session state from edited table
        st.session_state[UNMAPPED_EDITOR_KEY] = edited.copy().reset_index(drop=True)
        # Remove all trades for selected tickers
        trades = trades[~trades['scrip_code'].isin(delete_selection)]
        # Update mapping for new tickers
        new_mappings = {}
        failed_codes = []
        for _, row in edited.iterrows():
            code = str(row['scrip_code']).strip()
            ticker = str(row['yahoo_ticker']).strip().upper()
            if ticker and code not in delete_selection:
                if yahoo_ticker_valid(ticker):
                    new_mappings[code] = ticker
                else:
                    failed_codes.append((code, ticker))
        if new_mappings:
            user_mappings.update(new_mappings)
            save_user_mappings(user_mappings)
            # --- Push to GitHub ---
            token = st.secrets["GITHUB_TOKEN"] if "GITHUB_TOKEN" in st.secrets else None
            if token:
                push_mapping_to_github(
                    filepath=MAPPINGS_CSV,
                    repo=GITHUB_REPO,
                    branch=GITHUB_BRANCH,
                    token=token
                )
            else:
                st.warning("No GitHub token found in Streamlit secrets. File only saved locally (may be lost after restart).")
        if failed_codes:
            for code, ticker in failed_codes:
                st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo Finance and was not saved.")

        # Update mapped/unmapped
        trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)
        unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
        unmapped['yahoo_ticker'] = ""
        st.session_state[UNMAPPED_EDITOR_KEY] = unmapped.reset_index(drop=True)
        st.session_state[DELETE_LIST_KEY] = []
        st.success("Mapping and deletion complete. Tables updated.")

    mapped = trades[trades['yahoo_ticker'] != ""].copy()
    mapped['yahoo_stock_name'] = mapped['yahoo_ticker'].apply(get_yahoo_stock_name)

    st.subheader("Mapped Trades")
    st.write(mapped)

    st.markdown("---")
    st.subheader("Current User Ticker Mappings")
    mapping_df = pd.DataFrame([
        {"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()
    ])
    st.dataframe(mapping_df)
    st.info(f"Mappings are stored in `{MAPPINGS_CSV}` and pushed to GitHub for permanent persistence.")

else:
    st.info("Upload at least one Excel file or select trade reports from the repository to begin.")

st.markdown("---")
st.info(f"To add new trade reports for previous years, place `.xlsx` files in the `{TRADE_REPORTS_DIR}` folder in this repository. The app will pick them up automatically.")
