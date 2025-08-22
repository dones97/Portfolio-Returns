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
PORTFOLIO_HISTORY_CSV = "portfolio_history.csv"

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

def push_file_to_github(filepath, repo, branch, token, message="Update file via Streamlit app"):
    if not os.path.exists(filepath):
        st.error(f"File {filepath} does not exist, cannot push to GitHub.")
        return False
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
        "message": message,
        "content": b64_content,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        st.success(f"{os.path.basename(filepath)} updated and pushed to GitHub!")
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
TRADE_SELECTION_KEY = "trade_selection"
STATIC_UNMAPPED_KEY = "static_unmapped"
STATIC_MAPPED_KEY = "static_mapped"
STATIC_TRADES_KEY = "static_trades"
STATIC_MAPPING_KEY = "static_mapping"
DELETE_LIST_KEY = "delete_ticker_list"
LAST_SELECTION_KEY = "last_selection"
LAST_UPLOADED_KEY = "last_uploaded_files"
PORTFOLIO_HISTORY_KEY = "portfolio_history"

# --- MAIN APP UI ---

st.title("Portfolio Returns Tracker")

tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
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
    repo_file_names = [fname for fname, _ in repo_reports] if repo_reports else []
    selected_repo_files = st.multiselect(
        "Select previous trade reports to include:",
        repo_file_names,
        default=[],
        key=TRADE_SELECTION_KEY
    )

    # --- Always recalculate all_trades when selection changes ---
    all_trades = []
    uploaded_names = [f.name for f in uploaded_files] if uploaded_files else []

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

    # -- Helper to check session state existence
    def state_exists(keys):
        return all(k in st.session_state for k in keys)

    # --- Build trades DataFrame and mapping only on initial load or after update ---
    if all_trades:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['scrip_code'] = trades['scrip_code'].astype(str).str.strip()

        if (
            not state_exists([STATIC_TRADES_KEY, STATIC_UNMAPPED_KEY, STATIC_MAPPED_KEY, STATIC_MAPPING_KEY, LAST_SELECTION_KEY, LAST_UPLOADED_KEY])
            or st.session_state[LAST_SELECTION_KEY] != selected_repo_files
            or st.session_state[LAST_UPLOADED_KEY] != uploaded_names
        ):
            trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)
            unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
            unmapped['yahoo_ticker'] = ""
            mapped = trades[trades['yahoo_ticker'] != ""].copy()
            if not mapped.empty:
                mapped['yahoo_stock_name'] = mapped['yahoo_ticker'].apply(get_yahoo_stock_name)
            mapping_df = pd.DataFrame([
                {"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()
            ])
            # Save to session state
            st.session_state[STATIC_TRADES_KEY] = trades
            st.session_state[STATIC_UNMAPPED_KEY] = unmapped.reset_index(drop=True)
            st.session_state[STATIC_MAPPED_KEY] = mapped.reset_index(drop=True)
            st.session_state[STATIC_MAPPING_KEY] = mapping_df
            st.session_state[LAST_SELECTION_KEY] = selected_repo_files
            st.session_state[LAST_UPLOADED_KEY] = uploaded_names

        trades = st.session_state.get(STATIC_TRADES_KEY)
        unmapped = st.session_state.get(STATIC_UNMAPPED_KEY)
        mapped = st.session_state.get(STATIC_MAPPED_KEY)
        mapping_df = st.session_state.get(STATIC_MAPPING_KEY)

        st.success(f"Loaded {len(trades)} trades from {len(all_trades)} report(s).")
        st.write("First 5 rows of combined trades:", trades.head())

        st.subheader("Unmapped Scrip Codes")
        st.write("Enter Yahoo tickers for unmapped codes, or select tickers to delete all trades for them (e.g., delisted stocks).")
        edited = st.data_editor(
            unmapped,
            key="unmapped_editor",
            num_rows="dynamic"
        )

        delete_candidates = edited['scrip_code'].tolist()
        delete_selection = st.multiselect(
            "Select tickers to delete all trades for (delisted, not needed):",
            delete_candidates,
            default=[],
            key="delete_selection"
        )

        if st.button("Update Mapping and Delete Selected Tickers"):
            trades_new = trades[~trades['scrip_code'].isin(delete_selection)].copy()
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
                token = st.secrets["GITHUB_TOKEN"] if "GITHUB_TOKEN" in st.secrets else None
                if token:
                    push_file_to_github(
                        filepath=MAPPINGS_CSV,
                        repo=GITHUB_REPO,
                        branch=GITHUB_BRANCH,
                        token=token,
                        message="Update ticker_mappings.csv via Streamlit app"
                    )
                else:
                    st.warning("No GitHub token found in Streamlit secrets. File only saved locally (may be lost after restart).")
            if failed_codes:
                for code, ticker in failed_codes:
                    st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo Finance and was not saved.")

            # Recalculate mapping/unmapped/mapped/mapping table after update
            trades_new['yahoo_ticker'] = trades_new.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)
            unmapped_new = trades_new[trades_new['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
            unmapped_new['yahoo_ticker'] = ""
            mapped_new = trades_new[trades_new['yahoo_ticker'] != ""].copy()
            if not mapped_new.empty:
                mapped_new['yahoo_stock_name'] = mapped_new['yahoo_ticker'].apply(get_yahoo_stock_name)
            mapping_df_new = pd.DataFrame([
                {"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()
            ])
            st.session_state[STATIC_TRADES_KEY] = trades_new
            st.session_state[STATIC_UNMAPPED_KEY] = unmapped_new.reset_index(drop=True)
            st.session_state[STATIC_MAPPED_KEY] = mapped_new.reset_index(drop=True)
            st.session_state[STATIC_MAPPING_KEY] = mapping_df_new
            st.success("Mapping and deletion complete. Tables updated.")

        st.subheader("Mapped Trades")
        st.write(mapped)

        st.markdown("---")
        st.subheader("Current User Ticker Mappings")
        st.dataframe(mapping_df)
        st.info(f"Mappings are stored in `{MAPPINGS_CSV}` and pushed to GitHub for permanent persistence.")

        # --- Portfolio History Save Button ---
        st.markdown("---")
        st.subheader("Portfolio History")
        st.write("Once all trades are mapped, save your portfolio history to the repository. This will be used for returns calculation.")
        if st.button("Save Portfolio History"):
            # Save mapped trades to CSV
            mapped_to_save = mapped.copy()
            mapped_to_save.to_csv(PORTFOLIO_HISTORY_CSV, index=False)
            st.session_state[PORTFOLIO_HISTORY_KEY] = mapped_to_save
            token = st.secrets["GITHUB_TOKEN"] if "GITHUB_TOKEN" in st.secrets else None
            if token:
                push_file_to_github(
                    filepath=PORTFOLIO_HISTORY_CSV,
                    repo=GITHUB_REPO,
                    branch=GITHUB_BRANCH,
                    token=token,
                    message="Update portfolio_history.csv via Streamlit app"
                )
            else:
                st.warning("No GitHub token found in Streamlit secrets. History only saved locally (may be lost after restart).")
            st.success("Portfolio history saved and pushed to GitHub. Returns tab will now use this data.")

    else:
        st.info("Upload at least one Excel file or select trade reports from the repository to begin.")

    st.markdown("---")
    st.info(f"To add new trade reports for previous years, place `.xlsx` files in the `{TRADE_REPORTS_DIR}` folder in this repository. The app will pick them up automatically.")

with tabs[1]:
    st.header("Portfolio Returns")
    st.info("Returns are calculated only from the saved portfolio history. Click 'Save Portfolio History' in the previous tab after mapping trades.")
    # Load portfolio history from session or file
    portfolio_history = st.session_state.get(PORTFOLIO_HISTORY_KEY, None)
    if portfolio_history is None and os.path.exists(PORTFOLIO_HISTORY_CSV):
        portfolio_history = pd.read_csv(PORTFOLIO_HISTORY_CSV)
        st.session_state[PORTFOLIO_HISTORY_KEY] = portfolio_history
    if portfolio_history is not None:
        st.write("Portfolio history as saved:", portfolio_history)
        # --- Returns calculation will be implemented here ---
    else:
        st.warning("No portfolio history saved yet. Go to Trade Mapping tab and click 'Save Portfolio History' after mapping trades.")
