import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- CONFIGURATION ---

# Path to user ticker mappings CSV in the repository (relative or absolute)
MAPPINGS_CSV = "ticker_mappings.csv"

# --- HELPER FUNCTIONS ---

def load_user_mappings():
    """
    Load user mappings from a CSV file.
    The CSV must have columns: scrip_code, yahoo_ticker
    """
    if os.path.exists(MAPPINGS_CSV):
        df = pd.read_csv(MAPPINGS_CSV, dtype=str)
        df = df.dropna(subset=["scrip_code", "yahoo_ticker"])
        return dict(zip(df['scrip_code'].str.strip(), df['yahoo_ticker'].str.strip()))
    else:
        return {}

def save_user_mappings(mapping_dict):
    """
    Save user mappings to a CSV file.
    mapping_dict: dict of scrip_code -> yahoo_ticker
    """
    df = pd.DataFrame(
        [(code, ticker) for code, ticker in mapping_dict.items()],
        columns=["scrip_code", "yahoo_ticker"]
    )
    df.to_csv(MAPPINGS_CSV, index=False)

def default_bse_mapping(scrip_code):
    """
    Map scrip_code to Yahoo BSE ticker (NNNNNN.BO) if possible.
    Returns ticker or None.
    """
    if pd.isnull(scrip_code):
        return None
    s = str(scrip_code).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return f"{s.zfill(6)}.BO"
    return None

def yahoo_ticker_valid(ticker):
    """
    Check if yfinance returns data for the ticker.
    Returns True if valid, else False.
    """
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return not data.empty
    except Exception:
        return False

def map_ticker_for_row(row, user_mappings):
    code = str(row['scrip_code']).strip()
    # 1. User mapping takes priority
    if code in user_mappings:
        return user_mappings[code]
    # 2. Try default BSE mapping
    default_map = default_bse_mapping(code)
    if default_map and yahoo_ticker_valid(default_map):
        return default_map
    # 3. Not mapped yet
    return ""

# --- MAIN APP UI ---

st.title("Portfolio Ticker Mapping & Storage Demo")

st.markdown("""
This app maps your scrip codes to Yahoo Finance tickers.<br>
- **Numeric codes** are mapped to `.BO` tickers (e.g., `543306` → `543306.BO`).
- **If mapping fails**, you can edit unmapped tickers below and save your custom mapping.
- **Your mappings are saved in `ticker_mappings.csv`** for reuse and manual editing.
""", unsafe_allow_html=True)

# Load existing mappings
user_mappings = load_user_mappings()

# File upload
uploaded_file = st.file_uploader("Upload your trades CSV", type=["csv"])
if uploaded_file:
    # Read trades file
    trades = pd.read_csv(uploaded_file, dtype=str)
    # Normalize relevant columns (you can adjust as needed)
    # Accept column names: Scrip Code, Company, Quantity, Price, Side
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
    trades = trades.rename(columns={c: colmap[c] for c in trades.columns if c in colmap})
    # Show preview
    st.write("First 5 rows of uploaded trades:", trades.head())

    # Map tickers
    trades['scrip_code'] = trades['scrip_code'].astype(str).str.strip()
    trades['yahoo_ticker'] = trades.apply(lambda row: map_ticker_for_row(row, user_mappings), axis=1)

    # Show unmapped tickers
    unmapped = trades[trades['yahoo_ticker'] == ""][['scrip_code', 'company_name']].drop_duplicates()
    st.subheader("Unmapped Scrip Codes")
    if not unmapped.empty:
        st.write("For these codes, enter the correct Yahoo ticker (e.g., 'RELIANCE.NS' or '543306.BO').")
        edited = st.data_editor(
            unmapped.assign(yahoo_ticker=""),
            key="edit_tickers",
            num_rows="dynamic"
        )
        if st.button("Save new user mappings and try again"):
            new_mappings = {}
            for _, row in edited.iterrows():
                code = str(row['scrip_code']).strip()
                ticker = str(row['yahoo_ticker']).strip().upper()
                if ticker and yahoo_ticker_valid(ticker):
                    new_mappings[code] = ticker
                elif ticker:
                    st.warning(f"Ticker {ticker} for code {code} is not valid on Yahoo Finance and was not saved.")
            # Update and save mapping
            user_mappings.update(new_mappings)
            save_user_mappings(user_mappings)
            st.success("Saved new mappings. Please reload or re-upload your file.")

    else:
        st.success("All scrip codes mapped successfully!")

    # Show mapped trades and mapping summary
    mapped = trades[trades['yahoo_ticker'] != ""]
    st.subheader("Mapped Trades")
    st.write(mapped)

    # Show mapping file for reference/editing
    st.markdown("---")
    st.subheader("Current User Ticker Mappings")
    mapping_df = pd.DataFrame([
        {"scrip_code": k, "yahoo_ticker": v} for k, v in user_mappings.items()
    ])
    st.dataframe(mapping_df)
    st.info(f"Mappings are stored in `{MAPPINGS_CSV}`. You can edit this file by hand for persistent custom mappings.")

else:
    st.info("Please upload your trades CSV file to begin.")
