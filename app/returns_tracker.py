import streamlit as st
import pandas as pd
import yfinance as yf
import os
from glob import glob

# --- CONFIGURATION ---

MAPPINGS_CSV = "ticker_mappings.csv"
TRADE_REPORTS_DIR = "trade_reports"
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

# --- PORTFOLIO RETURNS HELPER FUNCTIONS ---

def get_current_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.history(period="1d")['Close']
        if not price.empty:
            return float(price.iloc[-1])
        info = ticker_obj.info
        return info.get("regularMarketPrice", None)
    except Exception:
        return None

def calc_realized_unrealized_avgcost(df):
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

        # Realized portion: total sell qty at avg buy price
        realized_qty = min(total_sell_qty, total_buy_qty)
        realized_buy_amt = realized_qty * avg_buy_price
        realized_sell_amt = realized_qty * avg_sell_price
        realized_pl_value = realized_sell_amt - realized_buy_amt
        realized_pl_pct = ((avg_sell_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price else 0

        if realized_qty > 0:
            result_realized.append({
                "Ticker": ticker,
                "Quantity": realized_qty,
                "Average Buy Price": round(avg_buy_price, 2),
                "Average Sell Price": round(avg_sell_price, 2),
                "Profit/Loss Value": round(realized_pl_value, 2),
                "Profit/Loss %": round(realized_pl_pct, 2)
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
                "Quantity": unrealized_qty,
                "Average Buy Price": round(avg_buy_price, 2),
                "Current Price": round(current_price, 2) if current_price is not None else "N/A",
                "Profit/Loss Value": round(unrealized_pl_value, 2) if unrealized_pl_value is not None else "N/A",
                "Profit/Loss %": round(unrealized_pl_pct, 2) if unrealized_pl_pct is not None else "N/A"
            })

    realized_df = pd.DataFrame(result_realized)
    unrealized_df = pd.DataFrame(result_unrealized)
    total_realized = realized_df["Profit/Loss Value"].sum() if not realized_df.empty else 0
    total_unrealized = unrealized_df["Profit/Loss Value"].replace("N/A", 0).astype(float).sum() if not unrealized_df.empty else 0

    return realized_df, unrealized_df, total_realized, total_unrealized

# --- MAIN APP UI ---

st.title("Portfolio Ticker Mapping & Storage Demo (Excel + Historical Reports)")

tabs = st.tabs(["Trade Mapping", "Portfolio Returns"])

with tabs[0]:
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
        if not unmapped.empty:
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

        # --- Portfolio History Save Button ---
        st.markdown("---")
        st.subheader("Portfolio History")
        st.write("Once all trades are mapped, save your portfolio history to the repository. This will be used for returns calculation.")
        if st.button("Save Portfolio History"):
            mapped_to_save = mapped.copy()
            mapped_to_save.to_csv(PORTFOLIO_HISTORY_CSV, index=False)
            st.success("Portfolio history saved. Returns tab will now use this data.")

    else:
        st.info("Upload at least one Excel file or select trade reports from the repository to begin.")

    st.markdown("---")
    st.info(f"To add new trade reports for previous years, place `.xlsx` files in the `{TRADE_REPORTS_DIR}` folder in this repository. The app will pick them up automatically.")

with tabs[1]:
    st.header("Portfolio Returns")
    st.info("Returns are calculated only from the saved portfolio history. Click 'Save Portfolio History' in the previous tab after mapping trades.")

    # Load portfolio history from csv
    if os.path.exists(PORTFOLIO_HISTORY_CSV):
        history_df = pd.read_csv(PORTFOLIO_HISTORY_CSV)
        required_cols = {"yahoo_ticker", "side", "quantity", "price"}
        if required_cols.issubset(history_df.columns):
            realized_df, unrealized_df, total_realized, total_unrealized = calc_realized_unrealized_avgcost(history_df)

            st.subheader("Realized Returns (Average Costing)")
            st.dataframe(realized_df)
            st.markdown(f"**Total Realized Profit/Loss:** {round(total_realized,2)}")

            st.subheader("Unrealized Returns (Average Costing)")
            st.dataframe(unrealized_df)
            st.markdown(f"**Total Unrealized Profit/Loss:** {round(total_unrealized,2)}")

            if not realized_df.empty or not unrealized_df.empty:
                main_takeaway = ""
                if total_realized >= 0 and total_unrealized >= 0:
                    main_takeaway = "Your portfolio is currently in profit (realized and unrealized)."
                elif total_realized < 0 and total_unrealized < 0:
                    main_takeaway = "Your portfolio is currently in loss (realized and unrealized)."
                elif total_realized >= 0 and total_unrealized < 0:
                    main_takeaway = "You have realized profits, but unrealized holdings are in loss."
                elif total_realized < 0 and total_unrealized >= 0:
                    main_takeaway = "You have realized losses, but unrealized holdings are in profit."
                st.success(main_takeaway)
        else:
            st.error(f"Portfolio history file missing required columns: {required_cols}")
    else:
        st.warning("No portfolio history saved yet. Go to Trade Mapping tab and click 'Save Portfolio History' after mapping trades.")
