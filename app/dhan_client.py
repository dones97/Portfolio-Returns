"""Fetch Dhan trades and convert them to the app's trade report schema."""
import datetime
import io

import pandas as pd
import requests

DHAN_API_URL = "https://api.dhan.co/v2"
DHAN_AUTH_URL = "https://auth.dhan.co"
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
# Dhan's history endpoint does not document a page size, so stop on an empty page.
DHAN_MAX_PAGES = 1000
DHAN_EQUITY_SEGMENTS = {"NSE_EQ": "NSE", "BSE_EQ": "BSE"}
TRADE_COLUMNS = [
    "date", "company_name", "Amount", "Exchange", "Segment", "scrip_code",
    "Instrument Type", "Strike Price", "Expiry", "Trade Num", "Trade Time",
    "side", "quantity", "price", "isin", "Source",
]


def _clean(value):
    """Return a stripped string, or "" for Dhan's null/"NA" placeholders."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.upper() == "NA" else value


def generate_dhan_login_url(client_id, api_key, api_secret):
    """Start a consent session and return the browser login URL for it."""
    resp = requests.post(
        f"{DHAN_AUTH_URL}/app/generate-consent",
        params={"client_id": client_id},
        headers={"app_id": api_key, "app_secret": api_secret},
        timeout=30,
    )
    resp.raise_for_status()
    consent_id = resp.json()["consentAppId"]
    return f"{DHAN_AUTH_URL}/login/consentApp-login?consentAppId={consent_id}"


def exchange_dhan_token_id(token_id, api_key, api_secret):
    """Exchange the tokenId from the login redirect for an access token."""
    resp = requests.get(
        f"{DHAN_AUTH_URL}/app/consumeApp-consent",
        params={"tokenId": token_id},
        headers={"app_id": api_key, "app_secret": api_secret},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["accessToken"]


def _get_json(url, access_token):
    """GET a Dhan endpoint and return its JSON list of records."""
    headers = {"Accept": "application/json", "access-token": access_token}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected Dhan response: {data}")
    return data


def fetch_dhan_trade_records(access_token, start_date, end_date):
    """Fetch raw trades, combining dated history with today's trade book.

    Any failure raises, so a partial trade list can never be saved as history.
    """
    records = []
    for page in range(DHAN_MAX_PAGES):
        url = f"{DHAN_API_URL}/trades/{start_date}/{end_date}/{page}"
        page_records = _get_json(url, access_token)
        if not page_records:
            break
        records.extend(page_records)
    else:
        raise ValueError(f"Dhan trade history exceeded {DHAN_MAX_PAGES} pages.")
    # History may not include today's trades, and the two can overlap.
    records.extend(_get_json(f"{DHAN_API_URL}/trades", access_token))
    unique = {}
    for r in records:
        key = (r.get("exchangeSegment"), r.get("orderId"), r.get("exchangeTradeId"))
        unique.setdefault(key, r)
    return list(unique.values())


def fetch_dhan_holdings(access_token):
    """Fetch current holdings, used to check the quantities built from trades."""
    return _get_json(f"{DHAN_API_URL}/holdings", access_token)


def fetch_dhan_scrip_master():
    """Download the equity rows of Dhan's scrip master."""
    resp = requests.get(DHAN_SCRIP_MASTER_URL, timeout=120)
    resp.raise_for_status()
    master = pd.read_csv(
        io.BytesIO(resp.content),
        dtype=str,
        usecols=[
            "EXCH_ID", "SEGMENT", "SECURITY_ID", "ISIN", "INSTRUMENT",
            "UNDERLYING_SYMBOL", "DISPLAY_NAME",
        ],
    )
    return master[(master["SEGMENT"] == "E") & (master["INSTRUMENT"] == "EQUITY")]


def dhan_trades_to_report(records, master, user_mappings):
    """Convert raw Dhan trades to the schema produced by read_trade_report.

    scrip_code is the BSE scrip code when user_mappings has it, matching the
    Upstox reports, so a stock keeps one yahoo_ticker across brokers. Otherwise
    it is the trading symbol, which the app lists as unmapped: an unmapped
    numeric code would become a default .BO ticker that Yahoo cannot price.
    NSE securityIds are never used as codes, being tokens rather than BSE codes.
    """
    by_security = {
        (row.EXCH_ID, row.SECURITY_ID): row for row in master.itertuples()
    }
    bse_code_by_isin = {
        row.ISIN: row.SECURITY_ID
        for row in master.itertuples()
        if row.EXCH_ID == "BSE" and _clean(row.ISIN)
    }

    rows = []
    for r in records:
        exchange = DHAN_EQUITY_SEGMENTS.get(r.get("exchangeSegment"))
        if exchange is None or _clean(r.get("instrument")).upper() == "DERIVATIVES":
            continue

        security_id = _clean(r.get("securityId"))
        listing = by_security.get((exchange, security_id))
        isin = _clean(r.get("isin")) or (_clean(listing.ISIN) if listing else "")
        symbol = _clean(r.get("tradingSymbol")) or (
            _clean(listing.UNDERLYING_SYMBOL) if listing else ""
        )
        custom_symbol = _clean(r.get("customSymbol"))
        bse_code = bse_code_by_isin.get(isin, "")
        if not bse_code and exchange == "BSE":
            bse_code = security_id
        if bse_code in user_mappings:
            scrip_code = bse_code
        else:
            scrip_code = symbol or bse_code or custom_symbol or isin

        timestamp = pd.NaT
        for field in ("exchangeTime", "updateTime", "createTime"):
            timestamp = pd.to_datetime(_clean(r.get(field)) or None, errors="coerce")
            if pd.notna(timestamp):
                break
        if pd.isna(timestamp) or not scrip_code:
            raise ValueError(f"Dhan trade missing date or security: {r}")

        quantity = float(r["tradedQuantity"])
        price = float(r["tradedPrice"])
        rows.append({
            "date": timestamp.normalize(),
            "company_name": (
                custom_symbol
                or (_clean(listing.DISPLAY_NAME) if listing else "")
                or symbol
            ),
            # Strings, as read_trade_report reads every descriptive column
            "Amount": str(round(quantity * price, 2)),
            "Exchange": exchange,
            "Segment": "EQ",
            "scrip_code": scrip_code,
            "Instrument Type": "Equity",
            "Strike Price": "0",
            "Expiry": None,
            "Trade Num": _clean(r.get("exchangeTradeId")),
            "Trade Time": timestamp.strftime("%H:%M:%S"),
            "side": _clean(r.get("transactionType")).capitalize(),
            "quantity": quantity,
            "price": price,
            "isin": isin,
            "Source": "Dhan",
        })
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def fetch_dhan_trades(access_token, start_date, user_mappings, end_date=None):
    """Fetch all Dhan equity trades in the report schema."""
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
    records = fetch_dhan_trade_records(access_token, start_date, end_date)
    if not records:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return dhan_trades_to_report(records, fetch_dhan_scrip_master(), user_mappings)
