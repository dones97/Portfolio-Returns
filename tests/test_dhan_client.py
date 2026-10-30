"""Tests for the Dhan trade fetch and conversion, using no network access."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import dhan_client  # noqa: E402

MASTER = pd.DataFrame(
    [
        ("BSE", "E", "532540", "INE467B01029", "EQUITY", "TCS", "TCS Ltd"),
        ("NSE", "E", "11536", "INE467B01029", "EQUITY", "TCS", "TCS Ltd"),
        ("NSE", "E", "99999", "INE000000001", "EQUITY", "NSEONLY", "NSE Only Ltd"),
    ],
    columns=[
        "EXCH_ID", "SEGMENT", "SECURITY_ID", "ISIN", "INSTRUMENT",
        "UNDERLYING_SYMBOL", "DISPLAY_NAME",
    ],
)
MAPPINGS = {"532540": "TCS.NS"}


def trade(**overrides):
    """Build a raw Dhan trade record, shaped like the documented example."""
    record = {
        "orderId": "1",
        "exchangeTradeId": "T1",
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "tradingSymbol": None,
        "customSymbol": "TCS Ltd",
        "securityId": "11536",
        "tradedQuantity": 2,
        "tradedPrice": 100.5,
        "isin": "INE467B01029",
        "instrument": "EQUITY",
        "createTime": "NA",
        "updateTime": "NA",
        "exchangeTime": "2026-10-01 10:00:46",
    }
    record.update(overrides)
    return record


def test_nse_trade_uses_bse_scrip_code_and_report_schema():
    row = dhan_client.dhan_trades_to_report([trade()], MASTER, MAPPINGS).iloc[0]
    assert row["scrip_code"] == "532540"
    assert row["side"] == "Buy"
    assert row["date"] == pd.Timestamp("2026-10-01")
    assert row["Trade Time"] == "10:00:46"
    assert row["Amount"] == "201.0"
    assert row["Exchange"] == "NSE"
    assert row["Segment"] == "EQ"


def test_unmapped_bse_code_uses_symbol_so_the_app_lists_it_as_unmapped():
    row = dhan_client.dhan_trades_to_report([trade()], MASTER, {}).iloc[0]
    assert row["scrip_code"] == "TCS"


def test_nse_only_stock_uses_symbol_never_the_numeric_token():
    record = trade(securityId="99999", isin="INE000000001", customSymbol=None)
    row = dhan_client.dhan_trades_to_report([record], MASTER, MAPPINGS).iloc[0]
    assert row["scrip_code"] == "NSEONLY"
    assert row["company_name"] == "NSE Only Ltd"


def test_unknown_nse_stock_does_not_fall_back_to_token():
    record = trade(securityId="123", isin="INE000000099", customSymbol="Unknown Co")
    row = dhan_client.dhan_trades_to_report([record], MASTER, MAPPINGS).iloc[0]
    assert row["scrip_code"] == "Unknown Co"


def test_trade_book_record_without_isin_resolves_through_security_id():
    record = trade(isin=None, customSymbol=None, tradingSymbol="TCS")
    row = dhan_client.dhan_trades_to_report([record], MASTER, MAPPINGS).iloc[0]
    assert row["scrip_code"] == "532540"


def test_bse_trade_not_in_master_uses_security_id():
    record = trade(exchangeSegment="BSE_EQ", securityId="500001", isin="X")
    mappings = {"500001": "ABC.NS"}
    row = dhan_client.dhan_trades_to_report([record], MASTER, mappings).iloc[0]
    assert row["scrip_code"] == "500001"


def test_non_equity_trades_are_excluded():
    records = [
        trade(exchangeSegment="NSE_FNO"),
        trade(instrument="DERIVATIVES"),
    ]
    assert dhan_client.dhan_trades_to_report(records, MASTER, MAPPINGS).empty


def test_trade_without_usable_time_raises():
    with pytest.raises(ValueError):
        dhan_client.dhan_trades_to_report([trade(exchangeTime="NA")], MASTER, MAPPINGS)


def test_fetch_pages_until_empty_and_deduplicates_trade_book(monkeypatch):
    responses = {
        "/trades/2026-04-01/2026-10-30/0": [trade(), trade(exchangeTradeId="T2")],
        "/trades/2026-04-01/2026-10-30/1": [trade(exchangeTradeId="T3")],
        "/trades/2026-04-01/2026-10-30/2": [],
        "/trades": [trade(exchangeTradeId="T3"), trade(exchangeTradeId="T4")],
    }
    monkeypatch.setattr(
        dhan_client,
        "_get_json",
        lambda url, token: responses[url.replace(dhan_client.DHAN_API_URL, "")],
    )
    records = dhan_client.fetch_dhan_trade_records("t", "2026-04-01", "2026-10-30")
    assert [r["exchangeTradeId"] for r in records] == ["T1", "T2", "T3", "T4"]


def test_fetch_failure_propagates_instead_of_returning_partial(monkeypatch):
    def fail_on_second_page(url, token):
        if url.endswith("/1"):
            raise ConnectionError("boom")
        return [trade()]

    monkeypatch.setattr(dhan_client, "_get_json", fail_on_second_page)
    with pytest.raises(ConnectionError):
        dhan_client.fetch_dhan_trade_records("t", "2026-04-01", "2026-10-30")


class FakeResponse:
    """A failed response with a JSON or plain text body."""

    ok = False
    status_code = 400

    def __init__(self, text):
        self.text = text

    def json(self):
        return json.loads(self.text)


def test_failure_message_keeps_the_reason_from_dhan():
    body = '{"errorCode": "DH-907", "errorMessage": "No data present"}'
    with pytest.raises(ValueError, match="HTTP 400. DH-907: No data present"):
        dhan_client._raise_for_dhan_error(FakeResponse(body))


def test_failure_message_handles_non_json_body():
    with pytest.raises(ValueError, match="HTTP 400. Bad gateway"):
        dhan_client._raise_for_dhan_error(FakeResponse("Bad gateway"))


def test_login_code_can_be_pasted_alone_or_as_the_redirect_url():
    assert dhan_client.extract_dhan_token_id(" abc-123 \n") == "abc-123"
    url = "https://app.streamlit.app/?tokenId=abc-123&other=1"
    assert dhan_client.extract_dhan_token_id(url) == "abc-123"
