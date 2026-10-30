"""Tests for the corporate action, dividend and holdings helpers."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import corporate_actions as ca  # noqa: E402


def trades(*rows):
    """Build Dhan trades from (date, side, quantity) tuples."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(d), "side": s, "quantity": float(q),
                "price": 100.0, "yahoo_ticker": "ABC.NS", "isin": "INE1",
                "Amount": "1", "Trade Num": "1", "Trade Time": "10:00:00",
            }
            for d, s, q in rows
        ]
    )


def series(**values):
    """Build a Yahoo-style series from d_YYYY_MM_DD keyword arguments."""
    index = [pd.Timestamp(k[2:].replace("_", "-")) for k in values]
    return pd.Series(list(values.values()), index=index, dtype=float)


def test_bonus_applies_only_to_shares_held_before_ex_date():
    df = trades(("2026-10-16", "Buy", 10), ("2026-11-05", "Buy", 7))
    rows = ca.bonus_rows(df, {"ABC.NS": series(d_2026_11_05=2.0)})
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["quantity"] == 10
    assert row["price"] == 0
    assert row["side"] == "Buy"
    assert row["Trade Num"] == "BONUS"
    assert row["isin"] == "INE1"


def test_splits_before_purchase_or_after_exit_are_ignored():
    df = trades(("2026-10-16", "Buy", 10), ("2026-11-01", "Sell", 10))
    splits = series(d_2026_01_01=2.0, d_2026_12_01=5.0)
    assert ca.bonus_rows(df, {"ABC.NS": splits}).empty


def test_consecutive_splits_compound_and_fractions_round_down():
    df = trades(("2026-10-16", "Buy", 3))
    splits = series(d_2026_11_01=1.5, d_2026_12_01=2.0)
    rows = ca.bonus_rows(df, {"ABC.NS": splits})
    assert rows["quantity"].tolist() == [1, 4]


def test_consolidation_adds_no_rows():
    df = trades(("2026-10-16", "Buy", 10))
    assert ca.bonus_rows(df, {"ABC.NS": series(d_2026_11_01=0.1)}).empty


def test_dividend_uses_quantity_on_ex_date_and_skips_future_dates():
    df = trades(("2026-10-16", "Buy", 10), ("2026-11-05", "Buy", 7))
    dividends = series(d_2026_11_05=2.0, d_2026_12_01=3.0, d_2027_06_01=9.0)
    rows = ca.dividend_rows(
        df, {"ABC.NS": dividends}, {}, pd.Timestamp("2027-01-01")
    )
    assert rows["Amount"].tolist() == [20.0, 51.0]


def test_dividend_uses_quantity_across_both_brokers_and_zero_priced_bonus():
    df = trades(
        ("2021-01-01", "Buy", 10),
        ("2021-02-01", "Buy", 10),
        ("2022-10-15", "Sell", 20),
        ("2022-10-16", "Buy", 20),
    )
    df.loc[1, "price"] = 0.0
    df["Source"] = [None, None, None, "Dhan"]
    dividends = series(d_2021_03_01=1.0, d_2022_10_16=5.0, d_2022_12_01=2.0)
    rows = ca.dividend_rows(
        df, {"ABC.NS": dividends}, {}, pd.Timestamp("2023-01-01")
    )
    # Nothing is held on the ex-date falling between the sale and the rebuy
    assert rows["Amount"].tolist() == [20.0, 40.0]


def test_dividend_is_scaled_back_up_for_later_split():
    df = trades(("2026-10-16", "Buy", 10))
    rows = ca.dividend_rows(
        df,
        {"ABC.NS": series(d_2026_11_01=1.0)},
        {"ABC.NS": series(d_2026_12_01=2.0)},
        pd.Timestamp("2027-01-01"),
    )
    assert rows.iloc[0]["Amount"] == 20.0


def test_holdings_check_flags_only_real_differences():
    df = trades(("2026-10-16", "Buy", 10), ("2026-10-30", "Buy", 5))
    as_of = pd.Timestamp("2026-10-30")
    matching = [{"isin": "INE1", "totalQty": 10, "tradingSymbol": "ABC"}]
    assert ca.holdings_mismatches(df, matching, as_of).empty

    extra = matching + [{"isin": "INE2", "totalQty": 4, "tradingSymbol": "NEW"}]
    rows = ca.holdings_mismatches(df, extra, as_of)
    assert rows["Stock"].tolist() == ["NEW"]
    assert rows["App Quantity"].tolist() == [0.0]
