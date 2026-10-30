"""Derive bonus/split rows, dividends and holding checks from Yahoo's history.

Dhan's APIs report neither corporate actions nor dividends, so these use Yahoo's
per-ticker history. Yahoo records bonuses as splits (a 1:1 bonus has ratio 2).
Bonus rows are for Dhan trades only, as the Upstox reports carry their own.
Dividends cover the whole trade history, whichever broker held the shares.
"""
import math

import numpy as np
import pandas as pd


def normalize_action_dates(series):
    """Return a Yahoo action series indexed by plain dates.

    Some yfinance versions stamp actions at the market open. Trade dates are at
    midnight, so a timestamped ex-date would count that day's trades as prior.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    series = series.copy()
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    return series


def _signed_quantity(trades):
    """Return quantities as positive for buys and negative for sells.

    Built with numpy because pandas 3 keeps a text dtype when mapping an empty
    text column, which then fails on multiplication.
    """
    sides = trades["side"].astype(object).str.lower()
    signs = np.where(sides == "buy", 1.0, np.where(sides == "sell", -1.0, 0.0))
    return pd.to_numeric(trades["quantity"]) * signs


def _quantity_held(trades, before_date):
    """Return the net quantity from trades dated before the given date."""
    return float(_signed_quantity(trades[trades["date"] < before_date]).sum())


def bonus_rows(dhan_trades, splits_by_ticker):
    """Build zero-priced Buy rows for bonuses and splits, as the reports record them.

    The rows are timed at the start of the ex-date because shares bought on the
    ex-date do not qualify. Consolidations (ratio below 1) are left to the
    holdings check, since a zero-priced Sell would book a false loss.
    """
    rows = []
    for ticker, trades in dhan_trades.groupby("yahoo_ticker"):
        splits = splits_by_ticker.get(ticker)
        if splits is None:
            continue
        added = 0.0
        for ex_date, ratio in splits.sort_index().items():
            held = _quantity_held(trades, ex_date) + added
            new_shares = math.floor(held * (ratio - 1) + 1e-9)
            if held <= 0 or ratio <= 1 or new_shares <= 0:
                continue
            added += new_shares
            row = trades.iloc[-1].to_dict()
            row.update({
                "date": ex_date,
                "Amount": "0",
                "Trade Num": "BONUS",
                "Trade Time": "00:00:00",
                "side": "Buy",
                "quantity": float(new_shares),
                "price": 0.0,
            })
            rows.append(row)
    return pd.DataFrame(rows, columns=dhan_trades.columns)


def dividend_rows(trades_df, dividends_by_ticker, splits_by_ticker, as_of):
    """Estimate gross dividends received, one row per ticker and ex-date.

    Yahoo restates past dividends for later splits, so the per-share amount is
    scaled back up to what was paid on the shares held at the time.
    """
    rows = []
    for ticker, trades in trades_df.groupby("yahoo_ticker"):
        dividends = dividends_by_ticker.get(ticker)
        if dividends is None:
            continue
        splits = splits_by_ticker.get(ticker, pd.Series(dtype=float))
        for ex_date, per_share in dividends.sort_index().items():
            held = _quantity_held(trades, ex_date)
            if ex_date > as_of or held <= 0:
                continue
            later_ratio = splits[splits.index > ex_date].prod() if len(splits) else 1.0
            per_share_paid = per_share * later_ratio
            rows.append({
                "Ticker": ticker,
                "Ex-Date": ex_date,
                "Quantity": held,
                "Dividend per Share": round(per_share_paid, 4),
                "Amount": held * per_share_paid,
            })
    return pd.DataFrame(
        rows,
        columns=["Ticker", "Ex-Date", "Quantity", "Dividend per Share", "Amount"],
    )


def holdings_mismatches(dhan_trades, holdings, as_of):
    """Compare quantities from trades with Dhan's holdings, matching on ISIN.

    Trades dated as_of are ignored because Dhan's holdings only cover previous
    trading sessions. Bonus shares can show here until they are credited.
    """
    prior = dhan_trades[dhan_trades["date"] < as_of]
    from_trades = _signed_quantity(prior).groupby(prior["isin"]).sum()
    at_dhan = {h.get("isin"): float(h.get("totalQty", 0)) for h in holdings}
    symbols = {h.get("isin"): h.get("tradingSymbol") for h in holdings}
    names = dict(zip(dhan_trades["isin"], dhan_trades["yahoo_ticker"]))

    rows = []
    for isin in sorted(set(from_trades.index) | set(at_dhan)):
        app_qty = float(from_trades.get(isin, 0.0))
        dhan_qty = at_dhan.get(isin, 0.0)
        if abs(app_qty - dhan_qty) > 1e-6:
            rows.append({
                "Stock": names.get(isin) or symbols.get(isin) or isin,
                "ISIN": isin,
                "App Quantity": app_qty,
                "Dhan Quantity": dhan_qty,
            })
    return pd.DataFrame(
        rows, columns=["Stock", "ISIN", "App Quantity", "Dhan Quantity"]
    )
