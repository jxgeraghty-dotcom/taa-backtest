"""Real-data loaders. These require network access you will have locally, not in
a sandbox. Nothing here is imported by the offline demo path.

Honest data constraint, stated so you do not forget it: liquid ETF proxies mostly
begin around 2003 to 2007 (TIP 2003, HYG 2007, DBC 2006). A clean all-ETF backtest
realistically starts circa 2007, which is roughly 220 monthly points dominated by a
handful of regimes. Treat this as a methods showcase, not proof a strategy works.
"""
from __future__ import annotations

import pandas as pd

from .panel import PriceHistory, MacroHistory, DataBundle

# Suggested ETF proxies by sleeve. Adjust to taste; document any change.
DEFAULT_ETFS = {
    "EQ_US": "VTI",
    "EQ_INTL": "VEA",
    "GOVT": "IEF",
    "CREDIT": "HYG",
    "TIPS": "TIP",
    "CMDTY": "DBC",
    "CASH": "BIL",
}

# FRED series ids for the macro nowcast. NOTE: the ICE BofA index series
# (BAML*) are restricted on FRED's free CSV endpoint to a rolling ~3-year window, so
# they silently truncate long backtests. We use Moody's Baa-Treasury spread (BAA10Y,
# full history back to the 1980s) as the credit-spread input instead.
DEFAULT_FRED = {
    "dgs10": "DGS10",
    "dgs2": "DGS2",
    "credit_spread": "BAA10Y",   # Moody's Baa corporate - 10y Treasury spread
    "tbill_3m": "DGS3MO",
}

# Per-sleeve carry from FRED, in percent. Cleanly defined for fixed income, credit,
# and cash as a published yield. Equities and commodities get their carry legs built
# from price data instead (dividend yield and a roll-yield proxy); see below.
# Credit uses Moody's Baa corporate yield (full history) rather than an ICE BofA HY
# yield, because the ICE BofA series on FRED are truncated to ~2023+; Baa is investment
# grade vs the HY sleeve proxy, a documented mismatch that co-moves closely enough to
# stand in as the credit-carry level.
CARRY_FRED = {
    "GOVT": "DGS10",   # 10y nominal Treasury yield
    "TIPS": "DFII10",  # 10y TIPS (real) yield
    "CREDIT": "BAA",   # Moody's Baa corporate bond yield (full history; IG proxy for HY)
    "CASH": "DGS3MO",  # 3m T-bill yield
}

# Equity carry = trailing dividend yield, computed from prices (no fundamentals feed
# needed). These sleeves' ETF tickers are read from the ETF map.
EQUITY_CARRY_SLEEVES = ("EQ_US", "EQ_INTL")

# Commodity carry = a roll-yield proxy: the futures-based ETF's total return over a
# broad commodity spot/benchmark, net of cash collateral. Positive in backwardation,
# negative in contango. A proxy, not a pure per-contract roll — see load_carry.
COMMODITY_SPOT_TICKER = "^SPGSCI"   # S&P GSCI commodity benchmark (long free history)


def load_etf_prices(tickers: dict[str, str] | None = None, start: str = "2007-01-01") -> PriceHistory:
    """Monthly total-return proxy from yfinance adjusted close. Requires `yfinance`."""
    import yfinance as yf  # local import so the offline path never needs it

    tickers = tickers or DEFAULT_ETFS
    raw = yf.download(list(tickers.values()), start=start, auto_adjust=True, progress=False)["Close"]
    monthly = raw.resample("ME").last()
    monthly.columns = [k for k, v in tickers.items()]
    return PriceHistory(monthly.dropna(how="all"))


def _fred_frame(series: dict[str, str], start: str) -> pd.DataFrame:
    """Monthly (last-of-month) frame of FRED series, columns renamed to the dict keys."""
    from pandas_datareader import data as pdr  # local import

    frames = {k: pdr.DataReader(v, "fred", start) for k, v in series.items()}
    out = pd.concat(frames, axis=1)
    out.columns = list(series.keys())
    return out.resample("ME").last()


def load_fred(series: dict[str, str] | None = None, start: str = "2007-01-01") -> MacroHistory:
    """Macro/rate series from FRED. Requires `pandas_datareader`. No API key needed."""
    return MacroHistory(_fred_frame(series or DEFAULT_FRED, start), release_lag=1)


# --- pure carry-leg computations (no network; unit-tested offline) --------------------

def trailing_income_yield(tr_monthly: pd.DataFrame, pr_monthly: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Trailing dividend/income yield from total-return vs price-return levels.

    Each month's income yield is (total-return %chg) - (price-return %chg); the reinvested
    dividend is exactly the gap between an adjusted-close series and a raw-close series.
    Summing `window` months gives a trailing annual dividend yield. Pure and PIT: month t
    uses only returns realized through month t.
    """
    monthly_income = tr_monthly.pct_change() - pr_monthly.pct_change()
    return monthly_income.rolling(window).sum()


def roll_yield_proxy(fut_tr_monthly, spot_monthly, cash_annual_monthly, window: int = 12):
    """Trailing commodity roll-yield proxy from a futures index vs a spot benchmark.

    monthly roll = (futures total return %chg) - (spot benchmark %chg) - (cash collateral).
    The futures ETF earns spot + roll + collateral; subtracting the spot move and the bill
    yield leaves the roll. Summed over `window` months = a trailing annual roll yield.
    It is a proxy: both indices are broad baskets, so basket differences leak into it, and
    it captures the term-structure state (backwardation vs contango) rather than a clean
    per-contract roll. Pure and PIT.
    """
    monthly_roll = fut_tr_monthly.pct_change() - spot_monthly.pct_change() - cash_annual_monthly / 12.0
    return monthly_roll.rolling(window).sum()


def _field_frame(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract one field (e.g. 'Adj Close') from a yfinance download as a ticker-columned frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[field]
    return raw[[field]]


def load_carry(
    start: str = "2007-01-01",
    tickers: dict[str, str] | None = None,
    prices: PriceHistory | None = None,
    window: int = 12,
) -> PriceHistory:
    """Assemble the full per-sleeve carry panel (decimals), one column per sleeve.

      fixed income / credit / cash : published FRED yield/spread (CARRY_FRED).
      equities                     : trailing dividend yield (trailing_income_yield).
      commodities                  : roll-yield proxy (roll_yield_proxy).

    Each leg is best-effort: if a fetch fails, that sleeve is omitted and the carry
    signal simply holds no view on it. Values are yields, wrapped in PriceHistory only
    because the carry signal needs history(as_of). If `prices` is given, the panel is
    reindexed to the price clock and forward-filled so every trade date has a carry read.
    """
    tickers = tickers or DEFAULT_ETFS
    cols: dict[str, pd.Series] = {}

    fred = _fred_frame(CARRY_FRED, start) / 100.0            # ME monthly, percent -> decimal
    for sleeve in CARRY_FRED:
        cols[sleeve] = fred[sleeve]
    cash_annual = fred["CASH"]                              # collateral for the commodity leg

    try:  # equity dividend yield
        import yfinance as yf
        eq = {s: tickers[s] for s in EQUITY_CARRY_SLEEVES if s in tickers}
        raw = yf.download(list(eq.values()), start=start, auto_adjust=False, progress=False)
        adj_m = _field_frame(raw, "Adj Close").resample("ME").last()
        px_m = _field_frame(raw, "Close").resample("ME").last()
        dy = trailing_income_yield(adj_m, px_m, window)
        for sleeve, sym in eq.items():
            cols[sleeve] = dy[sym]
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        print(f"[load_carry] equity carry unavailable, omitting: {e}")

    try:  # commodity roll-yield proxy
        import yfinance as yf
        fut = yf.download(tickers["CMDTY"], start=start, auto_adjust=True, progress=False)["Close"].squeeze()
        spot = yf.download(COMMODITY_SPOT_TICKER, start=start, auto_adjust=True, progress=False)["Close"].squeeze()
        fut_m = fut.resample("ME").last()
        spot_m = spot.resample("ME").last()
        cash_m = cash_annual.reindex(fut_m.index).ffill()
        cols["CMDTY"] = roll_yield_proxy(fut_m, spot_m, cash_m, window)
    except Exception as e:  # noqa: BLE001
        print(f"[load_carry] commodity carry unavailable, omitting: {e}")

    frame = pd.DataFrame(cols).sort_index()
    if prices is not None:
        frame = frame.reindex(prices.dates).ffill()

    # Surface silent truncation: FRED restricts some series (e.g. ICE BofA) to a
    # rolling recent window, which would quietly leave a sleeve blank for most of the
    # sample. Report first-valid date per sleeve so a short leg is never a surprise.
    cov = {c: (frame[c].first_valid_index(), int(frame[c].notna().sum())) for c in frame.columns}
    short = {c: v for c, v in cov.items() if v[1] < 0.8 * len(frame)}
    print(f"[load_carry] sleeves={list(frame.columns)} n={len(frame)}")
    if short:
        print(f"[load_carry] WARNING short-history legs (first_valid, n): "
              + ", ".join(f"{c}={d.date()}/{n}" for c, (d, n) in short.items() if d is not None))
    return PriceHistory(frame)


def load_bundle(
    tickers: dict[str, str] | None = None,
    start: str = "2007-01-01",
    with_macro: bool = True,
    with_carry: bool = True,
) -> DataBundle:
    """Assemble the real-data DataBundle: ETF total-return prices plus FRED carry/macro.

    Prices are required; carry and macro are best-effort. If a FRED pull fails (network,
    a discontinued series), that panel is dropped and the bundle still runs on the
    families that remain, so a flaky data source degrades the overlay rather than
    crashing the run. What was actually loaded is printed so you never silently think
    you had carry/macro when you did not.
    """
    prices = load_etf_prices(tickers, start)
    carry = macro = None
    if with_carry:
        try:
            carry = load_carry(start=start, tickers=tickers, prices=prices)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, report what failed
            print(f"[load_bundle] carry unavailable, continuing without it: {e}")
    if with_macro:
        try:
            macro = load_fred(start=start)
        except Exception as e:  # noqa: BLE001
            print(f"[load_bundle] macro unavailable, continuing without it: {e}")
    print(f"[load_bundle] prices={list(prices.assets)} "
          f"carry={list(carry.assets) if carry is not None else None} "
          f"macro={'yes' if macro is not None else None}")
    return DataBundle(prices=prices, carry=carry, macro=macro)
