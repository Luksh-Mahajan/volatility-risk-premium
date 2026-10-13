"""
Step 6a: daily option-chain snapshotter (free; yfinance only returns the CURRENT chain,
so history exists only if we save it ourselves).

Saves, per trading day, to data/snapshots/:
    chain_<TICKER>_<YYYY-MM-DD>.csv.gz   bid/ask/last/volume/OI/Yahoo IV for expiries <= MAX_DTE
    vix_<YYYY-MM-DD>.csv                 VIX9D/VIX/VIX3M/VIX6M/VIX1Y + ^IRX at snapshot time

Quote quality: Yahoo bid/ask are often 0 after the close, so the script only runs in a window
around 3:30-4:25pm ET. Two cron entries in the workflow cover EST and EDT; this gate makes
only one of them do work, and a per-day file check prevents duplicates.

Env vars:  FORCE=1 (ignore the time window and duplicate check, for manual testing)
           TICKERS=SPY,^SPX (default)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "snapshots"
ET = ZoneInfo("America/New_York")

SPOT_SYMBOL = {"SPY": "SPY", "^SPX": "^GSPC"}
VOL_INDICES = {"^VIX9D": "vix9d", "^VIX": "vix", "^VIX3M": "vix3m", "^VIX6M": "vix6m", "^VIX1Y": "vix1y"}
MAX_DTE = 250
MAX_MONEYNESS = 0.20  # keep strikes within +/-20% of spot
FORCE = os.environ.get("FORCE") == "1"
TICKERS = os.environ.get("TICKERS", "SPY,^SPX").split(",")


def last_price(symbol: str):
    """Most recent 1-minute close and its timestamp; None if unavailable."""
    for attempt in range(3):
        try:
            h = yf.Ticker(symbol).history(period="1d", interval="1m")
            if len(h):
                return float(h["Close"].iloc[-1]), h.index[-1]
        except Exception as e:  # noqa: BLE001
            print(f"  {symbol}: price attempt {attempt + 1} failed ({e})")
        time.sleep(2 * (attempt + 1))
    return None


def snapshot_chain(ticker: str, asof, spot: float) -> pd.DataFrame | None:
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
    except Exception as e:  # noqa: BLE001
        print(f"  {ticker}: could not list expiries ({e})")
        return None
    frames = []
    for exp in expiries:
        dte = (pd.Timestamp(exp).date() - asof.date()).days
        if dte < 1 or dte > MAX_DTE:
            continue
        for attempt in range(3):
            try:
                ch = t.option_chain(exp)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {ticker} {exp}: attempt {attempt + 1} failed ({e})")
                time.sleep(2 * (attempt + 1))
                ch = None
        if ch is None:
            continue
        for side, df in (("C", ch.calls), ("P", ch.puts)):
            d = df.copy()
            d["type"] = side
            d["expiry"] = exp
            d["dte"] = dte
            frames.append(d)
        time.sleep(0.3)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["strike"] / spot - 1).abs() <= MAX_MONEYNESS]
    keep = ["expiry", "dte", "type", "strike", "bid", "ask", "lastPrice", "volume",
            "openInterest", "impliedVolatility", "lastTradeDate"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.insert(0, "ticker", ticker)
    df.insert(1, "spot", spot)
    df.insert(2, "asof", asof.isoformat())
    return df


def main() -> int:
    now = datetime.now(ET)
    day = now.strftime("%Y-%m-%d")
    mins = now.hour * 60 + now.minute
    if not FORCE:
        if now.weekday() >= 5:
            print("Weekend; nothing to do.")
            return 0
        if not (15 * 60 + 30 <= mins <= 16 * 60 + 25):
            print(f"Outside 15:30-16:25 ET window (now {now:%H:%M}); skipping.")
            return 0
    OUT.mkdir(parents=True, exist_ok=True)

    # Rates and vol indices (also confirms the market was open: price timestamp must be today)
    vix_row = {"asof": now.isoformat()}
    for sym, name in {**VOL_INDICES, "^IRX": "irx"}.items():
        r = last_price(sym)
        vix_row[name] = r[0] if r else None
    spx = last_price("^GSPC")
    if spx is None or (spx[1].tz_convert(ET).strftime("%Y-%m-%d") != day and not FORCE):
        print("No fresh S&P price for today (market holiday?); skipping.")
        return 0
    vix_file = OUT / f"vix_{day}.csv"
    if vix_file.exists() and not FORCE:
        print("Already snapshotted today; skipping.")
        return 0

    saved = 0
    for ticker in TICKERS:
        px = last_price(SPOT_SYMBOL.get(ticker, ticker))
        if px is None:
            print(f"{ticker}: no price, skipping")
            continue
        spot, _ = px
        df = snapshot_chain(ticker, now, spot)
        if df is None or df.empty:
            print(f"{ticker}: no chain data")
            continue
        n_quoted = int(((df["bid"] > 0) & (df["ask"] > 0)).sum())
        path = OUT / f"chain_{ticker.replace('^', '')}_{day}.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        print(f"{ticker}: {len(df):,} rows ({n_quoted:,} with bid&ask>0) -> {path.name}")
        saved += 1

    if saved:
        pd.DataFrame([vix_row]).to_csv(vix_file, index=False)
        print(f"Saved vol indices -> {vix_file.name}")
        return 0
    print("Nothing saved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
