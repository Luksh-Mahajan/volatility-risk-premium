"""
Step 1: download and clean all free data for the VRP lab.

Sources (all free, no API keys):
  - CBOE VIX-family CSVs: VIX9D, VIX, VIX3M, VIX6M, VIX1Y (implied vol, % points)
  - yfinance: ^GSPC daily prices (long history)
  - FRED: DGS3MO (3-month Treasury yield, %), via the public CSV endpoint

Run:  python data.py
Output: data/raw/*.csv and data/processed/panel.csv
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_SERIES = {
    "VIX9D": "iv_9d",
    "VIX": "iv_30d",
    "VIX3M": "iv_3m",
    "VIX6M": "iv_6m",
    "VIX1Y": "iv_1y",
}
YF_TICKERS = {  # fallback if CBOE download fails
    "VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M", "VIX1Y": "^VIX1Y",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS = {"User-Agent": "Mozilla/5.0 (vrp-lab research script)"}


def _get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_cboe(name: str) -> pd.Series:
    """Daily close of a CBOE vol index. Falls back to yfinance on failure."""
    col = CBOE_SERIES[name]
    try:
        txt = _get(CBOE_URL.format(name=name))
        (RAW / f"{name}.csv").write_text(txt)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip().upper() for c in df.columns]
        date_col = "DATE"
        # VIX_History has OHLC columns; the others have a single CLOSE-like column
        close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = pd.Series(
            pd.to_numeric(df[close_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col], errors="coerce"),
            name=col,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  CBOE failed for {name} ({e}); falling back to yfinance")
        import yfinance as yf

        d = yf.download(YF_TICKERS[name], start="1990-01-01",
                        auto_adjust=False, progress=False)
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.rename(col)
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()


def fetch_sp500() -> pd.DataFrame:
    import yfinance as yf

    d = yf.download("^GSPC", start="1990-01-01", auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d.to_csv(RAW / "sp500.csv")
    return d.dropna()


def fetch_fred(sid: str, name: str) -> pd.Series:
    txt = _get(FRED_URL.format(sid=sid))
    (RAW / f"{sid}.csv").write_text(txt)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "value"]
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,  # "." -> NaN
        index=pd.to_datetime(df["date"]),
        name=name,
    )
    return s.dropna()


def build_panel() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    print("Downloading CBOE vol indices...")
    iv = pd.concat([fetch_cboe(n) for n in CBOE_SERIES], axis=1)

    print("Downloading S&P 500 from yfinance...")
    spx = fetch_sp500()

    print("Downloading 3M T-bill from FRED...")
    rf = fetch_fred("DGS3MO", "rf_3m")

    # Align on S&P trading days (the days we can actually hedge on)
    panel = spx.join(iv, how="left").join(rf, how="left")
    panel["rf_3m"] = panel["rf_3m"].ffill()  # FRED skips some holidays

    # Convert units: vol indices and yields are in percent -> decimals
    iv_cols = list(CBOE_SERIES.values())
    panel[iv_cols] = panel[iv_cols] / 100.0
    panel["rf_3m"] = panel["rf_3m"] / 100.0

    # Daily log return (used later for realized variance and hedging P&L)
    panel["ret"] = np.log(panel["close"]).diff()

    return panel


def validate(panel: pd.DataFrame) -> None:
    print("\n=== Validation ===")
    print(f"Rows: {len(panel):,}   {panel.index.min().date()} -> {panel.index.max().date()}")
    print("\nFirst valid date per series:")
    for c in ["iv_9d", "iv_30d", "iv_3m", "iv_6m", "iv_1y", "rf_3m"]:
        fv = panel[c].first_valid_index()
        miss = panel.loc[fv:, c].isna().sum() if fv is not None else "n/a"
        print(f"  {c:7s} start={fv.date() if fv is not None else None}  missing_after_start={miss}")

    # Sanity checks: implied vols should live in a plausible range
    for c in ["iv_9d", "iv_30d", "iv_3m", "iv_6m", "iv_1y"]:
        s = panel[c].dropna()
        if len(s) and (s.min() < 0.05 or s.max() > 1.5):
            print(f"  WARNING: {c} range [{s.min():.3f}, {s.max():.3f}] looks off (units?)")

    big = panel["ret"].abs() > 0.15
    if big.any():
        print(f"\nDays with |daily return| > 15% (check, e.g. Oct 1987 if history reaches it): {big.sum()}")
    stale = (panel["close"].diff() == 0).sum()
    print(f"Days with unchanged S&P close (possible stale data): {stale}")


if __name__ == "__main__":
    panel = build_panel()
    validate(panel)
    out = PROC / "panel.csv"
    panel.to_csv(out)
    print(f"\nSaved {out}")
