"""
Step 4: return proxies for selling volatility, built only from free data.

(1) Short variance swap, per unit vega notional (in vol points):
        payoff = (K^2 - RV^2) / (2K) * 100,   K = VIX at entry (decimal), RV forward realized.
(2) SYNTHETIC delta-hedged short ATM straddle (model-based, NOT real quotes):
        - every 21 trading days, sell a 30-calendar-day ATM straddle priced by Black-Scholes at
          sigma = scale * VIX (VIX over-states pure ATM vol, so scale<1 is a sensitivity check),
        - hedge daily (or every `hedge_every` days) with BS delta on real S&P closes,
        - cash earns the 3M T-bill, no dividends (small upward bias in call-heavy P&L), no costs,
        - hold to expiry; return = P&L / S0.
    Two hedge-delta vols: 'entry' (fixed at entry IV) and 'vix' (re-marked to that day's VIX).

Cycles are non-overlapping (every 21 trading days) so t-stats and drawdowns are not inflated.

Run:  python returns.py   (needs data/processed/panel.csv)
Output: data/processed/cycles.csv, returns_summary.csv, hedge_freq.csv
"""
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
H = 21  # trading days per cycle (~30 calendar days)


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_straddle(S, K, T, r, sig):
    """Black-Scholes ATM-style straddle value and delta (q = 0)."""
    if T <= 0:
        return abs(S - K), float(np.sign(S - K))
    d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    call = S * ncdf(d1) - K * exp(-r * T) * ncdf(d2)
    put = K * exp(-r * T) * ncdf(-d2) - S * ncdf(-d1)
    return call + put, 2.0 * ncdf(d1) - 1.0


def hedged_straddle(dates, S, vix, rf, i0, scale=1.0, delta_vol="entry", hedge_every=1):
    """Sell straddle at i0, expire at i0+H. Returns (P&L per $ of S0, premium/S0)."""
    i1 = i0 + H
    S0, K = S[i0], S[i0]
    sig0 = scale * vix[i0]
    T_total = (dates[i1] - dates[i0]).days / 365.0
    prem, delta = bs_straddle(S0, K, T_total, rf[i0], sig0)
    cash = prem - delta * S0           # short straddle: receive premium, buy `delta` shares
    shares = delta
    for j in range(i0 + 1, i1):
        dt = (dates[j] - dates[j - 1]).days / 365.0
        cash *= exp(rf[j - 1] * dt)
        if (j - i0) % hedge_every == 0:
            T_rem = (dates[i1] - dates[j]).days / 365.0
            sig = sig0 if delta_vol == "entry" else scale * vix[j]
            _, new = bs_straddle(S[j], K, T_rem, rf[j], sig)
            cash -= (new - shares) * S[j]
            shares = new
    dt = (dates[i1] - dates[i1 - 1]).days / 365.0
    cash *= exp(rf[i1 - 1] * dt)
    pnl = cash + shares * S[i1] - abs(S[i1] - K)
    return pnl / S0, prem / S0


def stats(x: pd.Series, name: str) -> dict:
    x = x.dropna()
    cum = x.cumsum()
    dd = (cum - cum.cummax()).min()
    mu, sd = x.mean(), x.std()
    return {"strategy": name, "n_cycles": len(x), "mean_per_cycle": mu, "sd": sd,
            "t_stat": mu / (sd / sqrt(len(x))), "ann_sharpe": mu / sd * sqrt(12),
            "pct_win": (x > 0).mean(), "worst": x.min(), "p05": x.quantile(0.05),
            "skew": x.skew(), "max_drawdown_additive": dd}


def build_cycles(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel.index.to_list()
    S = panel["close"].values
    vix = panel["iv_30d"].values
    rf = panel["rf_3m"].values
    rows = []
    for i0 in range(0, len(panel) - H, H):
        if np.isnan(vix[i0]) or np.isnan(rf[i0]):
            continue
        i1 = i0 + H
        r = np.diff(np.log(S[i0:i1 + 1]))
        rv2 = (r ** 2).sum() * 252.0 / H
        K = vix[i0]
        row = {"date": dates[i0], "vix": K, "rv": sqrt(rv2),
               "varswap_volpts": (K ** 2 - rv2) / (2 * K) * 100.0}
        for tag, kw in {
            "straddle_entry": dict(scale=1.0, delta_vol="entry"),
            "straddle_vix": dict(scale=1.0, delta_vol="vix"),
            "straddle_vix_x0.9": dict(scale=0.9, delta_vol="vix"),
        }.items():
            pnl, prem = hedged_straddle(dates, S, vix, rf, i0, **kw)
            row[tag] = pnl
            if tag == "straddle_entry":
                row["premium_pct"] = prem
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def hedge_frequency(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel.index.to_list()
    S, vix, rf = panel["close"].values, panel["iv_30d"].values, panel["rf_3m"].values
    out = []
    for he, label in [(1, "daily"), (5, "weekly"), (H, "never (unhedged straddle)")]:
        vals = []
        for i0 in range(0, len(panel) - H, H):
            if np.isnan(vix[i0]) or np.isnan(rf[i0]):
                continue
            pnl, _ = hedged_straddle(dates, S, vix, rf, i0, delta_vol="entry", hedge_every=he)
            vals.append(pnl)
        v = pd.Series(vals)
        out.append({"hedging": label, "mean": v.mean(), "sd": v.std(),
                    "ann_sharpe": v.mean() / v.std() * sqrt(12), "worst": v.min(), "p05": v.quantile(.05)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    panel = pd.read_csv(PROC / "panel.csv", index_col=0, parse_dates=True)
    cyc = build_cycles(panel)
    cyc.to_csv(PROC / "cycles.csv")
    pd.set_option("display.width", 220)

    series = {
        "Short var swap (vol pts / vega)": "varswap_volpts",
        "Hedged straddle, entry-vol delta": "straddle_entry",
        "Hedged straddle, VIX-marked delta": "straddle_vix",
        "Hedged straddle, sigma = 0.9*VIX": "straddle_vix_x0.9",
    }
    summ = pd.DataFrame([stats(cyc[c], n) for n, c in series.items()])
    summ.to_csv(PROC / "returns_summary.csv", index=False)
    print(f"{len(cyc)} non-overlapping cycles, {cyc.index.min().date()} -> {cyc.index.max().date()}")
    print("\nMean premium collected (straddle, % of S0):", round(cyc['premium_pct'].mean() * 100, 2))
    print("\n", summ.round(4).to_string(index=False))

    sub = cyc.loc["2009-09-18":]
    print("\nSince 2009-09 (post-GFC):")
    print(pd.DataFrame([stats(sub[c], n) for n, c in series.items()])[
        ["strategy", "n_cycles", "mean_per_cycle", "ann_sharpe", "pct_win", "worst", "max_drawdown_additive"]
    ].round(4).to_string(index=False))

    for name, (a, b) in {"2008-09 GFC": ("2008-01-01", "2009-12-31"), "2020 COVID": ("2020-02-01", "2020-12-31"),
                         "2022 bear": ("2022-01-01", "2022-12-31")}.items():
        w = cyc.loc[a:b]
        print(f"\n{name}: n={len(w)}  varswap mean={w['varswap_volpts'].mean():.2f}  "
              f"straddle_entry mean={w['straddle_entry'].mean()*100:.2f}%  worst={w['straddle_entry'].min()*100:.2f}%")

    hf = hedge_frequency(panel)
    hf.to_csv(PROC / "hedge_freq.csv", index=False)
    print("\nHedging frequency (entry-vol delta):\n", hf.round(4).to_string(index=False))
