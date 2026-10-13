"""
Step 2: forward realized variance and the variance risk premium (VRP).

For each date t and horizon h (trading days):
    IV_t^2            implied variance from the matching CBOE index (annualized)
    RV_{t->t+h}^2     (252/h) * sum_{i=1..h} r_{t+i}^2   (forward-looking, annualized)
    VRP_t             IV_t^2 - RV_{t->t+h}^2

Only RV uses future data; IV is known at t. Overlapping windows make VRP serially
correlated, so t-stats use Newey-West (HAC) errors with h lags.

Run:  python vrp.py
Output: data/processed/vrp.csv, data/processed/vrp_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

# column -> (label, horizon in trading days). Calendar days / ~1.45 ≈ trading days.
HORIZONS = {
    "iv_9d": ("9D", 6),
    "iv_30d": ("30D", 21),
    "iv_3m": ("3M", 63),
    "iv_6m": ("6M", 126),
    "iv_1y": ("1Y", 252),
}

REGIMES = {
    "Full sample": (None, None),
    "2008-09 GFC": ("2008-01-01", "2009-12-31"),
    "2020 COVID": ("2020-02-01", "2020-12-31"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "Ex-crisis": None,  # everything outside the three windows above
}


def forward_rv2(ret: pd.Series, h: int) -> pd.Series:
    """Annualized realized variance over the h trading days AFTER each date."""
    sq = ret ** 2
    fwd_sum = sq[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return fwd_sum * 252.0 / h


def build_vrp(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    for col, (lab, h) in HORIZONS.items():
        iv2 = panel[col] ** 2
        rv2 = forward_rv2(panel["ret"], h)
        out[f"iv2_{lab}"] = iv2
        out[f"rv2_{lab}"] = rv2
        out[f"vrp_{lab}"] = iv2 - rv2
        out[f"volpts_{lab}"] = panel[col] - np.sqrt(rv2)  # IV - RV in vol terms
    return out


def nw_mean(x: pd.Series, lags: int):
    """Mean of x with Newey-West (Bartlett kernel) t-stat. Same estimator as
    statsmodels OLS(x, const).fit(cov_type='HAC', maxlags=lags), minus the
    small-sample correction."""
    x = x.dropna().values
    n = len(x)
    if n < 3 * lags:  # too few independent windows for a reliable HAC t-stat
        return np.nan, np.nan, n
    mu = x.mean()
    e = x - mu
    lrv = e @ e / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(lrv / n)
    return float(mu), float(mu / se), n


def summarize(vrp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, (lab, h) in HORIZONS.items():
        for name, win in REGIMES.items():
            if name == "Ex-crisis":
                m = pd.Series(True, index=vrp.index)
                for a, b in [v for v in REGIMES.values() if v and v[0]]:
                    m &= ~((vrp.index >= a) & (vrp.index <= b))
            elif win[0] is None:
                m = pd.Series(True, index=vrp.index)
            else:
                m = (vrp.index >= win[0]) & (vrp.index <= win[1])
            s = vrp.loc[m, f"vrp_{lab}"].dropna()
            if len(s) < 30:
                continue
            mean, t, n = nw_mean(s, lags=h)
            vp = vrp.loc[m, f"volpts_{lab}"].dropna()
            rows.append({
                "horizon": lab, "regime": name, "n_days": n,
                "mean_vrp_var": mean, "nw_t": t,
                "mean_volpts": vp.mean(),
                "pct_days_vrp>0": (s > 0).mean(),
                "sd_vrp": s.std(),
                "p05": s.quantile(0.05), "min": s.min(),
                "sharpe_like": mean / s.std(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = pd.read_csv(PROC / "panel.csv", index_col=0, parse_dates=True)
    vrp = build_vrp(panel)
    vrp.to_csv(PROC / "vrp.csv")
    summ = summarize(vrp)
    summ.to_csv(PROC / "vrp_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
