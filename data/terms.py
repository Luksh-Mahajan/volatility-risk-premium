"""
Step 3: IV term-structure slopes, contango/backwardation regimes, and regressions
of the forward 30D VRP on information known at each date.

Run:  python terms.py   (needs data/processed/panel.csv and vrp.csv from steps 1-2)
Output: data/processed/terms.csv, regime_table.csv, regressions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int):
    """OLS with Newey-West (Bartlett) standard errors. Returns table + R^2 + n."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv = d["y"].values
    Xv = np.column_stack([np.ones(len(d)), d[X.columns].values])
    n, k = Xv.shape
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    e = yv - Xv @ beta
    U = Xv * e[:, None]
    S = U.T @ U
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = U[L:].T @ U[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((yv - yv.mean()) @ (yv - yv.mean()))
    tab = pd.DataFrame({"coef": beta, "se": se, "t": beta / se},
                       index=["const"] + list(X.columns))
    return tab, r2, n


def trailing_rv2(ret: pd.Series, h: int) -> pd.Series:
    """Annualized realized variance over the h days ending today (known at t)."""
    return (ret ** 2).rolling(h, min_periods=h).sum() * 252.0 / h


def build_terms(panel: pd.DataFrame, vrp: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=panel.index)
    # slopes in vol points (decimals): positive = contango
    t["slope_30d_9d"] = panel["iv_30d"] - panel["iv_9d"]
    t["slope_3m_30d"] = panel["iv_3m"] - panel["iv_30d"]
    t["slope_6m_3m"] = panel["iv_6m"] - panel["iv_3m"]
    t["slope_1y_30d"] = panel["iv_1y"] - panel["iv_30d"]
    t["ratio_3m_30d"] = panel["iv_3m"] / panel["iv_30d"]
    t["backwardation"] = (t["ratio_3m_30d"] < 1.0).astype(float).where(t["ratio_3m_30d"].notna())
    t["rv2_trail_21"] = trailing_rv2(panel["ret"], 21)
    t["iv2_30d"] = panel["iv_30d"] ** 2
    t["vrp_30D"] = vrp["vrp_30D"]
    t["vrp_3M"] = vrp["vrp_3M"]
    return t


def regime_table(t: pd.DataFrame) -> pd.DataFrame:
    d = t.dropna(subset=["ratio_3m_30d", "vrp_30D"]).copy()
    d["state"] = np.where(d["ratio_3m_30d"] < 1.0, "backwardation (3M<30D)",
                          np.where(d["ratio_3m_30d"] < 1.10, "flat/mild contango (1.00-1.10)",
                                   "steep contango (>1.10)"))
    g = d.groupby("state")["vrp_30D"]
    out = pd.DataFrame({"n_days": g.size(), "pct_of_sample": g.size() / len(d),
                        "mean_vrp_30D": g.mean(), "median": g.median(),
                        "pct_positive": g.apply(lambda s: (s > 0).mean()),
                        "p05": g.quantile(0.05), "min": g.min()})
    return out


def run_regressions(t: pd.DataFrame) -> pd.DataFrame:
    # Variables scaled for readable coefficients: slopes in vol points (x100)
    d = t.copy()
    d["slope_3m_30d_pts"] = d["slope_3m_30d"] * 100
    d["slope_30d_9d_pts"] = d["slope_30d_9d"] * 100
    specs = {
        "A: lagged RV only (1990+)":      ("vrp_30D", ["rv2_trail_21"], "1990-01-01"),
        "B: IV2 + lagged RV (1990+)":     ("vrp_30D", ["iv2_30d", "rv2_trail_21"], "1990-01-01"),
        "C: slope 3M-30D (2009+)":        ("vrp_30D", ["slope_3m_30d_pts"], "2009-09-18"),
        "D: slope + lagged RV (2009+)":   ("vrp_30D", ["slope_3m_30d_pts", "rv2_trail_21"], "2009-09-18"),
        "E: slope + RV, ex-crisis (2009+)": ("vrp_30D", ["slope_3m_30d_pts", "rv2_trail_21"], "2009-09-18"),
        "F: 9D slope + 3M slope + RV (2011+)": ("vrp_30D", ["slope_30d_9d_pts", "slope_3m_30d_pts", "rv2_trail_21"], "2011-01-04"),
    }
    rows = []
    for name, (yc, xs, start) in specs.items():
        dd = d.loc[start:]
        if name.startswith("E"):
            m = ~(((dd.index >= "2020-02-01") & (dd.index <= "2020-12-31")))
            dd = dd[m]
        tab, r2, n = hac_ols(dd[yc], dd[xs], lags=21)
        for var, r in tab.iterrows():
            rows.append({"model": name, "var": var, "coef": r["coef"], "se": r["se"],
                         "t": r["t"], "r2": r2, "n": n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = pd.read_csv(PROC / "panel.csv", index_col=0, parse_dates=True)
    vrp = pd.read_csv(PROC / "vrp.csv", index_col=0, parse_dates=True)
    t = build_terms(panel, vrp)
    t.to_csv(PROC / "terms.csv")
    pd.set_option("display.width", 200)
    rt = regime_table(t)
    rt.to_csv(PROC / "regime_table.csv")
    print("\nForward 30D VRP by term-structure state (2009-09 onward)\n")
    print(rt.round(4).to_string())
    reg = run_regressions(t)
    reg.to_csv(PROC / "regressions.csv", index=False)
    print("\nRegressions of forward 30D VRP (Newey-West, 21 lags)\n")
    print(reg.round(4).to_string(index=False))
