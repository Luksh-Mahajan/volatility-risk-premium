"""
Step 6b: analyze whatever the snapshotter has collected so far.

Answers, from real quotes:
  1. How far is VIX above the real ATM implied vol?  (replaces the 0.9x guess)
  2. How wide are ATM option spreads?               (real transaction cost)
  3. Does the chain-built term structure track VIX9D/VIX3M/VIX6M?
  4. How much skew is there (95% put vs ATM)?
  5. A real short-straddle position opened on the first snapshot day, delta-hedged and
     marked to market each day, compared with the synthetic model over the same dates.

IV is inverted from MID quotes (OTM options only) with Black-Scholes-Merton; Yahoo's own IV
field is ignored. SPY is American-style and pays dividends: pass --div-yield to match.

Run:  python analyze_snapshots.py [--ticker SPY] [--div-yield 0.013] [--scale 0.9]
"""
from __future__ import annotations

import argparse
import glob
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).parent
SNAP = ROOT / "data" / "snapshots"
PROC = ROOT / "data" / "processed"


# ----------------------------------------------------------------- Black-Scholes-Merton
def bsm(S, K, T, r, q, sig, cp):
    d1 = (log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    if cp == "C":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


def straddle_delta(S, K, T, r, q, sig):
    d1 = (log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    return exp(-q * T) * (2.0 * norm.cdf(d1) - 1.0)


def implied_vol(price, S, K, T, r, q, cp):
    lo = max(S * exp(-q * T) - K * exp(-r * T), 0.0) if cp == "C" else max(K * exp(-r * T) - S * exp(-q * T), 0.0)
    if price <= lo + 1e-10:
        return np.nan
    try:
        return brentq(lambda s: bsm(S, K, T, r, q, s, cp) - price, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------- loading
def load_days(ticker: str):
    days = {}
    for f in sorted(glob.glob(str(SNAP / f"chain_{ticker.replace('^', '')}_*.csv.gz"))):
        day = Path(f).name.split("_")[-1].replace(".csv.gz", "")
        vf = SNAP / f"vix_{day}.csv"
        if not vf.exists():
            continue
        days[pd.Timestamp(day)] = (pd.read_csv(f), pd.read_csv(vf).iloc[0])
    return days


def clean_quotes(df: pd.DataFrame, max_rel_spread: float = 0.5) -> pd.DataFrame:
    d = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    d["mid"] = (d["bid"] + d["ask"]) / 2.0
    d = d[(d["ask"] - d["bid"]) / d["mid"] <= max_rel_spread]
    return d[d["dte"] >= 2]


# ----------------------------------------------------------------- smile / term structure
def expiry_slices(df: pd.DataFrame, S: float, r: float, q: float):
    """Per expiry: (T, DataFrame[k, iv]) using OTM options relative to the forward."""
    out = []
    for exp_, g in clean_quotes(df).groupby("expiry"):
        T = g["dte"].iloc[0] / 365.0
        F = S * exp((r - q) * T)
        g = g[((g["type"] == "P") & (g["strike"] < F)) | ((g["type"] == "C") & (g["strike"] >= F))]
        ivs = [implied_vol(m, S, k, T, r, q, cp) for m, k, cp in zip(g["mid"], g["strike"], g["type"])]
        s = pd.DataFrame({"k": np.log(g["strike"].values / F), "iv": ivs}).dropna().sort_values("k")
        s = s[(s["iv"] > 0.02) & (s["iv"] < 3.0)].drop_duplicates("k")
        if len(s) >= 4 and s["k"].min() < 0 < s["k"].max():
            out.append((T, s))
    return sorted(out, key=lambda x: x[0])


def interp_at(s: pd.DataFrame, k: float):
    return float(np.interp(k, s["k"].values, s["iv"].values)) if s["k"].min() <= k <= s["k"].max() else np.nan


def const_maturity(slices, T_target: float, k: float = 0.0, max_gap: float = 2.0):
    """Interpolate IV at log-moneyness k to a target maturity, linear in total variance."""
    pts = [(T, interp_at(s, k)) for T, s in slices]
    pts = [(T, v) for T, v in pts if not np.isnan(v)]
    if not pts:
        return np.nan
    below = [p for p in pts if p[0] <= T_target]
    above = [p for p in pts if p[0] >= T_target]
    if not below or not above:
        return np.nan
    (T0, v0), (T1, v1) = below[-1], above[0]
    if T1 > T_target * max_gap and T0 < T_target * 0.5:
        return np.nan  # bracket too wide to trust
    if T1 == T0:
        return v0
    w0, w1 = v0 ** 2 * T0, v1 ** 2 * T1
    w = w0 + (w1 - w0) * (T_target - T0) / (T1 - T0)
    return sqrt(max(w, 1e-12) / T_target)


def atm_spread(df: pd.DataFrame, S: float, target_dte: int = 30):
    """Relative and dollar bid/ask spread of the ATM straddle on the expiry nearest target_dte."""
    d = df[(df["bid"] > 0) & (df["ask"] > 0)]
    if d.empty:
        return np.nan, np.nan, np.nan
    exps = d.groupby("expiry")["dte"].first()
    e = (exps - target_dte).abs().idxmin()
    g = d[d["expiry"] == e]
    k = g.loc[(g["strike"] - S).abs().idxmin(), "strike"]
    c, p = g[(g["strike"] == k) & (g["type"] == "C")], g[(g["strike"] == k) & (g["type"] == "P")]
    if c.empty or p.empty:
        return np.nan, np.nan, int(exps[e])
    sp = (c["ask"].iloc[0] - c["bid"].iloc[0]) + (p["ask"].iloc[0] - p["bid"].iloc[0])
    mid = (c["ask"].iloc[0] + c["bid"].iloc[0] + p["ask"].iloc[0] + p["bid"].iloc[0]) / 2
    return sp / mid, sp / S, int(exps[e])


# ----------------------------------------------------------------- analysis
def daily_table(days: dict, q: float) -> pd.DataFrame:
    rows = []
    for day, (df, vx) in days.items():
        S = float(df["spot"].iloc[0])
        r = float(vx["irx"]) / 100.0 if pd.notna(vx.get("irx")) else 0.04
        sl = expiry_slices(df, S, r, q)
        atm = {lab: const_maturity(sl, T / 365.0) for lab, T in [("9d", 9), ("30d", 30), ("90d", 91), ("180d", 182)]}
        k95 = log(0.95)
        sk = const_maturity(sl, 30 / 365.0, k=k95) - atm["30d"]
        rel, dol, dte_used = atm_spread(df, S)
        rows.append({"date": day, "spot": S, "rf": r,
                     "vix9d": vx.get("vix9d") / 100, "vix": vx.get("vix") / 100, "vix3m": vx.get("vix3m") / 100,
                     "vix6m": vx.get("vix6m") / 100,
                     "atm_9d": atm["9d"], "atm_30d": atm["30d"], "atm_90d": atm["90d"], "atm_180d": atm["180d"],
                     "skew_95put_minus_atm_30d": sk, "straddle_spread_rel": rel, "straddle_spread_pct_of_spot": dol,
                     "spread_expiry_dte": dte_used, "n_clean_quotes": len(clean_quotes(df))})
    t = pd.DataFrame(rows).set_index("date").sort_index()
    t["gap_vix_minus_atm30"] = t["vix"] - t["atm_30d"]
    t["ratio_atm30_over_vix"] = t["atm_30d"] / t["vix"]
    return t


def track_position(days: dict, q: float, scale: float, target_dte: int = 30):
    """Short ATM straddle opened on the first snapshot day, delta-hedged daily, marked at mid."""
    dates = sorted(days)
    d0 = dates[0]
    df0, _ = days[d0]
    S0 = float(df0["spot"].iloc[0])
    cand = df0[(df0["bid"] > 0) & (df0["ask"] > 0)]
    exps = cand.groupby("expiry")["dte"].first()
    expiry = (exps - target_dte).abs().idxmin()
    g = cand[cand["expiry"] == expiry]
    K = g.loc[(g["strike"] - S0).abs().idxmin(), "strike"]
    exp_ts = pd.Timestamp(expiry)

    def quotes(day):
        df, vx = days[day]
        x = df[(df["expiry"] == expiry) & (df["strike"] == K)]
        c, p = x[x["type"] == "C"], x[x["type"] == "P"]
        if c.empty or p.empty:
            return None
        return (float(df["spot"].iloc[0]), c.iloc[0], p.iloc[0],
                float(vx["irx"]) / 100 if pd.notna(vx.get("irx")) else 0.04, float(vx["vix"]) / 100)

    rows, prev = [], None
    cum_real = cum_syn = 0.0
    for day in dates:
        if day > exp_ts:
            break
        qd = quotes(day)
        if qd is None:
            continue
        S, c, p, r, vix = qd
        T = max((exp_ts - day).days, 0) / 365.0
        v_mid = c["bid"] / 2 + c["ask"] / 2 + p["bid"] / 2 + p["ask"] / 2
        if T > 0:
            iv_c = implied_vol((c["bid"] + c["ask"]) / 2, S, K, T, r, q, "C")
            iv_p = implied_vol((p["bid"] + p["ask"]) / 2, S, K, T, r, q, "P")
            sig = np.nanmean([iv_c, iv_p])
            delta = straddle_delta(S, K, T, r, q, sig) if not np.isnan(sig) else np.nan
            syn_sig = scale * vix
            v_syn = bsm(S, K, T, r, q, syn_sig, "C") + bsm(S, K, T, r, q, syn_sig, "P")
            d_syn = straddle_delta(S, K, T, r, q, syn_sig)
        else:
            v_mid, delta, v_syn, d_syn = abs(S - K), np.nan, abs(S - K), np.nan
        row = {"date": day, "spot": S, "straddle_mid": v_mid, "straddle_bid": c["bid"] + p["bid"],
               "straddle_ask": c["ask"] + p["ask"], "delta": delta, "synthetic_value": v_syn}
        if prev is not None and not np.isnan(prev["delta"]):
            dt = (day - prev["date"]).days / 365.0
            carry = (prev["straddle_mid"] - prev["delta"] * prev["spot"]) * prev["r"] * dt + prev["delta"] * prev["spot"] * q * dt
            real = -(v_mid - prev["straddle_mid"]) + prev["delta"] * (S - prev["spot"]) + carry
            carry_s = (prev["synthetic_value"] - prev["d_syn"] * prev["spot"]) * prev["r"] * dt + prev["d_syn"] * prev["spot"] * q * dt
            syn = -(v_syn - prev["synthetic_value"]) + prev["d_syn"] * (S - prev["spot"]) + carry_s
            cum_real += real
            cum_syn += syn
        row.update(cum_pnl_mid_pct=cum_real / S0 * 100, cum_pnl_synthetic_pct=cum_syn / S0 * 100)
        rows.append(row)
        prev = {**row, "r": r, "d_syn": d_syn}
    out = pd.DataFrame(rows).set_index("date")
    meta = {"expiry": expiry, "strike": float(K), "S0": S0, "entry_bid": float(out["straddle_bid"].iloc[0]),
            "entry_mid": float(out["straddle_mid"].iloc[0]), "entry_synthetic": float(out["synthetic_value"].iloc[0]),
            "last_date": out.index[-1], "reached_expiry": out.index[-1] >= exp_ts}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--div-yield", type=float, default=0.013)
    ap.add_argument("--scale", type=float, default=0.9)
    a = ap.parse_args()

    days = load_days(a.ticker)
    if not days:
        print(f"No snapshots for {a.ticker} in {SNAP}. Run snapshot.py first.")
        return
    PROC.mkdir(parents=True, exist_ok=True)
    t = daily_table(days, a.div_yield)
    t.to_csv(PROC / "snapshot_daily.csv")
    pd.set_option("display.width", 200)
    n = len(t)
    print(f"{a.ticker}: {n} snapshot days, {t.index.min().date()} -> {t.index.max().date()}\n")

    g = t["gap_vix_minus_atm30"].dropna()
    r_ = t["ratio_atm30_over_vix"].dropna()
    if len(g):
        print("1) VIX vs real 30D ATM IV")
        print(f"   VIX - ATM30: mean {g.mean()*100:.2f} vol pts (sd {g.std()*100:.2f}, n={len(g)})")
        print(f"   ATM30/VIX ratio: mean {r_.mean():.3f}  -> use as the 'scale' in the straddle backtest "
              f"(currently {a.scale})")
    for lab, vcol in [("9d", "vix9d"), ("90d", "vix3m"), ("180d", "vix6m")]:
        x = (t[vcol] - t[f"atm_{lab}"]).dropna()
        if len(x):
            print(f"   {vcol.upper()} - ATM{lab}: mean {x.mean()*100:.2f} pts (n={len(x)})")
    sp = t["straddle_spread_rel"].dropna()
    if len(sp):
        print(f"\n2) ATM straddle bid/ask spread (nearest-30D expiry): mean {sp.mean()*100:.2f}% of premium, "
              f"{t['straddle_spread_pct_of_spot'].mean()*100:.3f}% of spot")
    sk = t["skew_95put_minus_atm_30d"].dropna()
    if len(sk):
        print(f"\n3) Skew: 95% put IV - ATM IV (30D): mean {sk.mean()*100:.2f} vol pts")

    try:
        pos, meta = track_position(days, a.div_yield, a.scale)
        pos.to_csv(PROC / "snapshot_position.csv")
        print(f"\n4) Real short ATM straddle: expiry {meta['expiry']}, strike {meta['strike']:.0f}, "
              f"opened {pos.index[0].date()} at mid {meta['entry_mid']:.2f} (bid {meta['entry_bid']:.2f}); "
              f"synthetic model value {meta['entry_synthetic']:.2f}")
        last = pos.iloc[-1]
        half_entry = (pos["straddle_mid"].iloc[0] - pos["straddle_bid"].iloc[0]) / meta["S0"] * 100
        half_exit = (last["straddle_ask"] - last["straddle_mid"]) / meta["S0"] * 100
        print(f"   through {pos.index[-1].date()} ({'EXPIRED' if meta['reached_expiry'] else 'still open'}): "
              f"real cum P&L at mid {last['cum_pnl_mid_pct']:.2f}% of S0, "
              f"after entry/exit half-spreads {last['cum_pnl_mid_pct'] - half_entry - half_exit:.2f}%; "
              f"synthetic model {last['cum_pnl_synthetic_pct']:.2f}%")
        print("   (one trade = one data point: a calibration check, not evidence of an average return)")
    except Exception as e:  # noqa: BLE001
        print(f"\n4) Position tracking skipped: {e}")


if __name__ == "__main__":
    main()
