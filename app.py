"""
Step 5: Streamlit dashboard for the VRP lab.

Run:  streamlit run app.py
Needs the CSVs produced by data.py, vrp.py, terms.py and returns.py in data/processed/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import returns as rt  # reuses the straddle hedging engine from step 4

PROC = Path(__file__).parent / "data" / "processed"
HORIZONS = {"9D": "iv_9d", "30D": "iv_30d", "3M": "iv_3m", "6M": "iv_6m", "1Y": "iv_1y"}
CRISES = {"2008-09 GFC": ("2008-01-01", "2009-12-31"),
          "2020 COVID": ("2020-02-01", "2020-12-31"),
          "2022 bear": ("2022-01-01", "2022-12-31")}

st.set_page_config(page_title="Volatility Risk Premium Lab", layout="wide")


@st.cache_data
def load():
    rd = lambda f: pd.read_csv(PROC / f, index_col=0, parse_dates=True)
    return {
        "panel": rd("panel.csv"), "vrp": rd("vrp.csv"), "terms": rd("terms.csv"),
        "cycles": rd("cycles.csv"),
        "vrp_summary": pd.read_csv(PROC / "vrp_summary.csv"),
        "regimes": pd.read_csv(PROC / "regime_table.csv"),
        "regs": pd.read_csv(PROC / "regressions.csv"),
        "hedge": pd.read_csv(PROC / "hedge_freq.csv"),
    }


@st.cache_data
def straddle_cycles(scale: float, delta_vol: str, hedge_every: int) -> pd.Series:
    """Recompute hedged short-straddle returns for a given vol scale (slow path, cached)."""
    panel = load()["panel"]
    dates = panel.index.to_list()
    S, vix, rf = panel["close"].values, panel["iv_30d"].values, panel["rf_3m"].values
    out = {}
    for i0 in range(0, len(panel) - rt.H, rt.H):
        if np.isnan(vix[i0]) or np.isnan(rf[i0]):
            continue
        pnl, _ = rt.hedged_straddle(dates, S, vix, rf, i0, scale=scale,
                                    delta_vol=delta_vol, hedge_every=hedge_every)
        out[dates[i0]] = pnl
    return pd.Series(out).dropna()


def shade_crises(fig):
    for name, (a, b) in CRISES.items():
        fig.add_vrect(x0=a, x1=b, fillcolor="grey", opacity=0.15, line_width=0,
                      annotation_text=name, annotation_position="top left")
    return fig


d = load()
panel, vrp, terms, cycles = d["panel"], d["vrp"], d["terms"], d["cycles"]

st.title("Volatility Risk Premium Lab")
st.caption("IV term structure, delta-hedged option returns, and the price of volatility risk. "
           "Free data only: CBOE VIX family, S&P 500 (yfinance), FRED. Index-level and model-based, "
           "not real option quotes.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Term structure", "Variance risk premium", "Hedged returns", "Regimes & regressions", "Notes"])

# ---------------------------------------------------------------- Term structure
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        start = st.date_input("From", value=pd.Timestamp("2011-01-04"),
                              min_value=panel.index.min().date(), max_value=panel.index.max().date())
        sub = panel.loc[str(start):]
        fig = go.Figure()
        for lab, col in HORIZONS.items():
            fig.add_trace(go.Scatter(x=sub.index, y=sub[col] * 100, name=lab, mode="lines", line=dict(width=1)))
        fig.update_layout(title="Implied volatility by maturity (%)", height=420, hovermode="x unified")
        st.plotly_chart(shade_crises(fig), use_container_width=True)
    with c2:
        avail = panel.dropna(subset=["iv_9d", "iv_3m", "iv_6m", "iv_1y"]).index
        day = st.select_slider("Curve on date", options=list(avail.date), value=avail[-1].date())
        row = panel.loc[str(day)].iloc[0] if isinstance(panel.loc[str(day)], pd.DataFrame) else panel.loc[str(day)]
        xs = ["9D", "30D", "3M", "6M", "1Y"]
        ys = [row[HORIZONS[x]] * 100 for x in xs]
        f2 = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
        f2.update_layout(title=f"IV curve on {day}", height=420, yaxis_title="IV (%)")
        st.plotly_chart(f2, use_container_width=True)
        st.metric("3M/30D ratio", f"{row['iv_3m'] / row['iv_30d']:.3f}",
                  "backwardation" if row["iv_3m"] < row["iv_30d"] else "contango", delta_color="off")

    slope_choice = st.selectbox("Slope (vol points)", ["slope_3m_30d", "slope_30d_9d", "slope_6m_3m", "slope_1y_30d"], index=0)
    s = terms[slope_choice].dropna() * 100
    f3 = go.Figure(go.Scatter(x=s.index, y=s, mode="lines", line=dict(width=1)))
    f3.add_hline(y=0, line_dash="dash")
    f3.update_layout(title=f"{slope_choice} (negative = backwardation)", height=300)
    st.plotly_chart(shade_crises(f3), use_container_width=True)

# ---------------------------------------------------------------- VRP
with tab2:
    hz = st.radio("Horizon", list(HORIZONS), index=1, horizontal=True)
    win = st.slider("Rolling average window (trading days)", 21, 504, 252, step=21)
    v = vrp[f"vrp_{hz}"].dropna()
    f = go.Figure()
    f.add_trace(go.Scatter(x=v.index, y=v, name="VRP (IV² − fwd RV²)", line=dict(width=0.5, color="lightsteelblue")))
    f.add_trace(go.Scatter(x=v.index, y=v.rolling(win).mean(), name=f"{win}d rolling mean", line=dict(width=2)))
    f.add_hline(y=0, line_dash="dash")
    f.update_layout(title=f"{hz} variance risk premium", height=420, hovermode="x unified",
                    yaxis_range=[max(v.min(), -0.5), min(v.max(), 0.3)])
    st.plotly_chart(shade_crises(f), use_container_width=True)
    st.caption("Y-axis clipped to [-0.5, 0.3] for readability; the crash tails are in the table below.")

    both = pd.concat([vrp[f"iv2_{hz}"], vrp[f"rv2_{hz}"]], axis=1).dropna()
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=both.index, y=np.sqrt(both.iloc[:, 0]) * 100, name="Implied vol", line=dict(width=1)))
    f4.add_trace(go.Scatter(x=both.index, y=np.sqrt(both.iloc[:, 1]) * 100, name="Forward realized vol", line=dict(width=1)))
    f4.update_layout(title="Implied vs subsequently realized vol (%)", height=320, hovermode="x unified")
    st.plotly_chart(f4, use_container_width=True)

    summ = d["vrp_summary"]
    st.subheader("Summary by horizon and regime (Newey-West t-stats; blank = too few independent windows)")
    st.dataframe(summ[summ["horizon"] == hz].round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Returns
with tab3:
    c1, c2, c3 = st.columns(3)
    scale = c1.slider("Straddle sigma = scale × VIX", 0.80, 1.00, 0.90, step=0.01,
                      help="VIX includes skew and overstates pure ATM vol; <1 is a more realistic haircut.")
    dvol = c2.selectbox("Delta vol", ["entry", "vix"], index=0,
                        format_func=lambda x: "Fixed at entry IV" if x == "entry" else "Re-marked to daily VIX")
    he = c3.selectbox("Hedge frequency", [1, 5, rt.H], index=0,
                      format_func=lambda x: {1: "Daily", 5: "Weekly", rt.H: "Never"}[x])
    with st.spinner("Re-running hedged straddle backtest..."):
        sret = straddle_cycles(scale, dvol, he)
    since = st.date_input("Start date", value=pd.Timestamp("1990-01-02"),
                          min_value=sret.index.min().date(), max_value=sret.index.max().date())
    sret = sret.loc[str(since):]
    series = {"Hedged short straddle (live settings above)": sret * 100,
              "Short variance swap (vol pts / vega)": cycles["varswap_volpts"].loc[str(since):]}
    pick = st.multiselect("Show", list(series), default=list(series))
    f5 = go.Figure()
    for name in pick:
        x = series[name].dropna()
        f5.add_trace(go.Scatter(x=x.index, y=x.cumsum(), name=name,
                                yaxis="y2" if "variance" in name else "y"))
    f5.update_layout(title="Cumulative additive P&L (non-overlapping 21-day cycles)", height=420,
                     yaxis=dict(title="Straddle: cumulative % of index"),
                     yaxis2=dict(title="Var swap: cumulative vol pts", overlaying="y", side="right"),
                     hovermode="x unified")
    st.plotly_chart(shade_crises(f5), use_container_width=True)

    stat = rt.stats(sret, "Hedged short straddle")
    m = st.columns(5)
    m[0].metric("Mean / cycle", f"{stat['mean_per_cycle'] * 100:.2f}%")
    m[1].metric("Annualized Sharpe", f"{stat['ann_sharpe']:.2f}")
    m[2].metric("Cycles up", f"{stat['pct_win'] * 100:.0f}%")
    m[3].metric("Worst cycle", f"{stat['worst'] * 100:.1f}%")
    m[4].metric("Max drawdown", f"{stat['max_drawdown_additive'] * 100:.1f}%")
    st.caption("Model-based: Black-Scholes at scale × VIX, real S&P moves, no dividends, no transaction costs.")

    st.subheader("Hedging frequency (scale = 1.0, entry-vol delta, all cycles)")
    st.dataframe(d["hedge"].round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Regimes & regressions
with tab4:
    st.subheader("Forward 30D VRP by term-structure state (Sept 2009 onward)")
    st.dataframe(d["regimes"].round(4), use_container_width=True, hide_index=True)
    st.subheader("Regressions of forward 30D VRP (Newey-West, 21 lags)")
    regs = d["regs"].copy()
    model = st.selectbox("Model", regs["model"].unique())
    r = regs[regs["model"] == model]
    st.dataframe(r[["var", "coef", "se", "t"]].round(4), use_container_width=True, hide_index=True)
    st.caption(f"R² = {r['r2'].iloc[0]:.3f},  n = {int(r['n'].iloc[0]):,}")

# ---------------------------------------------------------------- Notes
with tab5:
    st.markdown("""
**What this is.** An index-level study of the variance risk premium using only free data: CBOE
VIX9D / VIX / VIX3M / VIX6M / VIX1Y, S&P 500 closes, and the 3-month T-bill.

**Method.** VRP = IV² − forward realized variance (the next h trading days, annualized). Windows
overlap, so t-stats use Newey-West errors. Return backtests use non-overlapping 21-day cycles.

**Caveats.**
- VIX is a variance-swap-style measure that includes skew, so it overstates pure ATM vol. The
  straddle backtest at scale = 1.0 is flattering; use the slider.
- No bid/ask, dividends, or financing spreads. Numbers are upper bounds.
- The straddle is *synthetic* (Black-Scholes priced), not traded quotes.
- Regime results rest on a handful of episodes; do not over-read the tails.
""")
