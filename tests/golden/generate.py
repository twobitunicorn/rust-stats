"""Generate golden reference values for rust-stats parity tests.

Covers OLS, classical seasonal_decompose, STL, and LOESS — for each, fits
the corresponding statsmodels routine and dumps inputs + outputs as JSON.

Run manually:
    python3 tests/golden/generate.py

Pinned versions documented below to make outputs reproducible.
Do NOT run this from cargo test. Outputs are committed to source control.
"""
# Tested with: numpy 2.4+, scipy 1.17+, statsmodels 0.14+, pandas 3.0+

import json
import os
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.datasets as smds
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL, seasonal_decompose

OUT_DIR = Path(__file__).parent

COV_TYPES = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]


def fit_and_dump(name, y, x_no_intercept, x_predict_no_intercept, intercept=True):
    if intercept:
        x = sm.add_constant(x_no_intercept, has_constant="add")
        x_pred = sm.add_constant(x_predict_no_intercept, has_constant="add")
    else:
        x = x_no_intercept
        x_pred = x_predict_no_intercept

    out = {
        "y": list(map(float, y.flatten())),
        "x": x_no_intercept.tolist(),
        "intercept": bool(intercept),
        "x_predict": x_predict_no_intercept.tolist(),
    }

    base = sm.OLS(y, x).fit()
    out["coef"]          = list(map(float, base.params))
    out["residuals"]     = list(map(float, base.resid))
    out["fitted"]        = list(map(float, base.fittedvalues))
    out["rss"]           = float(base.ssr)
    out["sigma"]         = float(np.sqrt(base.scale))
    out["r_squared"]     = float(base.rsquared)
    out["adj_r_squared"] = float(base.rsquared_adj)
    out["fvalue"]        = float(base.fvalue)
    out["f_pvalue"]      = float(base.f_pvalue)

    out["per_cov_type"] = {}
    for ct in COV_TYPES:
        if ct == "nonrobust":
            r = base
        else:
            r = sm.OLS(y, x).fit(cov_type=ct)
        out["per_cov_type"][ct] = {
            "std_err":  list(map(float, r.bse)),
            "t_values": list(map(float, r.tvalues)),
            "p_values": list(map(float, r.pvalues)),
            "conf_int_95": [list(map(float, row)) for row in r.conf_int(alpha=0.05)],
        }

    pred = base.get_prediction(x_pred)
    out["predict_point"]      = list(map(float, pred.predicted_mean))
    pi = pred.summary_frame(alpha=0.05)
    out["predict_interval_95"] = [
        [float(pi["mean"][i]), float(pi["obs_ci_lower"][i]), float(pi["obs_ci_upper"][i])]
        for i in range(len(pi))
    ]

    target = OUT_DIR / f"{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def longley():
    df = smds.longley.load_pandas().data
    y = df["TOTEMP"].to_numpy()
    x = df.drop(columns=["TOTEMP"]).to_numpy()
    x_pred = x[:3]  # arbitrary held-out slice
    fit_and_dump("longley", y, x, x_pred, intercept=True)


def mtcars():
    df = smds.get_rdataset("mtcars", "datasets").data
    y = df["mpg"].to_numpy()
    x = df[["cyl", "hp", "wt"]].to_numpy().astype(float)
    x_pred = x[:5]
    fit_and_dump("mtcars", y, x, x_pred, intercept=True)


def synthetic():
    rng = np.random.default_rng(20260509)
    n, p = 200, 4
    x = rng.standard_normal((n, p))
    beta = np.array([0.5, -1.2, 2.1, 0.3])
    y = 1.0 + x @ beta + rng.standard_normal(n) * 0.5
    x_pred = rng.standard_normal((10, p))
    fit_and_dump("synthetic", y, x, x_pred, intercept=True)


def heteroskedastic():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.uniform(0.5, 5.0, size=(n, 1))
    eps = rng.standard_normal(n) * x[:, 0]   # variance ∝ x²
    y = 2.0 + 3.0 * x[:, 0] + eps
    x_pred = np.array([[1.0], [2.5], [4.0]])
    fit_and_dump("heteroskedastic", y, x, x_pred, intercept=True)


def rank_deficient_input():
    """Only saves the input — there is no reference fit because statsmodels
    will silently use a pseudoinverse and we want the rust side to error."""
    rng = np.random.default_rng(7)
    n = 25
    x_base = rng.standard_normal((n, 2))
    x = np.column_stack([x_base[:, 0], x_base[:, 1], x_base[:, 0]])  # col 2 == col 0
    y = rng.standard_normal(n)
    out = {
        "y": list(map(float, y)),
        "x": x.tolist(),
    }
    target = OUT_DIR / "rank_deficient.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


# ── Time-series fixtures ────────────────────────────────────────────────────


def _quarterly_series(n_years=8, seed=20260510):
    """Quarterly synthetic series: linear trend + sin-driven seasonal + noise."""
    rng = np.random.default_rng(seed)
    n = n_years * 4
    t = np.arange(n, dtype=float)
    trend = 10.0 + 0.25 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / 4.0) + 1.5 * np.cos(4 * np.pi * t / 4.0)
    noise = rng.standard_normal(n) * 0.5
    return trend + seasonal + noise


def _airpassengers():
    """Classic monthly AirPassengers (1949-01..1960-12, n=144, period=12)."""
    df = smds.get_rdataset("AirPassengers", "datasets").data
    return df["value"].to_numpy().astype(float)


def _ts_fit_decompose(name, y, period, mode):
    """statsmodels.tsa.seasonal.seasonal_decompose → JSON.
    NaN edges (first/last period/2) are preserved as nulls."""
    res = seasonal_decompose(y, period=period, model=mode, two_sided=True,
                             extrapolate_trend=0)
    out = {
        "y": list(map(float, y)),
        "period": int(period),
        "mode": mode,
        "trend":    [None if np.isnan(v) else float(v) for v in res.trend],
        "seasonal": [None if np.isnan(v) else float(v) for v in res.seasonal],
        "residual": [None if np.isnan(v) else float(v) for v in res.resid],
    }
    target = OUT_DIR / f"seasonal_decompose_{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def seasonal_decompose_goldens():
    y_q = _quarterly_series()
    _ts_fit_decompose("quarterly_additive",          y_q, 4,  "additive")
    _ts_fit_decompose("quarterly_multiplicative",    y_q + 50.0, 4, "multiplicative")
    y_air = _airpassengers()
    _ts_fit_decompose("airpassengers_additive",      y_air, 12, "additive")
    _ts_fit_decompose("airpassengers_multiplicative", y_air, 12, "multiplicative")


def _stl_fit(name, y, period, seasonal_window, mode):
    """statsmodels.tsa.seasonal.STL with no robustness, inner_iter=2.
    rust-stats matches: Cleveland 1990 trend-window default, robust=False."""
    if mode == "multiplicative":
        work = np.log(y)
    else:
        work = y
    stl = STL(work, period=period, seasonal=seasonal_window,
              robust=False).fit(inner_iter=2, outer_iter=0)
    if mode == "multiplicative":
        trend, seasonal, resid = np.exp(stl.trend), np.exp(stl.seasonal), np.exp(stl.resid)
    else:
        trend, seasonal, resid = stl.trend, stl.seasonal, stl.resid
    out = {
        "y": list(map(float, y)),
        "period": int(period),
        "seasonal_window": int(seasonal_window),
        "mode": mode,
        "trend":    list(map(float, trend)),
        "seasonal": list(map(float, seasonal)),
        "residual": list(map(float, resid)),
    }
    target = OUT_DIR / f"stl_{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def stl_goldens():
    y_q = _quarterly_series()
    _stl_fit("quarterly_additive",       y_q,        4,  7, "additive")
    _stl_fit("quarterly_multiplicative", y_q + 50.0, 4,  7, "multiplicative")
    y_air = _airpassengers()
    _stl_fit("airpassengers_additive",       y_air, 12, 7, "additive")
    _stl_fit("airpassengers_multiplicative", y_air, 12, 7, "multiplicative")


def _loess_fit(name, y, span):
    """statsmodels.nonparametric.smoothers_lowess.lowess with no robustness.
    rust-stats matches at degree=1 (lowess is degree-1 only)."""
    n = len(y)
    x = np.arange(n, dtype=float)
    smoothed = lowess(y, x, frac=span, it=0, return_sorted=False)
    out = {
        "y": list(map(float, y)),
        "span": float(span),
        "degree": 1,
        "smoothed": list(map(float, smoothed)),
    }
    target = OUT_DIR / f"loess_{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def loess_goldens():
    rng = np.random.default_rng(20260510)
    n = 200
    t = np.linspace(0.0, 4.0 * np.pi, n)
    smooth = np.sin(t) + 0.5 * np.cos(2.0 * t)
    noisy = smooth + rng.standard_normal(n) * 0.3
    _loess_fit("smooth_span30", smooth, 0.30)
    _loess_fit("smooth_span50", smooth, 0.50)
    _loess_fit("noisy_span30",  noisy,  0.30)
    _loess_fit("noisy_span50",  noisy,  0.50)


def main():
    longley()
    mtcars()
    synthetic()
    heteroskedastic()
    rank_deficient_input()
    seasonal_decompose_goldens()
    stl_goldens()
    loess_goldens()


if __name__ == "__main__":
    main()
