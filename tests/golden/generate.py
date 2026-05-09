"""Generate golden reference values for rust-stats OLS tests.

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


def main():
    longley()
    mtcars()
    synthetic()
    heteroskedastic()
    rank_deficient_input()


if __name__ == "__main__":
    main()
