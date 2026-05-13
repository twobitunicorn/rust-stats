"""Generate golden reference values for rust-stats parity tests
(seasonal_decompose, STL, LOESS).

Run manually:
    python3 tests/golden/generate.py

Pinned versions documented below to make outputs reproducible.
Do NOT run this from cargo test. Outputs are committed to source control.
"""
# Tested with: numpy 2.4+, scipy 1.17+, statsmodels 0.14+, pandas 3.0+

import json
from pathlib import Path

import numpy as np
import statsmodels.datasets as smds
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL, seasonal_decompose

OUT_DIR = Path(__file__).parent


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


def _catch22_fit(name, y, *, catch24=True, short_names=False):
    import pycatch22  # noqa: imported lazily so STL/LOESS goldens still work without it
    ref = pycatch22.catch22_all(
        list(map(float, y)),
        catch24=bool(catch24),
        short_names=bool(short_names),
    )
    # Map NaN / ±inf to JSON null so the file stays valid JSON
    # (serde_json rejects Python's non-standard NaN/Infinity literals).
    # The Rust side decodes None back to NaN.
    def jsonable(v):
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    out = {
        "y": list(map(float, y)),
        "names": list(ref["names"]),
        "values": [jsonable(v) for v in ref["values"]],
    }
    if short_names:
        out["short_names"] = list(ref["short_names"])
    target = OUT_DIR / f"catch22_{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def catch22_goldens():
    # Same seeds and sample sizes as the polars-timeseries Python tests,
    # so the rust-stats integration test exercises the canonical
    # pycatch22 surface on inputs known to compare cleanly.
    for seed in (0, 1, 7, 42, 1234):
        rng = np.random.default_rng(seed)
        _catch22_fit(f"normal_n200_seed{seed}", rng.standard_normal(200))

    # Edge-case goldens beyond the n=200 random-normal panel.
    # Pin the behavior pycatch22 produces on these so a future numeric
    # change in our kernels can't drift silently.
    _catch22_fit("constant_n200", np.full(200, 3.0))                          # std == 0 fallback
    rng = np.random.default_rng(0)
    near = 3.0 + 1e-12 * rng.standard_normal(200)
    _catch22_fit("near_constant_n200_seed0", near)                            # tiny-std stress
    _catch22_fit("normal_n20_seed0",   np.random.default_rng(0).standard_normal(20))   # short
    _catch22_fit("normal_n50_seed0",   np.random.default_rng(0).standard_normal(50))   # short
    _catch22_fit("normal_n10000_seed0", np.random.default_rng(0).standard_normal(10000))  # large

    # Round-trip pycatch22's short_names=True output so we can verify
    # CATCH22_SHORT_NAMES line up index-for-index with the canonical mapping.
    _catch22_fit(
        "normal_n200_seed0_shortnames",
        np.random.default_rng(0).standard_normal(200),
        short_names=True,
    )

    # Periodic series (sine etc.) are intentionally NOT included — they
    # produce integer-quantised features (SB_BinaryStats_*_longstretch*,
    # SB_TransitionMatrix_*) that sit on a threshold. A sub-ULP shift
    # between the rust-stats standalone build and an LTO'd downstream
    # build can change the count by ±1. The random-normal goldens
    # exercise every feature code path without this brittleness.


def main():
    seasonal_decompose_goldens()
    stl_goldens()
    loess_goldens()
    catch22_goldens()


if __name__ == "__main__":
    main()
