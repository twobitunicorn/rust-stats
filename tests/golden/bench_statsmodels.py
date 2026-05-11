"""Time the same operations as `examples/bench.rs` against statsmodels.

Methodology:
  - One warmup call per configuration (excluded).
  - Median of N iterations of a single call (no inner loop) using
    perf_counter so the timer resolution is sub-microsecond.
  - Reports milliseconds per call so the output lines up with the rust
    bench output.

Run:  python3 tests/golden/bench_statsmodels.py
"""
import time
from statistics import median

import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL, seasonal_decompose


def time_iters(iters, fn):
    fn()  # warmup
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def report(label, n, extra, secs):
    print(f"{label:<22} n={n:<6} {extra:<20} {secs * 1e3:>10.3f} ms")


def ols_inputs(n, p, seed=0xC0FFEE):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = 0.5 + np.arange(p) * 0.1
    y = 1.0 + x @ beta + rng.standard_normal(n) * 0.5
    return y, sm.add_constant(x, has_constant="add")


def series_with_seasonality(n, period, seed=0xCAFE):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    trend = 10.0 + 0.05 * i
    phase = 2.0 * np.pi * (i % period) / period
    seasonal = 3.0 * np.sin(phase) + 1.5 * np.cos(2.0 * phase)
    return trend + seasonal + rng.standard_normal(n) * 0.5


def bench_ols():
    for n, p, iters in [(100, 5, 200), (1_000, 10, 100), (10_000, 20, 30)]:
        y, x = ols_inputs(n, p)

        def fn():
            r = sm.OLS(y, x).fit()
            _ = r.bse  # force inference materialisation
            _ = r.rsquared
        secs = time_iters(iters, fn)
        report("ols + nonrobust inf", n, f"p={p}", secs)

    for n, p, iters in [(1_000, 10, 100), (10_000, 20, 30)]:
        y, x = ols_inputs(n, p)

        def fn():
            r = sm.OLS(y, x).fit(cov_type="HC3")
            _ = r.bse
        secs = time_iters(iters, fn)
        report("ols + HC3 inf", n, f"p={p}", secs)


def bench_loess():
    for n, span, iters in [(100, 0.3, 100), (1_000, 0.3, 30), (5_000, 0.3, 10)]:
        rng = np.random.default_rng(0xBEEF)
        y = rng.standard_normal(n)
        x = np.arange(n, dtype=float)

        def fn():
            _ = lowess(y, x, frac=span, it=0, return_sorted=False)
        secs = time_iters(iters, fn)
        report("loess (deg=1)", n, f"span={span}", secs)


def bench_stl():
    for n, period, iters in [(144, 12, 50), (720, 12, 20), (2_880, 24, 10)]:
        y = series_with_seasonality(n, period)

        def fn():
            _ = STL(y, period=period, robust=False).fit(inner_iter=2, outer_iter=0)
        secs = time_iters(iters, fn)
        report("stl", n, f"period={period}", secs)


def bench_seasonal_decompose():
    for n, period, iters in [(144, 12, 200), (720, 12, 100), (2_880, 24, 50)]:
        y = series_with_seasonality(n, period)

        def fn_add():
            _ = seasonal_decompose(y, period=period, model="additive",
                                   two_sided=True, extrapolate_trend=0)
        secs = time_iters(iters, fn_add)
        report("seasonal_decompose +", n, f"period={period}", secs)

        def fn_mul():
            _ = seasonal_decompose(y, period=period, model="multiplicative",
                                   two_sided=True, extrapolate_trend=0)
        secs = time_iters(iters, fn_mul)
        report("seasonal_decompose *", n, f"period={period}", secs)


def bench_batched():
    """Loop the single-series statsmodels routines over P series — the
    equivalent of rust-stats' stl_batch / seasonal_decompose_batch /
    loess_batch. statsmodels has no built-in multi-column variant."""
    rng = np.random.default_rng(0xABCD)

    def make_series_set(n, p, period):
        # Match the rust-side generator structure (trend + sin + noise).
        i = np.arange(n)
        out = np.empty((p, n))
        for j in range(p):
            phase = 2 * np.pi * (i % period) / period
            out[j] = (
                10.0 + 0.05 * i + 3.0 * np.sin(phase)
                + 0.5 * rng.standard_normal(n)
            )
        return out

    for n, p, period, iters in [(1_000, 50, 12, 20), (720, 50, 12, 30), (2_880, 50, 24, 10)]:
        series = make_series_set(n, p, period)

        def fn():
            for j in range(p):
                STL(series[j], period=period, robust=False).fit(
                    inner_iter=2, outer_iter=0
                )
        secs = time_iters(iters, fn)
        report("stl_batch (loop)",   n, f"p={p} period={period}", secs)

    for n, p, period, iters in [(1_000, 50, 12, 20), (720, 50, 12, 30), (2_880, 50, 24, 10)]:
        series = make_series_set(n, p, period)

        def fn():
            for j in range(p):
                seasonal_decompose(
                    series[j], period=period, model="additive",
                    two_sided=True, extrapolate_trend=0,
                )
        secs = time_iters(iters, fn)
        report("seasonal_decompose_batch", n, f"p={p} period={period}", secs)

    for n, p, iters in [(1_000, 50, 20), (5_000, 50, 5)]:
        series = make_series_set(n, p, 12)
        x = np.arange(n, dtype=float)

        def fn():
            for j in range(p):
                lowess(series[j], x, frac=0.3, it=0, return_sorted=False)
        secs = time_iters(iters, fn)
        report("loess_batch (loop)", n, f"p={p} span=0.3", secs)


def main():
    print("# statsmodels benchmark")
    print()
    bench_ols()
    print()
    bench_loess()
    print()
    bench_stl()
    print()
    bench_seasonal_decompose()
    print()
    bench_batched()


if __name__ == "__main__":
    main()
