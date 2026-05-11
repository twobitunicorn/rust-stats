"""Time the rust-stats bench from examples/bench.rs against statsmodels."""
import time
from statistics import median

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL, seasonal_decompose


def time_iters(iters, fn):
    fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def report(label, n, extra, secs):
    print(f"{label:<22} n={n:<6} {extra:<20} {secs * 1e3:>10.3f} ms")


def series_with_seasonality(n, period, seed=0xCAFE):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    trend = 10.0 + 0.05 * i
    phase = 2.0 * np.pi * (i % period) / period
    seasonal = 3.0 * np.sin(phase) + 1.5 * np.cos(2.0 * phase)
    return trend + seasonal + rng.standard_normal(n) * 0.5


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
    rng = np.random.default_rng(0xABCD)

    def make_series_set(n, p, period):
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
                STL(series[j], period=period, robust=False).fit(inner_iter=2, outer_iter=0)
        secs = time_iters(iters, fn)
        report("stl_batch (loop)", n, f"p={p} period={period}", secs)

    for n, p, period, iters in [(1_000, 50, 12, 20), (720, 50, 12, 30), (2_880, 50, 24, 10)]:
        series = make_series_set(n, p, period)
        def fn():
            for j in range(p):
                seasonal_decompose(series[j], period=period, model="additive",
                                   two_sided=True, extrapolate_trend=0)
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


# ── center  ↔  sklearn StandardScaler(with_std=False) ───────────────────

def bench_center():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0xC1)
    for n, iters in [(10_000, 200), (100_000, 100), (1_000_000, 30)]:
        y = rng.standard_normal(n).reshape(-1, 1)
        sc = StandardScaler(with_std=False)
        def fn():
            sc.fit_transform(y)
        secs = time_iters(iters, fn)
        report("center (sklearn)", n, "", secs)


# ── z_score  ↔  sklearn StandardScaler (ddof=0; differs slightly from R) ─

def bench_z_score():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0xC2)
    for n, iters in [(10_000, 200), (100_000, 100), (1_000_000, 30)]:
        y = rng.standard_normal(n).reshape(-1, 1)
        sc = StandardScaler()
        def fn():
            sc.fit_transform(y)
        secs = time_iters(iters, fn)
        report("z_score (sklearn)", n, "", secs)


# ── min_max_scale  ↔  sklearn MinMaxScaler ─────────────────────────────

def bench_min_max():
    from sklearn.preprocessing import MinMaxScaler
    rng = np.random.default_rng(0xC3)
    for n, iters in [(10_000, 200), (100_000, 100), (1_000_000, 30)]:
        y = rng.standard_normal(n).reshape(-1, 1)
        sc = MinMaxScaler()
        def fn():
            sc.fit_transform(y)
        secs = time_iters(iters, fn)
        report("min_max_scale (sklearn)", n, "", secs)


# ── box_cox  ↔  scipy.stats.boxcox(x, lmbda=) ─────────────────────────
#
# scipy.stats.boxcox raises on non-positive y. We use exp(rnorm) + 0.5
# (matching the Rust bench) to keep everything strictly positive.

def bench_box_cox():
    from scipy import stats
    rng = np.random.default_rng(0xC4)
    for n, iters in [(10_000, 100), (100_000, 30), (1_000_000, 5)]:
        y = np.exp(rng.standard_normal(n)) + 0.5
        for lmbda in (0.0, 0.5, 2.0):
            def fn(_l=lmbda):
                stats.boxcox(y, lmbda=_l)
            secs = time_iters(iters, fn)
            report("box_cox (scipy)", n, f"lambda={lmbda}", secs)


# ── Holt-Winters  ↔  statsmodels ExponentialSmoothing(...).fit(optimized=False) ─

def bench_holt_winters():
    from statsmodels.tsa.holtwinters import (
        ExponentialSmoothing,
        Holt,
        SimpleExpSmoothing,
    )

    for n, period, iters in [(144, 12, 200), (720, 12, 100), (2_880, 24, 30)]:
        y = series_with_seasonality(n, period, seed=0xC5)
        y_pos = np.abs(y) + 1.0  # strictly positive for multiplicative

        def fn_ses():
            SimpleExpSmoothing(y, initialization_method="heuristic").fit(
                smoothing_level=0.5, optimized=False
            )
        secs = time_iters(iters, fn_ses)
        report("hw SES", n, f"period={period}", secs)

        def fn_holt():
            Holt(y, initialization_method="heuristic").fit(
                smoothing_level=0.5,
                smoothing_trend=0.1,
                optimized=False,
            )
        secs = time_iters(iters, fn_holt)
        report("hw Holt linear", n, f"period={period}", secs)

        def fn_add():
            ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=period,
                initialization_method="heuristic",
            ).fit(
                smoothing_level=0.5,
                smoothing_trend=0.1,
                smoothing_seasonal=0.2,
                optimized=False,
            )
        secs = time_iters(iters, fn_add)
        report("hw additive", n, f"period={period}", secs)

        def fn_mul():
            ExponentialSmoothing(
                y_pos,
                trend="add",
                seasonal="mul",
                seasonal_periods=period,
                initialization_method="heuristic",
            ).fit(
                smoothing_level=0.5,
                smoothing_trend=0.1,
                smoothing_seasonal=0.2,
                optimized=False,
            )
        secs = time_iters(iters, fn_mul)
        report("hw multiplicative", n, f"period={period}", secs)


def main():
    print("# statsmodels benchmark (rust-stats subset)")
    print()
    bench_loess()
    print()
    bench_stl()
    print()
    bench_seasonal_decompose()
    print()
    bench_batched()
    print()
    bench_center()
    print()
    bench_z_score()
    print()
    bench_min_max()
    print()
    bench_box_cox()
    print()
    bench_holt_winters()


if __name__ == "__main__":
    main()
