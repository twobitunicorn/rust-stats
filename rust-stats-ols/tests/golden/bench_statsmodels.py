"""Time the OLS bench from `examples/bench.rs` against statsmodels."""
import time
from statistics import median

import numpy as np
import statsmodels.api as sm


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


def ols_inputs(n, p, seed=0xC0FFEE):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = 0.5 + np.arange(p) * 0.1
    y = 1.0 + x @ beta + rng.standard_normal(n) * 0.5
    return y, sm.add_constant(x, has_constant="add")


def bench_ols():
    for n, p, iters in [(100, 5, 200), (1_000, 10, 100), (10_000, 20, 30)]:
        y, x = ols_inputs(n, p)
        def fn():
            r = sm.OLS(y, x).fit()
            _ = r.bse
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


def main():
    print("# statsmodels benchmark (rust-stats-ols)")
    print()
    bench_ols()


if __name__ == "__main__":
    main()
