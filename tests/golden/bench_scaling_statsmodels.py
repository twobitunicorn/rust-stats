"""Scaling sweep: statsmodels SARIMAX (Kalman + L-BFGS) on
ARIMA(1, 1, 1) at n = 10^4, 10^5, 10^6, 10^7. Pair with
examples/bench_scaling.rs.

The largest size will take a long time — statsmodels' Cython-backed
SARIMAX is competitive at small n but the per-iteration Python wrapping
overhead piles up.

Run with:

    python3 tests/golden/bench_scaling_statsmodels.py
"""

import sys
import time
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def simulate_arima_111(n, seed=0x5CA1ED):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    phi, theta = 0.5, -0.3
    diff = np.zeros(n)
    for t in range(1, n):
        diff[t] = 0.1 + phi * diff[t - 1] + theta * eps[t - 1] + eps[t]
    y = np.empty(n)
    y[0] = 100.0
    for t in range(1, n):
        y[t] = y[t - 1] + diff[t]
    return y


def main():
    print("# scaling sweep: SARIMAX(1, 1, 1) — one fit per cell\n")
    print("  n              time          throughput")
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        y = simulate_arima_111(n)
        t0 = time.perf_counter()
        try:
            SARIMAX(y, order=(1, 1, 1), trend="n").fit(disp=False)
            err = ""
        except Exception as e:  # noqa: BLE001
            err = f"  (errored: {e!r})"
        secs = time.perf_counter() - t0
        rate = secs * 1e6 / n
        print(f"  n={n:<10}    {secs:>10.3f} s    {rate:>7.2f} us/pt{err}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
