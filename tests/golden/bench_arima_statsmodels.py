"""Time statsmodels' SARIMAX on the rust-stats ARIMA bench workloads.

Pair with `examples/bench_arima.rs` for a side-by-side comparison.

statsmodels has no plain-CSS option in SARIMAX — everything goes through
Kalman-filter Gaussian MLE. We use the `lbfgs` optimizer (the SARIMAX
default), which is the closest analog to rust-stats' MLE path.

Run with:

    python3 tests/golden/bench_arima_statsmodels.py
"""

import time
import warnings
from statistics import median

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def time_iters(iters, fn):
    fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def report(label, n, extra, secs):
    print(f"{label:<32} n={n:<6} {extra:<20} {secs * 1e3:>10.2f} ms")


def simulate_arma(n, phi, theta, sigma, seed):
    burn = 200
    total = n + burn
    rng = np.random.default_rng(seed)
    eps = sigma * rng.standard_normal(total)
    y = np.zeros(total)
    p = len(phi)
    q = len(theta)
    for t in range(total):
        yt = eps[t]
        for i in range(min(p, t)):
            yt += phi[i] * y[t - 1 - i]
        for i in range(min(q, t)):
            yt += theta[i] * eps[t - 1 - i]
        y[t] = yt
    return y[burn:]


def integrate_once(y, start=100.0):
    out = np.empty_like(y)
    running = start
    for i, v in enumerate(y):
        running += v
        out[i] = running
    return out


def fit_sarimax(y, order, seasonal_order=(0, 0, 0, 0)):
    """statsmodels' SARIMAX with default Kalman + L-BFGS."""
    return SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 and seasonal_order[1] == 0 else "n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False)


def bench_one(label, y, order, iters, seasonal_order=(0, 0, 0, 0)):
    def fn():
        fit_sarimax(y, order, seasonal_order)
    secs = time_iters(iters, fn)
    report(f"{label} (statsmodels)", len(y), "", secs)


def bench_ar1():
    for n, iters in [(144, 50), (720, 20), (2880, 5)]:
        y = simulate_arma(n, [0.6], [], 1.0, 0xA1)
        bench_one("ARIMA(1,0,0)", y, (1, 0, 0), iters)


def bench_ma1():
    for n, iters in [(144, 50), (720, 20), (2880, 5)]:
        y = simulate_arma(n, [], [0.5], 1.0, 0xA2)
        bench_one("ARIMA(0,0,1)", y, (0, 0, 1), iters)


def bench_arma11():
    for n, iters in [(144, 30), (720, 15), (2880, 3)]:
        y = simulate_arma(n, [0.5], [0.3], 1.0, 0xA3)
        bench_one("ARIMA(1,0,1)", y, (1, 0, 1), iters)


def bench_ima11():
    for n, iters in [(144, 30), (720, 15), (2880, 3)]:
        arma = simulate_arma(n, [], [-0.4], 1.0, 0xA4)
        y = integrate_once(arma)
        bench_one("ARIMA(0,1,1)", y, (0, 1, 1), iters)


def bench_arima111():
    for n, iters in [(144, 20), (720, 10), (2880, 3)]:
        arma = simulate_arma(n, [0.5], [-0.3], 1.0, 0xA5)
        y = integrate_once(arma)
        bench_one("ARIMA(1,1,1)", y, (1, 1, 1), iters)


def bench_sarima_airline():
    for n, iters in [(144, 5), (288, 3)]:
        arma = simulate_arma(n, [], [-0.4], 1.0, 0xA6)
        i = np.arange(n)
        trend = 0.05 * i
        phase = 2 * np.pi * (i % 12) / 12
        seasonal = 3.0 * np.sin(phase)
        y = arma + trend + seasonal + 100.0
        for k in range(1, n):
            y[k] += y[k - 1] * 0.001
        bench_one("SARIMA(0,1,1)(0,1,1)[12]", y, (0, 1, 1), iters,
                  seasonal_order=(0, 1, 1, 12))


def main():
    print("# statsmodels SARIMAX benchmark (matches examples/bench_arima.rs)")
    print()
    bench_ar1()
    print()
    bench_ma1()
    print()
    bench_arma11()
    print()
    bench_ima11()
    print()
    bench_arima111()
    print()
    bench_sarima_airline()


if __name__ == "__main__":
    main()
