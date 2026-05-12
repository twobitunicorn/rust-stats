"""Time pmdarima.auto_arima against rust-stats `auto_arima`.

Both run a Hyndman-Khandakar stepwise search over (p, d, q)(P, D, Q).
Per-candidate, pmdarima uses statsmodels SARIMAX (Kalman + L-BFGS).
rust-stats can use CSS (default), MLE, or CSS-ML — pmdarima's
default-equivalent is rust-stats with `ArimaMethod::Mle`.

Run with:

    python3 tests/golden/bench_auto_arima_pmdarima.py
"""

import time
import warnings
from statistics import median

import numpy as np
import pmdarima as pm

warnings.filterwarnings("ignore")


def time_iters(iters, fn):
    fn()  # warmup
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
    p, q = len(phi), len(theta)
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


def run_auto(label, y, seasonal=False, m=1, iters=3):
    """Bench pmdarima.auto_arima with the conventional defaults."""
    def fn():
        kwargs = dict(
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5, max_d=2,
        )
        if seasonal:
            kwargs.update(max_P=2, max_Q=2, max_D=1)
        # When seasonal=False, pmdarima ignores P/Q/D bounds — leave defaults.
        pm.auto_arima(y, **kwargs)
    secs = time_iters(iters, fn)
    report(f"{label} (pmdarima auto)", len(y), "", secs)


def main():
    print("# pmdarima.auto_arima benchmark (stepwise; statsmodels SARIMAX underneath)\n")

    # Non-seasonal: same workloads as bench_arima.rs
    print("Non-seasonal:")
    for n, iters in [(144, 5), (720, 3), (2880, 1)]:
        y = simulate_arma(n, [0.5], [-0.3], 1.0, 0xAA1)
        y = integrate_once(y)   # so auto_arima has a chance to pick d > 0
        run_auto("auto_arima", y, seasonal=False, m=1, iters=iters)

    print("\nSeasonal (airline):")
    for n, iters in [(144, 2), (288, 1)]:
        arma = simulate_arma(n, [], [-0.4], 1.0, 0xAA6)
        i = np.arange(n)
        trend = 0.05 * i
        phase = 2 * np.pi * (i % 12) / 12
        seasonal = 3.0 * np.sin(phase)
        y = arma + trend + seasonal + 100.0
        for k in range(1, n):
            y[k] += y[k - 1] * 0.001
        run_auto("auto_arima [m=12]", y, seasonal=True, m=12, iters=iters)


if __name__ == "__main__":
    main()
