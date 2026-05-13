"""Time pycatch22 on the same series sizes that `examples/bench_catch22.rs`
times. Pair the two outputs for an apples-to-apples README table.

The input distribution is Gaussian; for the README we only care about
wall-clock per call, not feature parity. (Parity is covered by
`tests/catch22_pycatch22.rs`.)

Run with:
    python3 tests/golden/bench_catch22_pycatch22.py
"""

import time
from statistics import median

import numpy as np
import pycatch22


def time_iters(iters, fn):
    fn()  # warmup
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples)


def main():
    print("\npycatch22.catch22_all (catch24=True)\n")
    print(f"{'n':>8s}  {'ms/call':>10s}")
    print("-" * 22)

    rng = np.random.default_rng(0xCAFEBABE)
    for n, iters in [
        (200,        50),
        (1_000,      20),
        (5_000,      10),
        (20_000,      5),
        (50_000,      3),
        (100_000,     3),
    ]:
        y = rng.standard_normal(n).tolist()
        secs = time_iters(iters, lambda: pycatch22.catch22_all(y, catch24=True))
        print(f"{n:>8}  {secs * 1000:>10.3f}")


if __name__ == "__main__":
    main()
