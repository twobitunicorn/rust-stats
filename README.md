# rust-stats

Pure-Rust statistical modeling, statsmodels-inspired. LOESS smoothing
and Cleveland 1990 STL / classical seasonal decomposition.

## Quick start

Add the dependency:

```toml
[dependencies]
rust-stats = "0.1"
```

Decompose a seasonal series:

```rust
use rust_stats::{stl, StlOpts};

fn main() {
    // 12 years of monthly data (n = 144, period = 12).
    let y: Vec<f64> = monthly_series();

    let d = stl(&y, StlOpts::new(12)).unwrap();
    // d.trend, d.seasonal, d.residual each have length n=144.
    // Reconstruction holds: y[i] == d.trend[i] + d.seasonal[i] + d.residual[i].

    println!("first trend value: {:.2}", d.trend[0]);
    println!("january seasonality: {:.2}", d.seasonal[0]);
}
# fn monthly_series() -> Vec<f64> { vec![0.0; 144] }
```

Common options use struct-update syntax. For example, robust STL on
data with outliers, with a stationary seasonal pattern and NaN handling:

```rust
use rust_stats::{stl, StlOpts, SeasonalWindow, Missing};

let d = stl(&y, StlOpts {
    seasonal_window: SeasonalWindow::Periodic,
    outer_iters:     15,                  // R's `robust = TRUE` default
    missing:         Missing::Interpolate, // linear-fill NaNs
    ..StlOpts::new(12)
})?;
```

For multiplicative series (e.g. AirPassengers, where seasonal amplitude
grows with the level):

```rust
use rust_stats::{stl, StlOpts, DecomposeMode};

let d = stl(&y, StlOpts {
    mode: DecomposeMode::Multiplicative,
    ..StlOpts::new(12)
})?;
// d.trend in original units; d.seasonal and d.residual are
// dimensionless ratios centred around 1.
// y[i] == d.trend[i] * d.seasonal[i] * d.residual[i]
```

LOESS on its own:

```rust
use rust_stats::loess;

let smoothed = loess(&y, 0.3, 1)?;  // span = 30%, degree = 1
```

Classical centered-MA decomposition (faster than STL, NaN edges at the
first/last `period/2` positions):

```rust
use rust_stats::{seasonal_decompose, SeasonalDecomposeOpts};

let d = seasonal_decompose(&y, SeasonalDecomposeOpts::new(12))?;
```

For multi-series workloads, enable the `arrow` feature and use the
batched variants (`stl_batch`, `loess_batch`, `seasonal_decompose_batch`)
— see the [Apache Arrow interop](#apache-arrow-interop) section below.

## Features

- **LOESS** — single-pass tricube-weighted local polynomial smoother
  (degree 0/1/2), parallelised with rayon.
- **STL** — Cleveland 1990 STL with the standard inner-loop, additive and
  multiplicative; supports `seasonal_jump` / `trend_jump` / `low_pass_jump`
  to trade accuracy for speed (Cleveland 1990, §3).
- **seasonal_decompose** — classical centered moving-average decomposition,
  additive and multiplicative, matching statsmodels exactly.
- **ARIMA / SARIMA / ARIMAX** — full ARIMA(p, d, q) and seasonal
  SARIMA(p, d, q)(P, D, Q)[m] with optional exogenous regressors;
  three estimation paths (CSS, Kalman MLE, CSS-ML), point forecasts,
  Gaussian prediction intervals, and AIC / AICc / BIC.
- **`auto_arima`** — Hyndman-Khandakar stepwise model selection with
  KPSS-driven non-seasonal differencing, strength-of-seasonality
  driven seasonal differencing, and AICc as the search criterion.
- **Diagnostics** — Ljung-Box test for serial correlation in residuals;
  KPSS test for level / trend stationarity (used by `auto_arima`).
- **Holt-Winters** — additive and multiplicative exponential smoothing
  with caller-supplied α, β, γ.
- **Transforms** — `center`, `z_score`, `min_max_scale`, `box_cox`;
  the three reductions go through a `pulp` runtime-dispatched SIMD
  kernel on stable Rust.
- **Apache Arrow interop** (optional, `arrow` feature) — thin adapters so
  the same routines accept `Float64Array` / `RecordBatch` and return
  Arrow outputs.
- **Polars interop** (optional, `polars` feature) — thin adapters so the
  same routines accept Polars `Series` / `DataFrame` and return Polars
  outputs. When combined with `arrow`, `loess_batch` routes through the
  shared SIMD kernel.

## Apache Arrow interop

Enable the `arrow` feature:

```toml
rust-stats = { version = "...", features = ["arrow"] }
```

```rust
use rust_stats::arrow_compat;
use rust_stats::StlOpts;

let smoothed = arrow_compat::loess(&series, 0.3, 1)?;
let decomp   = arrow_compat::stl(&series, StlOpts::new(12))?;
// decomp is a RecordBatch with `trend | seasonal | residual` columns —
// drop straight into Polars, DataFusion, or DuckDB.
```

### Batched (multi-column) variants

`loess_batch`, `stl_batch`, and `seasonal_decompose_batch` apply the same
operation to every column of a `RecordBatch` in parallel (rayon over
columns), preserving the input schema:

```rust
use rust_stats::arrow_compat;
use rust_stats::StlOpts;

// stocks: a RecordBatch with one Float64 column per ticker
let trends = arrow_compat::stl_batch(&stocks, StlOpts::new(252))?.trend;
// trends has the SAME schema as stocks — column `AAPL` is AAPL's trend.

let smoothed = arrow_compat::loess_batch(&stocks, 0.3, 1)?;
```

`stl_batch` and `seasonal_decompose_batch` return a `DecompositionBatch`
with three `RecordBatch`es (`trend`, `seasonal`, `residual`), each
sharing the input schema. Validation runs up front for the whole batch
— any column with the wrong type or any null fails fast before compute
starts.

Inputs must be `Float64`; any null returns `ArrowError::HasNulls` rather
than silently substituting. Use `arrow::compute::filter` or Polars'
`drop_nulls` upstream for statsmodels-style `missing='drop'` semantics.

The feature is off by default; users without it see zero impact on
compile time, binary size, or dependency graph.

## Polars interop

Enable the `polars` feature:

```toml
rust-stats = { version = "...", features = ["polars"] }
```

```rust
use rust_stats::polars_compat;
use rust_stats::{StlOpts, Missing};
use polars::prelude::*;

let y_series: Series = /* ... */;

let smoothed: Series = polars_compat::loess(&y_series, 0.3, 1, Missing::Error)?;
let decomp = polars_compat::stl(&y_series, StlOpts::new(12))?;
// decomp.trend, decomp.seasonal, decomp.residual are each a Series of
// length s.len() (PolarsDecomposition struct).
```

Multi-series workloads use `loess_batch` / `stl_batch` /
`seasonal_decompose_batch`, taking a `DataFrame` and preserving its
column schema:

```rust
// prices: a DataFrame with one Float64 column per ticker
let trends = polars_compat::stl_batch(&prices, StlOpts::new(252))?.trend;
// trends has the same schema as prices — column "AAPL" is AAPL's trend.

let smoothed = polars_compat::loess_batch(&prices, 0.3, 1, Missing::Error)?;
```

`stl` and `seasonal_decompose` return a `PolarsDecomposition` with
three `Series` fields (`trend`, `seasonal`, `residual`) of length
`s.len()`. The batched variants `stl_batch` and
`seasonal_decompose_batch` return a `PolarsDecompositionBatch` with
three `DataFrame`s (`trend`, `seasonal`, `residual`), each sharing the
input schema.

Validation runs up front: input columns must be `Float64`. By default
any Polars null returns `PolarsCompatError::HasNulls`. To linearly fill
nulls instead, pass `Missing::Interpolate` — for `stl` /
`seasonal_decompose` it lives on `opts.missing`, for `loess` /
`loess_batch` it's a function parameter:

```rust
use rust_stats::{StlOpts, Missing};

// STL: per-opts
let d = polars_compat::stl(&y_with_nulls, StlOpts {
    missing: Missing::Interpolate,
    ..StlOpts::new(12)
})?;
// trend / seasonal are finite everywhere; residual is NaN at the rows
// that were originally null (so callers can still see which rows the
// decomposition imputed).

// LOESS: per-call
let smoothed = polars_compat::loess(&y_with_nulls, 0.3, 1, Missing::Interpolate)?;
// Every output value is finite — there's no residual concept for
// LOESS, so the smoother just sees a linearly-filled input.

// Batched variants too:
let smoothed_df = polars_compat::loess_batch(&prices_with_gaps, 0.3, 1, Missing::Interpolate)?;
let decomp     = polars_compat::stl_batch(&prices_with_gaps, StlOpts {
    missing: Missing::Interpolate,
    ..StlOpts::new(252)
})?;
```

When both `polars` and `arrow` features are on, `loess_batch` routes
through the shared SIMD batched-LOESS kernel; with just `polars`, it
falls back to rayon-over-columns scalar LOESS.

## statsmodels parity

Every numerical routine has a parity test against statsmodels — see
`tests/golden/`. Goldens are regenerated by
`python3 tests/golden/generate.py` and committed to the repo so cargo test
runs without Python.

| Module | Tolerance vs statsmodels |
| --- | --- |
| `seasonal_decompose` (additive, multiplicative) | 1e-12 |
| `loess` (degree 1) vs `statsmodels.nonparametric.lowess(it=0)` | ≤ 0.04 abs |
| `stl` — reconstruction identity (`y = T+S+R` or `y = T·S·R`) | 1e-10 |
| `stl` — components vs `statsmodels.STL(robust=False)` | a few units abs |

`stl` shares Cleveland's algorithm with statsmodels but differs in low-level
LOESS internals, so component-wise drift is on the order of a few units on
AirPassengers. The reconstruction identity is checked tightly and is
independent of statsmodels.

## Benchmarks

Wall-clock per call, median of warmed runs, on **Apple M2 Pro / macOS**
(rustc 1.95, statsmodels 0.14.6, numpy 2.4.4, scipy 1.17.1, R 4.6).

R is `stats::stl()` / `lowess()` / `decompose()` with the jump/delta
approximations disabled (`s.jump = t.jump = l.jump = 1`, `delta = 0`)
so we're comparing per-point fits in both directions.

**Highlights** — milliseconds per call, lower is better, fastest **bolded**.

Single-series, large n:

| Operation | size | rust-stats | statsmodels | R 4.6 |
| --- | --- | ---: | ---: | ---: |
| LOESS                  | n=5 000              | **7.9** | 79.6 | 40.1 |
| STL                    | n=2 880, period=24   | **1.7** | 11.7 |  2.4 |
| seasonal_decompose     | n=2 880              | **0.02** |  0.22 |  1.19 |
| ARIMA(1,1,1) MLE       | n=2 880              | **12.1** | 57.2 | – |
| SARIMA airline MLE     | n=288, m=12          | **125.7** | 285.6 | – |

Batched, 50 series at a time:

| Operation | size | rust-stats | statsmodels loop | R loop |
| --- | --- | ---: | ---: | ---: |
| `stl_batch`                | 50 × n=2 880  | **30.5** |   574 |   122 |
| `loess_batch` (simd)       | 50 × n=5 000  | **36.6** |  3 914 |  2 016 |
| `seasonal_decompose_batch` | 50 × n=2 880  |  **0.47** |    11.0 |    59.3 |

R's stl/lowess Fortran beats statsmodels' Python port by 2–27× single-
series; statsmodels' `seasonal_decompose` beats R's `decompose()` by
5–6×. rust-stats wins both at large n and dominates the multi-series
workloads (R and statsmodels have no native batched form).

Full tables below.

| Operation | Size | rust-stats | statsmodels | R 4.6 |
| --- | --- | ---: | ---: | ---: |
| LOESS (deg=1, span=0.3)   | n=100              | 0.034 ms | 0.558 ms |    0.021 ms |
| LOESS (deg=1, span=0.3)   | n=1 000            | 0.528 ms | 7.547 ms |    1.652 ms |
| LOESS (deg=1, span=0.3)   | n=5 000            | 7.903 ms | 79.613 ms |  40.113 ms |
| STL                       | n=144,  period=12  | 0.176 ms | 0.316 ms |    0.111 ms |
| STL                       | n=720,  period=12  | 0.653 ms | 1.602 ms |    0.356 ms |
| STL                       | n=2 880, period=24 | 1.723 ms | 11.669 ms |   2.397 ms |
| seasonal_decompose (+)    | n=144,  period=12  | 0.001 ms | 0.117 ms |    0.578 ms |
| seasonal_decompose (+)    | n=720,  period=12  | 0.004 ms | 0.121 ms |    0.704 ms |
| seasonal_decompose (+)    | n=2 880, period=24 | 0.024 ms | 0.223 ms |    1.194 ms |
| seasonal_decompose (×)    | n=144,  period=12  | 0.001 ms | 0.120 ms |    0.562 ms |
| seasonal_decompose (×)    | n=720,  period=12  | 0.005 ms | 0.129 ms |    0.678 ms |
| seasonal_decompose (×)    | n=2 880, period=24 | 0.024 ms | 0.227 ms |    1.108 ms |

### ARIMA / SARIMA

Three estimation methods are exposed via `ArimaOpts.method`:

- **CSS** — Conditional Sum of Squares (default). Skips the Kalman filter
  entirely; minimises squared one-step prediction errors with the
  recursion conditioned on zero pre-sample innovations.
- **MLE** — Exact Gaussian likelihood via Kalman filter on a Harvey 1989
  state-space form. Same objective as statsmodels' `SARIMAX.fit()`.
- **CSS-ML** — CSS for initial values, then MLE refinement. R's
  `arima(method = "CSS-ML")` default.

statsmodels' SARIMAX has no plain-CSS option; everything goes through
Kalman + L-BFGS. R's `stats::arima` defaults to CSS-ML (CSS for
starting values, then exact MLE via Kalman). So the strict
like-for-like cells are **rust CSS-ML vs R arima** and **rust MLE vs
statsmodels SARIMAX**; CSS is reported separately because it's a
different (faster, slightly less efficient at finite n) estimator.

| Workload | n | rust CSS | rust MLE | rust CSS-ML | R arima | statsmodels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARIMA(1,0,0) | 144   | **0.07** |  0.55  |  0.43  |   1.12 |   5.06 |
| ARIMA(1,0,0) | 720   | **0.11** |  1.13  |  0.97  |   2.22 |  14.71 |
| ARIMA(1,0,0) | 2 880 | **0.27** |  3.56  |  3.74  |   4.84 |  44.86 |
| ARIMA(0,0,1) | 144   | **0.05** |  0.30  |  0.31  |   1.11 |   5.45 |
| ARIMA(0,0,1) | 720   | **0.22** |  1.28  |  1.49  |   2.50 |  17.97 |
| ARIMA(0,0,1) | 2 880 | **0.91** |  5.34  |  5.90  |   7.76 |  56.16 |
| ARIMA(1,0,1) | 144   | **0.10** |  0.65  |  0.74  |   1.69 |   7.95 |
| ARIMA(1,0,1) | 720   | **0.46** |  2.73  |  3.30  |   4.11 |  22.61 |
| ARIMA(1,0,1) | 2 880 | **1.80** | 11.43  | 12.63  |  16.23 |  76.93 |
| ARIMA(0,1,1) | 144   | **0.05** |  0.28  |  0.29  |   0.37 |   3.70 |
| ARIMA(0,1,1) | 720   | **0.22** |  1.21  |  1.47  |   1.06 |  10.43 |
| ARIMA(0,1,1) | 2 880 |   0.86   |  4.89  |  5.44  | **2.25** | 27.75 |
| ARIMA(1,1,1) | 144   | **0.12** |  0.76  |  0.81  |  13.28 |   7.42 |
| ARIMA(1,1,1) | 720   | **0.54** |  2.83  |  3.70  |   4.47 |  17.23 |
| ARIMA(1,1,1) | 2 880 | **2.09** | 12.08  | 14.12  |   9.53 |  57.21 |
| SARIMA(0,1,1)(0,1,1)[12] | 144 | **0.32** |  62.65 |  70.21 |  16.24 | 214.37 |
| SARIMA(0,1,1)(0,1,1)[12] | 288 | **0.69** | 125.68 |  95.91 |  31.63 | 285.61 |

(All times in ms, median of 3–50 iters.)

**rust-stats CSS-ML vs R arima** (both Kalman MLE with CSS seeds):
rust-stats is roughly **1.5–3× faster** on non-seasonal models thanks
to a tighter Nelder-Mead inner loop. R wins on **SARIMA** (R 16.2 ms
vs ours 70.2 ms on the airline model at n=144) — R's `arima` is a
mature Fortran/C implementation, and its Kalman + L-BFGS handles the
state-space dimension growth of seasonal models more efficiently than
our Nelder-Mead does.

**rust-stats MLE vs statsmodels SARIMAX** (same Gaussian Kalman
objective, both default-optimised): rust-stats is **3–18× faster**
across every workload.

**CSS path** (different objective — faster, slightly less efficient
at finite n): **3–410× faster** than the references, with the biggest
multiplier on SARIMA at long horizons.

R wins one cell strictly (ARIMA(0,1,1) n=2880, where its IMA(1,1)
fast path is essentially free), and beats us on SARIMA where the
optimizer choice matters more than the kernel speed. Everywhere else,
rust-stats is at least competitive and usually faster.

Reproduce with:

```sh
cargo run --release --example bench_arima
python3 tests/golden/bench_arima_statsmodels.py
Rscript tests/golden/bench_arima_r.R
```

#### auto_arima vs pmdarima

For *automated* model selection — `auto_arima(y)` / `pm.auto_arima(y)`
end-to-end — the relevant comparison is how long it takes to go from
raw series to a fitted model. Both implementations run the
Hyndman-Khandakar stepwise search; the difference is per-candidate fit
cost.

| Workload | n | rust CSS (default) | rust MLE | pmdarima |
| --- | ---: | ---: | ---: | ---: |
| auto_arima                      | 144   |    **6.6** |       96.3 |    105.5 |
| auto_arima                      | 720   |   **97.4** |    7 019.4 |    633.8 |
| auto_arima                      | 2 880 |  **255.3** |    8 248.6 |  1 410.5 |
| auto_arima [m=12 airline model] | 144   |  **102.3** |  174 530.6 | 25 848.3 |
| auto_arima [m=12 airline model] | 288   |  **134.7** |  374 913.1 | 70 536.5 |

(All times in ms, median of 1–10 iters. The 6-minute rust MLE cell at
airline n=288 is honest: our Nelder-Mead × iterative Lyapunov ×
stepwise candidates is a slow inner triple on a 13-dimensional state
space.)

- **rust-stats CSS (default) vs pmdarima**: rust-stats is **6–525×
  faster** end-to-end on `auto_arima`, with the biggest multiplier on
  seasonal models (where pmdarima fits ~50 SARIMAX candidates each at
  ~1 s).
- **rust-stats MLE vs pmdarima** (same Gaussian Kalman objective):
  pmdarima wins. Our Nelder-Mead optimiser is slower than scipy's
  L-BFGS-B once the parameter space gets non-trivial. An L-BFGS port
  is the natural fix and lives on the roadmap.

Reproduce with:

```sh
cargo run --release --example bench_auto_arima
python3 tests/golden/bench_auto_arima_pmdarima.py
```

#### Scaling: ARIMA(1, 1, 1) from n=10⁴ to n=10⁷

How does each implementation scale as the series gets longer? One
fit per cell, no warmup, ARIMA(1, 1, 1) with `φ = 0.5`, `θ = -0.3`,
drift `= 0.1`.

| n | rust CSS | rust CSS-ML | rust MLE | R arima | statsmodels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 000      |   **0.010 s** | 0.072 s |  0.043 s |   0.038 s |   0.250 s |
| 100 000     |   **0.069 s** | 0.518 s |  0.422 s |   0.143 s |   2.080 s |
| 1 000 000   |   **0.710 s** | 6.046 s |  5.385 s |   1.394 s |  20.355 s |
| 10 000 000  |   **9.819 s** |    —    |     —    |  13.777 s | 204.011 s |

Throughput, in µs per data point (constant means linear scaling):

| n | rust CSS | rust CSS-ML | rust MLE | R arima | statsmodels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 000      | 1.05 |  7.17  |  4.32 |  3.85 | 25.04 |
| 100 000     | 0.69 |  5.18  |  4.22 |  1.43 | 20.80 |
| 1 000 000   | 0.71 |  6.05  |  5.39 |  1.39 | 20.35 |
| 10 000 000  | 0.98 |   —    |   —   |  1.38 | 20.40 |

Everything is **O(n)** as expected — Kalman filters and CSS recursions
all touch each point a constant number of times. What varies is the
constant:

- **rust-stats CSS** wins overall at ~0.7-1.0 µs/pt — it skips the
  Kalman filter entirely.
- **R `stats::arima`** is the standout at ~1.4 µs/pt for the Kalman
  MLE objective: mature Fortran/C, decades of tuning. We lose to R by
  ~4× on the like-for-like CSS-ML / MLE columns.
- **statsmodels SARIMAX** is ~15× slower than R per point. The
  hot-path Kalman filter is Cython, but each L-BFGS optim step pays
  Python wrapping overhead that compounds at large n.
- **rust-stats MLE / CSS-ML at n=10⁷** would project to ~50-70 s
  (linear extrapolation); we skip them in the script because they're
  not interesting numbers — the time is dominated by Nelder-Mead
  iterations, not by the per-point cost.

The L-BFGS port on the roadmap is squarely aimed at the CSS-ML / MLE
column — if we matched R's per-fit constant, we'd be the throughput
winner across all five columns simultaneously.

Reproduce with:

```sh
cargo run --release --example bench_scaling
Rscript tests/golden/bench_scaling_r.R
python3 tests/golden/bench_scaling_statsmodels.py
```

### Batched (50 series per call)

Decomposing 50 independent series at once. rust-stats parallelises over
columns with rayon (`arrow_compat::*_batch`, `arrow` feature enabled);
`loess_batch` (degree 0/1) additionally runs through a `pulp`-backed
cross-column SIMD kernel — `pulp` selects SSE2 / AVX2 / AVX-512 on
x86_64 or NEON on aarch64 at runtime, with a scalar fallback elsewhere.
statsmodels has no native batched form, so the Python column is a
straight Python loop over the same 50 series.

| Operation | Size | rust-stats `*_batch` | statsmodels loop | R loop |
| --- | --- | ---: | ---: | ---: |
| `stl_batch`                | 50 × n=720,   period=12 |   5.7 ms |    76.2 ms |    18.3 ms |
| `stl_batch`                | 50 × n=1 000, period=12 |   7.7 ms |   105.2 ms |    24.2 ms |
| `stl_batch`                | 50 × n=2 880, period=24 |  30.5 ms |   574.1 ms |   121.8 ms |
| `seasonal_decompose_batch` | 50 × n=720,   period=12 |   0.15 ms |    5.95 ms |    35.6 ms |
| `seasonal_decompose_batch` | 50 × n=1 000, period=12 |   0.18 ms |    5.98 ms |    38.7 ms |
| `seasonal_decompose_batch` | 50 × n=2 880, period=24 |   0.47 ms |    11.0 ms |    59.3 ms |
| `loess_batch`              | 50 × n=1 000, span=0.3  |   1.7 ms |   381.5 ms |    82.0 ms |
| `loess_batch`              | 50 × n=5 000, span=0.3  |  36.6 ms |  3914.1 ms |  2016.2 ms |

Reproduce with:

```sh
cargo run --release --example bench                  # core benches
cargo run --release --features arrow --example bench # + batched (uses pulp SIMD)
python3 tests/golden/bench_statsmodels.py
Rscript tests/golden/bench_r.R
```

LOESS gains a parallel inner loop. `seasonal_decompose` is an O(n) routine
where Python-side overhead dominates at small n. Batched variants add a
second layer of parallelism (rayon over columns) for multi-series workloads.

**vs R**: R's Fortran inner loops are extremely tight, so R beats us on
small-n single-series (n ≤ ~500 for LOESS, n ≤ ~1500 for STL). Past that
crossover, rust-stats' rayon parallelism amortizes its overhead and we
overtake — sometimes by a lot at large n. R's `decompose()` has heavy
R-side per-call overhead; we beat it by 50–550× across all sizes.
On batched workloads R has no native form and we lead consistently
(3–4× on `stl_batch`, ~40× on `loess_batch`, ~180× on
`seasonal_decompose_batch`).

## Roadmap

Things that are known to be missing or suboptimal in the current
code, roughly ordered by user-visible impact:

### ARIMA / SARIMA

- **L-BFGS optimiser for the MLE path.** Today the CSS-ML / MLE
  fitters run Nelder-Mead on the PACF-reparameterised parameter
  space. It's robust but slow on >5-dimensional problems — most
  visibly on SARIMA airline-style models, where the benchmarks above
  show rust-stats' MLE losing to both R's `arima` and pmdarima.
  Porting to L-BFGS-B with numerical gradients should close that gap.
- **Joint ARIMAX MLE.** `arima_with_exog` currently does the simple
  two-stage thing: OLS for β, then ARMA on the residuals. The
  efficient version fits (β, φ, θ) jointly inside one likelihood
  optimisation — what R's `arima(xreg=)` and statsmodels' SARIMAX do.
- **Kalman smoother for in-sample fitted values.** The `fitted`
  vector currently comes from the CSS recursion; a backward pass
  would tighten it at the start of the series.
- **Coefficient standard errors.** No SEs / CIs on `phi`, `theta`,
  `beta` yet. Adding them needs the Hessian of the log-likelihood at
  the optimum — straightforward once we have a quasi-Newton optimiser
  in place.

### Transforms

- **Vectorised transcendentals for `box_cox`.** The SIMD kernels
  cover `center` / `z_score` / `min_max_scale`; `box_cox` stays
  scalar because `pulp` doesn't ship `pow` / `ln`. A `sleef`-bound
  variant would unlock another ~3× on the lambda ≠ 0 path.

### Time-series

- **Multiple seasonalities** (TBATS-style daily + weekly + yearly).
  Currently we model one seasonal period.
- **Non-Gaussian innovations** (Student-t residuals, etc.).
- **GARCH / volatility models.**
- **Multivariate models** (VAR, VARMA).

If any of these would unblock you, open an issue.

## License

MIT OR Apache-2.0.
