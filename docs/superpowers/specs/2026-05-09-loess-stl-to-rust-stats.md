# Move LOESS + STL into rust-stats; add seasonal_decompose

**Status**: design / awaiting user review
**Date**: 2026-05-09
**Spans**: `../rust-stats`, `polars-timeseries`

## Context

`polars-timeseries` currently embeds LOESS and STL implementations directly in
`src/expressions.rs`. They were grown organically next to the polars
`#[polars_expr]` glue. Both belong in a reusable Rust library: LOESS is a
general-purpose smoother, and STL is a general-purpose seasonal-trend
decomposer. A sibling crate, `../rust-stats`, already exists as the home for
"pure-Rust statistical modeling, statsmodels-inspired" — it currently has
the `regression::ols` module scaffolded with Faer-style types
(`ColRef<'a, f64>`, `MatRef<'a, f64>`) and a thiserror error enum.

This spec migrates LOESS and STL into rust-stats, adds a classical
moving-average `seasonal_decompose` alongside STL, and makes
`polars-timeseries` consume them. All public Rust APIs use Faer types.

## Goals

1. **Move LOESS to `rust_stats::smoothing`.** Free-function API. Tested.
2. **Move STL to `rust_stats::tsa::seasonal::stl`.** Free-function API.
   Tested.
3. **Add `rust_stats::tsa::seasonal::seasonal_decompose`.** Classical
   moving-average decomposition, alongside STL. Tested.
4. **`polars-timeseries` depends on rust-stats** (`path = "../rust-stats"`)
   and uses the imported implementations. Existing Python tests pass with
   no API drift.
5. **Add `pl.col("y").ts.seasonal_decompose(period=..., seasonal=...)`** as
   a new polars expression backed by `rust_stats::seasonal_decompose`.
6. **Time comparison against `statsmodels.tsa.seasonal.STL`** in
   `polars-timeseries/bench/bench_stl.py`, mirroring the existing
   `bench_loess.py` pattern.

## Non-goals

- Robust outer-loop iterations for STL (Tukey biweight reweighting).
  Out of scope; can be added later as `n_outer_iters: u32` on `StlOpts`.
- Exposing rust-stats' OLS through polars-timeseries.
- Replacing rust-stats' Faer dependency or upgrading Faer.

## Design

### rust-stats crate layout

```
rust-stats/
├── src/
│   ├── lib.rs                      # adds `pub mod smoothing` and `pub mod tsa`
│   ├── error.rs                    # adds LoessError, StlError, SeasonalDecomposeError
│   ├── smoothing/
│   │   ├── mod.rs                  # re-exports loess, loess_at
│   │   └── loess.rs                # public free fns + private helpers
│   └── tsa/
│       ├── mod.rs                  # re-exports `seasonal::*`
│       └── seasonal/
│           ├── mod.rs              # shared Decomposition + DecomposeMode + re-exports
│           ├── stl.rs              # Cleveland 1990 LOESS-based STL
│           └── decompose.rs        # Classical MA-based seasonal_decompose
├── tests/
│   ├── loess.rs                    # ported LOESS unit tests
│   ├── stl.rs                      # ported STL unit tests
│   └── seasonal_decompose.rs       # MA-decomp unit tests
└── Cargo.toml                      # adds rayon = "1.10"
```

`tsa` is reserved for time-series analysis (mirrors statsmodels' `tsa.*`).
LOESS lives under `smoothing` because it's a generic smoother that STL
happens to consume.

### Public Rust API

All public functions and types use Faer types where applicable. Inputs are
`ColRef<'_, f64>` (zero-copy borrowed view); outputs are `Col<f64>` (owned)
inside result types. No `Vec<f64>` or `&[f64]` in the public surface.

```rust
// rust_stats::smoothing
pub fn loess(
    y: ColRef<'_, f64>,
    span: f64,
    degree: u8,
) -> Result<Col<f64>, LoessError>;

pub fn loess_at(
    y: ColRef<'_, f64>,
    xq: f64,
    span: f64,
    degree: u8,
) -> Result<f64, LoessError>;

// rust_stats::tsa::seasonal
pub fn stl(
    y: ColRef<'_, f64>,
    opts: StlOpts,
) -> Result<Decomposition, StlError>;

pub fn seasonal_decompose(
    y: ColRef<'_, f64>,
    opts: SeasonalDecomposeOpts,
) -> Result<Decomposition, SeasonalDecomposeError>;

pub struct StlOpts {
    pub period: u32,
    pub seasonal_window: u32,           // odd, >= 7. Default 7.
    pub trend_window: Option<u32>,      // None = next_odd >= 1.5*period/(1-1.5/n_s)
    pub inner_iters: u32,               // default 2
    pub mode: DecomposeMode,            // default Additive
}

pub struct SeasonalDecomposeOpts {
    pub period: u32,
    pub mode: DecomposeMode,            // default Additive
}

pub enum DecomposeMode {
    Additive,
    Multiplicative,
}

pub struct Decomposition {
    pub trend:    Col<f64>,
    pub seasonal: Col<f64>,
    pub residual: Col<f64>,
}

impl StlOpts             { pub fn new(period: u32) -> Self; }
impl SeasonalDecomposeOpts { pub fn new(period: u32) -> Self; }
```

`Decomposition` and `DecomposeMode` are shared between STL and
`seasonal_decompose` since both produce identically-shaped output. STL's
result has finite values everywhere; `seasonal_decompose`'s `trend` and
`residual` are NaN at the first/last `period/2` positions (centered
moving-average edge band) — documented per function.

### Error types

```rust
// src/error.rs (added enums)

#[derive(Debug, Error, PartialEq)]
pub enum LoessError {
    #[error("span must be in (0, 1]; got {0}")]
    InvalidSpan(f64),
    #[error("degree must be 0, 1, or 2; got {0}")]
    InvalidDegree(u8),
    #[error("input is empty")]
    Empty,
    #[error("input contains non-finite values")]
    NonFinite,
}

#[derive(Debug, Error, PartialEq)]
pub enum StlError {
    #[error("period must be >= 2; got {0}")]
    InvalidPeriod(u32),
    #[error("seasonal_window must be odd and >= 7; got {0}")]
    InvalidSeasonalWindow(u32),
    #[error("trend_window must be odd; got {0}")]
    InvalidTrendWindow(u32),
    #[error("inner_iters must be >= 1; got 0")]
    InvalidInnerIters,
    #[error("series too short: needs >= 2*period samples, got {n} < {min}")]
    SeriesTooShort { n: usize, min: usize },
    #[error("multiplicative mode requires strictly positive values; got {min}")]
    NonPositiveForMultiplicative { min: f64 },
    #[error("input contains non-finite values")]
    NonFinite,
    #[error(transparent)]
    Loess(#[from] LoessError),
}

#[derive(Debug, Error, PartialEq)]
pub enum SeasonalDecomposeError {
    #[error("period must be >= 2; got {0}")]
    InvalidPeriod(u32),
    #[error("series too short: needs >= 2*period samples, got {n} < {min}")]
    SeriesTooShort { n: usize, min: usize },
    #[error("multiplicative mode requires strictly positive values; got {min}")]
    NonPositiveForMultiplicative { min: f64 },
    #[error("input contains non-finite values")]
    NonFinite,
}
```

All three are `Debug + thiserror::Error + PartialEq`, matching `OlsError`.

### Internal implementation strategy

- **Boundary I/O**: `ColRef::try_as_slice()` to obtain `&[f64]` for the input
  (Faer columns are contiguous in our use cases; non-contiguous returns
  an error wrapped as `LoessError::NonFinite` / `StlError::NonFinite`
  variant — actually we'll add a separate `NonContiguous` variant if it
  becomes a real concern; for the polars caller this is always contiguous).
  Output is built into a `Vec<f64>` and wrapped via `Col::from_iter(...)`
  before returning.

- **Length-n persistent state**: `Col<f64>` for `trend`, `seasonal`,
  `residual` carried across STL inner-loop iterations and handed to the
  caller in `Decomposition`.

- **Inner LOESS local fit**: small fixed-size `[f64; 9]` for the m×m
  normal-equations matrix (m ≤ 3, so up to 3×3 = 9 entries) and `[f64; 3]`
  for the right-hand side. Manual Gaussian elimination with partial
  pivoting for the 1×1 / 2×2 / 3×3 solve. Faer's `partial_piv_lu` would
  carry ~µs-scale per-call overhead which compounds across thousands of
  LOESS query points; manual elimination is ~10ns per solve and matches
  the existing implementation bit-for-bit. **Decided.**

- **Parallelism**: `rayon::par_iter` over the LOESS outer (per-query-point)
  loop, with the same `n >= 256` threshold used today. STL inherits this
  via its calls to LOESS.

- **STL helpers** (`cycle_subseries_smooth`, `low_pass_filter`, `valid_ma`,
  `stl_inner_loop`, `next_odd_ceil`) move verbatim from polars-timeseries.
  The only intra-crate call site change is `cycle_subseries_smooth` calling
  `loess_at` (was `local_poly_fit_at_xf64`) for the one-period
  extrapolation, and `low_pass_filter` / `stl_inner_loop` calling `loess`
  (was `loess_compute`).

- **`seasonal_decompose` helpers**: `centered_ma` (re-introduced — it was
  removed from polars-timeseries when STL switched to LOESS) and per-phase
  mean accumulation. Length-n outputs as `Col<f64>`. The phase-mean
  buffer (length = `period`) is a transient `Vec<f64>`.

### Tests in rust-stats

All tests use `approx::assert_relative_eq!` for float comparisons (the
`approx` crate is already a dev dep) and build inputs via
`Col::<f64>::from_iter(...)`.

**`tests/loess.rs`** ports the ten LOESS unit tests currently in
`polars-timeseries/tests/test_transforms.py`:

- Constant-signal recovery
- Exact linear recovery (degree=1)
- Exact quadratic recovery (degree=2)
- Wider span ⇒ smaller residual variance from underlying trend
- Step-function smoothing within bounded overshoot
- Constant-signal preservation with degree=2
- Short-series graceful fallback (n < degree+2)
- Boundary recovery exact on linear input
- No-extreme-overshoot
- Reproducibility (deterministic parallel reduction)

Plus the validation paths: `InvalidSpan`, `InvalidDegree`, `Empty`,
`NonFinite`.

**`tests/stl.rs`** ports the STL unit tests:

- Pure linear trend recovery (every position, no NaN)
- Pure seasonal pattern (every position)
- Additive reconstruction `y = T + S + R` (every position)
- Multiplicative reconstruction `y = T * S * R` (every position)
- Seasonal sums to zero across one period (additive)
- Seasonal product to one across one period (multiplicative)
- All validation paths (`InvalidPeriod`, `InvalidSeasonalWindow`,
  `InvalidTrendWindow`, `InvalidInnerIters`, `SeriesTooShort`,
  `NonPositiveForMultiplicative`)

**`tests/seasonal_decompose.rs`** is new, structured similarly to
`tests/stl.rs` but checking only the inner non-NaN band:

- Pure linear trend recovery on the inner band
- Pure seasonal pattern recovery on the inner band
- Additive reconstruction `y = T + S + R` on the inner band
- Multiplicative reconstruction on the inner band
- Seasonal pattern sums to zero (additive) / products to one
  (multiplicative) within one inner cycle
- Edge band (first/last `period/2`) is NaN — explicit assertion
- Validation: `InvalidPeriod`, `SeriesTooShort`,
  `NonPositiveForMultiplicative`, `NonFinite`

### polars-timeseries changes

**Cargo.toml**: add `rust-stats = { path = "../rust-stats" }`.

**`src/expressions.rs`**: delete the ten private LOESS + STL helpers
(`local_poly_fit_at_xf64`, `local_poly_fit_at`, `loess_window_f`,
`gauss_solve_n`, `loess_compute`, `cycle_subseries_smooth`,
`low_pass_filter`, `valid_ma`, `stl_inner_loop`, `next_odd_ceil`).
Rewrite the three `#[polars_expr]` entry points as thin wrappers:

- `loess(...)`: parse `LoessKwargs`, build a `ColRef<'_, f64>` view of the
  input column, call `rust_stats::loess`, convert the returned
  `Col<f64>` back to a `Float64Chunked`, return as `Series`.
- `stl(...)`: parse `StlKwargs`, build `StlOpts`, call `rust_stats::stl`,
  unpack `Decomposition` into a 3-field `StructChunked` with field names
  `trend / seasonal / residual` (rename — see below).
- **NEW** `seasonal_decompose(...)`: parse a new `SeasonalDecomposeKwargs`,
  build `SeasonalDecomposeOpts`, call `rust_stats::seasonal_decompose`,
  unpack into a 3-field `StructChunked` with the same `trend / seasonal /
  residual` field names as `stl`.

**API break: `noise` → `residual`.** The existing `stl()` polars expression
returns a Struct with a field named `noise`. We rename it to `residual` to
match rust-stats and statsmodels. Affected sites:

- The `stl` polars expression's output Struct field (Rust side).
- `python/polars_timeseries/__init__.py`: rename the `noise` free function
  and the `.ts.noise` namespace method to `residual`. Drop `noise` from
  `__all__`.
- `tests/test_transforms.py`: any test reading `result["x"][0]["noise"]`
  becomes `["residual"]`; `test_noise_method_returns_float_series` becomes
  `test_residual_method_returns_float_series`.
- `README.md`: replace `noise` with `residual` in the transforms table,
  the usage example, and the worked-example output table.

This is a breaking change but it's contained to four user-visible names
(`stl()` Struct field, `noise()` function, `.ts.noise()` method, and the
`SeasonalDecomposeKwargs.seasonal` value never had `noise` in it). No
deprecation shim — clean rename.

**`python/polars_timeseries/__init__.py`**:

- Add `seasonal_decompose(expr, *, period, seasonal="additive") -> pl.Expr`
  free function and `__all__` entry.
- Add `.ts.seasonal_decompose(...)` namespace method on
  `TimeSeriesNamespace`.
- Existing `loess` and `stl` Python wrappers are unchanged.

**`tests/test_transforms.py`**: existing LOESS and STL tests stay (they
test the unchanged Python API). New tests for the polars
`seasonal_decompose` expression — schema / shape, additive
reconstruction in the inner band, multiplicative reconstruction,
edge-band NaN, all validation paths — modelled on the existing STL tests.

**`bench/bench_stl.py`** (new): mirrors `bench_loess.py`. Compares
`pl.col("y").ts.stl(period=m)` against
`statsmodels.tsa.seasonal.STL(y, period=m, robust=False).fit()` on the
same statsmodels datasets used in `bench_loess.py` (sunspots, co2,
macrodata.realgdp, nile, elnino), with `period` chosen per-dataset
(11 for sunspots, 12 for co2, 4 for macrodata, 12 for elnino, 5 or 11
for nile depending on what statsmodels accepts). Reports timing
(median of 5 runs after warmup) and per-component max/mean |Δ| of
trend, seasonal, residual.

### Parallelism story

- `rust_stats::loess` parallelises across query points via `rayon` for
  `n >= 256` — same threshold as today. Below that, serial. This avoids
  paying rayon overhead on tiny inputs.
- `rust_stats::stl` calls `loess` repeatedly (cycle-subseries, low-pass,
  trend) and inherits the threshold.
- `rust_stats::seasonal_decompose` is plain Vec ops; no parallelism.
- `polars-timeseries` already enables rayon for catch22; no double-pool
  concerns since both use the global rayon pool.

### Migration order

1. **rust-stats**: add `smoothing/loess.rs` + tests.
2. **rust-stats**: add `tsa/seasonal/{mod,stl,decompose}.rs` + tests.
3. **rust-stats**: `cargo test` green.
4. **polars-timeseries**: add path dep, refactor `loess` + `stl` polars
   expressions to call into rust-stats, run all Python tests.
5. **polars-timeseries**: add `seasonal_decompose` polars expression +
   tests, run all Python tests.
6. **polars-timeseries**: add `bench/bench_stl.py`, run, capture results.
7. README updates: brief "powered by rust-stats" mention; add STL bench
   table to the Performance section; document the new
   `seasonal_decompose` row in the transforms table.

Each step ends in a green `cargo test` (rust-stats) or `uv run pytest` /
build (polars-timeseries) before the next begins.

## Risks & open questions

- **`Col::from_iter` allocation paths.** Faer's `Col::from_iter` may copy
  out of the iterator into a fresh allocation. We could instead build a
  `Vec<f64>` and convert via `Col::from_fn` or similar — same effective
  cost. Choose at implementation time.
- **Cross-crate path dependency in CI.** The user runs everything locally
  and there's no CI yet, so a `path = "../rust-stats"` dependency is
  fine. If we ever publish to crates.io this becomes a real workspace
  dependency or a published crate version.
- **Bench dataset periods.** `statsmodels.tsa.seasonal.STL` requires
  `period >= 2` and the series length to be enough for its low-pass
  filter (similar to ours). For `nile` (n=100, no inherent periodicity)
  and `macrodata.realgdp` (n=203, quarterly) we'll pick reasonable
  periods or skip those rows in the STL bench.
