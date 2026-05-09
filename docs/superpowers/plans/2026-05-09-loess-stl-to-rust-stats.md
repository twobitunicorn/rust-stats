# Move LOESS + STL into rust-stats; add seasonal_decompose — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the LOESS and STL implementations from `polars-timeseries/src/expressions.rs` into the sibling `../rust-stats` crate behind a Faer-typed free-function API, add a sibling `seasonal_decompose` (classical moving-average decomposition), then refactor `polars-timeseries` to consume them and rename the `noise` Struct field to `residual`.

**Architecture:** Two crates linked by a path dependency. `rust-stats` becomes the home of the algorithm; `polars-timeseries` keeps only the `#[polars_expr]` glue. Public Rust API uses `ColRef<'_, f64>` in / `Col<f64>` (inside result types) out, mirroring the existing `regression::ols` module. Inner LOESS solve uses manual Gaussian elimination on a fixed-size 3×3 system (decided in the spec — Faer's LU has µs-scale per-call overhead that compounds across thousands of LOESS query points).

**Tech Stack:**
- Rust 1.95+ (no MSRV bump needed)
- `faer = "0.22"` (rust-stats already pins this)
- `rayon = "1.10"` (new in rust-stats; polars-timeseries already has it)
- `thiserror = "2"` (rust-stats already pins it)
- `approx = "0.5"` (rust-stats dev dep, already there)
- `pyo3-polars = "0.26"` (polars-timeseries, unchanged)
- `statsmodels >= 0.14` (polars-timeseries dev dep, already installed via uv) — for the bench

**Spec:** `docs/superpowers/specs/2026-05-09-loess-stl-to-rust-stats.md`

---

## File Structure

### rust-stats (../rust-stats/)

```
src/
├── lib.rs                                  # MODIFY: declare smoothing + tsa modules and re-export
├── error.rs                                # MODIFY: add LoessError, StlError, SeasonalDecomposeError
├── distributions.rs                        # unchanged
├── regression/                             # unchanged
└── (new) smoothing/
    ├── mod.rs                              # CREATE: re-export loess + loess_at
    └── loess.rs                            # CREATE: public + private LOESS code
└── (new) tsa/
    ├── mod.rs                              # CREATE: re-export seasonal::*
    └── seasonal/
        ├── mod.rs                          # CREATE: shared types + re-exports
        ├── stl.rs                          # CREATE: Cleveland 1990 STL
        └── decompose.rs                    # CREATE: classical seasonal_decompose

tests/
├── loess.rs                                # CREATE: ten unit tests
├── stl.rs                                  # CREATE: STL unit tests
└── seasonal_decompose.rs                   # CREATE: decompose unit tests

Cargo.toml                                  # MODIFY: add rayon = "1.10"
```

### polars-timeseries

```
Cargo.toml                                  # MODIFY: add rust-stats = { path = "../rust-stats" }
src/expressions.rs                          # MODIFY: delete LOESS+STL internals; thin wrappers
python/polars_timeseries/__init__.py        # MODIFY: rename noise→residual; add seasonal_decompose
tests/test_transforms.py                    # MODIFY: rename noise→residual; add seasonal_decompose tests
README.md                                   # MODIFY: noise→residual; new seasonal_decompose row; STL bench
bench/bench_stl.py                          # CREATE: STL vs statsmodels.tsa.seasonal.STL
```

---

## Task 1: rust-stats scaffolding (errors, modules, deps)

Set up empty modules, error enums, and the rayon dep so subsequent tasks have a place to land code. This task only touches rust-stats.

**Files:**
- Modify: `../rust-stats/Cargo.toml`
- Modify: `../rust-stats/src/lib.rs`
- Modify: `../rust-stats/src/error.rs`
- Create: `../rust-stats/src/smoothing/mod.rs`
- Create: `../rust-stats/src/smoothing/loess.rs`
- Create: `../rust-stats/src/tsa/mod.rs`
- Create: `../rust-stats/src/tsa/seasonal/mod.rs`
- Create: `../rust-stats/src/tsa/seasonal/stl.rs`
- Create: `../rust-stats/src/tsa/seasonal/decompose.rs`

- [ ] **Step 1: Add rayon to rust-stats deps**

Open `../rust-stats/Cargo.toml`. Find the `[dependencies]` block and add `rayon = "1.10"`. The block becomes:

```toml
[dependencies]
faer = "0.22"
statrs = "0.18"
thiserror = "2"
once_cell = "1"
rayon = "1.10"
```

- [ ] **Step 2: Add the three error enums**

Open `../rust-stats/src/error.rs`. The existing file defines `OlsError` only. Append these three enums after it:

```rust
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

- [ ] **Step 3: Create empty `smoothing/loess.rs` placeholder**

Create `../rust-stats/src/smoothing/loess.rs` with a doc comment so the file is non-empty and compiles:

```rust
//! Locally estimated scatterplot smoothing (LOESS).
//!
//! Public API: `loess` (smooth at all integer positions) and `loess_at`
//! (single fractional query, used for boundary extrapolation in STL).
//! See module-level docs in `smoothing::mod` for usage notes.
```

- [ ] **Step 4: Create `smoothing/mod.rs`**

Create `../rust-stats/src/smoothing/mod.rs`:

```rust
//! Smoothing — currently LOESS.

pub mod loess;
```

- [ ] **Step 5: Create `tsa/seasonal/{stl,decompose}.rs` placeholders**

Create `../rust-stats/src/tsa/seasonal/stl.rs`:

```rust
//! Cleveland 1990 STL (LOESS-based seasonal-trend decomposition).
```

Create `../rust-stats/src/tsa/seasonal/decompose.rs`:

```rust
//! Classical (moving-average) seasonal-trend decomposition.
```

- [ ] **Step 6: Create `tsa/seasonal/mod.rs`**

Create `../rust-stats/src/tsa/seasonal/mod.rs`:

```rust
//! Seasonal decomposition: STL and classical moving-average.

pub mod decompose;
pub mod stl;
```

- [ ] **Step 7: Create `tsa/mod.rs`**

Create `../rust-stats/src/tsa/mod.rs`:

```rust
//! Time-series analysis. Currently: seasonal decomposition.

pub mod seasonal;
```

- [ ] **Step 8: Wire the new modules into `lib.rs`**

Open `../rust-stats/src/lib.rs`. Replace its contents with:

```rust
//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS) and LOESS-based smoothing /
//! seasonal decomposition. See `regression::Ols`, `smoothing::loess`,
//! `tsa::seasonal::stl`, and `tsa::seasonal::seasonal_decompose`.

pub mod distributions;
pub mod error;
pub mod regression;
pub mod smoothing;
pub mod tsa;

pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};
pub use error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
pub use regression::{CovType, Inference, Ols, OlsResults};
```

- [ ] **Step 9: Verify everything compiles**

Run: `cd ../rust-stats && cargo build`
Expected: clean build, no warnings except possibly `dead_code` for the new (empty) modules.

- [ ] **Step 10: Verify existing tests still pass**

Run: `cd ../rust-stats && cargo test`
Expected: 1 passed; 0 failed (the existing `crate_links` smoke test).

- [ ] **Step 11: Commit**

```bash
cd ../rust-stats
git add Cargo.toml src/error.rs src/lib.rs src/smoothing src/tsa
git commit -m "feat: scaffold smoothing + tsa modules and error enums

Adds empty smoothing/{mod.rs,loess.rs} and tsa/seasonal/{mod,stl,decompose}.rs
files; defines LoessError, StlError, SeasonalDecomposeError in error.rs;
adds rayon = \"1.10\" dependency. Module bodies will be filled in by
follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: rust-stats LOESS implementation + tests

Port the LOESS implementation from `polars-timeseries/src/expressions.rs` (lines 539–730 of the current file) into `rust-stats/src/smoothing/loess.rs` behind the Faer-typed free-function API. Tests live in `tests/loess.rs`.

**Files:**
- Modify: `../rust-stats/src/smoothing/loess.rs`
- Create: `../rust-stats/tests/loess.rs`

- [ ] **Step 1: Implement private helpers + public API in `smoothing/loess.rs`**

Replace the contents of `../rust-stats/src/smoothing/loess.rs` with:

```rust
//! Locally estimated scatterplot smoothing (LOESS).
//!
//! For each output index `i`, takes the `span * n` nearest indices in the
//! input, weights them with the tricube `w(d) = (1 - |d|^3)^3` of the
//! distance from `i` (normalised by the furthest window point), and fits a
//! weighted polynomial of `degree` to those points. The fitted value at
//! index `i` (the intercept of the centred fit) is the smoothed output.
//!
//! `degree = 1` is the classic LOWESS smoother; `degree = 0` reduces to a
//! tricube-weighted moving average; `degree = 2` is Cleveland's default.
//! No robustness iterations — this is single-pass (non-robust) LOESS.

use crate::error::LoessError;
use faer::{Col, ColRef};
use rayon::prelude::*;

/// Smooth the input series at every integer position `0..n`.
///
/// `span` is the fraction of points used in each local fit (0 < span <= 1);
/// `degree` is the polynomial degree (0, 1, or 2).
pub fn loess(
    y: ColRef<'_, f64>,
    span: f64,
    degree: u8,
) -> Result<Col<f64>, LoessError> {
    validate_loess_args(y, span, degree)?;
    let slice = y.as_slice();
    let n = slice.len();
    if n == 0 {
        return Ok(Col::zeros(0));
    }
    let degree_us = degree as usize;
    let window = ((span * n as f64).ceil() as usize)
        .max(degree_us + 2)
        .min(n);
    let smoothed = loess_compute(slice, window, degree_us);
    Ok(Col::<f64>::from_fn(smoothed.len(), |i| smoothed[i]))
}

/// Fitted LOESS value at a single (possibly fractional) query point `xq`.
/// `xq` may be outside `[0, n-1]` — the window snaps to the nearest
/// boundary slice, giving LOESS extrapolation by extension of the
/// boundary fit. Used by STL's cycle-subseries one-period extrapolation.
pub fn loess_at(
    y: ColRef<'_, f64>,
    xq: f64,
    span: f64,
    degree: u8,
) -> Result<f64, LoessError> {
    validate_loess_args(y, span, degree)?;
    let slice = y.as_slice();
    let n = slice.len();
    if n == 0 {
        return Err(LoessError::Empty);
    }
    let degree_us = degree as usize;
    let window = ((span * n as f64).ceil() as usize)
        .max(degree_us + 2)
        .min(n);
    Ok(local_poly_fit_at_xf64(slice, xq, window, degree_us))
}

fn validate_loess_args(
    y: ColRef<'_, f64>,
    span: f64,
    degree: u8,
) -> Result<(), LoessError> {
    if !(span > 0.0 && span <= 1.0) {
        return Err(LoessError::InvalidSpan(span));
    }
    if degree > 2 {
        return Err(LoessError::InvalidDegree(degree));
    }
    if y.nrows() == 0 {
        return Err(LoessError::Empty);
    }
    if y.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(LoessError::NonFinite);
    }
    Ok(())
}

/// Window of size `k` (clipped to `n`) centred around the integer floor of
/// `xq`. `xq` may be outside `[0, n-1]`, in which case the window snaps to
/// the nearest boundary slice.
pub(crate) fn loess_window_f(n: usize, xq: f64, k: usize) -> (usize, usize) {
    if k >= n {
        return (0, n);
    }
    let half = k / 2;
    let xq_clamp = xq.max(0.0).min((n - 1) as f64);
    let lo_unclamped = (xq_clamp - half as f64).floor();
    let lo = (lo_unclamped.max(0.0) as usize).min(n - k);
    (lo, lo + k)
}

/// Solve an `n x n` linear system `mat * x = rhs` via Gaussian elimination
/// with partial pivoting. Returns `None` if the matrix is singular.
pub(crate) fn gauss_solve_n(
    n: usize,
    mut mat: Vec<f64>,
    mut rhs: Vec<f64>,
) -> Option<Vec<f64>> {
    for i in 0..n {
        let mut pivot = i;
        let mut best = mat[i * n + i].abs();
        for r in (i + 1)..n {
            if mat[r * n + i].abs() > best {
                best = mat[r * n + i].abs();
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != i {
            for c in i..n {
                mat.swap(i * n + c, pivot * n + c);
            }
            rhs.swap(i, pivot);
        }
        for j in (i + 1)..n {
            let factor = mat[j * n + i] / mat[i * n + i];
            rhs[j] -= factor * rhs[i];
            for c in i..n {
                mat[j * n + c] -= factor * mat[i * n + c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..n {
            s -= mat[i * n + j] * x[j];
        }
        x[i] = s / mat[i * n + i];
    }
    Some(x)
}

/// Local polynomial fit at (possibly fractional) position `xq`, using the
/// `k` closest indices in `y` and tricube distance weights. Returns the
/// fitted value at `xq` (the intercept of the centred fit). Falls back to
/// a weighted mean if the polynomial system is singular.
pub(crate) fn local_poly_fit_at_xf64(
    y: &[f64],
    xq: f64,
    k: usize,
    degree: usize,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return f64::NAN;
    }
    let (lo, hi) = loess_window_f(n, xq, k);

    // Furthest distance from xq to any window point. Bumped by 1 so the
    // boundary point doesn't get exactly zero weight, which preserves
    // (degree+1) effective points and keeps the normal-equations matrix
    // non-singular for centred windows.
    let max_dist = {
        let left = (xq - lo as f64).abs();
        let right = ((hi - 1) as f64 - xq).abs();
        left.max(right).max(1.0) + 1.0
    };

    let m = degree + 1;
    let nearest_idx = || -> usize { xq.round().max(0.0).min((n - 1) as f64) as usize };

    if degree == 0 {
        let mut wsum = 0.0;
        let mut wysum = 0.0;
        for i in lo..hi {
            let d = (i as f64 - xq).abs() / max_dist;
            let w = if d >= 1.0 {
                0.0
            } else {
                let u = 1.0 - d * d * d;
                u * u * u
            };
            wsum += w;
            wysum += w * y[i];
        }
        return if wsum > 0.0 { wysum / wsum } else { y[nearest_idx()] };
    }

    let mut xtwx = vec![0.0; m * m];
    let mut xtwy = vec![0.0; m];
    let p_len = 2 * m - 1;
    let mut powers = vec![0.0; p_len];

    let mut wsum = 0.0;
    for i in lo..hi {
        let dx = i as f64 - xq;
        let abs_d = dx.abs() / max_dist;
        let w = if abs_d >= 1.0 {
            0.0
        } else {
            let u = 1.0 - abs_d * abs_d * abs_d;
            u * u * u
        };
        if w == 0.0 {
            continue;
        }
        wsum += w;

        powers[0] = 1.0;
        for r in 1..p_len {
            powers[r] = powers[r - 1] * dx;
        }
        let yi = y[i];
        for r in 0..m {
            xtwy[r] += w * powers[r] * yi;
            for c in 0..m {
                xtwx[r * m + c] += w * powers[r + c];
            }
        }
    }

    if wsum == 0.0 {
        return y[nearest_idx()];
    }

    match gauss_solve_n(m, xtwx, xtwy) {
        Some(coefs) => coefs[0],
        None => {
            let mut wsum = 0.0;
            let mut wysum = 0.0;
            for i in lo..hi {
                let d = (i as f64 - xq).abs() / max_dist;
                let w = if d >= 1.0 {
                    0.0
                } else {
                    let u = 1.0 - d * d * d;
                    u * u * u
                };
                wsum += w;
                wysum += w * y[i];
            }
            if wsum > 0.0 {
                wysum / wsum
            } else {
                y[nearest_idx()]
            }
        }
    }
}

pub(crate) fn local_poly_fit_at(y: &[f64], xq: usize, k: usize, degree: usize) -> f64 {
    local_poly_fit_at_xf64(y, xq as f64, k, degree)
}

/// LOESS smoother that takes an integer window directly. Used by `loess`
/// (which converts a fractional span first) and STL (which uses integer
/// windows throughout). Parallelises via rayon when n >= 256.
pub(crate) fn loess_compute(y: &[f64], window: usize, degree: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let k = window.max(degree + 2).min(n);
    if n >= 256 {
        (0..n)
            .into_par_iter()
            .map(|i| local_poly_fit_at(y, i, k, degree))
            .collect()
    } else {
        (0..n).map(|i| local_poly_fit_at(y, i, k, degree)).collect()
    }
}
```

- [ ] **Step 2: Re-export public functions from `smoothing/mod.rs`**

Replace the contents of `../rust-stats/src/smoothing/mod.rs` with:

```rust
//! Smoothing — currently LOESS.

pub mod loess;

pub use loess::{loess, loess_at};
```

- [ ] **Step 3: Verify it compiles**

Run: `cd ../rust-stats && cargo build`
Expected: clean build (no errors).

- [ ] **Step 4: Create `tests/loess.rs` with the ten ported tests**

Create `../rust-stats/tests/loess.rs`:

```rust
//! Unit tests for `rust_stats::smoothing::loess`. Ported from
//! `polars-timeseries/tests/test_transforms.py` LOESS section.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::smoothing::{loess, loess_at};
use rust_stats::error::LoessError;

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn constant_signal_returns_constant() {
    let y = col_from(vec![3.0; 20]);
    let out = loess(y.as_ref(), 0.5, 1).unwrap();
    for i in 0..out.nrows() {
        assert_relative_eq!(out[i], 3.0, epsilon = 1e-9);
    }
}

#[test]
fn linear_signal_exact_recovery_degree_one() {
    let n = 50;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let out = loess(y.as_ref(), 0.5, 1).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], i as f64, epsilon = 1e-9);
    }
}

#[test]
fn quadratic_signal_exact_recovery_degree_two() {
    let n = 30;
    let y = col_from((0..n).map(|i| (i as f64).powi(2)).collect());
    let out = loess(y.as_ref(), 0.5, 2).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], (i as f64).powi(2), epsilon = 1e-9, max_relative = 1e-9);
    }
}

#[test]
fn wider_span_smooths_more() {
    // Noisy linear series; wider span should reduce residual variance
    // relative to the underlying line.
    use rand::{SeedableRng, rngs::StdRng, Rng};
    let n = 300;
    let mut rng = StdRng::seed_from_u64(0);
    let y: Vec<f64> = (0..n)
        .map(|i| i as f64 + rng.gen_range(-1.0..1.0))
        .collect();
    let y_col = col_from(y);

    let narrow = loess(y_col.as_ref(), 0.05, 1).unwrap();
    let wide = loess(y_col.as_ref(), 0.5, 1).unwrap();
    let narrow_var: f64 = (0..n).map(|i| (narrow[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    let wide_var: f64 = (0..n).map(|i| (wide[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    assert!(wide_var < narrow_var, "wide_var={} not < narrow_var={}", wide_var, narrow_var);
}

#[test]
fn step_function_smooths_with_bounded_overshoot() {
    let n = 100;
    let half = n / 2;
    let mut v = vec![0.0; half];
    v.extend(vec![1.0; n - half]);
    let y = col_from(v);
    let out = loess(y.as_ref(), 0.2, 1).unwrap();
    assert!(out[0] < 0.05);
    assert!(out[n - 1] > 0.95);
    for i in 0..n {
        assert!((-0.1..=1.1).contains(&out[i]), "overshoot at {}: {}", i, out[i]);
    }
}

#[test]
fn constant_signal_preserved_with_degree_two() {
    let n = 50;
    let y = col_from(vec![4.2; n]);
    let out = loess(y.as_ref(), 0.4, 2).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], 4.2, epsilon = 1e-9);
    }
}

#[test]
fn short_series_falls_back_gracefully() {
    let y = col_from(vec![1.0, 2.0, 3.0]);
    let out = loess(y.as_ref(), 1.0, 1).unwrap();
    assert!(out.as_slice().iter().all(|v| v.is_finite()));
    assert_relative_eq!(out[0], 1.0, epsilon = 1e-9);
    assert_relative_eq!(out[1], 2.0, epsilon = 1e-9);
    assert_relative_eq!(out[2], 3.0, epsilon = 1e-9);
}

#[test]
fn boundary_recovery_exact_on_linear_input() {
    let n = 100;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let out = loess(y.as_ref(), 0.3, 1).unwrap();
    assert_relative_eq!(out[0], 0.0, epsilon = 1e-9);
    assert_relative_eq!(out[n - 1], (n - 1) as f64, epsilon = 1e-9);
}

#[test]
fn loess_at_extrapolates_past_boundary() {
    let n = 50;
    let y = col_from((0..n).map(|i| i as f64).collect());
    // Linear y = i; LOESS at x = -1.0 should extrapolate to ~ -1.0.
    let v = loess_at(y.as_ref(), -1.0, 0.3, 1).unwrap();
    assert_relative_eq!(v, -1.0, epsilon = 1e-6);
    // At x = n (one past the end) → ~ n.
    let v2 = loess_at(y.as_ref(), n as f64, 0.3, 1).unwrap();
    assert_relative_eq!(v2, n as f64, epsilon = 1e-6);
}

#[test]
fn validation_rejects_bad_span_and_degree() {
    let y = col_from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(loess(y.as_ref(), 0.0, 1), Err(LoessError::InvalidSpan(0.0)));
    assert_eq!(loess(y.as_ref(), 1.5, 1), Err(LoessError::InvalidSpan(1.5)));
    assert_eq!(loess(y.as_ref(), 0.5, 3), Err(LoessError::InvalidDegree(3)));
}

#[test]
fn rejects_non_finite_input() {
    let y = col_from(vec![1.0, f64::NAN, 3.0]);
    assert_eq!(loess(y.as_ref(), 0.5, 1), Err(LoessError::NonFinite));
}
```

The `wider_span_smooths_more` test uses `rand::{SeedableRng, rngs::StdRng, Rng}`. `rand` isn't in rust-stats' deps yet. Replace it with a deterministic noisy series so we don't pull in another dep:

Replace the body of `wider_span_smooths_more` (the rand-using version) with this simpler deterministic version:

```rust
#[test]
fn wider_span_smooths_more() {
    // Deterministic "noise" via a simple LCG so we don't pull in rand.
    let n = 300;
    let mut state: u64 = 1;
    let next = || -> f64 {
        // Linear congruential — output is just for noise, not statistical fitness.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as i32 as f64 / (1u64 << 31) as f64 // ≈ uniform in [-1, 1]
    };
    let y: Vec<f64> = {
        let mut state: u64 = 1;
        (0..n)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as i32 as f64) / (1u64 << 31) as f64;
                i as f64 + noise
            })
            .collect()
    };
    let y_col = col_from(y);

    let narrow = loess(y_col.as_ref(), 0.05, 1).unwrap();
    let wide = loess(y_col.as_ref(), 0.5, 1).unwrap();
    let narrow_var: f64 = (0..n).map(|i| (narrow[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    let wide_var: f64 = (0..n).map(|i| (wide[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    assert!(wide_var < narrow_var, "wide_var={} not < narrow_var={}", wide_var, narrow_var);
    let _ = next; // keep nested closure off the unused-warning list
}
```

(The first `let next = ||` block is unused boilerplate — kept only to mirror the structure cleanly. We can simplify in a follow-up; for now the test compiles and passes.)

- [ ] **Step 5: Run the tests**

Run: `cd ../rust-stats && cargo test --test loess`
Expected: 11 passed; 0 failed.

If a test fails, read the assertion message, compare the `local_poly_fit_at_xf64` body against `polars-timeseries/src/expressions.rs:596–702`, and verify the port is byte-for-byte equivalent.

- [ ] **Step 6: Run the full test suite to make sure nothing else regressed**

Run: `cd ../rust-stats && cargo test`
Expected: all tests pass (1 OLS smoke + 11 LOESS = 12 total).

- [ ] **Step 7: Commit**

```bash
cd ../rust-stats
git add src/smoothing tests/loess.rs
git commit -m "feat(smoothing): port LOESS from polars-timeseries

Adds public free functions \`loess\` and \`loess_at\` taking \`ColRef<'_, f64>\`
in and \`Col<f64>\` / \`f64\` out, behind \`LoessError\` validation. Private
helpers (\`loess_window_f\`, \`gauss_solve_n\`, \`local_poly_fit_at_xf64\`,
\`local_poly_fit_at\`, \`loess_compute\`) are byte-for-byte ports of the
existing implementation in polars-timeseries/src/expressions.rs. Inner
local-fit solve uses a fixed-size 3x3 manual Gaussian elimination — matches
the existing implementation and avoids per-call \`Faer LU\` overhead across
thousands of query points. Outer per-point loop parallelises via rayon for
n >= 256.

Eleven tests in tests/loess.rs cover constant / linear / quadratic exact
recovery, span sensitivity, step-function smoothness, boundary recovery on
linear input, fractional-query extrapolation past both ends, the short-
series fallback, and the InvalidSpan / InvalidDegree / NonFinite validation
paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Shared seasonal types in rust-stats

Add the shared types (`Decomposition`, `DecomposeMode`, `StlOpts`, `SeasonalDecomposeOpts`) and re-exports. Keep `stl.rs` and `decompose.rs` mostly empty for now — they get filled in by Tasks 4 and 5.

**Files:**
- Modify: `../rust-stats/src/tsa/seasonal/mod.rs`
- Modify: `../rust-stats/src/tsa/mod.rs`
- Modify: `../rust-stats/src/lib.rs`

- [ ] **Step 1: Define shared types in `tsa/seasonal/mod.rs`**

Replace the contents of `../rust-stats/src/tsa/seasonal/mod.rs` with:

```rust
//! Seasonal-trend decomposition: Cleveland 1990 STL and the classical
//! moving-average `seasonal_decompose`. Both produce identically-shaped
//! `Decomposition` output.

pub mod decompose;
pub mod stl;

use faer::Col;

pub use decompose::seasonal_decompose;
pub use stl::stl;

/// Output of a seasonal-trend decomposition. The components reconstruct
/// the input where defined:
///   additive:        `y[i] = trend[i] + seasonal[i] + residual[i]`
///   multiplicative:  `y[i] = trend[i] * seasonal[i] * residual[i]`
///
/// STL produces finite values everywhere; classical `seasonal_decompose`
/// has NaN at the first/last `period/2` positions where the centered
/// moving average can't be computed.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub trend: Col<f64>,
    pub seasonal: Col<f64>,
    pub residual: Col<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposeMode {
    Additive,
    Multiplicative,
}

/// Options for `stl`. Construct via `StlOpts::new(period)` for Cleveland
/// defaults and override individual fields with struct-update syntax.
#[derive(Debug, Clone)]
pub struct StlOpts {
    pub period: u32,
    /// LOESS span (in points) for cycle-subseries smoothing.
    /// Must be odd and >= 7.
    pub seasonal_window: u32,
    /// LOESS span for the trend smoother. `None` uses Cleveland's
    /// recommended formula: smallest odd >=
    /// `1.5 * period / (1 - 1.5 / seasonal_window)`.
    pub trend_window: Option<u32>,
    /// Number of inner-loop iterations. Cleveland recommends 2.
    pub inner_iters: u32,
    pub mode: DecomposeMode,
}

impl StlOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            seasonal_window: 7,
            trend_window: None,
            inner_iters: 2,
            mode: DecomposeMode::Additive,
        }
    }
}

/// Options for `seasonal_decompose`. Construct via
/// `SeasonalDecomposeOpts::new(period)` for additive defaults.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposeOpts {
    pub period: u32,
    pub mode: DecomposeMode,
}

impl SeasonalDecomposeOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            mode: DecomposeMode::Additive,
        }
    }
}
```

- [ ] **Step 2: Re-export from `tsa/mod.rs`**

Replace the contents of `../rust-stats/src/tsa/mod.rs` with:

```rust
//! Time-series analysis. Currently: seasonal decomposition.

pub mod seasonal;

pub use seasonal::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, SeasonalDecomposeOpts, StlOpts,
};
```

- [ ] **Step 3: Re-export from `lib.rs`**

In `../rust-stats/src/lib.rs`, extend the `pub use` block at the bottom. Replace the existing:

```rust
pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};
pub use error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
pub use regression::{CovType, Inference, Ols, OlsResults};
```

with:

```rust
pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};
pub use error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
pub use regression::{CovType, Inference, Ols, OlsResults};
pub use smoothing::{loess, loess_at};
pub use tsa::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, SeasonalDecomposeOpts, StlOpts,
};
```

- [ ] **Step 4: Build, expecting failures from missing function bodies**

Run: `cd ../rust-stats && cargo build`
Expected: errors like `cannot find function \`stl\` in module \`stl\`` — the re-exports point at functions that don't exist yet. This will be fixed in Task 4.

- [ ] **Step 5: Add stub bodies to `stl.rs` and `decompose.rs` so the crate compiles**

Replace the contents of `../rust-stats/src/tsa/seasonal/stl.rs` with:

```rust
//! Cleveland 1990 STL (LOESS-based seasonal-trend decomposition).
//!
//! See module-level docs in `tsa::seasonal` for usage notes.

use crate::error::StlError;
use crate::tsa::seasonal::{Decomposition, StlOpts};
use faer::ColRef;

/// Cleveland 1990 STL — full implementation lands in Task 4.
pub fn stl(_y: ColRef<'_, f64>, _opts: StlOpts) -> Result<Decomposition, StlError> {
    unimplemented!("stl: implemented in Task 4")
}
```

Replace the contents of `../rust-stats/src/tsa/seasonal/decompose.rs` with:

```rust
//! Classical (moving-average) seasonal-trend decomposition.
//!
//! See module-level docs in `tsa::seasonal` for usage notes.

use crate::error::SeasonalDecomposeError;
use crate::tsa::seasonal::{Decomposition, SeasonalDecomposeOpts};
use faer::ColRef;

/// Classical decomposition — full implementation lands in Task 5.
pub fn seasonal_decompose(
    _y: ColRef<'_, f64>,
    _opts: SeasonalDecomposeOpts,
) -> Result<Decomposition, SeasonalDecomposeError> {
    unimplemented!("seasonal_decompose: implemented in Task 5")
}
```

- [ ] **Step 6: Verify compile and existing tests still pass**

Run: `cd ../rust-stats && cargo test`
Expected: 12 passed (OLS smoke + 11 LOESS), no new errors.

- [ ] **Step 7: Commit**

```bash
cd ../rust-stats
git add src/tsa src/lib.rs
git commit -m "feat(tsa): scaffold seasonal types and re-exports

Adds shared Decomposition, DecomposeMode, StlOpts, and SeasonalDecomposeOpts
types in tsa::seasonal. Function bodies are unimplemented!() stubs that
land in Tasks 4 and 5; this commit just lets the rest of the crate compile
against the public type surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: rust-stats STL implementation + tests

Port the STL code from `polars-timeseries/src/expressions.rs:288–533` into `rust-stats/src/tsa/seasonal/stl.rs` behind the Faer-typed `stl(y, opts)` free function.

**Files:**
- Modify: `../rust-stats/src/tsa/seasonal/stl.rs`
- Create: `../rust-stats/tests/stl.rs`

- [ ] **Step 1: Replace the stub body in `stl.rs` with the full implementation**

Replace the contents of `../rust-stats/src/tsa/seasonal/stl.rs` with:

```rust
//! Cleveland 1990 STL — seasonal-trend decomposition by LOESS.
//!
//! Inner loop:
//!   1. Detrend                 `D = Y − T`
//!   2. Cycle-subseries LOESS   one-period extrapolation each end → `C` of length n+2*period
//!   3. Low-pass filter         `MA(period) → MA(period) → MA(3) → LOESS` → `L` of length n
//!   4. Seasonal                `S = C[period..period+n] − L`
//!   5. Deseasonalize           `Y − S`
//!   6. Trend LOESS             `T = LOESS(Y − S)`
//! repeated `inner_iters` times. No outer robustness loop.
//!
//! Multiplicative mode: log-transform → additive STL → exp components.

use crate::error::StlError;
use crate::smoothing::loess::{local_poly_fit_at_xf64, loess_compute};
use crate::tsa::seasonal::{DecomposeMode, Decomposition, StlOpts};
use faer::{Col, ColRef};

/// Cleveland 1990 STL.
///
/// Returns a `Decomposition` whose three columns reconstruct `y` exactly
/// (additive: `y = T + S + R`; multiplicative: `y = T * S * R`).
/// LOESS-based — no NaN edges.
pub fn stl(y: ColRef<'_, f64>, opts: StlOpts) -> Result<Decomposition, StlError> {
    if opts.period < 2 {
        return Err(StlError::InvalidPeriod(opts.period));
    }
    let period = opts.period as usize;

    let n_s = opts.seasonal_window as usize;
    if n_s < 7 || n_s % 2 == 0 {
        return Err(StlError::InvalidSeasonalWindow(opts.seasonal_window));
    }

    let n_l = if period % 2 == 0 { period + 1 } else { period };

    let n_t = match opts.trend_window {
        None => next_odd_ceil(1.5 * period as f64 / (1.0 - 1.5 / n_s as f64)),
        Some(t) => {
            if t % 2 == 0 {
                return Err(StlError::InvalidTrendWindow(t));
            }
            t as usize
        }
    };

    let n_i = opts.inner_iters as usize;
    if n_i == 0 {
        return Err(StlError::InvalidInnerIters);
    }

    if y.nrows() == 0 {
        return Err(StlError::SeriesTooShort {
            n: 0,
            min: 2 * period,
        });
    }

    let raw = y.as_slice();
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(StlError::NonFinite);
    }
    let n = raw.len();

    if n < 2 * period {
        return Err(StlError::SeriesTooShort {
            n,
            min: 2 * period,
        });
    }

    let multiplicative = matches!(opts.mode, DecomposeMode::Multiplicative);
    if multiplicative {
        let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
        if min <= 0.0 {
            return Err(StlError::NonPositiveForMultiplicative { min });
        }
    }

    let work: Vec<f64> = if multiplicative {
        raw.iter().map(|v| v.ln()).collect()
    } else {
        raw.to_vec()
    };

    let (trend, seasonal) = stl_inner_loop(&work, period, n_s, n_l, n_t, n_i);

    let residual: Vec<f64> = (0..n)
        .map(|i| work[i] - trend[i] - seasonal[i])
        .collect();

    let (trend, seasonal, residual) = if multiplicative {
        (
            trend.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            seasonal.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            residual.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
        )
    } else {
        (trend, seasonal, residual)
    };

    Ok(Decomposition {
        trend: Col::<f64>::from_fn(n, |i| trend[i]),
        seasonal: Col::<f64>::from_fn(n, |i| seasonal[i]),
        residual: Col::<f64>::from_fn(n, |i| residual[i]),
    })
}

/// Smallest odd integer >= x.
fn next_odd_ceil(x: f64) -> usize {
    let n = x.ceil() as usize;
    if n % 2 == 0 {
        (n + 1).max(1)
    } else {
        n.max(1)
    }
}

/// Valid (non-padded) moving average. Input length n, output length
/// `n - window + 1`. Output[k] is the mean of input[k..k+window].
fn valid_ma(y: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    if window == 0 || n < window {
        return Vec::new();
    }
    let out_n = n - window + 1;
    let mut out = Vec::with_capacity(out_n);
    let inv = 1.0 / window as f64;
    let mut sum: f64 = y[..window].iter().sum();
    out.push(sum * inv);
    for i in window..n {
        sum += y[i] - y[i - window];
        out.push(sum * inv);
    }
    out
}

/// Cycle-subseries smoothing — Step 2 of STL.
fn cycle_subseries_smooth(d: &[f64], period: usize, span: usize, degree: usize) -> Vec<f64> {
    let n = d.len();
    let mut c = vec![0.0; n + 2 * period];

    for phase in 0..period {
        let subs: Vec<f64> = (phase..n).step_by(period).map(|i| d[i]).collect();
        let sub_n = subs.len();
        if sub_n == 0 {
            continue;
        }
        let k = span.max(degree + 2).min(sub_n);

        c[phase] = local_poly_fit_at_xf64(&subs, -1.0, k, degree);

        for j in 0..sub_n {
            let orig = phase + j * period;
            c[period + orig] = local_poly_fit_at_xf64(&subs, j as f64, k, degree);
        }

        let after = phase + sub_n * period;
        c[period + after] = local_poly_fit_at_xf64(&subs, sub_n as f64, k, degree);
    }
    c
}

/// Low-pass filter — Step 3 of STL.
fn low_pass_filter(c: &[f64], period: usize, span: usize, degree: usize) -> Vec<f64> {
    let ma1 = valid_ma(c, period);
    let ma2 = valid_ma(&ma1, period);
    let ma3 = valid_ma(&ma2, 3);
    loess_compute(&ma3, span, degree)
}

/// One inner loop pass repeated `n_i` times. Returns `(trend, seasonal)`.
fn stl_inner_loop(
    y: &[f64],
    period: usize,
    n_s: usize,
    n_l: usize,
    n_t: usize,
    n_i: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut trend = vec![0.0f64; n];
    let mut seasonal = vec![0.0f64; n];

    for _ in 0..n_i {
        let detrended: Vec<f64> = (0..n).map(|i| y[i] - trend[i]).collect();
        let c = cycle_subseries_smooth(&detrended, period, n_s, 1);
        let l = low_pass_filter(&c, period, n_l, 1);
        seasonal = (0..n).map(|i| c[period + i] - l[i]).collect();
        let deseasonalized: Vec<f64> = (0..n).map(|i| y[i] - seasonal[i]).collect();
        trend = loess_compute(&deseasonalized, n_t, 1);
    }
    (trend, seasonal)
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd ../rust-stats && cargo build`
Expected: clean build. Note: `local_poly_fit_at_xf64` and `loess_compute` need to be `pub(crate)` in `smoothing/loess.rs` — they already are from Task 2.

- [ ] **Step 3: Create `tests/stl.rs`**

Create `../rust-stats/tests/stl.rs`:

```rust
//! Unit tests for `rust_stats::tsa::seasonal::stl`. Ported from
//! `polars-timeseries/tests/test_transforms.py` STL section.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::error::StlError;
use rust_stats::tsa::{stl, DecomposeMode, StlOpts};

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn pure_linear_trend_recovered_everywhere() {
    let n = 24usize;
    let period = 4u32;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        assert_relative_eq!(d.trend[i], i as f64, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], 0.0, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn pure_seasonal_pattern_recovered_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let pattern_mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % pattern.len()]).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        assert_relative_eq!(d.trend[i], pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], pattern[i % 4] - pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn additive_reconstruction_exact_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = 24usize;
    let y_vec: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let y = col_from(y_vec.clone());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y_vec[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_exact_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let y_vec: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let y = col_from(y_vec.clone());
    let d = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(period)
        },
    )
    .unwrap();
    for i in 0..y_vec.len() {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y_vec[i], max_relative = 1e-9);
    }
}

#[test]
fn additive_seasonal_pattern_sums_to_zero() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    let inner: f64 = (8..12).map(|i| d.seasonal[i]).sum();
    assert_relative_eq!(inner, 0.0, epsilon = 1e-9);
}

#[test]
fn multiplicative_seasonal_pattern_products_to_one() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(period)
        },
    )
    .unwrap();
    let prod: f64 = (8..12).map(|i| d.seasonal[i]).product();
    assert_relative_eq!(prod, 1.0, max_relative = 1e-9);
}

#[test]
fn validation_paths() {
    let y = col_from(vec![1.0; 24]);
    assert!(matches!(
        stl(y.as_ref(), StlOpts::new(1)),
        Err(StlError::InvalidPeriod(1))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                seasonal_window: 8,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidSeasonalWindow(8))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                trend_window: Some(10),
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidTrendWindow(10))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                inner_iters: 0,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidInnerIters)
    ));
    let short = col_from(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        stl(short.as_ref(), StlOpts::new(4)),
        Err(StlError::SeriesTooShort { .. })
    ));
}

#[test]
fn multiplicative_rejects_non_positive() {
    let y = col_from(vec![1.0, 2.0, 0.0, 1.5].repeat(6));
    let err = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(4)
        },
    )
    .unwrap_err();
    assert!(matches!(err, StlError::NonPositiveForMultiplicative { .. }));
}

#[test]
fn rejects_non_finite() {
    let mut v = vec![1.0; 24];
    v[5] = f64::NAN;
    let y = col_from(v);
    assert_eq!(
        stl(y.as_ref(), StlOpts::new(4)),
        Err(StlError::NonFinite)
    );
}
```

- [ ] **Step 4: Run STL tests**

Run: `cd ../rust-stats && cargo test --test stl`
Expected: 9 passed; 0 failed.

- [ ] **Step 5: Run full test suite**

Run: `cd ../rust-stats && cargo test`
Expected: 1 OLS smoke + 11 LOESS + 9 STL = 21 passed.

- [ ] **Step 6: Commit**

```bash
cd ../rust-stats
git add src/tsa/seasonal/stl.rs tests/stl.rs
git commit -m "feat(tsa): port STL from polars-timeseries

Adds the public free function \`tsa::seasonal::stl(y, opts) -> Decomposition\`
implementing Cleveland 1990 STL via the LOESS smoother in \`smoothing\`.
Private helpers (\`valid_ma\`, \`cycle_subseries_smooth\`, \`low_pass_filter\`,
\`stl_inner_loop\`, \`next_odd_ceil\`) are byte-for-byte ports of the
existing implementation in polars-timeseries/src/expressions.rs.
Multiplicative mode log-transforms input, runs additive STL, and
exp-transforms the components — the y = T*S*R reconstruction is then
exact by construction.

Nine tests in tests/stl.rs cover pure-linear and pure-seasonal exact
recovery, additive and multiplicative reconstruction, seasonal-zero-sum
and seasonal-unit-product invariants, and all five validation paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: rust-stats seasonal_decompose implementation + tests

Implement classical (centered moving-average) decomposition. This is the algorithm `polars-timeseries` used to ship before STL replaced it; we re-introduce `centered_ma` here.

**Files:**
- Modify: `../rust-stats/src/tsa/seasonal/decompose.rs`
- Create: `../rust-stats/tests/seasonal_decompose.rs`

- [ ] **Step 1: Replace the stub with the full implementation**

Replace the contents of `../rust-stats/src/tsa/seasonal/decompose.rs` with:

```rust
//! Classical (moving-average) seasonal-trend decomposition.
//!
//! Trend: centered moving average of length `period`.
//! Seasonal: per-phase mean of detrended values, centred so the seasonal
//! pattern sums to zero (additive) or products to one (multiplicative).
//! Residual: `y - trend - seasonal` (additive) or `y / (trend * seasonal)`
//! (multiplicative).
//!
//! The first/last `period/2` positions of `trend` and `residual` are NaN
//! (the centred moving-average edge band).

use crate::error::SeasonalDecomposeError;
use crate::tsa::seasonal::{DecomposeMode, Decomposition, SeasonalDecomposeOpts};
use faer::{Col, ColRef};

pub fn seasonal_decompose(
    y: ColRef<'_, f64>,
    opts: SeasonalDecomposeOpts,
) -> Result<Decomposition, SeasonalDecomposeError> {
    if opts.period < 2 {
        return Err(SeasonalDecomposeError::InvalidPeriod(opts.period));
    }
    let period = opts.period as usize;

    if y.nrows() == 0 {
        return Err(SeasonalDecomposeError::SeriesTooShort {
            n: 0,
            min: 2 * period,
        });
    }
    let raw = y.as_slice();
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(SeasonalDecomposeError::NonFinite);
    }
    let n = raw.len();
    if n < 2 * period {
        return Err(SeasonalDecomposeError::SeriesTooShort {
            n,
            min: 2 * period,
        });
    }

    let multiplicative = matches!(opts.mode, DecomposeMode::Multiplicative);
    if multiplicative {
        let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
        if min <= 0.0 {
            return Err(SeasonalDecomposeError::NonPositiveForMultiplicative { min });
        }
    }

    // Work in log-space for multiplicative mode.
    let work: Vec<f64> = if multiplicative {
        raw.iter().map(|v| v.ln()).collect()
    } else {
        raw.to_vec()
    };

    let trend = centered_ma(&work, period);

    let detrended: Vec<f64> = work
        .iter()
        .zip(trend.iter())
        .map(|(yi, ti)| if ti.is_nan() { f64::NAN } else { yi - ti })
        .collect();

    let mut phase_sums = vec![0.0f64; period];
    let mut phase_counts = vec![0usize; period];
    for (i, &d) in detrended.iter().enumerate() {
        if !d.is_nan() {
            phase_sums[i % period] += d;
            phase_counts[i % period] += 1;
        }
    }
    let phase_means: Vec<f64> = (0..period)
        .map(|k| {
            if phase_counts[k] > 0 {
                phase_sums[k] / phase_counts[k] as f64
            } else {
                0.0
            }
        })
        .collect();
    let pattern_mean: f64 = phase_means.iter().sum::<f64>() / period as f64;
    let centered_pattern: Vec<f64> = phase_means.iter().map(|m| m - pattern_mean).collect();

    let seasonal: Vec<f64> = (0..n).map(|i| centered_pattern[i % period]).collect();

    let residual: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_nan() {
                f64::NAN
            } else {
                work[i] - trend[i] - seasonal[i]
            }
        })
        .collect();

    let (trend, seasonal, residual) = if multiplicative {
        (
            trend.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            seasonal.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            residual.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
        )
    } else {
        (trend, seasonal, residual)
    };

    Ok(Decomposition {
        trend: Col::<f64>::from_fn(n, |i| trend[i]),
        seasonal: Col::<f64>::from_fn(n, |i| seasonal[i]),
        residual: Col::<f64>::from_fn(n, |i| residual[i]),
    })
}

/// Centered moving average of length `window`. For odd window: standard
/// `(2k+1)`-MA at index i averages `y[i-k..=i+k]`. For even window m: a
/// `(m, 2)`-MA — equivalent to taking the m-MA twice and averaging — which
/// weights the endpoints by `1/(2m)` and the m-1 middle points by `1/m`.
/// Returns NaN at the first/last `m/2` positions where the centred window
/// doesn't fit.
fn centered_ma(y: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window {
        return out;
    }
    if window % 2 == 1 {
        let half = window / 2;
        let inv = 1.0 / window as f64;
        for i in half..(n - half) {
            let sum: f64 = y[i - half..=i + half].iter().sum();
            out[i] = sum * inv;
        }
    } else {
        let half = window / 2;
        let inv = 1.0 / (2 * window) as f64;
        for i in half..(n - half) {
            let mut sum = y[i - half] + y[i + half];
            for j in (i - half + 1)..(i + half) {
                sum += 2.0 * y[j];
            }
            out[i] = sum * inv;
        }
    }
    out
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd ../rust-stats && cargo build`
Expected: clean build.

- [ ] **Step 3: Create `tests/seasonal_decompose.rs`**

Create `../rust-stats/tests/seasonal_decompose.rs`:

```rust
//! Unit tests for `rust_stats::tsa::seasonal::seasonal_decompose`.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::error::SeasonalDecomposeError;
use rust_stats::tsa::{seasonal_decompose, DecomposeMode, SeasonalDecomposeOpts};

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn linear_trend_recovered_in_inner_band() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        assert_relative_eq!(d.trend[i], i as f64, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], 0.0, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn seasonal_pattern_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let pattern_mean = pattern.iter().sum::<f64>() / 4.0;
    let n = pattern.len() * 6;
    let half = (period as usize) / 2;
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        assert_relative_eq!(d.trend[i], pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], pattern[i % 4] - pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn additive_reconstruction_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = 24usize;
    let half = (period as usize) / 2;
    let y_vec: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let y = col_from(y_vec.clone());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y_vec[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let half = (period as usize) / 2;
    let y_vec: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let n = y_vec.len();
    let y = col_from(y_vec.clone());
    let d = seasonal_decompose(
        y.as_ref(),
        SeasonalDecomposeOpts {
            mode: DecomposeMode::Multiplicative,
            ..SeasonalDecomposeOpts::new(period)
        },
    )
    .unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y_vec[i], max_relative = 1e-9);
    }
}

#[test]
fn edges_are_nan() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
    for i in 0..half {
        assert!(d.trend[i].is_nan(), "trend[{}] not NaN", i);
        assert!(d.residual[i].is_nan(), "residual[{}] not NaN", i);
    }
    for i in (n - half)..n {
        assert!(d.trend[i].is_nan(), "trend[{}] not NaN", i);
        assert!(d.residual[i].is_nan(), "residual[{}] not NaN", i);
    }
}

#[test]
fn validation_paths() {
    let y = col_from(vec![1.0; 24]);
    assert!(matches!(
        seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(1)),
        Err(SeasonalDecomposeError::InvalidPeriod(1))
    ));
    let short = col_from(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        seasonal_decompose(short.as_ref(), SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::SeriesTooShort { .. })
    ));
    let bad = col_from(vec![1.0, 2.0, 0.0, 1.5].repeat(6));
    assert!(matches!(
        seasonal_decompose(
            bad.as_ref(),
            SeasonalDecomposeOpts {
                mode: DecomposeMode::Multiplicative,
                ..SeasonalDecomposeOpts::new(4)
            }
        ),
        Err(SeasonalDecomposeError::NonPositiveForMultiplicative { .. })
    ));
    let mut v = vec![1.0; 24];
    v[5] = f64::NAN;
    assert!(matches!(
        seasonal_decompose(col_from(v).as_ref(), SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::NonFinite)
    ));
}
```

- [ ] **Step 4: Run decompose tests**

Run: `cd ../rust-stats && cargo test --test seasonal_decompose`
Expected: 6 passed; 0 failed.

- [ ] **Step 5: Run the entire rust-stats test suite**

Run: `cd ../rust-stats && cargo test`
Expected: 1 OLS smoke + 11 LOESS + 9 STL + 6 decompose = 27 passed.

- [ ] **Step 6: Commit**

```bash
cd ../rust-stats
git add src/tsa/seasonal/decompose.rs tests/seasonal_decompose.rs
git commit -m "feat(tsa): add classical seasonal_decompose

Implements \`tsa::seasonal::seasonal_decompose(y, opts)\` — the classical
centered-moving-average decomposition that polars-timeseries used to ship
before STL replaced it. Trend is a length-\`period\` centred MA (with the
even-m double-pass correction), seasonal is the per-phase mean of the
detrended series re-centred to zero (additive) or unit-product
(multiplicative), and residual is the leftover. The first and last
\`period/2\` positions of trend and residual are NaN by construction.

Six tests in tests/seasonal_decompose.rs cover the inner-band recovery
of pure linear and pure seasonal signals, additive and multiplicative
reconstruction within the inner band, the explicit NaN-edge
contract, and all four validation paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: polars-timeseries — add path dep, delete obsolete code, refactor `loess` polars expression

Wire polars-timeseries to the new rust-stats and remove the now-duplicated LOESS internals. Refactor the `loess` polars_expr to call `rust_stats::loess`.

**Files:**
- Modify: `Cargo.toml` (polars-timeseries)
- Modify: `src/expressions.rs` (polars-timeseries)

- [ ] **Step 1: Add the path dependency**

In `/Users/joseph/Projects/polars-timeseries/Cargo.toml`, add the rust-stats path dep after `realfft`. The `[dependencies]` block becomes:

```toml
[dependencies]
polars = { version = "0.53", default-features = false, features = ["fmt", "dtype-full"] }
polars-arrow = { version = "0.53", default-features = false }
pyo3 = { version = "0.27", features = ["extension-module", "abi3-py39"] }
pyo3-polars = { version = "0.26", features = ["derive"] }
rayon = "1.10"
realfft = "3.4"
rust-stats = { path = "../rust-stats" }
serde = { version = "1", features = ["derive"] }
```

- [ ] **Step 2: Delete the LOESS + STL private helpers from `src/expressions.rs`**

Open `/Users/joseph/Projects/polars-timeseries/src/expressions.rs`. Delete these functions in their entirety (line ranges per the current file):
- `next_odd_ceil` (lines 288–296)
- `valid_ma` (lines 298–315)
- `cycle_subseries_smooth` (lines 317–354)
- `low_pass_filter` (lines 356–363)
- `stl_inner_loop` (lines 365–394)
- `loess_window_f` (lines 537–550)
- `gauss_solve_n` (lines 552–594)
- `local_poly_fit_at_xf64` (lines 596–702)
- `local_poly_fit_at` (lines 704–707)
- `loess_compute` (lines 709–730)

Keep the `LoessKwargs`, `StlKwargs`, `stl_output`, `#[polars_expr] fn stl(...)`, and `#[polars_expr] fn loess(...)` — they get rewritten in the next steps.

Also delete from the top of the file:
- `use rayon::prelude::*;` (no longer used in this file — rust-stats handles rayon internally)

- [ ] **Step 3: Rewrite the `loess` polars expression to call `rust_stats::loess`**

Find the `#[polars_expr] fn loess` body in `src/expressions.rs` and replace it with:

```rust
/// Locally estimated scatterplot smoothing (LOESS).
///
/// Thin wrapper over `rust_stats::loess`. See that crate for algorithm
/// details and tunables.
#[polars_expr(output_type = Float64)]
fn loess(inputs: &[Series], kwargs: LoessKwargs) -> PolarsResult<Series> {
    let s = inputs[0].cast(&DataType::Float64)?;
    let ca = s.f64()?;
    let name = s.name().clone();

    if ca.null_count() > 0 {
        return Err(polars_err!(
            ComputeError: "loess: input contains nulls; fill or drop them upstream"
        ));
    }

    let raw: Vec<f64> = ca.into_no_null_iter().collect();
    let y_col = faer::Col::<f64>::from_fn(raw.len(), |i| raw[i]);

    let smoothed = rust_stats::loess(y_col.as_ref(), kwargs.span, kwargs.degree as u8)
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let smoothed_vec: Vec<f64> = (0..smoothed.nrows()).map(|i| smoothed[i]).collect();
    Ok(Float64Chunked::from_slice(name, &smoothed_vec).into_series())
}
```

(Note the explicit `faer::Col::...` to avoid having to add another `use` to the top of the file. We'll do the same in the `stl` rewrite in Task 8.)

- [ ] **Step 4: Verify the crate builds**

Run: `cd /Users/joseph/Projects/polars-timeseries && uv run maturin develop --release`
Expected: clean build (Rust crate compiles, wheel installs).

The previous build cached most polars deps; this should take ~1–2 min, not 5 min.

- [ ] **Step 5: Run the LOESS Python tests**

Run: `uv run pytest tests/test_transforms.py -k loess`
Expected: all LOESS tests pass (including the statsmodels comparison).

- [ ] **Step 6: Run the full Python test suite to make sure nothing else broke**

Run: `uv run pytest tests/`
Expected: 83 passed (the count just before this refactor).

If `stl` tests fail at this step, that's expected — Task 7 hasn't refactored the STL polars expression yet. STL is currently broken between Step 2 (helpers deleted) and Task 7's Step 1 below. Continue to Task 7 before running STL tests.

**Important:** if STL tests fail here, do NOT commit. Move on to Task 7. We commit at the end of Task 7 once both LOESS and STL go through rust-stats.

---

## Task 7: polars-timeseries — refactor `stl` polars expression and rename `noise` → `residual` in the Struct field

Rewrite the `stl` polars_expr to call `rust_stats::stl`. The output Struct field is renamed `noise` → `residual`. Python-side renames (`tests/test_transforms.py`, `__init__.py`, README) come in Task 9.

**Files:**
- Modify: `src/expressions.rs` (polars-timeseries)

- [ ] **Step 1: Update `stl_output` to use `residual`**

Find `fn stl_output(...)` in `src/expressions.rs`. The current third `Field::new` reads `"noise".into()`. Change it to `"residual".into()`:

```rust
fn stl_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("trend".into(), DataType::Float64),
        Field::new("seasonal".into(), DataType::Float64),
        Field::new("residual".into(), DataType::Float64),
    ];
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Struct(fields),
    ))
}
```

- [ ] **Step 2: Replace the `stl` polars expression body**

Find `#[polars_expr(output_type_func = stl_output)] fn stl(...)` and replace its body with:

```rust
/// Cleveland 1990 STL via `rust_stats::stl`. See that crate for algorithm
/// details and tunables.
#[polars_expr(output_type_func = stl_output)]
fn stl(inputs: &[Series], kwargs: StlKwargs) -> PolarsResult<Series> {
    let s = inputs[0].cast(&DataType::Float64)?;
    let ca = s.f64()?;
    let name = s.name().clone();

    let mode = match kwargs.seasonal.as_str() {
        "additive" => rust_stats::DecomposeMode::Additive,
        "multiplicative" => rust_stats::DecomposeMode::Multiplicative,
        other => {
            return Err(polars_err!(
                ComputeError:
                "stl: seasonal must be 'additive' or 'multiplicative', got '{}'",
                other
            ));
        }
    };

    if ca.null_count() > 0 {
        return Err(polars_err!(
            ComputeError: "stl: input contains nulls; fill or drop them upstream"
        ));
    }

    let raw: Vec<f64> = ca.into_no_null_iter().collect();
    let n = raw.len();
    let y_col = faer::Col::<f64>::from_fn(n, |i| raw[i]);

    let opts = rust_stats::StlOpts {
        period: kwargs.period,
        seasonal_window: kwargs.seasonal_window,
        trend_window: if kwargs.trend_window == 0 {
            None
        } else {
            Some(kwargs.trend_window)
        },
        inner_iters: kwargs.inner_iters,
        mode,
    };

    let decomp = rust_stats::stl(y_col.as_ref(), opts)
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let trend_vec: Vec<f64> = (0..n).map(|i| decomp.trend[i]).collect();
    let seasonal_vec: Vec<f64> = (0..n).map(|i| decomp.seasonal[i]).collect();
    let residual_vec: Vec<f64> = (0..n).map(|i| decomp.residual[i]).collect();

    let trend_s = Series::new("trend".into(), trend_vec);
    let seasonal_s = Series::new("seasonal".into(), seasonal_vec);
    let residual_s = Series::new("residual".into(), residual_vec);
    let series_vec = vec![trend_s, seasonal_s, residual_s];
    let struct_ca =
        StructChunked::from_series(name, n, series_vec.iter())?;
    Ok(struct_ca.into_series())
}
```

- [ ] **Step 3: Build**

Run: `uv run maturin develop --release`
Expected: clean build.

- [ ] **Step 4: Run STL tests — these will fail because of the noise→residual rename**

Run: `uv run pytest tests/test_transforms.py -k stl`
Expected: failures like `KeyError: 'noise'` because the Python-side tests still expect `"noise"` as the Struct field name. This is the exact rename that Task 9 fixes.

- [ ] **Step 5: Hold off on commit until Task 9 fixes the Python side**

Don't commit yet — the test suite is currently red. Tasks 8 and 9 are the natural commit boundaries.

---

## Task 8: polars-timeseries — rename `noise` → `residual` in Python wrappers

Rename the user-visible `noise` symbols (`noise()` free function, `.ts.noise()` namespace method, `__all__` entry) to `residual`.

**Files:**
- Modify: `python/polars_timeseries/__init__.py`

- [ ] **Step 1: Rename `noise` to `residual` in `__all__`**

Open `python/polars_timeseries/__init__.py`. Find the `__all__` list and replace `"noise"` with `"residual"`. The list becomes:

```python
__all__ = [
    "box_cox",
    "catch22",
    "center",
    "holt_winters",
    "loess",
    "min_max_scale",
    "residual",
    "seasons",
    "stl",
    "trend",
    "z_score",
]
```

- [ ] **Step 2: Rename the `noise` free function to `residual`**

Find the `def noise(...)` function. Rename it to `residual` and update its body to access the `"residual"` struct field (was `"noise"`):

```python
def residual(
    expr: IntoExpr,
    *,
    period: int,
    seasonal: str = "additive",
    seasonal_window: int = 7,
    trend_window: int | None = None,
    inner_iters: int = 2,
) -> pl.Expr:
    """Residual component of the STL decomposition (see :func:`stl`)."""
    return stl(
        expr,
        period=period,
        seasonal=seasonal,
        seasonal_window=seasonal_window,
        trend_window=trend_window,
        inner_iters=inner_iters,
    ).struct.field("residual")
```

- [ ] **Step 3: Rename the `noise` namespace method to `residual`**

Find `def noise(self, ...)` inside `class TimeSeriesNamespace` and rename it to `residual`. The body remains a call to the now-renamed top-level `residual()`:

```python
    def residual(
        self,
        *,
        period: int,
        seasonal: str = "additive",
        seasonal_window: int = 7,
        trend_window: int | None = None,
        inner_iters: int = 2,
    ) -> pl.Expr:
        return residual(
            self._expr,
            period=period,
            seasonal=seasonal,
            seasonal_window=seasonal_window,
            trend_window=trend_window,
            inner_iters=inner_iters,
        )
```

- [ ] **Step 4: Verify the imports module loads cleanly**

Run: `uv run python -c "import polars_timeseries as pts; print(sorted(pts.__all__))"`
Expected: list shown including `"residual"` and not including `"noise"`.

---

## Task 9: polars-timeseries — update Python tests to use `residual`

Mass-rename the test file's `noise` references to `residual` so the suite goes green again.

**Files:**
- Modify: `tests/test_transforms.py`

- [ ] **Step 1: Replace `["noise"]` with `["residual"]` everywhere in the test file**

Run from the polars-timeseries dir:

```bash
sed -i.bak 's/\["noise"\]/["residual"]/g' tests/test_transforms.py
rm tests/test_transforms.py.bak
```

- [ ] **Step 2: Replace `.ts.noise(` with `.ts.residual(` everywhere in the test file**

Run:

```bash
sed -i.bak 's/\.ts\.noise(/\.ts\.residual(/g' tests/test_transforms.py
rm tests/test_transforms.py.bak
```

- [ ] **Step 3: Rename `test_noise_method_returns_float_series` → `test_residual_method_returns_float_series`**

Open `tests/test_transforms.py`, find `def test_noise_method_returns_float_series(`, and rename it to `def test_residual_method_returns_float_series(`. Update any internal references to `"noise"` if you spot them (the sed pass should have handled most).

- [ ] **Step 4: Run the STL tests**

Run: `uv run pytest tests/test_transforms.py -k 'stl or seasons or trend or residual'`
Expected: all pass.

- [ ] **Step 5: Run the full Python test suite**

Run: `uv run pytest tests/`
Expected: 83 passed.

- [ ] **Step 6: Commit Tasks 6 + 7 + 8 + 9 together**

We held the commit through these four because the suite was red between Step 2 of Task 6 and now. Squash into one feat commit:

```bash
git add Cargo.toml Cargo.lock src/expressions.rs python/polars_timeseries/__init__.py tests/test_transforms.py
git commit -m "refactor(loess+stl): consume rust-stats; rename noise → residual

Adds rust-stats = { path = \"../rust-stats\" } as a path dep and rewires
the \`loess\` and \`stl\` polars expressions to call into the new crate.
The ten private LOESS+STL helpers in src/expressions.rs (\`local_poly_fit_at_xf64\`,
\`local_poly_fit_at\`, \`loess_window_f\`, \`gauss_solve_n\`, \`loess_compute\`,
\`cycle_subseries_smooth\`, \`low_pass_filter\`, \`valid_ma\`, \`stl_inner_loop\`,
\`next_odd_ceil\`) are removed — they live in rust-stats now.

API break: \`stl()\` Struct field renamed \`noise\` → \`residual\`. Python:
\`pl.col(\"x\").ts.noise(...)\` and the \`noise\` free function are renamed to
\`residual\`. Tests and \`__all__\` updated accordingly. Matches the
upstream rust-stats / statsmodels convention.

All 83 Python tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: polars-timeseries — add `seasonal_decompose` polars expression

Add a new polars expression backed by `rust_stats::seasonal_decompose`. The Python wrapper mirrors `stl` but with fewer params.

**Files:**
- Modify: `src/expressions.rs`
- Modify: `python/polars_timeseries/__init__.py`
- Modify: `tests/test_transforms.py`

- [ ] **Step 1: Add `SeasonalDecomposeKwargs` and the polars expression in `src/expressions.rs`**

Append to the bottom of `src/expressions.rs`:

```rust
#[derive(Deserialize)]
struct SeasonalDecomposeKwargs {
    period: u32,
    seasonal: String,
}

fn seasonal_decompose_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("trend".into(), DataType::Float64),
        Field::new("seasonal".into(), DataType::Float64),
        Field::new("residual".into(), DataType::Float64),
    ];
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Struct(fields),
    ))
}

/// Classical (moving-average) seasonal-trend decomposition via
/// `rust_stats::seasonal_decompose`. See that crate for details.
#[polars_expr(output_type_func = seasonal_decompose_output)]
fn seasonal_decompose(
    inputs: &[Series],
    kwargs: SeasonalDecomposeKwargs,
) -> PolarsResult<Series> {
    let s = inputs[0].cast(&DataType::Float64)?;
    let ca = s.f64()?;
    let name = s.name().clone();

    let mode = match kwargs.seasonal.as_str() {
        "additive" => rust_stats::DecomposeMode::Additive,
        "multiplicative" => rust_stats::DecomposeMode::Multiplicative,
        other => {
            return Err(polars_err!(
                ComputeError:
                "seasonal_decompose: seasonal must be 'additive' or 'multiplicative', got '{}'",
                other
            ));
        }
    };

    if ca.null_count() > 0 {
        return Err(polars_err!(
            ComputeError: "seasonal_decompose: input contains nulls; fill or drop them upstream"
        ));
    }

    let raw: Vec<f64> = ca.into_no_null_iter().collect();
    let n = raw.len();
    let y_col = faer::Col::<f64>::from_fn(n, |i| raw[i]);

    let opts = rust_stats::SeasonalDecomposeOpts {
        period: kwargs.period,
        mode,
    };

    let decomp = rust_stats::seasonal_decompose(y_col.as_ref(), opts)
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let trend_vec: Vec<f64> = (0..n).map(|i| decomp.trend[i]).collect();
    let seasonal_vec: Vec<f64> = (0..n).map(|i| decomp.seasonal[i]).collect();
    let residual_vec: Vec<f64> = (0..n).map(|i| decomp.residual[i]).collect();

    let trend_s = Series::new("trend".into(), trend_vec);
    let seasonal_s = Series::new("seasonal".into(), seasonal_vec);
    let residual_s = Series::new("residual".into(), residual_vec);
    let series_vec = vec![trend_s, seasonal_s, residual_s];
    let struct_ca =
        StructChunked::from_series(name, n, series_vec.iter())?;
    Ok(struct_ca.into_series())
}
```

- [ ] **Step 2: Add the Python wrapper**

In `python/polars_timeseries/__init__.py`:

(a) Add `"seasonal_decompose"` to `__all__` (alphabetical position: between `"residual"` and `"seasons"`):

```python
__all__ = [
    "box_cox",
    "catch22",
    "center",
    "holt_winters",
    "loess",
    "min_max_scale",
    "residual",
    "seasonal_decompose",
    "seasons",
    "stl",
    "trend",
    "z_score",
]
```

(b) Add the `seasonal_decompose` free function. Place it after the `stl` function definition:

```python
def seasonal_decompose(
    expr: IntoExpr,
    *,
    period: int,
    seasonal: str = "additive",
) -> pl.Expr:
    """Classical (moving-average) seasonal-trend decomposition.

    Returns a Struct column with three Float64 fields per row: ``trend``
    (centered moving-average), ``seasonal`` (per-phase mean of the
    detrended series, re-centred), and ``residual`` (leftover). The
    first/last ``period // 2`` positions of ``trend`` and ``residual``
    are NaN.

    Equivalent to statsmodels.tsa.seasonal.seasonal_decompose for the
    additive and multiplicative modes. Multiplicative mode requires
    strictly positive input.
    """
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="seasonal_decompose",
        args=[expr],
        kwargs={"period": int(period), "seasonal": str(seasonal)},
        is_elementwise=False,
    )
```

(c) Add the namespace method, alphabetically after `residual` in `class TimeSeriesNamespace`:

```python
    def seasonal_decompose(
        self,
        *,
        period: int,
        seasonal: str = "additive",
    ) -> pl.Expr:
        return seasonal_decompose(
            self._expr,
            period=period,
            seasonal=seasonal,
        )
```

- [ ] **Step 3: Build**

Run: `uv run maturin develop --release`
Expected: clean build.

- [ ] **Step 4: Add tests for the new polars expression**

Append to `tests/test_transforms.py`, after the existing STL tests:

```python
# ---------- seasonal_decompose ----------


def test_seasonal_decompose_returns_struct_with_three_fields():
    n = 24
    df = pl.DataFrame({"x": [float(i) for i in range(n)]})
    result = df.select(pl.col("x").ts.seasonal_decompose(period=4))
    assert result.height == n
    assert isinstance(result.schema["x"], pl.Struct)
    field_names = [f.name for f in result.schema["x"].fields]
    assert field_names == ["trend", "seasonal", "residual"]


def test_seasonal_decompose_additive_reconstruction_in_inner_band():
    period = 4
    pattern = [1.0, 2.0, 3.0, 2.0]
    n = 24
    half = period // 2
    y = [float(i) + pattern[i % period] for i in range(n)]
    df = pl.DataFrame({"x": y})
    rows = df.select(pl.col("x").ts.seasonal_decompose(period=period))["x"].to_list()
    for i in range(half, n - half):
        recon = rows[i]["trend"] + rows[i]["seasonal"] + rows[i]["residual"]
        assert math.isclose(recon, y[i], abs_tol=1e-9)


def test_seasonal_decompose_multiplicative_reconstruction_in_inner_band():
    period = 4
    pattern = [1.0, 2.0, 0.5, 1.5]
    n = 24
    half = period // 2
    y = [(1.0 + 0.05 * i) * pattern[i % period] for i in range(n)]
    df = pl.DataFrame({"x": y})
    rows = df.select(
        pl.col("x").ts.seasonal_decompose(period=period, seasonal="multiplicative")
    )["x"].to_list()
    for i in range(half, n - half):
        recon = rows[i]["trend"] * rows[i]["seasonal"] * rows[i]["residual"]
        assert math.isclose(recon, y[i], rel_tol=1e-9)


def test_seasonal_decompose_edges_are_nan():
    period = 4
    n = 24
    half = period // 2
    df = pl.DataFrame({"x": [float(i) for i in range(n)]})
    rows = df.select(pl.col("x").ts.seasonal_decompose(period=period))["x"].to_list()
    for i in list(range(half)) + list(range(n - half, n)):
        assert math.isnan(rows[i]["trend"]), f"trend[{i}] expected NaN"
        assert math.isnan(rows[i]["residual"]), f"residual[{i}] expected NaN"


def test_seasonal_decompose_period_too_small():
    df = pl.DataFrame({"x": [float(i) for i in range(20)]})
    with pytest.raises(Exception, match="(?i)period"):
        df.select(pl.col("x").ts.seasonal_decompose(period=1))


def test_seasonal_decompose_too_short():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="(?i)samples|seasonal_decompose|short"):
        df.select(pl.col("x").ts.seasonal_decompose(period=4))


def test_seasonal_decompose_multiplicative_rejects_non_positive():
    df = pl.DataFrame({"x": [1.0, 2.0, 0.0, 1.5] * 6})
    with pytest.raises(Exception, match="(?i)positive"):
        df.select(
            pl.col("x").ts.seasonal_decompose(period=4, seasonal="multiplicative")
        )


def test_seasonal_decompose_invalid_seasonal_mode():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.0] * 6})
    with pytest.raises(Exception, match="(?i)additive|multiplicative"):
        df.select(pl.col("x").ts.seasonal_decompose(period=4, seasonal="rotational"))


def test_seasonal_decompose_rejects_nulls():
    df = pl.DataFrame({"x": [1.0, None, 3.0] + [1.0, 2.0, 3.0] * 7})
    with pytest.raises(Exception, match="(?i)null"):
        df.select(pl.col("x").ts.seasonal_decompose(period=3))
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/`
Expected: 92 passed (was 83; +9 for `seasonal_decompose`).

- [ ] **Step 6: Commit**

```bash
git add src/expressions.rs python/polars_timeseries/__init__.py tests/test_transforms.py
git commit -m "feat: add seasonal_decompose polars expression

Adds \`pl.col(\"y\").ts.seasonal_decompose(period=..., seasonal=...)\` —
a thin wrapper over \`rust_stats::seasonal_decompose\`. Returns a Struct
with \`trend / seasonal / residual\` fields (same shape as \`stl\`); first
and last \`period/2\` positions of trend and residual are NaN.

Nine new tests cover schema, additive and multiplicative reconstruction
in the inner band, the explicit NaN-edge contract, and all five
validation paths (period < 2, series too short, multiplicative with
non-positive input, invalid mode, null input).

All 92 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: polars-timeseries — bench_stl.py

Add a Python benchmark script comparing our STL against statsmodels' STL on the statsmodels datasets.

**Files:**
- Create: `bench/bench_stl.py`

- [ ] **Step 1: Create the bench script**

Create `bench/bench_stl.py`:

```python
"""Benchmark polars-timeseries' STL against statsmodels' STL on real
statsmodels time-series datasets.

Both run as Cleveland 1990 LOESS-based STL with no robustness loop:
- ``robust=False`` for statsmodels.tsa.seasonal.STL
- inner_iters=2 (Cleveland default) for both
- Same period per dataset (chosen below)
- Same seasonal_window (statsmodels default 7; we match)

Run with::

    uv run python bench/bench_stl.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL as SmSTL

import polars_timeseries  # noqa: F401  registers the `ts` namespace


def load_datasets() -> dict[str, tuple[np.ndarray, int]]:
    """Returns {name: (series, period)}. Periods chosen per dataset
    based on its sampling frequency / what statsmodels accepts."""
    out: dict[str, tuple[np.ndarray, int]] = {}

    sunspots = sm.datasets.sunspots.load_pandas().data
    out["sunspots"] = (
        sunspots["SUNACTIVITY"].dropna().to_numpy(dtype=np.float64),
        11,  # ~11-year solar cycle
    )

    co2 = sm.datasets.co2.load_pandas().data
    out["co2"] = (
        co2["co2"].dropna().to_numpy(dtype=np.float64),
        52,  # weekly data with annual seasonality
    )

    macro = sm.datasets.macrodata.load_pandas().data
    out["macrodata.realgdp"] = (
        macro["realgdp"].dropna().to_numpy(dtype=np.float64),
        4,  # quarterly
    )

    elnino = sm.datasets.elnino.load_pandas().data
    monthly = elnino.iloc[:, 1:].to_numpy(dtype=np.float64).flatten()
    out["elnino"] = (monthly[~np.isnan(monthly)], 12)  # monthly

    return out


def median_time_ms(callable_, iters: int, repeats: int = 5) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            callable_()
        samples.append((time.perf_counter() - t0) / iters)
    return statistics.median(samples) * 1000.0


def main() -> None:
    print("\nSTL comparison: polars-timeseries vs statsmodels.tsa.seasonal.STL")
    print("(both Cleveland LOESS-based, robust=False, inner_iters=2, "
          "seasonal_window=7)\n")
    print(
        f"{'dataset':>20s}  {'n':>5s}  {'period':>6s}  "
        f"{'mine ms':>10s}  {'sm ms':>10s}  {'speedup':>8s}  "
        f"{'max |Δ T|':>10s}  {'max |Δ S|':>10s}  {'max |Δ R|':>10s}"
    )
    print("-" * 110)

    for name, (y, period) in load_datasets().items():
        n = y.size
        df = pl.DataFrame({"y": y})

        # Warmup both
        df.select(pl.col("y").ts.stl(period=period))
        SmSTL(y, period=period, robust=False).fit()

        iters = 20 if n < 500 else 10 if n < 5000 else 3

        mine_ms = median_time_ms(
            lambda: df.select(pl.col("y").ts.stl(period=period)),
            iters=iters,
        )
        sm_ms = median_time_ms(
            lambda: SmSTL(y, period=period, robust=False).fit(),
            iters=iters,
        )

        # Value comparison: take fitted components from one fresh call
        ours = df.select(pl.col("y").ts.stl(period=period))["y"].to_list()
        ours_trend = np.array([r["trend"] for r in ours])
        ours_seasonal = np.array([r["seasonal"] for r in ours])
        ours_residual = np.array([r["residual"] for r in ours])
        sm_res = SmSTL(y, period=period, robust=False).fit()

        d_t = np.abs(ours_trend - sm_res.trend.to_numpy()).max()
        d_s = np.abs(ours_seasonal - sm_res.seasonal.to_numpy()).max()
        d_r = np.abs(ours_residual - sm_res.resid.to_numpy()).max()

        ratio = sm_ms / mine_ms if mine_ms > 0 else float("nan")
        print(
            f"{name:>20s}  {n:>5d}  {period:>6d}  "
            f"{mine_ms:>10.3f}  {sm_ms:>10.3f}  {ratio:>7.2f}x  "
            f"{d_t:>10.4f}  {d_s:>10.4f}  {d_r:>10.4f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench**

Run: `uv run python bench/bench_stl.py`
Expected: a table with the four datasets, speedup ratios, and per-component max |Δ|. The bench will take ~30s to ~2 min depending on dataset sizes. Capture the output for the README.

If statsmodels rejects a particular `(n, period)` combination (rare; usually requires n >= 2*period), the script will throw — adjust periods in `load_datasets()` accordingly.

- [ ] **Step 3: Run pytest one more time to make sure the bench file didn't accidentally regress anything**

Run: `uv run pytest tests/`
Expected: 92 passed.

- [ ] **Step 4: Commit**

```bash
git add bench/bench_stl.py
git commit -m "bench: STL vs statsmodels.tsa.seasonal.STL

Adds bench/bench_stl.py mirroring the existing bench_loess.py shape:
loads four statsmodels time-series datasets (sunspots, co2,
macrodata.realgdp, elnino), runs both implementations with matching
parameters (Cleveland LOESS-based, robust=False, inner_iters=2,
seasonal_window=7), and reports timing (median of 5 runs after warmup)
plus per-component max |Δ| of trend, seasonal, residual.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: README updates

Document the new `seasonal_decompose` polars expression, the `noise → residual` rename, the rust-stats dependency, and the STL bench numbers.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `seasonal_decompose` to the transforms table**

In `README.md`, find the transforms table (the one that already lists `stl(...)`, `trend(...)`, `seasons(...)`, `noise(...)`). Replace `noise(...)` with `residual(...)` (in-place rename of the row) and insert a new row for `seasonal_decompose` directly above the STL row.

The relevant rows become:

```markdown
| `seasonal_decompose(period, seasonal="additive")` | Classical (moving-average) seasonal-trend decomposition as `pl.Struct{trend, seasonal, residual}`. NaN at first/last `period/2`. Powered by `rust-stats::tsa::seasonal::seasonal_decompose`. |
| `stl(period, seasonal="additive", seasonal_window=7, trend_window=None, inner_iters=2)` | Cleveland 1990 STL via LOESS — `pl.Struct{trend, seasonal, residual}`. Powered by `rust-stats::tsa::seasonal::stl`. |
| `trend(period, seasonal="additive")` | Trend component from `stl()`. |
| `seasons(period, seasonal="additive")` | Seasonal component from `stl()`. |
| `residual(period, seasonal="additive")` | Residual component from `stl()`. |
```

- [ ] **Step 2: Update the usage example**

Find the usage example that currently shows `pl.col("x").ts.noise(...)`. Replace with `residual(...)`. Add a `seasonal_decompose` line nearby:

```python
df.select(pl.col("x").ts.stl(period=12))
df.select(pl.col("x").ts.trend(period=12))
df.select(pl.col("x").ts.seasons(period=12))
df.select(pl.col("x").ts.residual(period=12, seasonal="multiplicative"))

# Classical (moving-average) decomposition. Same Struct shape; NaN at edges.
df.select(pl.col("x").ts.seasonal_decompose(period=12))
```

- [ ] **Step 3: Update the worked example output table**

The existing worked example table has `noise` as a column header. Replace it with `residual`.

(If you can't easily locate this table, just `grep -n "noise" README.md` and replace each user-visible occurrence; the README has few enough that visual inspection suffices.)

- [ ] **Step 4: Add the STL bench numbers**

Find the existing "LOESS vs `statsmodels` LOWESS" subsection in Performance. Add a new subsection right after it called "STL vs `statsmodels` STL" with the bench table you captured in Task 11. Skeleton:

```markdown
### STL vs `statsmodels` STL

`pl.col("y").ts.stl(period=m)` versus `statsmodels.tsa.seasonal.STL(y, period=m, robust=False).fit()` on the same statsmodels datasets (Apple Silicon, single process, median of 5 timing samples after warmup; reproduce with `uv run python bench/bench_stl.py`):

| dataset           |     n | period |  mine ms | statsmodels ms |  speedup | max |Δ T| | max |Δ S| | max |Δ R| |
| ---               |   --: |   ---: |     ---: |           ---: |     ---: |       ---: |       ---: |       ---: |
| `macrodata.realgdp` |  203 |     4 |    [TBD] |          [TBD] |    [TBD] |      [TBD] |      [TBD] |      [TBD] |
| `sunspots`        |   309 |    11 |    [TBD] |          [TBD] |    [TBD] |      [TBD] |      [TBD] |      [TBD] |
| `elnino`          |   732 |    12 |    [TBD] |          [TBD] |    [TBD] |      [TBD] |      [TBD] |      [TBD] |
| `co2`             | 2,225 |    52 |    [TBD] |          [TBD] |    [TBD] |      [TBD] |      [TBD] |      [TBD] |

Both implementations are Cleveland 1990 LOESS-based STL with the robustness outer loop disabled and `inner_iters=2`. Numerical agreement is within Cleveland's documented bit-level differences; see the `bench_loess.py` discussion above for why.
```

Replace the `[TBD]` cells with the numbers you captured in Task 11 Step 2.

- [ ] **Step 5: Add a brief "powered by rust-stats" mention**

At the top of the README under the project description (before the Features section), add a one-liner:

```markdown
LOESS, STL, and seasonal_decompose are powered by the sibling [`rust-stats`](../rust-stats) crate, which exposes them as Faer-typed free functions. polars-timeseries is the polars-flavored frontend.
```

(Adjust the relative link as appropriate for the repo layout — for now `../rust-stats` is correct since both crates live as siblings.)

- [ ] **Step 6: Final pytest run**

Run: `uv run pytest tests/`
Expected: 92 passed.

- [ ] **Step 7: Final cargo check across both crates**

Run: `cd ../rust-stats && cargo test && cd /Users/joseph/Projects/polars-timeseries && uv run maturin develop --release && uv run pytest tests/`
Expected: 27 Rust tests pass; clean build; 92 Python tests pass.

- [ ] **Step 8: Commit the README changes**

```bash
cd /Users/joseph/Projects/polars-timeseries
git add README.md
git commit -m "docs: document seasonal_decompose, residual rename, and STL bench

Updates the transforms table to add the new \`seasonal_decompose\` row and
rename the \`noise\` row to \`residual\`. Updates the usage example and the
worked-example output table for the same rename. Adds an \"STL vs
statsmodels STL\" subsection under Performance with the bench numbers
from bench/bench_stl.py. Adds a one-liner up top crediting rust-stats as
the home of the LOESS / STL / seasonal_decompose implementations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (from `docs/superpowers/specs/2026-05-09-loess-stl-to-rust-stats.md`):
- LOESS into rust-stats with Faer-typed free fns: Tasks 1, 2 ✓
- STL into rust-stats: Tasks 3, 4 ✓
- seasonal_decompose: Tasks 3, 5 ✓
- Path dep + polars-timeseries refactor: Tasks 6, 7 ✓
- seasonal_decompose polars expression: Task 10 ✓
- bench_stl.py vs statsmodels.tsa.seasonal.STL: Task 11 ✓
- noise → residual rename: Tasks 7, 8, 9 ✓
- Manual elimination for inner LOESS solve: Task 2 (`gauss_solve_n` ports verbatim) ✓
- Faer types at API boundary throughout: Tasks 2, 3, 4, 5 ✓

**Placeholder scan**: the only `[TBD]` is intentional — it's where Task 12 Step 4 instructs the engineer to paste the actual bench numbers from Task 11. No silent gaps.

**Type / signature consistency**:
- `Decomposition { trend, seasonal, residual }` — referenced in Tasks 3, 4, 5, 7, 10 with the same field names. Consistent.
- `StlOpts` fields (period, seasonal_window, trend_window, inner_iters, mode) — Tasks 3, 4, 7. Consistent.
- `SeasonalDecomposeOpts { period, mode }` — Tasks 3, 5, 10. Consistent.
- `loess(y, span, degree)` — Task 2 defines, Task 6 calls. Same signature.
- `loess_at(y, xq, span, degree)` — defined Task 2, called by `cycle_subseries_smooth` in Task 4 (via the private `local_poly_fit_at_xf64` it shadows). Consistent.
- `local_poly_fit_at_xf64` is `pub(crate)` so STL in Task 4 can reach it. Confirmed in Task 2 Step 1.
- `loess_compute` is `pub(crate)` — same. Confirmed.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-loess-stl-to-rust-stats.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each task has explicit file paths, complete code, and a single commit, so a fresh subagent can pick it up cold.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
