# rust-stats OLS v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 of `rust-stats` — a pure-Rust OLS regression library over `faer`, with classical inference, HC0–HC3 robust covariance, point and interval prediction, and a statsmodels-style summary, validated against statsmodels reference values.

**Architecture:** Single crate. Concrete `Ols` builder + `OlsResults` struct (no traits). Borrowed inputs (`MatRef<'_, f64>` / `ColRef<'_, f64>`); owned results that retain the augmented design and the QR `R` factor. Numerical core is column-pivoted QR via faer; rank deficiency is an error. Inference distributions come from `statrs`. Lazy caching for heavier derived quantities (`OnceCell`).

**Tech Stack:** Rust 2021, `faer`, `statrs`, `thiserror`, `once_cell`. Dev: `serde`, `serde_json`, `approx`. Python 3 + statsmodels for golden generation (committed but not run by `cargo test`).

**Reference spec:** `docs/superpowers/specs/2026-05-09-rust-stats-ols-design.md` — read it before starting. Open it for any field/method whose semantics aren't restated in a task.

**A note on the faer API.** faer evolves quickly and method names may differ slightly from version to version. Tasks pin a specific version (`faer = "0.22"` — adjust if a newer 0.x is available). For each task that uses faer's QR / triangular-solve / matrix-arithmetic APIs, the steps describe the operations conceptually and show a likely call shape; if a method name doesn't compile, consult `https://docs.rs/faer/<version>/faer/` for the equivalent. Functionality is what matters: column-pivoted QR producing `R` (upper-triangular `p×p`), the column permutation, and a way to apply `Q'` to a vector and to read row norms of `Q`.

---

## File structure

Files created in this plan, with single-purpose responsibilities:

```
rust-stats/
├── Cargo.toml                               # crate manifest, deps
├── .gitignore                               # already exists from spec commit
├── src/
│   ├── lib.rs                               # crate root, re-exports public API
│   ├── error.rs                             # OlsError enum
│   ├── distributions.rs                     # statrs wrappers: t_cdf, t_quantile, f_sf
│   └── regression/
│       ├── mod.rs                           # `pub mod` declarations + re-exports
│       ├── ols.rs                           # Ols<'a> builder + fit() entry point
│       ├── design.rs                        # build_design_matrix() helper
│       ├── results.rs                       # OlsResults struct + accessors + CovType + Inference
│       ├── robust.rs                        # cov_hc0..hc3
│       ├── predict.rs                       # predict() + predict_interval()
│       └── summary.rs                       # summary string formatter
├── tests/
│   ├── golden/
│   │   ├── generate.py                      # statsmodels reference generator
│   │   ├── longley.json
│   │   ├── mtcars.json
│   │   ├── synthetic.json
│   │   ├── heteroskedastic.json
│   │   └── rank_deficient.json              # only the input data; no reference fit
│   ├── ols_golden.rs                        # loads JSON, asserts vs OlsResults
│   ├── properties.rs                        # orthogonality, recovery, permutation
│   └── negative.rs                          # error-path tests
└── examples/
    └── longley.rs
```

Splits worth flagging:
- `regression/design.rs` is its own file because building `X̃` is small but distinct (and easier to test in isolation than as a private function nested inside `fit`).
- `regression/results.rs` holds `OlsResults`, `CovType`, `Inference` together — they are tightly coupled.
- Robust covariance, prediction, and summary formatting are siblings of `results.rs`, each in its own file. They all take `&OlsResults` and produce derived values.

---

## Task 1: Cargo project skeleton and dependencies

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `tests/smoke.rs`

- [ ] **Step 1: Initialize the Cargo manifest**

Create `Cargo.toml`:

```toml
[package]
name = "rust-stats"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Pure-Rust statistical modeling, statsmodels-inspired"

[dependencies]
faer = "0.22"
statrs = "0.18"
thiserror = "2"
once_cell = "1"

[dev-dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
approx = "0.5"
```

- [ ] **Step 2: Create the crate root**

Create `src/lib.rs`:

```rust
//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS). See `regression::Ols`.

pub mod error;
pub mod distributions;
pub mod regression;

pub use error::OlsError;
pub use regression::{CovType, Inference, Ols, OlsResults};
```

This won't compile yet (modules don't exist). That's intentional — the next steps stub them.

- [ ] **Step 3: Create empty module stubs so the crate compiles**

Create `src/error.rs`:

```rust
//! Error types for rust-stats.

#[derive(Debug)]
pub enum OlsError {}
```

Create `src/distributions.rs`:

```rust
//! Thin wrappers over `statrs` distributions used for inference.
```

Create `src/regression/mod.rs`:

```rust
//! Regression models. v1: OLS only.

pub struct Ols<'a> {
    _phantom: core::marker::PhantomData<&'a ()>,
}

pub struct OlsResults;

pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference;
```

These are placeholders to make `lib.rs` compile. Each will be replaced as real types arrive.

- [ ] **Step 4: Add a smoke test**

Create `tests/smoke.rs`:

```rust
#[test]
fn crate_links() {
    let _ = rust_stats::CovType::NonRobust;
}
```

- [ ] **Step 5: Verify it builds and the smoke test passes**

Run: `cargo test --test smoke`
Expected: `test crate_links ... ok` (1 passed).

If `faer = "0.22"` doesn't resolve, try `cargo search faer` and pin the latest 0.x. Update the manifest and re-run.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/ tests/smoke.rs
git commit -m "Bootstrap rust-stats crate skeleton"
```

---

## Task 2: OlsError enum

**Files:**
- Modify: `src/error.rs`
- Create: `tests/negative.rs` (initial — extended in later tasks)

- [ ] **Step 1: Write a test that the error variants exist and Display correctly**

Create `tests/negative.rs`:

```rust
use rust_stats::OlsError;

#[test]
fn error_variants_display_correctly() {
    let cases: Vec<(OlsError, &str)> = vec![
        (
            OlsError::DimensionMismatch { y: 10, x: 8 },
            "dimension mismatch: y has 10 rows but X has 8",
        ),
        (
            OlsError::InsufficientObservations { n: 3, p: 5 },
            "not enough observations: n=3 must exceed p=5",
        ),
        (
            OlsError::RankDeficient { rank: 2, p: 3 },
            "rank deficient design matrix: rank 2 < p 3",
        ),
        (OlsError::NonFinite, "input contains non-finite values"),
        (
            OlsError::NewXShapeMismatch { got: 4, expected: 3 },
            "predict X has 4 columns, expected 3",
        ),
        (
            OlsError::InvalidAlpha(1.5),
            "invalid alpha 1.5: must be in (0, 1)",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(format!("{}", err), expected);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --test negative error_variants_display_correctly`
Expected: compile error — `OlsError` has no variants.

- [ ] **Step 3: Implement OlsError**

Replace `src/error.rs`:

```rust
//! Error types for rust-stats.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum OlsError {
    #[error("dimension mismatch: y has {y} rows but X has {x}")]
    DimensionMismatch { y: usize, x: usize },

    #[error("not enough observations: n={n} must exceed p={p}")]
    InsufficientObservations { n: usize, p: usize },

    #[error("rank deficient design matrix: rank {rank} < p {p}")]
    RankDeficient { rank: usize, p: usize },

    #[error("input contains non-finite values")]
    NonFinite,

    #[error("predict X has {got} columns, expected {expected}")]
    NewXShapeMismatch { got: usize, expected: usize },

    #[error("invalid alpha {0}: must be in (0, 1)")]
    InvalidAlpha(f64),
}
```

`PartialEq` is added to make assertions in later tests easier (the `f64` payload in `InvalidAlpha` is fine for exact equality in tests where we construct both sides ourselves).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --test negative error_variants_display_correctly`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/error.rs tests/negative.rs
git commit -m "Add OlsError enum with Display messages"
```

---

## Task 3: Distributions wrappers

`statrs` provides `StudentsT` and `FisherSnedecor`. We wrap them in tiny helper functions so the call sites in `results.rs` stay readable and so we can adjust the underlying implementation without touching consumers.

**Files:**
- Modify: `src/distributions.rs`
- Create: `tests/distributions.rs`

- [ ] **Step 1: Write tests for the four helpers we need**

Create `tests/distributions.rs`:

```rust
use approx::assert_relative_eq;
use rust_stats::distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};

#[test]
fn t_cdf_at_zero_is_half() {
    assert_relative_eq!(t_cdf(0.0, 10.0), 0.5, epsilon = 1e-12);
}

#[test]
fn t_two_sided_pvalue_known_values() {
    // df=10, |t|=2.228 corresponds to roughly p=0.05
    let p = t_two_sided_pvalue(2.228, 10.0);
    assert_relative_eq!(p, 0.05, epsilon = 1e-3);
}

#[test]
fn t_quantile_symmetry() {
    let df = 12.0;
    let q_upper = t_quantile(0.975, df);
    let q_lower = t_quantile(0.025, df);
    assert_relative_eq!(q_upper, -q_lower, epsilon = 1e-10);
}

#[test]
fn f_survival_at_one_for_df1_df2() {
    // Sanity: F(1, 1) survival at 1.0 is 0.5.
    assert_relative_eq!(f_sf(1.0, 1.0, 1.0), 0.5, epsilon = 1e-10);
}
```

This requires re-exporting the functions through `lib.rs`. Update `src/lib.rs` to add:

```rust
pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test distributions`
Expected: compile errors — functions don't exist.

- [ ] **Step 3: Implement the wrappers**

Replace `src/distributions.rs`:

```rust
//! Thin wrappers over `statrs` distributions used for inference.
//!
//! Centralizing these makes call sites readable and the implementation
//! swappable.

use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

/// CDF of Student's t with `df` degrees of freedom at `x`.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    StudentsT::new(0.0, 1.0, df)
        .expect("df must be > 0")
        .cdf(x)
}

/// Inverse CDF (quantile) of Student's t with `df` degrees of freedom at `p`.
pub fn t_quantile(p: f64, df: f64) -> f64 {
    StudentsT::new(0.0, 1.0, df)
        .expect("df must be > 0")
        .inverse_cdf(p)
}

/// Two-sided p-value for a t-statistic with `df` degrees of freedom.
pub fn t_two_sided_pvalue(t: f64, df: f64) -> f64 {
    let dist = StudentsT::new(0.0, 1.0, df).expect("df must be > 0");
    2.0 * (1.0 - dist.cdf(t.abs()))
}

/// Survival function (1 - CDF) of F-distribution with (df1, df2) at `x`.
pub fn f_sf(x: f64, df1: f64, df2: f64) -> f64 {
    let dist = FisherSnedecor::new(df1, df2).expect("df1, df2 must be > 0");
    1.0 - dist.cdf(x)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --test distributions`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/distributions.rs src/lib.rs tests/distributions.rs
git commit -m "Add distribution wrappers (t and F)"
```

---

## Task 4: `Ols` builder + `fit()` signature

The first real type. We define the constructor, the `without_intercept` option, and a `fit()` that immediately returns a sentinel error. Subsequent tasks fill in real behavior. This task locks in the public shape.

**Files:**
- Modify: `src/regression/mod.rs`
- Create: `src/regression/ols.rs`
- Create: `tests/builder.rs`

- [ ] **Step 1: Write tests for the builder shape**

Create `tests/builder.rs`:

```rust
use faer::{Col, Mat};
use rust_stats::Ols;

#[test]
fn builder_constructs_with_intercept_by_default() {
    let y: Col<f64> = Col::from_fn(3, |i| i as f64);
    let x: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);

    let ols = Ols::new(y.as_ref(), x.as_ref());
    assert!(ols.has_intercept());
}

#[test]
fn without_intercept_disables_intercept() {
    let y: Col<f64> = Col::from_fn(3, |_| 1.0);
    let x: Mat<f64> = Mat::from_fn(3, 2, |_, _| 0.0);

    let ols = Ols::new(y.as_ref(), x.as_ref()).without_intercept();
    assert!(!ols.has_intercept());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test builder`
Expected: compile errors — `Ols::new` doesn't exist.

- [ ] **Step 3: Implement the builder**

Create `src/regression/ols.rs`:

```rust
//! `Ols` builder and the `fit()` entry point.

use crate::error::OlsError;
use crate::regression::results::OlsResults;
use faer::{ColRef, MatRef};

/// Ordinary least squares model builder.
///
/// Construct with `Ols::new(y, X)`; an intercept column is auto-prepended
/// at fit time unless `without_intercept` is called.
pub struct Ols<'a> {
    pub(crate) y: ColRef<'a, f64>,
    pub(crate) x: MatRef<'a, f64>,
    pub(crate) intercept: bool,
}

impl<'a> Ols<'a> {
    pub fn new(y: ColRef<'a, f64>, x: MatRef<'a, f64>) -> Self {
        Self { y, x, intercept: true }
    }

    pub fn without_intercept(mut self) -> Self {
        self.intercept = false;
        self
    }

    pub fn has_intercept(&self) -> bool {
        self.intercept
    }

    pub fn fit(&self) -> Result<OlsResults, OlsError> {
        // Filled in by Task 5 onward.
        unimplemented!("fit() implemented in Task 5+")
    }
}
```

Replace `src/regression/mod.rs`:

```rust
//! Regression models. v1: OLS only.

pub mod ols;
pub mod results;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
```

Create `src/regression/results.rs` (placeholder; real implementation arrives in later tasks):

```rust
//! Fitted-model results object.

pub struct OlsResults;

pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference;
```

- [ ] **Step 4: Run the builder tests to verify they pass**

Run: `cargo test --test builder`
Expected: 2 passed.

(The smoke test from Task 1 also still passes.)

- [ ] **Step 5: Commit**

```bash
git add src/regression/ tests/builder.rs
git commit -m "Add Ols builder with intercept toggle"
```

---

## Task 5: Input validation in `fit()`

Three validation errors before any numerics: dimension mismatch, insufficient observations, non-finite values.

**Files:**
- Modify: `src/regression/ols.rs`
- Modify: `tests/negative.rs`

- [ ] **Step 1: Write the validation tests**

Append to `tests/negative.rs`:

```rust
use faer::{Col, Mat};
use rust_stats::Ols;

#[test]
fn fit_rejects_mismatched_y_x_rows() {
    let y: Col<f64> = Col::from_fn(5, |_| 0.0);
    let x: Mat<f64> = Mat::from_fn(4, 2, |_, _| 1.0);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::DimensionMismatch { y: 5, x: 4 });
}

#[test]
fn fit_rejects_insufficient_observations() {
    // n=2, intercept=true, so p=3 ⇒ n <= p
    let y: Col<f64> = Col::from_fn(2, |_| 1.0);
    let x: Mat<f64> = Mat::from_fn(2, 2, |_, _| 1.0);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::InsufficientObservations { n: 2, p: 3 });
}

#[test]
fn fit_rejects_non_finite_in_y() {
    let mut y_data = vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0];
    let y: Col<f64> = Col::from_fn(5, |i| y_data[i]);
    let _ = &mut y_data; // silence unused
    let x: Mat<f64> = Mat::from_fn(5, 2, |i, j| (i + j) as f64);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::NonFinite);
}

#[test]
fn fit_rejects_non_finite_in_x() {
    let y: Col<f64> = Col::from_fn(5, |i| i as f64);
    let x: Mat<f64> = Mat::from_fn(5, 2, |i, j| {
        if i == 2 && j == 1 { f64::INFINITY } else { 1.0 }
    });
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::NonFinite);
}
```

- [ ] **Step 2: Run tests to verify they fail (with `unimplemented!` panic)**

Run: `cargo test --test negative`
Expected: the four new tests panic with "fit() implemented in Task 5+".

- [ ] **Step 3: Implement the validation**

Update `src/regression/ols.rs` `fit()`:

```rust
pub fn fit(&self) -> Result<OlsResults, OlsError> {
    let n_y = self.y.nrows();
    let n_x = self.x.nrows();
    if n_y != n_x {
        return Err(OlsError::DimensionMismatch { y: n_y, x: n_x });
    }

    let n = n_y;
    let p = self.x.ncols() + usize::from(self.intercept);
    if n <= p {
        return Err(OlsError::InsufficientObservations { n, p });
    }

    if !all_finite_col(self.y) || !all_finite_mat(self.x) {
        return Err(OlsError::NonFinite);
    }

    // Numerical fit fills in here in Task 6+.
    unimplemented!("numerical fit in Task 6+")
}

fn all_finite_col(c: ColRef<'_, f64>) -> bool {
    (0..c.nrows()).all(|i| c[i].is_finite())
}

fn all_finite_mat(m: MatRef<'_, f64>) -> bool {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}
```

Note on faer indexing: `c[i]` and `m[(i, j)]` work on borrowed views in current faer. If the version you pinned uses `c.read(i)` / `m.read(i, j)`, substitute those.

- [ ] **Step 4: Run negative tests to verify they pass**

Run: `cargo test --test negative`
Expected: all 5 tests pass (the original `error_variants_display_correctly` plus the 4 new validation tests).

- [ ] **Step 5: Commit**

```bash
git add src/regression/ols.rs tests/negative.rs
git commit -m "Add input validation in Ols::fit"
```

---

## Task 6: `build_design_matrix` helper

Small standalone function that produces `X̃`. Easier to test in isolation than as a private nested step.

**Files:**
- Create: `src/regression/design.rs`
- Modify: `src/regression/mod.rs`
- Create: `tests/design.rs`

- [ ] **Step 1: Write tests for both branches**

Create `tests/design.rs`:

```rust
use approx::assert_relative_eq;
use faer::Mat;
use rust_stats::regression::design::build_design_matrix;

#[test]
fn with_intercept_prepends_column_of_ones() {
    let x: Mat<f64> = Mat::from_fn(4, 2, |i, j| (i * 10 + j) as f64);
    let xt = build_design_matrix(x.as_ref(), true);
    assert_eq!(xt.nrows(), 4);
    assert_eq!(xt.ncols(), 3);
    for i in 0..4 {
        assert_relative_eq!(xt[(i, 0)], 1.0);
        assert_relative_eq!(xt[(i, 1)], (i * 10) as f64);
        assert_relative_eq!(xt[(i, 2)], (i * 10 + 1) as f64);
    }
}

#[test]
fn without_intercept_copies_x_unchanged() {
    let x: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
    let xt = build_design_matrix(x.as_ref(), false);
    assert_eq!(xt.nrows(), 3);
    assert_eq!(xt.ncols(), 2);
    for i in 0..3 {
        for j in 0..2 {
            assert_relative_eq!(xt[(i, j)], (i + j) as f64);
        }
    }
}
```

This requires `regression::design` to be public. We'll re-export the module.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test design`
Expected: compile error — module/function don't exist.

- [ ] **Step 3: Implement the helper**

Create `src/regression/design.rs`:

```rust
//! Design-matrix construction.

use faer::{Mat, MatRef};

/// Build the augmented design matrix `X̃` from `x`. If `intercept`, prepends
/// a column of ones; otherwise returns an owned copy of `x`.
pub fn build_design_matrix(x: MatRef<'_, f64>, intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let p_in = x.ncols();
    let p_out = p_in + usize::from(intercept);
    Mat::from_fn(n, p_out, |i, j| {
        if intercept {
            if j == 0 {
                1.0
            } else {
                x[(i, j - 1)]
            }
        } else {
            x[(i, j)]
        }
    })
}
```

Update `src/regression/mod.rs` to expose the module:

```rust
pub mod design;
pub mod ols;
pub mod results;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test design`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/design.rs src/regression/mod.rs tests/design.rs
git commit -m "Add build_design_matrix helper"
```

---

## Task 7: Pivoted QR + rank check + solve for β̂

This is the heart of the fit. We use faer's column-pivoted QR, detect rank deficiency, solve for β̂ via back-substitution on `R`, and store the pieces we'll need later. The `OlsResults` struct grows real fields here.

**Files:**
- Modify: `src/regression/results.rs`
- Modify: `src/regression/ols.rs`
- Create: `tests/fit_basic.rs`
- Modify: `tests/negative.rs`

- [ ] **Step 1: Write tests for fit happy path and rank deficiency**

Create `tests/fit_basic.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::Ols;

/// Synthetic: y = 2 + 3*x1 - 1*x2 exactly, no noise, with intercept.
#[test]
fn recovers_known_coefficients_exactly() {
    let n = 50;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        if j == 0 { i as f64 * 0.1 } else { (i as f64 * 0.05).sin() }
    });
    let y: Col<f64> = Col::from_fn(n, |i| {
        2.0 + 3.0 * x[(i, 0)] - 1.0 * x[(i, 1)]
    });
    let res = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap();
    let beta = res.coef();
    assert_relative_eq!(beta[0],  2.0, epsilon = 1e-10);
    assert_relative_eq!(beta[1],  3.0, epsilon = 1e-10);
    assert_relative_eq!(beta[2], -1.0, epsilon = 1e-10);
    assert_eq!(res.n_obs(), n);
    assert_eq!(res.df_resid(), n - 3);
    assert_eq!(res.df_model(), 2);
}

#[test]
fn without_intercept_recovers_known_coefficients() {
    let n = 30;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        if j == 0 { (i + 1) as f64 } else { (i as f64).cos() }
    });
    let y: Col<f64> = Col::from_fn(n, |i| 0.5 * x[(i, 0)] + 1.5 * x[(i, 1)]);
    let res = Ols::new(y.as_ref(), x.as_ref())
        .without_intercept()
        .fit()
        .unwrap();
    let beta = res.coef();
    assert_relative_eq!(beta[0], 0.5, epsilon = 1e-10);
    assert_relative_eq!(beta[1], 1.5, epsilon = 1e-10);
}
```

Append to `tests/negative.rs`:

```rust
#[test]
fn fit_rejects_rank_deficient_x() {
    let n = 10;
    // Two identical columns ⇒ rank-deficient even with intercept.
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, _| i as f64);
    let y: Col<f64> = Col::from_fn(n, |i| i as f64);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    match err {
        OlsError::RankDeficient { rank, p } => {
            assert!(rank < p);
            assert_eq!(p, 3);
        }
        other => panic!("expected RankDeficient, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test fit_basic --test negative`
Expected: compile errors / panics — `coef()`, `n_obs()`, `df_resid()`, `df_model()` don't exist; fit panics.

- [ ] **Step 3: Implement the fit numerics + grow `OlsResults`**

Replace `src/regression/results.rs`:

```rust
//! Fitted-model results object.

use faer::{Col, Mat};
use once_cell::sync::OnceCell;

/// Owned result of fitting an OLS model. All accessors are read-only.
pub struct OlsResults {
    // Eagerly computed by fit():
    pub(crate) coef: Col<f64>,
    pub(crate) fitted: Col<f64>,
    pub(crate) residuals: Col<f64>,
    pub(crate) x_design: Mat<f64>,    // X̃: includes intercept column if has_intercept
    pub(crate) r_factor: Mat<f64>,    // R from pivoted QR (p×p, upper triangular)
    pub(crate) perm: Vec<usize>,      // column permutation
    pub(crate) leverage: Col<f64>,    // h_ii (diag of hat matrix)
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) rank: usize,
    pub(crate) sigma2: f64,
    pub(crate) rss: f64,
    pub(crate) tss: f64,
    pub(crate) has_intercept: bool,
    pub(crate) names: Option<Vec<String>>,

    // Lazy caches (filled in later tasks):
    pub(crate) cov_unscaled: OnceCell<Mat<f64>>,
    pub(crate) std_err_classical: OnceCell<Col<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

pub struct Inference {
    pub std_err: Col<f64>,
    pub t_values: Col<f64>,
    pub p_values: Col<f64>,
}
```

Now implement the numerics in `src/regression/ols.rs`. Replace the `fit()` body:

```rust
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use faer::linalg::triangular_solve::{
    solve_upper_triangular_in_place,
    solve_upper_triangular_transpose_in_place,
};
use faer::{Col, ColRef, Mat, MatRef, Par, Side};
use once_cell::sync::OnceCell;

// ... existing struct/builder code ...

impl<'a> Ols<'a> {
    // ... new(), without_intercept(), has_intercept() unchanged ...

    pub fn fit(&self) -> Result<OlsResults, OlsError> {
        // 1. validation (existing) ...

        // 2. Build X̃ (owned).
        let x_design = build_design_matrix(self.x, self.intercept);
        let n = x_design.nrows();
        let p = x_design.ncols();

        // 3. Column-pivoted QR.
        let qr = x_design.col_piv_qr();
        let r_factor: Mat<f64> = qr.compute_r();
        let perm: Vec<usize> = qr.col_perm().arrays().0.to_vec();

        // 4. Rank detection.
        let r00 = r_factor[(0, 0)].abs();
        let tol = (n.max(p) as f64) * f64::EPSILON * r00;
        let rank = (0..p).filter(|&i| r_factor[(i, i)].abs() > tol).count();
        if rank < p {
            return Err(OlsError::RankDeficient { rank, p });
        }

        // 5. Solve R β̂_p = Q' y, then unpermute to get β̂.
        let y_owned: Col<f64> = Col::from_fn(n, |i| self.y[i]);
        let qty: Col<f64> = qr.q_transpose_times(&y_owned).subrows(0, p).to_owned();
        let mut beta_p = qty.clone();
        solve_upper_triangular_in_place(
            r_factor.as_ref(),
            beta_p.as_mut().as_mat_mut(),
            Par::Seq,
        );
        let beta = unpermute(&beta_p, &perm);

        // 6. fitted, residuals, sigma2, rss, tss.
        let fitted: Col<f64> = &x_design * &beta;
        let residuals: Col<f64> = &y_owned - &fitted;
        let rss: f64 = (0..n).map(|i| residuals[i] * residuals[i]).sum();
        let sigma2 = rss / (n - p) as f64;
        let tss = if self.intercept {
            let mean: f64 = (0..n).map(|i| y_owned[i]).sum::<f64>() / n as f64;
            (0..n).map(|i| (y_owned[i] - mean).powi(2)).sum()
        } else {
            (0..n).map(|i| y_owned[i].powi(2)).sum()
        };

        // 7. Hat-matrix diagonal h_ii = ‖Q_i,*‖² (using the thin Q's first p cols).
        //    We compute it without materializing all of Q by applying Q' to each
        //    standard basis vector and reading the first p entries. For a thin
        //    QR of size n×p this costs O(n p²); fine for our sizes.
        //    Alternative: faer often exposes `qr.compute_thin_q()` returning the
        //    n×p Q matrix, in which case h_ii = sum_j Q[i,j]².
        let leverage: Col<f64> = {
            let q_thin = qr.compute_thin_q(); // Mat<f64>, n×p
            Col::from_fn(n, |i| {
                (0..p).map(|j| q_thin[(i, j)].powi(2)).sum::<f64>()
            })
        };

        Ok(OlsResults {
            coef: beta,
            fitted,
            residuals,
            x_design,
            r_factor,
            perm,
            leverage,
            n,
            p,
            rank,
            sigma2,
            rss,
            tss,
            has_intercept: self.intercept,
            names: None,
            cov_unscaled: OnceCell::new(),
            std_err_classical: OnceCell::new(),
        })
    }
}

/// Apply the inverse of a column permutation: given `beta_p[i] = β̂[perm[i]]`,
/// produce β̂.
fn unpermute(beta_p: &Col<f64>, perm: &[usize]) -> Col<f64> {
    let p = perm.len();
    let mut beta = Col::<f64>::zeros(p);
    for i in 0..p {
        beta[perm[i]] = beta_p[i];
    }
    beta
}
```

**Notes for the implementer:**
- `compute_r`, `col_perm`, `q_transpose_times`, `compute_thin_q` are the operations needed; method names may vary slightly. If `col_piv_qr()` isn't the constructor name, look for `ColPivQr::new(x_design.as_ref())` or `x_design.col_pivoted_qr()` in the pinned faer version.
- The triangular-solve module path may differ; the conceptual operation is "solve `R β = b` in place where `R` is upper-triangular".
- If `&Mat * &Col` operator overloading isn't available, use `faer::linalg::matmul::matmul` directly.

Add the minimal accessors needed by the tests to `OlsResults`. Append to `src/regression/results.rs`:

```rust
use faer::ColRef;

impl OlsResults {
    pub fn coef(&self) -> ColRef<'_, f64> { self.coef.as_ref() }
    pub fn n_obs(&self) -> usize { self.n }
    pub fn df_resid(&self) -> usize { self.n - self.p }
    pub fn df_model(&self) -> usize {
        if self.has_intercept { self.p - 1 } else { self.p }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: all tests pass (smoke, distributions, builder, design, fit_basic, negative).

- [ ] **Step 5: Commit**

```bash
git add src/regression/ tests/fit_basic.rs tests/negative.rs
git commit -m "Implement OLS fit: pivoted QR, rank check, β̂ solve"
```

---

## Task 8: Remaining point-estimate accessors and goodness-of-fit

Add `fitted_values`, `residuals`, `sigma`, plus `r_squared`, `adj_r_squared`, `f_statistic`, `f_pvalue`. The values are already cached on `OlsResults` from Task 7; this task exposes them.

**Files:**
- Modify: `src/regression/results.rs`
- Create: `tests/goodness.rs`

- [ ] **Step 1: Write tests**

Create `tests/goodness.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::Ols;

/// Build a deterministic small problem with non-zero residuals so we can
/// assert specific values.
fn small_fit() -> rust_stats::OlsResults {
    let y: Col<f64> = Col::from_fn(6, |i| match i {
        0 => 1.0, 1 => 2.0, 2 => 1.5, 3 => 3.0, 4 => 2.5, _ => 4.0,
    });
    let x: Mat<f64> = Mat::from_fn(6, 1, |i, _| (i as f64) + 1.0);
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn fitted_plus_residuals_recovers_y() {
    let res = small_fit();
    let f = res.fitted_values();
    let e = res.residuals();
    let y_true = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    for i in 0..6 {
        assert_relative_eq!(f[i] + e[i], y_true[i], epsilon = 1e-12);
    }
}

#[test]
fn r_squared_in_zero_one() {
    let res = small_fit();
    let r2 = res.r_squared();
    assert!(r2 > 0.0 && r2 < 1.0);
    let adj = res.adj_r_squared();
    assert!(adj <= r2);
}

#[test]
fn f_statistic_is_positive_with_nonzero_signal() {
    let res = small_fit();
    let f = res.f_statistic();
    let p = res.f_pvalue();
    assert!(f > 0.0);
    assert!(p > 0.0 && p < 1.0);
}

#[test]
fn sigma_squared_matches_rss_over_df_resid() {
    let res = small_fit();
    let rss: f64 = (0..res.n_obs())
        .map(|i| res.residuals()[i].powi(2))
        .sum();
    let expected_sigma = (rss / res.df_resid() as f64).sqrt();
    assert_relative_eq!(res.sigma(), expected_sigma, epsilon = 1e-12);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test goodness`
Expected: compile errors — methods don't exist.

- [ ] **Step 3: Implement the accessors**

Append to `src/regression/results.rs`:

```rust
use crate::distributions::f_sf;

impl OlsResults {
    pub fn fitted_values(&self) -> ColRef<'_, f64> { self.fitted.as_ref() }
    pub fn residuals(&self) -> ColRef<'_, f64> { self.residuals.as_ref() }
    pub fn sigma(&self) -> f64 { self.sigma2.sqrt() }

    pub fn r_squared(&self) -> f64 {
        if self.tss == 0.0 { 1.0 } else { 1.0 - self.rss / self.tss }
    }

    pub fn adj_r_squared(&self) -> f64 {
        let n = self.n as f64;
        let dfm = self.df_model() as f64;
        let dfr = self.df_resid() as f64;
        if dfr == 0.0 || self.tss == 0.0 {
            return self.r_squared();
        }
        1.0 - (1.0 - self.r_squared()) * (n - 1.0) / dfr
            * if self.has_intercept { 1.0 } else { (n - 1.0) / n }
        // Note: with intercept this matches the standard formula
        // 1 - (1 - R²) * (n - 1) / (n - p).
        // The `if !has_intercept` correction matches statsmodels' behavior.
    }

    pub fn f_statistic(&self) -> f64 {
        let dfm = self.df_model() as f64;
        let dfr = self.df_resid() as f64;
        ((self.tss - self.rss) / dfm) / (self.rss / dfr)
    }

    pub fn f_pvalue(&self) -> f64 {
        f_sf(self.f_statistic(), self.df_model() as f64, self.df_resid() as f64)
    }
}
```

The adj-R² expression should reduce to `1 - (1 - R²) * (n-1)/(n-p)` for the intercept case. Double-check by hand on the small fit; if the formula misbehaves without intercept, simplify to the standard one and update the test note.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test goodness`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/results.rs tests/goodness.rs
git commit -m "Add fitted/residuals/sigma and R²/F accessors"
```

---

## Task 9: Classical covariance + std_err / t / p / CI (NonRobust path)

Lazy `(X̃'X̃)⁻¹` and the classical inference quantities. We keep `cov_hc{0..3}` for Task 10.

**Files:**
- Modify: `src/regression/results.rs`
- Create: `tests/inference_classical.rs`

- [ ] **Step 1: Write tests**

Create `tests/inference_classical.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};

/// Reference values computed by hand from a 5×1 problem with intercept:
/// y = [1, 2, 3, 4, 5], x = [1, 2, 3, 4, 5]. Perfect fit ⇒ residuals=0,
/// std_err=0, t=∞, p=0. Use a noisy variant for non-degenerate inference.
fn noisy_small() -> rust_stats::OlsResults {
    let y: Col<f64> = Col::from_fn(5, |i| (i as f64 + 1.0) + 0.1 * (i as f64 - 2.0));
    let x: Mat<f64> = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn std_err_positive_finite() {
    let res = noisy_small();
    let se = res.std_err();
    assert!(se[0].is_finite() && se[0] > 0.0);
    assert!(se[1].is_finite() && se[1] > 0.0);
}

#[test]
fn t_value_equals_coef_over_std_err() {
    let res = noisy_small();
    let beta = res.coef();
    let se = res.std_err();
    let t = res.t_values();
    assert_relative_eq!(t[0], beta[0] / se[0], epsilon = 1e-12);
    assert_relative_eq!(t[1], beta[1] / se[1], epsilon = 1e-12);
}

#[test]
fn p_value_in_zero_one() {
    let res = noisy_small();
    let p = res.p_values();
    for i in 0..res.coef().nrows() {
        assert!(p[i] >= 0.0 && p[i] <= 1.0);
    }
}

#[test]
fn conf_int_brackets_coefficient() {
    let res = noisy_small();
    let beta = res.coef();
    let ci = res.conf_int(0.05);
    for i in 0..beta.nrows() {
        assert!(ci[(i, 0)] <= beta[i]);
        assert!(ci[(i, 1)] >= beta[i]);
    }
}

#[test]
fn cov_nonrobust_diagonal_matches_std_err_squared() {
    let res = noisy_small();
    let cov = res.cov(CovType::NonRobust);
    let se = res.std_err();
    for i in 0..res.coef().nrows() {
        assert_relative_eq!(cov[(i, i)], se[i] * se[i], epsilon = 1e-12);
    }
}

#[test]
fn invalid_alpha_returns_error_or_panics_consistently() {
    // We chose Result for invalid alpha. Adjust if API changed.
    let res = noisy_small();
    // conf_int doesn't return Result in the spec; conf_int_with does not either.
    // We validate alpha at the boundary inside; non-(0,1) alpha is a panic.
    // This test documents that contract.
    let result = std::panic::catch_unwind(|| {
        res.conf_int(0.0);
    });
    assert!(result.is_err());
}
```

Note: the spec uses `OlsError::InvalidAlpha` returned from `conf_int_with`/`predict_interval`. For the convenience `conf_int(alpha)` wrapper we panic on bad alpha to keep the wrapper return type clean. Document this in the rustdoc.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test inference_classical`
Expected: compile errors.

- [ ] **Step 3: Implement classical covariance and inference**

Append to `src/regression/results.rs`:

```rust
use crate::distributions::{t_quantile, t_two_sided_pvalue};
use faer::{Mat, MatRef};
use faer::linalg::triangular_solve::{
    solve_upper_triangular_in_place,
    solve_upper_triangular_transpose_in_place,
};
use faer::{Par, Side};

impl OlsResults {
    /// Classical (X̃'X̃)⁻¹, computed lazily and cached.
    fn cov_unscaled_inner(&self) -> &Mat<f64> {
        self.cov_unscaled.get_or_init(|| {
            let p = self.p;
            // Build I_p, then solve R'·X = I (in-place) to get R'⁻¹,
            // then R·Y = X to finish (R'R)⁻¹ = R⁻¹·R'⁻¹.
            let mut a: Mat<f64> = Mat::identity(p, p);
            // Solve R' a = I  ⇒  a = R'⁻¹
            solve_upper_triangular_transpose_in_place(
                self.r_factor.as_ref(), a.as_mut(), Par::Seq);
            // Solve R b = a  ⇒  b = R⁻¹ R'⁻¹  =  (R'R)⁻¹
            solve_upper_triangular_in_place(
                self.r_factor.as_ref(), a.as_mut(), Par::Seq);
            // a is now (Π R'R Πᵀ)⁻¹ in pivoted coordinates; unpermute rows
            // and columns by perm. Build the unpermuted matrix.
            let mut out: Mat<f64> = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    out[(self.perm[i], self.perm[j])] = a[(i, j)];
                }
            }
            out
        })
    }

    fn classical_std_err_inner(&self) -> &Col<f64> {
        self.std_err_classical.get_or_init(|| {
            let cov = self.cov_unscaled_inner();
            Col::from_fn(self.p, |i| (cov[(i, i)] * self.sigma2).sqrt())
        })
    }

    pub fn std_err(&self) -> ColRef<'_, f64> {
        self.classical_std_err_inner().as_ref()
    }

    pub fn t_values(&self) -> Col<f64> {
        let beta = self.coef.as_ref();
        let se = self.classical_std_err_inner();
        Col::from_fn(self.p, |i| beta[i] / se[i])
    }

    pub fn p_values(&self) -> Col<f64> {
        let t = self.t_values();
        let df = self.df_resid() as f64;
        Col::from_fn(self.p, |i| t_two_sided_pvalue(t[i], df))
    }

    pub fn conf_int(&self, alpha: f64) -> Mat<f64> {
        assert!(alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1); use conf_int_with for a Result-returning version");
        let crit = t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64);
        let beta = self.coef.as_ref();
        let se = self.classical_std_err_inner();
        Mat::from_fn(self.p, 2, |i, j| match j {
            0 => beta[i] - crit * se[i],
            _ => beta[i] + crit * se[i],
        })
    }

    pub fn cov(&self, cov: CovType) -> Mat<f64> {
        match cov {
            CovType::NonRobust => {
                let unscaled = self.cov_unscaled_inner();
                Mat::from_fn(self.p, self.p,
                    |i, j| unscaled[(i, j)] * self.sigma2)
            }
            // HC0..HC3 implemented in Task 10.
            _ => unimplemented!("robust covariance arrives in Task 10"),
        }
    }
}
```

**Implementer note on the unpermutation step:** the QR factorization is of `X̃·P`, so the directly-computed `(R'R)⁻¹` corresponds to the *permuted* design. To obtain `(X̃'X̃)⁻¹` we apply `P · A · Pᵀ` (which the code above does by writing into `out[(perm[i], perm[j])]`). Verify on a tiny example by comparing to a direct `(X̃'X̃)⁻¹` you compute by another method during development.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inference_classical`
Expected: 6 passed (the panic test on `conf_int(0.0)` passes via `catch_unwind`).

- [ ] **Step 5: Commit**

```bash
git add src/regression/results.rs tests/inference_classical.rs
git commit -m "Add classical covariance and inference (NonRobust)"
```

---

## Task 10: Robust covariance (HC0–HC3)

Sandwich estimators against `(X̃'X̃)⁻¹`. We add `cov_hc0..hc3()`, then plumb them through `cov(CovType)`.

**Files:**
- Create: `src/regression/robust.rs`
- Modify: `src/regression/results.rs`
- Modify: `src/regression/mod.rs`
- Create: `tests/robust.rs`

- [ ] **Step 1: Write tests**

Create `tests/robust.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};

fn small_heteroskedastic() -> rust_stats::OlsResults {
    // y = 1 + 2x + ε with Var(ε) ∝ x²
    let n = 30;
    let x: Mat<f64> = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1 + 0.5);
    let y: Col<f64> = Col::from_fn(n, |i| {
        let xi = x[(i, 0)];
        1.0 + 2.0 * xi + 0.05 * xi * ((i as f64).sin())
    });
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn hc1_equals_hc0_times_n_over_n_minus_p() {
    let res = small_heteroskedastic();
    let hc0 = res.cov_hc0();
    let hc1 = res.cov_hc1();
    let scale = res.n_obs() as f64 / res.df_resid() as f64;
    for i in 0..res.coef().nrows() {
        for j in 0..res.coef().nrows() {
            assert_relative_eq!(hc1[(i, j)], hc0[(i, j)] * scale, epsilon = 1e-10);
        }
    }
}

#[test]
fn hc_diagonals_are_positive() {
    let res = small_heteroskedastic();
    for cov in [
        res.cov_hc0(), res.cov_hc1(), res.cov_hc2(), res.cov_hc3(),
    ] {
        for i in 0..res.coef().nrows() {
            assert!(cov[(i, i)] > 0.0);
        }
    }
}

#[test]
fn cov_dispatches_to_robust_variants() {
    let res = small_heteroskedastic();
    for (variant, direct) in [
        (CovType::HC0, res.cov_hc0()),
        (CovType::HC1, res.cov_hc1()),
        (CovType::HC2, res.cov_hc2()),
        (CovType::HC3, res.cov_hc3()),
    ] {
        let via = res.cov(variant);
        for i in 0..res.coef().nrows() {
            for j in 0..res.coef().nrows() {
                assert_relative_eq!(via[(i, j)], direct[(i, j)], epsilon = 1e-12);
            }
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test robust`
Expected: compile errors.

- [ ] **Step 3: Implement robust covariance**

Create `src/regression/robust.rs`:

```rust
//! Heteroskedasticity-consistent (HC) covariance estimators.

use crate::regression::results::OlsResults;
use faer::linalg::triangular_solve::{
    solve_upper_triangular_in_place,
    solve_upper_triangular_transpose_in_place,
};
use faer::{Mat, Par};

/// Compute Cov_HC = (X'X)⁻¹ · M · (X'X)⁻¹  where  M = Σ_i ω_i x_i x_i'.
/// `weights[i]` is ω_i.
pub(crate) fn sandwich(res: &OlsResults, weights: &[f64]) -> Mat<f64> {
    let n = res.n;
    let p = res.p;

    // Build M = X̃' diag(ω) X̃ as Σ ω_i x_i x_i'.
    let x = &res.x_design;
    let mut m: Mat<f64> = Mat::zeros(p, p);
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 { continue; }
        for j in 0..p {
            let xij = x[(i, j)];
            for k in 0..p {
                m[(j, k)] += w * xij * x[(i, k)];
            }
        }
    }

    // Apply (X̃'X̃)⁻¹ on both sides via triangular solves on R, with permutation.
    // (X̃'X̃)⁻¹ = P · (R'R)⁻¹ · Pᵀ. We compute (R'R)⁻¹ · M_permuted · (R'R)⁻¹
    // in pivoted coordinates, then unpermute.

    // Permute rows and cols of M by perm⁻¹: m_p[i,j] = m[perm[i], perm[j]].
    let mut m_p: Mat<f64> = Mat::from_fn(p, p, |i, j| m[(res.perm[i], res.perm[j])]);

    // Apply (R'R)⁻¹ on the right: solve R' Y = M_p ⇒ Y = R'⁻¹ M_p, then
    //                              solve R Z = Y ⇒ Z = R⁻¹ Y = (R'R)⁻¹ M_p.
    solve_upper_triangular_transpose_in_place(res.r_factor.as_ref(), m_p.as_mut(), Par::Seq);
    solve_upper_triangular_in_place(res.r_factor.as_ref(), m_p.as_mut(), Par::Seq);

    // Apply (R'R)⁻¹ on the left: transpose, repeat, transpose back.
    let mut m_pt: Mat<f64> = Mat::from_fn(p, p, |i, j| m_p[(j, i)]);
    solve_upper_triangular_transpose_in_place(res.r_factor.as_ref(), m_pt.as_mut(), Par::Seq);
    solve_upper_triangular_in_place(res.r_factor.as_ref(), m_pt.as_mut(), Par::Seq);
    let result_p: Mat<f64> = Mat::from_fn(p, p, |i, j| m_pt[(j, i)]);

    // Unpermute.
    let mut out: Mat<f64> = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            out[(res.perm[i], res.perm[j])] = result_p[(i, j)];
        }
    }
    out
}

pub(crate) fn weights_hc0(res: &OlsResults) -> Vec<f64> {
    (0..res.n).map(|i| res.residuals[i].powi(2)).collect()
}

pub(crate) fn weights_hc1(res: &OlsResults) -> Vec<f64> {
    let scale = res.n as f64 / res.df_resid() as f64;
    (0..res.n).map(|i| res.residuals[i].powi(2) * scale).collect()
}

pub(crate) fn weights_hc2(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| res.residuals[i].powi(2) / (1.0 - res.leverage[i]))
        .collect()
}

pub(crate) fn weights_hc3(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| {
            let one_minus_h = 1.0 - res.leverage[i];
            res.residuals[i].powi(2) / (one_minus_h * one_minus_h)
        })
        .collect()
}
```

Update `src/regression/mod.rs`:

```rust
pub mod design;
pub mod ols;
pub mod results;
pub mod robust;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
```

Append to `src/regression/results.rs`:

```rust
use crate::regression::robust::{sandwich, weights_hc0, weights_hc1, weights_hc2, weights_hc3};

impl OlsResults {
    pub fn cov_hc0(&self) -> Mat<f64> { sandwich(self, &weights_hc0(self)) }
    pub fn cov_hc1(&self) -> Mat<f64> { sandwich(self, &weights_hc1(self)) }
    pub fn cov_hc2(&self) -> Mat<f64> { sandwich(self, &weights_hc2(self)) }
    pub fn cov_hc3(&self) -> Mat<f64> { sandwich(self, &weights_hc3(self)) }
}
```

Replace the `unimplemented!` arm in the existing `cov(...)`:

```rust
pub fn cov(&self, cov: CovType) -> Mat<f64> {
    match cov {
        CovType::NonRobust => {
            let unscaled = self.cov_unscaled_inner();
            Mat::from_fn(self.p, self.p,
                |i, j| unscaled[(i, j)] * self.sigma2)
        }
        CovType::HC0 => self.cov_hc0(),
        CovType::HC1 => self.cov_hc1(),
        CovType::HC2 => self.cov_hc2(),
        CovType::HC3 => self.cov_hc3(),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test robust`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/ tests/robust.rs
git commit -m "Add HC0–HC3 robust covariance"
```

---

## Task 11: `inference()`, `conf_int_with()` — generic over CovType

Helper that returns SE/t/p for any covariance type, plus the Result-returning CI variant.

**Files:**
- Modify: `src/regression/results.rs`
- Modify: `src/error.rs` (no change but referenced)
- Create: `tests/inference_helper.rs`

- [ ] **Step 1: Write tests**

Create `tests/inference_helper.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols, OlsError};

fn fit() -> rust_stats::OlsResults {
    let n = 25;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| (i + j * 3) as f64 * 0.1);
    let y: Col<f64> = Col::from_fn(n, |i| 1.0 + 2.0 * (i as f64) * 0.1
        + 0.05 * ((i as f64).cos()));
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn inference_nonrobust_matches_direct_accessors() {
    let res = fit();
    let inf = res.inference(CovType::NonRobust);
    let se = res.std_err();
    let t = res.t_values();
    let p = res.p_values();
    for i in 0..res.coef().nrows() {
        assert_relative_eq!(inf.std_err[i],  se[i], epsilon = 1e-12);
        assert_relative_eq!(inf.t_values[i], t[i],  epsilon = 1e-12);
        assert_relative_eq!(inf.p_values[i], p[i],  epsilon = 1e-12);
    }
}

#[test]
fn inference_hc1_differs_from_nonrobust() {
    let res = fit();
    let nr  = res.inference(CovType::NonRobust);
    let hc1 = res.inference(CovType::HC1);
    let mut any_diff = false;
    for i in 0..res.coef().nrows() {
        if (nr.std_err[i] - hc1.std_err[i]).abs() > 1e-8 {
            any_diff = true;
        }
    }
    assert!(any_diff, "HC1 SEs should differ from classical on this dataset");
}

#[test]
fn conf_int_with_invalid_alpha_returns_error() {
    let res = fit();
    let err = res.conf_int_with(CovType::NonRobust, 1.5).unwrap_err();
    assert_eq!(err, OlsError::InvalidAlpha(1.5));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test inference_helper`
Expected: compile errors.

- [ ] **Step 3: Implement the helpers**

Append to `src/regression/results.rs`:

```rust
use crate::error::OlsError;

impl OlsResults {
    pub fn inference(&self, cov: CovType) -> Inference {
        let cov_mat = self.cov(cov);
        let beta = self.coef.as_ref();
        let df = self.df_resid() as f64;
        let std_err = Col::from_fn(self.p, |i| cov_mat[(i, i)].sqrt());
        let t_values = Col::from_fn(self.p, |i| beta[i] / std_err[i]);
        let p_values = Col::from_fn(self.p, |i| t_two_sided_pvalue(t_values[i], df));
        Inference { std_err, t_values, p_values }
    }

    pub fn conf_int_with(&self, cov: CovType, alpha: f64) -> Result<Mat<f64>, OlsError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(OlsError::InvalidAlpha(alpha));
        }
        let inf = self.inference(cov);
        let crit = t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64);
        let beta = self.coef.as_ref();
        Ok(Mat::from_fn(self.p, 2, |i, j| match j {
            0 => beta[i] - crit * inf.std_err[i],
            _ => beta[i] + crit * inf.std_err[i],
        }))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test inference_helper`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/results.rs tests/inference_helper.rs
git commit -m "Add inference() and conf_int_with() over any CovType"
```

---

## Task 12: `predict()` and `predict_interval()`

**Files:**
- Create: `src/regression/predict.rs`
- Modify: `src/regression/mod.rs`
- Modify: `src/regression/results.rs`
- Create: `tests/predict.rs`

- [ ] **Step 1: Write tests**

Create `tests/predict.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{Ols, OlsError};

fn fit_simple() -> (rust_stats::OlsResults, Mat<f64>) {
    let n = 20;
    let x: Mat<f64> = Mat::from_fn(n, 1, |i, _| i as f64 * 0.1);
    let y: Col<f64> = Col::from_fn(n, |i| 1.0 + 2.0 * (i as f64) * 0.1);
    let res = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap();
    let x_new: Mat<f64> = Mat::from_fn(3, 1, |i, _| i as f64);
    (res, x_new)
}

#[test]
fn predict_matches_known_function() {
    let (res, x_new) = fit_simple();
    let yhat = res.predict(x_new.as_ref()).unwrap();
    for i in 0..3 {
        assert_relative_eq!(yhat[i], 1.0 + 2.0 * (i as f64), epsilon = 1e-10);
    }
}

#[test]
fn predict_rejects_wrong_column_count() {
    let (res, _) = fit_simple();
    let bad: Mat<f64> = Mat::from_fn(2, 5, |_, _| 0.0);
    let err = res.predict(bad.as_ref()).unwrap_err();
    assert_eq!(err, OlsError::NewXShapeMismatch { got: 5, expected: 1 });
}

#[test]
fn predict_interval_brackets_point_estimate() {
    let (res, x_new) = fit_simple();
    let band = res.predict_interval(x_new.as_ref(), 0.05).unwrap();
    for i in 0..3 {
        let fit = band[(i, 0)];
        let lo = band[(i, 1)];
        let hi = band[(i, 2)];
        assert!(lo < fit, "lower bound must be below fit");
        assert!(hi > fit, "upper bound must be above fit");
    }
}

#[test]
fn predict_interval_rejects_invalid_alpha() {
    let (res, x_new) = fit_simple();
    let err = res.predict_interval(x_new.as_ref(), 0.0).unwrap_err();
    assert_eq!(err, OlsError::InvalidAlpha(0.0));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test predict`
Expected: compile errors.

- [ ] **Step 3: Implement predict + predict_interval**

Create `src/regression/predict.rs`:

```rust
//! Prediction on new observations.

use crate::distributions::t_quantile;
use crate::error::OlsError;
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use faer::linalg::triangular_solve::solve_upper_triangular_transpose_in_place;
use faer::{Col, Mat, MatRef, Par};

pub(crate) fn predict(res: &OlsResults, x_new: MatRef<'_, f64>) -> Result<Col<f64>, OlsError> {
    let expected = res.p - usize::from(res.has_intercept);
    if x_new.ncols() != expected {
        return Err(OlsError::NewXShapeMismatch { got: x_new.ncols(), expected });
    }
    let x_aug = build_design_matrix(x_new, res.has_intercept);
    let yhat: Col<f64> = &x_aug * &res.coef;
    Ok(yhat)
}

pub(crate) fn predict_interval(
    res: &OlsResults,
    x_new: MatRef<'_, f64>,
    alpha: f64,
) -> Result<Mat<f64>, OlsError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(OlsError::InvalidAlpha(alpha));
    }
    let expected = res.p - usize::from(res.has_intercept);
    if x_new.ncols() != expected {
        return Err(OlsError::NewXShapeMismatch { got: x_new.ncols(), expected });
    }

    let x_aug = build_design_matrix(x_new, res.has_intercept);
    let yhat: Col<f64> = &x_aug * &res.coef;
    let crit = t_quantile(1.0 - alpha / 2.0, res.df_resid() as f64);

    let n_new = x_aug.nrows();
    let p = res.p;

    let mut out: Mat<f64> = Mat::zeros(n_new, 3);
    for i in 0..n_new {
        // x_i in original ordering; permute to pivoted coords.
        let mut x_p: Col<f64> = Col::from_fn(p, |k| x_aug[(i, res.perm[k])]);
        // Solve R' z = x_p ⇒ z = R'⁻¹ x_p; then x_p' (R'R)⁻¹ x_p = ‖z‖².
        let mut z_mat: Mat<f64> = Mat::from_fn(p, 1, |r, _| x_p[r]);
        solve_upper_triangular_transpose_in_place(
            res.r_factor.as_ref(), z_mat.as_mut(), Par::Seq);
        let quad: f64 = (0..p).map(|k| z_mat[(k, 0)].powi(2)).sum();
        let se_pred = (res.sigma2 * (1.0 + quad)).sqrt();
        out[(i, 0)] = yhat[i];
        out[(i, 1)] = yhat[i] - crit * se_pred;
        out[(i, 2)] = yhat[i] + crit * se_pred;
    }
    Ok(out)
}
```

Update `src/regression/mod.rs`:

```rust
pub mod design;
pub mod ols;
pub mod predict;
pub mod results;
pub mod robust;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
```

Append to `src/regression/results.rs`:

```rust
use crate::regression::predict::{predict as predict_impl, predict_interval as predict_interval_impl};

impl OlsResults {
    pub fn predict(&self, x_new: MatRef<'_, f64>) -> Result<Col<f64>, OlsError> {
        predict_impl(self, x_new)
    }
    pub fn predict_interval(&self, x_new: MatRef<'_, f64>, alpha: f64)
        -> Result<Mat<f64>, OlsError>
    {
        predict_interval_impl(self, x_new, alpha)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test predict`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/ tests/predict.rs
git commit -m "Add predict() and predict_interval()"
```

---

## Task 13: `with_names()` / `names()`

**Files:**
- Modify: `src/regression/results.rs`
- Create: `tests/names.rs`

- [ ] **Step 1: Write tests**

Create `tests/names.rs`:

```rust
use faer::{Col, Mat};
use rust_stats::Ols;

fn fit() -> rust_stats::OlsResults {
    let y: Col<f64> = Col::from_fn(5, |i| i as f64);
    let x: Mat<f64> = Mat::from_fn(5, 2, |i, j| (i + j) as f64);
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn names_default_is_none() {
    let res = fit();
    assert!(res.names().is_none());
}

#[test]
fn with_names_stores_them() {
    let res = fit().with_names(vec![
        "const".to_string(), "age".to_string(), "income".to_string(),
    ]);
    let names = res.names().unwrap();
    assert_eq!(names, ["const", "age", "income"]);
}

#[test]
#[should_panic(expected = "names length")]
fn with_names_wrong_length_panics() {
    let _ = fit().with_names(vec!["only_one".to_string()]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test names`
Expected: compile errors.

- [ ] **Step 3: Implement**

Append to `src/regression/results.rs`:

```rust
impl OlsResults {
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        assert!(names.len() == self.p,
            "names length {} != p {}", names.len(), self.p);
        self.names = Some(names);
        self
    }

    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test names`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/results.rs tests/names.rs
git commit -m "Add OlsResults::with_names / names accessors"
```

---

## Task 14: `summary()` / `summary_with()` / Display

A pure-formatting layer. We don't snapshot-test the exact byte-for-byte output (that's brittle); we test structural properties — header lines present, all coefficient names appear with finite numbers, line lengths consistent.

**Files:**
- Create: `src/regression/summary.rs`
- Modify: `src/regression/mod.rs`
- Modify: `src/regression/results.rs`
- Create: `tests/summary.rs`

- [ ] **Step 1: Write tests**

Create `tests/summary.rs`:

```rust
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};

fn fit() -> rust_stats::OlsResults {
    let n = 16;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| (i + j * 7) as f64 * 0.13);
    let y: Col<f64> = Col::from_fn(n, |i| {
        1.0 + 0.5 * (i as f64) * 0.13 + 0.1 * ((i as f64).sin())
    });
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
        .with_names(vec!["const".to_string(), "x1".to_string(), "x2".to_string()])
}

#[test]
fn summary_contains_required_headers() {
    let s = fit().summary();
    for needle in [
        "OLS Regression Results",
        "Dep. Variable",
        "R-squared",
        "Adj. R-squared",
        "F-statistic",
        "No. Observations",
        "Df Residuals",
        "Df Model",
        "Covariance Type:",
        "coef",
        "std err",
        "P>|t|",
    ] {
        assert!(s.contains(needle), "summary missing {needle:?}\n---\n{s}");
    }
}

#[test]
fn summary_lists_each_coefficient_name() {
    let s = fit().summary();
    for name in ["const", "x1", "x2"] {
        assert!(s.contains(name), "summary missing coef name {name}");
    }
}

#[test]
fn summary_with_changes_covariance_label() {
    let s = fit().summary_with(CovType::HC1);
    assert!(s.contains("HC1"), "expected covariance label HC1\n---\n{s}");
}

#[test]
fn display_is_summary() {
    let res = fit();
    let s_disp = format!("{res}");
    let s_summ = res.summary();
    assert_eq!(s_disp, s_summ);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test summary`
Expected: compile errors.

- [ ] **Step 3: Implement the summary formatter**

Create `src/regression/summary.rs`:

```rust
//! statsmodels-style text summary.

use crate::regression::results::{CovType, OlsResults};
use std::fmt::Write;

pub(crate) fn render(res: &OlsResults, cov: CovType) -> String {
    let inf = res.inference(cov);
    let beta = res.coef();
    // Use 95% CI via conf_int_with (alpha=0.05); fallback to no CI if unavailable.
    let ci = res.conf_int_with(cov, 0.05).expect("alpha 0.05 valid");

    let mut s = String::new();
    let line_eq: String = "=".repeat(78);
    let line_dash: String = "-".repeat(78);

    let _ = writeln!(s, "{:^78}", "OLS Regression Results");
    let _ = writeln!(s, "{line_eq}");
    let _ = writeln!(s,
        "Dep. Variable:      {:>14}   R-squared:         {:>16.4}",
        "y", res.r_squared());
    let _ = writeln!(s,
        "Model:              {:>14}   Adj. R-squared:    {:>16.4}",
        "OLS", res.adj_r_squared());
    let _ = writeln!(s,
        "Method:             {:>14}   F-statistic:       {:>16.3}",
        "Least Squares", res.f_statistic());
    let _ = writeln!(s,
        "No. Observations:   {:>14}   Prob (F-statistic):{:>16.3e}",
        res.n_obs(), res.f_pvalue());
    let _ = writeln!(s, "Df Residuals:       {:>14}", res.df_resid());
    let _ = writeln!(s, "Df Model:           {:>14}", res.df_model());
    let _ = writeln!(s, "Covariance Type:    {:>14}", cov_label(cov));
    let _ = writeln!(s, "{line_eq}");
    let _ = writeln!(s,
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]");
    let _ = writeln!(s, "{line_dash}");

    let default_names: Vec<String> = (0..res.df_model())
        .map(|i| format!("x{}", i + 1))
        .collect();
    let names: Vec<&str> = match res.names() {
        Some(ns) => ns.iter().map(|s| s.as_str()).collect(),
        None => {
            let mut v: Vec<&str> = Vec::with_capacity(beta.nrows());
            if res.has_intercept() { v.push("const"); }
            for n in &default_names { v.push(n.as_str()); }
            v
        }
    };

    for i in 0..beta.nrows() {
        let _ = writeln!(s,
            "{:<10} {:>10.4} {:>10.4} {:>10.3} {:>10.3} {:>10.4} {:>10.4}",
            names[i],
            beta[i],
            inf.std_err[i],
            inf.t_values[i],
            inf.p_values[i],
            ci[(i, 0)],
            ci[(i, 1)],
        );
    }
    let _ = writeln!(s, "{line_eq}");
    s
}

fn cov_label(cov: CovType) -> &'static str {
    match cov {
        CovType::NonRobust => "nonrobust",
        CovType::HC0 => "HC0",
        CovType::HC1 => "HC1",
        CovType::HC2 => "HC2",
        CovType::HC3 => "HC3",
    }
}
```

`has_intercept()` needs to exist on `OlsResults` (it doesn't yet — only on `Ols`). Add it. Append to `src/regression/results.rs`:

```rust
impl OlsResults {
    pub fn has_intercept(&self) -> bool { self.has_intercept }

    pub fn summary(&self) -> String {
        crate::regression::summary::render(self, CovType::NonRobust)
    }
    pub fn summary_with(&self, cov: CovType) -> String {
        crate::regression::summary::render(self, cov)
    }
}

impl std::fmt::Display for OlsResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary())
    }
}

impl std::fmt::Debug for OlsResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OlsResults")
            .field("n", &self.n)
            .field("p", &self.p)
            .field("rank", &self.rank)
            .field("has_intercept", &self.has_intercept)
            .finish_non_exhaustive()
    }
}
```

Update `src/regression/mod.rs`:

```rust
pub mod design;
pub mod ols;
pub mod predict;
pub mod results;
pub mod robust;
pub mod summary;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test summary`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regression/ tests/summary.rs
git commit -m "Add statsmodels-style summary() and Display"
```

---

## Task 15: Golden-test scaffold + Longley dataset

A committed Python script generates JSON reference values from statsmodels. The Rust loader and the Longley assertion live here. Subsequent datasets append in Task 16.

**Files:**
- Create: `tests/golden/generate.py`
- Create: `tests/golden/longley.json` (output of running the script)
- Create: `tests/ols_golden.rs`

- [ ] **Step 1: Write the Python generator**

Create `tests/golden/generate.py`:

```python
"""Generate golden reference values for rust-stats OLS tests.

Run manually:
    python3 tests/golden/generate.py

Pinned versions documented below to make outputs reproducible.
Do NOT run this from cargo test. Outputs are committed to source control.
"""
# Tested with: numpy 1.26, scipy 1.12, statsmodels 0.14, pandas 2.2

import json
import os
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.datasets as smds

OUT_DIR = Path(__file__).parent

COV_TYPES = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]


def fit_and_dump(name, y, x_no_intercept, x_predict_no_intercept, intercept=True):
    if intercept:
        x = sm.add_constant(x_no_intercept, has_constant="add")
        x_pred = sm.add_constant(x_predict_no_intercept, has_constant="add")
    else:
        x = x_no_intercept
        x_pred = x_predict_no_intercept

    out = {
        "y": list(map(float, y.flatten())),
        "x": x_no_intercept.tolist(),
        "intercept": bool(intercept),
        "x_predict": x_predict_no_intercept.tolist(),
    }

    base = sm.OLS(y, x).fit()
    out["coef"]          = list(map(float, base.params))
    out["residuals"]     = list(map(float, base.resid))
    out["fitted"]        = list(map(float, base.fittedvalues))
    out["rss"]           = float(base.ssr)
    out["sigma"]         = float(np.sqrt(base.scale))
    out["r_squared"]     = float(base.rsquared)
    out["adj_r_squared"] = float(base.rsquared_adj)
    out["fvalue"]        = float(base.fvalue)
    out["f_pvalue"]      = float(base.f_pvalue)

    out["per_cov_type"] = {}
    for ct in COV_TYPES:
        if ct == "nonrobust":
            r = base
        else:
            r = sm.OLS(y, x).fit(cov_type=ct)
        out["per_cov_type"][ct] = {
            "std_err":  list(map(float, r.bse)),
            "t_values": list(map(float, r.tvalues)),
            "p_values": list(map(float, r.pvalues)),
            "conf_int_95": [list(map(float, row)) for row in r.conf_int(alpha=0.05)],
        }

    pred = base.get_prediction(x_pred)
    out["predict_point"]      = list(map(float, pred.predicted_mean))
    pi = pred.summary_frame(alpha=0.05)
    out["predict_interval_95"] = [
        [float(pi["mean"][i]), float(pi["obs_ci_lower"][i]), float(pi["obs_ci_upper"][i])]
        for i in range(len(pi))
    ]

    target = OUT_DIR / f"{name}.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")


def longley():
    df = smds.longley.load_pandas().data
    y = df["TOTEMP"].to_numpy()
    x = df.drop(columns=["TOTEMP"]).to_numpy()
    x_pred = x[:3]  # arbitrary held-out slice
    fit_and_dump("longley", y, x, x_pred, intercept=True)


def main():
    longley()
    # Task 16 will add: mtcars(), synthetic(), heteroskedastic()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the Longley JSON**

Run from project root:

```bash
python3 tests/golden/generate.py
```

Expected: `wrote tests/golden/longley.json` (file appears, ~5 KB).

If statsmodels isn't installed, the implementer should `pip install statsmodels` (or use a venv); document this in the script's docstring (already noted).

- [ ] **Step 3: Write the Rust loader and Longley assertion**

Create `tests/ols_golden.rs`:

```rust
use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct PerCov {
    std_err:    Vec<f64>,
    t_values:   Vec<f64>,
    p_values:   Vec<f64>,
    conf_int_95: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Golden {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    intercept: bool,
    x_predict: Vec<Vec<f64>>,

    coef:          Vec<f64>,
    residuals:     Vec<f64>,
    fitted:        Vec<f64>,
    rss:           f64,
    sigma:         f64,
    r_squared:     f64,
    adj_r_squared: f64,
    fvalue:        f64,
    f_pvalue:      f64,

    per_cov_type: std::collections::BTreeMap<String, PerCov>,

    predict_point:        Vec<f64>,
    predict_interval_95:  Vec<Vec<f64>>,
}

fn load(name: &str) -> Golden {
    let path: PathBuf = ["tests", "golden", &format!("{name}.json")].iter().collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e|
        panic!("failed to read {path:?}: {e}; did you run tests/golden/generate.py?"));
    serde_json::from_slice(&bytes).expect("invalid golden JSON")
}

fn col_from(v: &[f64]) -> Col<f64> { Col::from_fn(v.len(), |i| v[i]) }
fn mat_from(rows: &[Vec<f64>]) -> Mat<f64> {
    let n = rows.len();
    let p = if n == 0 { 0 } else { rows[0].len() };
    Mat::from_fn(n, p, |i, j| rows[i][j])
}

fn cov_type(name: &str) -> CovType {
    match name {
        "nonrobust" => CovType::NonRobust,
        "HC0" => CovType::HC0,
        "HC1" => CovType::HC1,
        "HC2" => CovType::HC2,
        "HC3" => CovType::HC3,
        other => panic!("unknown cov_type {other}"),
    }
}

fn assert_dataset(name: &str) {
    let g = load(name);
    let y = col_from(&g.y);
    let x = mat_from(&g.x);
    let model = Ols::new(y.as_ref(), x.as_ref());
    let res = (if g.intercept { model } else { model.without_intercept() })
        .fit().expect("fit failed");

    // Coefficients, residuals, fitted, rss, sigma, R², adj R², F, F-pvalue.
    let beta = res.coef();
    for i in 0..g.coef.len() {
        assert_relative_eq!(beta[i], g.coef[i], epsilon = 1e-10, max_relative = 1e-10);
    }
    let resid = res.residuals();
    for i in 0..g.residuals.len() {
        assert_relative_eq!(resid[i], g.residuals[i], epsilon = 1e-10, max_relative = 1e-10);
    }
    let fit = res.fitted_values();
    for i in 0..g.fitted.len() {
        assert_relative_eq!(fit[i], g.fitted[i], epsilon = 1e-10, max_relative = 1e-10);
    }
    let rss: f64 = (0..g.residuals.len()).map(|i| resid[i].powi(2)).sum();
    assert_relative_eq!(rss, g.rss, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(res.sigma(), g.sigma, max_relative = 1e-8);
    assert_relative_eq!(res.r_squared(), g.r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.adj_r_squared(), g.adj_r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.f_statistic(), g.fvalue, max_relative = 1e-8);
    assert_relative_eq!(res.f_pvalue(), g.f_pvalue, max_relative = 1e-6);

    // Per-CovType inference.
    for (ct_name, ref_) in &g.per_cov_type {
        let ct = cov_type(ct_name);
        let inf = res.inference(ct);
        for i in 0..ref_.std_err.len() {
            assert_relative_eq!(inf.std_err[i], ref_.std_err[i],
                max_relative = 1e-8);
            assert_relative_eq!(inf.t_values[i], ref_.t_values[i],
                max_relative = 1e-8);
            assert_relative_eq!(inf.p_values[i], ref_.p_values[i],
                max_relative = 1e-6);
        }
        let ci = res.conf_int_with(ct, 0.05).unwrap();
        for i in 0..ref_.conf_int_95.len() {
            assert_relative_eq!(ci[(i, 0)], ref_.conf_int_95[i][0],
                max_relative = 1e-7);
            assert_relative_eq!(ci[(i, 1)], ref_.conf_int_95[i][1],
                max_relative = 1e-7);
        }
    }

    // Predict.
    let x_new = mat_from(&g.x_predict);
    let yhat = res.predict(x_new.as_ref()).unwrap();
    for i in 0..g.predict_point.len() {
        assert_relative_eq!(yhat[i], g.predict_point[i], max_relative = 1e-9);
    }
    let band = res.predict_interval(x_new.as_ref(), 0.05).unwrap();
    for i in 0..g.predict_interval_95.len() {
        assert_relative_eq!(band[(i, 0)], g.predict_interval_95[i][0],
            max_relative = 1e-9);
        assert_relative_eq!(band[(i, 1)], g.predict_interval_95[i][1],
            max_relative = 1e-7);
        assert_relative_eq!(band[(i, 2)], g.predict_interval_95[i][2],
            max_relative = 1e-7);
    }
}

#[test] fn longley() { assert_dataset("longley"); }
```

- [ ] **Step 4: Run the golden test**

Run: `cargo test --test ols_golden`
Expected: 1 passed.

If a tolerance fails, investigate the underlying computation rather than loosening tolerances. Common culprits: HC1 scaling factor, F-statistic formula for no-intercept case, mishandled column permutation in `(X̃'X̃)⁻¹`.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/generate.py tests/golden/longley.json tests/ols_golden.rs
git commit -m "Add statsmodels golden-test scaffold + Longley fixture"
```

---

## Task 16: Add mtcars, synthetic, heteroskedastic, rank_deficient datasets

**Files:**
- Modify: `tests/golden/generate.py`
- Create: `tests/golden/mtcars.json`
- Create: `tests/golden/synthetic.json`
- Create: `tests/golden/heteroskedastic.json`
- Create: `tests/golden/rank_deficient.json`
- Modify: `tests/ols_golden.rs`
- Modify: `tests/negative.rs`

- [ ] **Step 1: Extend the Python script with the new datasets**

Append to `tests/golden/generate.py` before `main`:

```python
def mtcars():
    df = smds.get_rdataset("mtcars", "datasets").data
    y = df["mpg"].to_numpy()
    x = df[["cyl", "hp", "wt"]].to_numpy().astype(float)
    x_pred = x[:5]
    fit_and_dump("mtcars", y, x, x_pred, intercept=True)


def synthetic():
    rng = np.random.default_rng(20260509)
    n, p = 200, 4
    x = rng.standard_normal((n, p))
    beta = np.array([0.5, -1.2, 2.1, 0.3])
    y = 1.0 + x @ beta + rng.standard_normal(n) * 0.5
    x_pred = rng.standard_normal((10, p))
    fit_and_dump("synthetic", y, x, x_pred, intercept=True)


def heteroskedastic():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.uniform(0.5, 5.0, size=(n, 1))
    eps = rng.standard_normal(n) * x[:, 0]   # variance ∝ x²
    y = 2.0 + 3.0 * x[:, 0] + eps
    x_pred = np.array([[1.0], [2.5], [4.0]])
    fit_and_dump("heteroskedastic", y, x, x_pred, intercept=True)


def rank_deficient_input():
    """Only saves the input — there is no reference fit because statsmodels
    will silently use a pseudoinverse and we want the rust side to error."""
    rng = np.random.default_rng(7)
    n = 25
    x_base = rng.standard_normal((n, 2))
    x = np.column_stack([x_base[:, 0], x_base[:, 1], x_base[:, 0]])  # col 2 == col 0
    y = rng.standard_normal(n)
    out = {
        "y": list(map(float, y)),
        "x": x.tolist(),
    }
    target = OUT_DIR / "rank_deficient.json"
    with target.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {target}")
```

Update `main()`:

```python
def main():
    longley()
    mtcars()
    synthetic()
    heteroskedastic()
    rank_deficient_input()
```

- [ ] **Step 2: Regenerate goldens**

Run: `python3 tests/golden/generate.py`
Expected: prints 5 `wrote tests/golden/...json` lines.

- [ ] **Step 3: Add Rust tests for the new datasets**

Append to `tests/ols_golden.rs`:

```rust
#[test] fn mtcars()          { assert_dataset("mtcars"); }
#[test] fn synthetic()       { assert_dataset("synthetic"); }
#[test] fn heteroskedastic() { assert_dataset("heteroskedastic"); }
```

Append to `tests/negative.rs`:

```rust
#[test]
fn rank_deficient_golden_dataset_errors() {
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Rd { y: Vec<f64>, x: Vec<Vec<f64>> }

    let path: PathBuf = ["tests", "golden", "rank_deficient.json"].iter().collect();
    let bytes = std::fs::read(path).unwrap();
    let rd: Rd = serde_json::from_slice(&bytes).unwrap();

    let y = Col::from_fn(rd.y.len(), |i| rd.y[i]);
    let n = rd.x.len();
    let p = rd.x[0].len();
    let x = Mat::from_fn(n, p, |i, j| rd.x[i][j]);

    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    match err {
        OlsError::RankDeficient { .. } => {}
        other => panic!("expected RankDeficient, got {:?}", other),
    }
}
```

The existing `tests/negative.rs` already imports `faer::{Col, Mat}` and `Ols`/`OlsError`; if not, add the imports.

- [ ] **Step 4: Run all golden + negative tests**

Run: `cargo test --test ols_golden --test negative`
Expected: 4 golden tests pass, all negative tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/ tests/ols_golden.rs tests/negative.rs
git commit -m "Add mtcars/synthetic/heteroskedastic/rank-deficient goldens"
```

---

## Task 17: Property tests

Cheap invariants that catch silent regressions the goldens might miss.

**Files:**
- Create: `tests/properties.rs`

- [ ] **Step 1: Write the property tests**

Create `tests/properties.rs`:

```rust
use approx::assert_abs_diff_eq;
use faer::{Col, Mat};
use rust_stats::Ols;

/// Residuals must be orthogonal to every column of X̃ (including the intercept
/// column when present).
#[test]
fn residuals_orthogonal_to_design_with_intercept() {
    let n = 40;
    let x: Mat<f64> = Mat::from_fn(n, 3, |i, j| ((i + 2 * j) as f64).sin() + (j as f64));
    let y: Col<f64> = Col::from_fn(n, |i| (i as f64).cos() + 0.5 * (i as f64));
    let res = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap();
    let e = res.residuals();
    let sum_e: f64 = (0..n).map(|i| e[i]).sum();
    assert_abs_diff_eq!(sum_e, 0.0, epsilon = 1e-10); // intercept ⇒ residuals sum to 0
    for j in 0..x.ncols() {
        let dot: f64 = (0..n).map(|i| e[i] * x[(i, j)]).sum();
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn residuals_orthogonal_to_design_without_intercept() {
    let n = 30;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| (i as f64) + (j as f64) * 0.7);
    let y: Col<f64> = Col::from_fn(n, |i| 0.3 * (i as f64) + 0.05 * ((i as f64).sin()));
    let res = Ols::new(y.as_ref(), x.as_ref()).without_intercept().fit().unwrap();
    let e = res.residuals();
    for j in 0..x.ncols() {
        let dot: f64 = (0..n).map(|i| e[i] * x[(i, j)]).sum();
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn permuting_columns_of_x_preserves_predictions_and_r_squared() {
    let n = 40;
    let x: Mat<f64> = Mat::from_fn(n, 3, |i, j| (i as f64) * (j as f64 + 1.0).sin());
    let y: Col<f64> = Col::from_fn(n, |i| (i as f64).cos());
    let r1 = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap();

    // Swap columns 0 and 2.
    let x2: Mat<f64> = Mat::from_fn(n, 3, |i, j| match j {
        0 => x[(i, 2)],
        2 => x[(i, 0)],
        _ => x[(i, j)],
    });
    let r2 = Ols::new(y.as_ref(), x2.as_ref()).fit().unwrap();

    assert_abs_diff_eq!(r1.r_squared(), r2.r_squared(), epsilon = 1e-12);
    let f1 = r1.fitted_values();
    let f2 = r2.fitted_values();
    for i in 0..n {
        assert_abs_diff_eq!(f1[i], f2[i], epsilon = 1e-10);
    }
}
```

- [ ] **Step 2: Run them**

Run: `cargo test --test properties`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/properties.rs
git commit -m "Add OLS property tests (orthogonality, permutation invariance)"
```

---

## Task 18: Longley example

Demonstrates the public API end-to-end and gives users a copy-paste starting point.

**Files:**
- Create: `examples/longley.rs`

- [ ] **Step 1: Write the example**

Create `examples/longley.rs`:

```rust
//! Run with: `cargo run --example longley`
//!
//! Uses a hard-coded copy of the classic Longley macroeconomic dataset
//! and prints the OLS fit summary.

use faer::{Col, Mat};
use rust_stats::Ols;

const Y: [f64; 16] = [
    60323.0, 61122.0, 60171.0, 61187.0, 63221.0, 63639.0, 64989.0, 63761.0,
    66019.0, 67857.0, 68169.0, 66513.0, 68655.0, 69564.0, 69331.0, 70551.0,
];

// Columns: GNP_DEFLATOR, GNP, UNEMPLOYED, ARMED_FORCES, POPULATION, YEAR
const X: [[f64; 6]; 16] = [
    [ 83.0, 234289.0, 2356.0, 1590.0, 107608.0, 1947.0],
    [ 88.5, 259426.0, 2325.0, 1456.0, 108632.0, 1948.0],
    [ 88.2, 258054.0, 3682.0, 1616.0, 109773.0, 1949.0],
    [ 89.5, 284599.0, 3351.0, 1650.0, 110929.0, 1950.0],
    [ 96.2, 328975.0, 2099.0, 3099.0, 112075.0, 1951.0],
    [ 98.1, 346999.0, 1932.0, 3594.0, 113270.0, 1952.0],
    [ 99.0, 365385.0, 1870.0, 3547.0, 115094.0, 1953.0],
    [100.0, 363112.0, 3578.0, 3350.0, 116219.0, 1954.0],
    [101.2, 397469.0, 2904.0, 3048.0, 117388.0, 1955.0],
    [104.6, 419180.0, 2822.0, 2857.0, 118734.0, 1956.0],
    [108.4, 442769.0, 2936.0, 2798.0, 120445.0, 1957.0],
    [110.8, 444546.0, 4681.0, 2637.0, 121950.0, 1958.0],
    [112.6, 482704.0, 3813.0, 2552.0, 123366.0, 1959.0],
    [114.2, 502601.0, 3931.0, 2514.0, 125368.0, 1960.0],
    [115.7, 518173.0, 4806.0, 2572.0, 127852.0, 1961.0],
    [116.9, 554894.0, 4007.0, 2827.0, 130081.0, 1962.0],
];

fn main() {
    let y: Col<f64> = Col::from_fn(Y.len(), |i| Y[i]);
    let x: Mat<f64> = Mat::from_fn(X.len(), 6, |i, j| X[i][j]);
    let res = Ols::new(y.as_ref(), x.as_ref())
        .fit()
        .expect("Longley fit")
        .with_names(vec![
            "const".into(), "deflator".into(), "gnp".into(),
            "unemp".into(), "armed".into(), "pop".into(), "year".into(),
        ]);
    println!("{res}");
}
```

- [ ] **Step 2: Run it**

Run: `cargo run --example longley`
Expected: prints a summary table that looks like the statsmodels output (header, coefficient table, footer line of `=`).

- [ ] **Step 3: Make sure all tests still pass**

Run: `cargo test`
Expected: every test passes (smoke, distributions, builder, design, fit_basic, goodness, inference_classical, inference_helper, predict, names, summary, robust, ols_golden, properties, negative).

- [ ] **Step 4: Commit**

```bash
git add examples/longley.rs
git commit -m "Add Longley OLS example"
```

---

## Wrap-up

After Task 18, the v1 surface from the spec is implemented and validated against statsmodels reference values. Run a final full test pass and confirm there are no `cargo` warnings:

```bash
cargo test && cargo build --all-targets 2>&1 | grep -E 'warning|error' || echo "clean build"
```

Then verify the spec's open-questions list (§9 of the design) is still all v2-deferred — no scope crept in.

---

## Self-review (plan author's pass)

**Spec coverage check:**

| Spec section | Plan coverage |
| --- | --- |
| §1 Goals | Tasks 1–18 collectively |
| §2 Public API: `Ols` | Task 4 (builder), Task 5 (validation), Tasks 7–8 (fit numerics + accessors) |
| §2 Public API: `OlsResults` point estimates | Task 7 (coef), Task 8 (fitted/residuals/sigma/df) |
| §2 Public API: goodness-of-fit | Task 8 |
| §2 Public API: CovType + Inference + cov/inference/conf_int_with/summary_with | Task 9 (NonRobust path), Task 10 (HC0–HC3), Task 11 (helpers) |
| §2 Public API: cov_hc0..hc3 shortcuts | Task 10 |
| §2 Public API: predict / predict_interval | Task 12 |
| §2 Public API: with_names / names | Task 13 |
| §2 Public API: summary / summary_with / Display | Task 14 |
| §3 Numerical algorithm | Task 7 (fit), Task 9 (classical cov), Task 10 (sandwich) |
| §4 Error model | Task 2 (enum), Task 5 (validation paths), Task 7 (RankDeficient), Task 12 (NewXShapeMismatch + InvalidAlpha) |
| §5 Summary format | Task 14 |
| §6 Module layout | Task 1 + later tasks each create the listed file |
| §7 Dependencies | Task 1 |
| §8 Testing strategy: golden | Tasks 15–16 |
| §8 Testing strategy: properties | Task 17 |
| §8 Testing strategy: negative | Task 2 + appended tests in Tasks 5, 7, 12, 16 |

**Placeholder scan:** No "TBD" / "TODO" / "implement later" in any step. Two narrative notes say "Task N" cross-references for context — those are pointers, not placeholder code.

**Type/method consistency:**
- `Ols` lifetime parameter `'a`, fields `y`/`x`/`intercept`, methods `new`/`without_intercept`/`has_intercept`/`fit` — used identically in Tasks 4, 5, 7.
- `OlsResults` field names (`coef`, `fitted`, `residuals`, `x_design`, `r_factor`, `perm`, `leverage`, `n`, `p`, `rank`, `sigma2`, `rss`, `tss`, `has_intercept`, `names`, `cov_unscaled`, `std_err_classical`) consistent across Tasks 7, 8, 9, 10, 11, 12, 13, 14.
- `CovType` variants `NonRobust`, `HC0`, `HC1`, `HC2`, `HC3` consistent.
- `Inference` field names `std_err`, `t_values`, `p_values` consistent.
- `OlsError` variant payload names (`y`/`x`, `n`/`p`, `rank`/`p`, `got`/`expected`, `f64`) consistent across Tasks 2 and the test files.

**One ambiguity to flag in implementation:** the `adj_r_squared` formula in Task 8 has a no-intercept correction whose exact statsmodels semantics depend on the dataset; the golden tests in Task 15 will catch any divergence. If the formula fails the Longley golden, drop the correction term and use the canonical `1 - (1 - R²) · (n-1)/(n-p)`.
