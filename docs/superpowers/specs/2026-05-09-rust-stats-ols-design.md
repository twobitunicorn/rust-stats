# rust-stats — OLS v1 Design

**Date:** 2026-05-09
**Status:** Draft (awaiting user review)
**Scope:** First feature of `rust-stats`, a pure-Rust statistical modeling library inspired by Python's `statsmodels`. v1 ships Ordinary Least Squares regression with full classical inference, heteroskedasticity-robust covariance, prediction (point + interval), and a statsmodels-style text summary.

## 1. Goals and non-goals

### Goals

- Pure-Rust OLS with **numerically sound** estimation (QR-based, not normal equations).
- Full classical inference: standard errors, t-statistics, p-values, confidence intervals, R², adjusted R², F-statistic and its p-value, residuals, fitted values, σ̂.
- Heteroskedasticity-robust covariance: HC0, HC1, HC2, HC3.
- Point prediction and prediction intervals on new X.
- statsmodels-style monospace `summary()` text output.
- Validated against statsmodels reference outputs to tight tolerance.

### Non-goals (v1)

- WLS, GLS, GLM, mixed effects, time series, or any non-OLS estimator.
- Polars/`ndarray`/DataFrame integration. Inputs are `faer` types only.
- R-style formula API (`"y ~ x1 + x2"`).
- Categorical encoding, interaction terms, automatic transformations.
- Influence/diagnostic measures (leverage, Cook's distance, DFFITS, DFBETAS).
- Diagnostic tests (Durbin–Watson, Jarque–Bera, Omnibus, Breusch–Pagan).
- ANOVA tables, Wald tests, contrasts.
- Pseudoinverse / rank-deficient fallback. Rank-deficient input is an error.
- `f32` or generic numeric types. `f64` only.
- Trait abstractions (`Model`, `RegressionResults`, etc.). Concrete types only; we'll abstract once a second model arrives and tells us what the abstraction needs.

## 2. Public API

### `Ols` — model builder

```rust
pub struct Ols<'a> {
    y: ColRef<'a, f64>,
    x: MatRef<'a, f64>,
    intercept: bool,
}

impl<'a> Ols<'a> {
    /// Construct an OLS model. Intercept column added automatically by `fit`.
    pub fn new(y: ColRef<'a, f64>, x: MatRef<'a, f64>) -> Self;

    /// Disable the auto-prepended intercept column.
    pub fn without_intercept(self) -> Self;

    /// Fit the model. Returns owned results or a typed error.
    pub fn fit(&self) -> Result<OlsResults, OlsError>;
}
```

Inputs are **borrowed** (`MatRef`/`ColRef`) per faer convention. The caller retains ownership of `X` and `y`. The fitted results object owns the (possibly intercept-augmented) design matrix `X̃` and the QR factor `R` — enough for all downstream computation without re-borrowing from the caller.

### `OlsResults` — owned fit result

```rust
pub struct OlsResults {
    // eagerly computed
    coef: Col<f64>,             // β̂  (length p)
    fitted: Col<f64>,           // ŷ
    residuals: Col<f64>,        // y − ŷ
    x_design: Mat<f64>,         // X̃: original X with intercept column prepended if has_intercept
    r_factor: Mat<f64>,         // R from pivoted QR of X̃  (p × p, upper triangular)
    perm: Vec<usize>,           // column permutation from pivoted QR
    leverage: Col<f64>,         // h_ii = ‖Q_i,*‖² (hat-matrix diagonals; for HC2/HC3)
    n: usize,
    p: usize,                   // includes intercept
    rank: usize,
    sigma2: f64,                // RSS / (n − p)
    rss: f64,
    tss: f64,
    has_intercept: bool,
    names: Option<Vec<String>>,

    // lazy caches
    cov_unscaled: OnceCell<Mat<f64>>,   // (X̃'X̃)⁻¹
    std_err_classical: OnceCell<Col<f64>>,
}
```

`x_design` owns the (possibly intercept-prepended) design matrix. We keep it because the robust covariance estimators need to form `X̃' diag(ω) X̃` against the actual rows. Storage is `O(n·p)` — equivalent to keeping `Q` from a thin QR. We do not retain a borrow of the caller's `X`.

#### Point estimates

```rust
pub fn coef(&self) -> ColRef<'_, f64>;
pub fn fitted_values(&self) -> ColRef<'_, f64>;
pub fn residuals(&self) -> ColRef<'_, f64>;
```

#### Goodness of fit

```rust
pub fn r_squared(&self)     -> f64;
pub fn adj_r_squared(&self) -> f64;
pub fn f_statistic(&self)   -> f64;   // overall F vs intercept-only model
pub fn f_pvalue(&self)      -> f64;
pub fn sigma(&self)         -> f64;   // √σ̂²
pub fn n_obs(&self)         -> usize;
pub fn df_resid(&self)      -> usize; // n − p
pub fn df_model(&self)      -> usize; // p − 1 if intercept else p
```

`r_squared` is the centered version when an intercept is present and the uncentered version otherwise (matches statsmodels and R).

#### Inference (covariance-parameterized)

```rust
pub enum CovType { NonRobust, HC0, HC1, HC2, HC3 }

pub struct Inference {
    pub std_err:  Col<f64>,
    pub t_values: Col<f64>,
    pub p_values: Col<f64>,
}

impl OlsResults {
    pub fn cov(&self, cov: CovType)            -> Mat<f64>;        // p × p
    pub fn inference(&self, cov: CovType)      -> Inference;
    pub fn conf_int_with(&self, cov: CovType, alpha: f64) -> Mat<f64>;  // p × 2 [lower, upper]
    pub fn summary_with(&self, cov: CovType)   -> String;
}
```

**Alpha convention** (matches statsmodels): `alpha` is the *significance level*, so `alpha = 0.05` produces a 95% interval using `t_{1 − α/2, n−p}`. `alpha` must be in `(0, 1)` exclusive; otherwise `OlsError::InvalidAlpha`. The same convention applies to `predict_interval`.

```rust
```

Convenience wrappers over `CovType::NonRobust`:

```rust
pub fn std_err(&self)  -> ColRef<'_, f64>;
pub fn t_values(&self) -> Col<f64>;
pub fn p_values(&self) -> Col<f64>;
pub fn conf_int(&self, alpha: f64) -> Mat<f64>;
pub fn summary(&self) -> String;
```

Direct access to robust covariance matrices:

```rust
pub fn cov_hc0(&self) -> Mat<f64>;
pub fn cov_hc1(&self) -> Mat<f64>;
pub fn cov_hc2(&self) -> Mat<f64>;
pub fn cov_hc3(&self) -> Mat<f64>;
```

#### Prediction

```rust
pub fn predict(&self, x_new: MatRef<'_, f64>) -> Result<Col<f64>, OlsError>;

/// Returns an n_new × 3 matrix with columns [fit, lower, upper] using a t-based
/// prediction interval (not a confidence interval on the mean).
pub fn predict_interval(
    &self,
    x_new: MatRef<'_, f64>,
    alpha: f64,
) -> Result<Mat<f64>, OlsError>;
```

`x_new.ncols()` must equal the original `X.ncols()` (i.e., excluding the intercept). The intercept column is prepended internally when `has_intercept`.

#### Naming

```rust
pub fn with_names(self, names: Vec<String>) -> Self;
pub fn names(&self) -> Option<&[String]>;
```

`names.len()` must equal `p` (intercept name first if present); otherwise `with_names` panics. Default summary labels are `const`, `x1`, `x2`, …

#### Display

`impl Display for OlsResults` calls `summary()`. `Debug` derived (terse).

## 3. Numerical algorithm

### Fit

1. **Validate inputs.**
   - `y.nrows() == x.nrows()` else `DimensionMismatch`.
   - All entries finite (no NaN/Inf) else `NonFinite`.
   - `n > p` (where `p = x.ncols() + has_intercept as usize`) else `InsufficientObservations`.
2. **Build design matrix `X̃`.** Allocate a fresh owned `Mat<f64>`. If `has_intercept`, shape is `(n, x.ncols + 1)` with column 0 = ones and remaining columns copied from `x`. Otherwise shape is `(n, x.ncols)` and contents are copied from `x`. We always own `X̃` so robust-covariance computations can re-scale rows without touching the caller's data.
3. **Pivoted QR.** `X̃ · P = Q · R` via faer's column-pivoted QR. Detect rank with tolerance `tol = max(n, p) · ε · |R[0,0]|`. If `rank < p`, return `RankDeficient`.
4. **Solve.** `R · β̂_p = Q'y` via back-substitution; un-permute to get `β̂`.
5. **Residuals & σ̂².** `fitted = X̃ · β̂`; `residuals = y − fitted`; `rss = ‖residuals‖²`; `sigma2 = rss / (n − p)`.
6. **TSS for R².** `tss = Σ(y_i − ȳ)²` if `has_intercept` else `Σ y_i²`.
7. **Hat-diagonal cache.** Compute `leverage[i] = ‖Q_i,*‖²` (the hat-matrix diagonal `h_ii`, used by HC2/HC3). Cost is `O(np)` and avoids recomputing it on each robust-covariance call. `Q` itself is not retained after this step.

### Classical covariance

`Cov(β̂) = σ̂² (X̃'X̃)⁻¹ = σ̂² (R'R)⁻¹` (after un-permutation). Computed lazily: form `(R'R)⁻¹` once via two triangular solves on `R` against the identity; cache. `std_err[i] = sqrt(cov_unscaled[i,i] · σ̂²)`.

### Robust covariance (HC0–HC3)

Let `ω_i` be the per-observation weight:

| Variant | `ω_i`              |
| ------- | ------------------ |
| HC0     | `e_i²`             |
| HC1     | `e_i² · n/(n−p)`   |
| HC2     | `e_i² / (1 − h_ii)`|
| HC3     | `e_i² / (1 − h_ii)²` |

where `e_i` is residual `i` and `h_ii = leverage[i]`.

Sandwich:
`Cov_HC = (X̃'X̃)⁻¹ · (X̃' diag(ω) X̃) · (X̃'X̃)⁻¹`.

Implementation: form a working copy of `x_design` with rows scaled by `√ω_i`, compute the meat `M = X̃'_scaled · X̃_scaled`, then sandwich `Cov_HC = (X̃'X̃)⁻¹ · M · (X̃'X̃)⁻¹` by applying `(X̃'X̃)⁻¹ = (R'R)⁻¹` to each side via triangular solves on `R`. The explicit `M` is `p × p` (cheap); the explicit inverse is never formed for the final product.

### Inference distributions

- t-statistic: `t_i = β̂_i / SE_i`. p-value: two-sided `2 · (1 − T_cdf(|t_i|; n−p))`.
- Confidence interval: `β̂_i ± t_{1−α/2, n−p} · SE_i`.
- Overall F: classic `F = (TSS − RSS)/(p − 1) / (RSS/(n − p))` when intercept; degrees `(p − 1, n − p)`. p-value via F-distribution survival.

Distributions provided by `statrs` (StudentsT, FisherSnedecor).

### Predict

- `predict`: prepend ones to `x_new` if `has_intercept`, multiply by `β̂`. Validate `x_new.ncols() == p − has_intercept as usize` else `NewXShapeMismatch`.
- `predict_interval`: per row,
  `ŷ_new ± t_{1−α/2, n−p} · sqrt(σ̂² · (1 + x_new' (X̃'X̃)⁻¹ x_new))`.
  The quadratic form is computed via `R'⁻¹ · x_new` (one triangular solve) then squared norm.

## 4. Error model

```rust
#[derive(Debug, thiserror::Error)]
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

Programmer errors (e.g., `with_names` length mismatch) panic; data errors (rank, dimensions, non-finite) return `Result`.

## 5. Summary format

statsmodels-style monospace text, three blocks, plain ASCII, no color or unicode box-drawing:

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.997
Model:                            OLS   Adj. R-squared:                  0.996
Method:                 Least Squares   F-statistic:                     330.3
No. Observations:                  16   Prob (F-statistic):           4.98e-10
Df Residuals:                      10
Df Model:                           5
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          ...        ...        ...        ...        ...        ...
x1             ...        ...        ...        ...        ...        ...
==============================================================================
```

Number formatting:
- Coefficients/SE/t/CIs: `%10.4f` (or `%10.4e` when `|x| < 1e-3` or `|x| ≥ 1e6`).
- p-values: `%7.3f` clamped at `0.000`; switches to `%7.3e` below `1e-3`.
- The "Dep. Variable" label is `"y"` unless the user supplied a dependent-variable name (out of v1 scope; default `"y"`).

Implementation lives in `regression/summary.rs`. No external formatting deps.

## 6. Module layout

```
rust-stats/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs               // re-exports: Ols, OlsResults, CovType, Inference, OlsError
│   ├── error.rs
│   ├── distributions.rs     // thin wrappers around statrs (t, F)
│   └── regression/
│       ├── mod.rs
│       ├── ols.rs           // Ols struct + fit
│       ├── results.rs       // OlsResults struct + accessors
│       ├── robust.rs        // HC0–HC3
│       ├── predict.rs       // predict + predict_interval
│       └── summary.rs       // summary string formatter
├── tests/
│   ├── golden/
│   │   ├── generate.py      // committed Python script
│   │   ├── longley.json
│   │   ├── mtcars.json
│   │   ├── synthetic.json
│   │   ├── heteroskedastic.json
│   │   └── rank_deficient.json
│   ├── ols_golden.rs
│   └── properties.rs
└── examples/
    └── longley.rs
```

## 7. Dependencies

Runtime:

- `faer` (latest) — linear algebra, QR, triangular solves.
- `statrs` (latest) — Student's t and F distributions.
- `thiserror` — error enum derive.
- `once_cell` — `OnceCell` for lazy caches.

Dev:

- `serde` + `serde_json` — load golden JSON.
- `approx` — `assert_relative_eq!` / `assert_abs_diff_eq!`.

## 8. Testing strategy

### Golden-value tests vs statsmodels

`tests/golden/generate.py` is a committed Python script that runs statsmodels OLS on a fixed set of datasets and dumps reference JSON. The script is idempotent and pinned to specific statsmodels/numpy/scipy versions documented at its top. Regenerating goldens is a manual step, not run during `cargo test`.

Datasets:

1. **Longley** — classic well-conditioned macroeconomic series.
2. **mtcars** — small mixed-magnitude regressors.
3. **synthetic_well_conditioned** — `n=200`, `p=4`, β known, Gaussian noise, fixed seed.
4. **heteroskedastic** — synthetic with `Var(ε_i) ∝ x_i²` so HC0–HC3 differ visibly from classical.
5. **rank_deficient** — `X` with a duplicated column; tests that fit returns `OlsError::RankDeficient`.

Reference fields per dataset: `coef`, `residuals`, `fitted`, `rss`, `sigma`, `r_squared`, `adj_r_squared`, `fvalue`, `f_pvalue`, then for each `cov_type ∈ {nonrobust, HC0, HC1, HC2, HC3}`: `std_err`, `t_values`, `p_values`, `conf_int_95`. Plus `predict` and `predict_interval_95` on a held-out X.

### Tolerances

- `coef`, `residuals`, `fitted`: `abs_tol = 1e-10`, `rel_tol = 1e-10`.
- σ, R², adj-R², F, SE: `rel_tol = 1e-8`.
- t-values: `rel_tol = 1e-8`.
- p-values: `rel_tol = 1e-6` (distribution tail evaluation differs slightly between scipy and statrs).
- CI bounds, prediction intervals: `rel_tol = 1e-7`.

### Property tests

`tests/properties.rs`:

- Residuals are orthogonal to every column of `X̃` (`max |X̃' e| < 1e-10`).
- Recovery: known β plus small Gaussian noise, fitted β̂ within 3 SE of true.
- Permutation invariance: permuting columns of X (and matching the names) leaves predictions and R² unchanged.
- Without-intercept path: residuals orthogonal to `X` (no centering).

### Negative tests

- Mismatched `y` / `X` rows → `DimensionMismatch`.
- `n ≤ p` → `InsufficientObservations`.
- Rank-deficient input → `RankDeficient`.
- NaN/Inf in inputs → `NonFinite`.
- `predict` with wrong column count → `NewXShapeMismatch`.
- `conf_int(0.0)` and `conf_int(1.5)` → `InvalidAlpha`.

## 9. Open questions deferred to v2

- WLS / GLS.
- Influence and diagnostic measures.
- Diagnostic tests (DW, JB, etc.).
- Cluster-robust and HAC covariance.
- Trait abstraction for cross-model inference.
- Polars/`ndarray` interop.
- Formula API.
- `f32` / generic numeric backend.
- Pseudoinverse fallback for rank-deficient designs.
