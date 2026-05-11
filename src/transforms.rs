//! Per-element transforms over `&[f64]`.
//!
//! Standardisation / scaling helpers that consume a slice and produce a
//! `Vec<f64>` of the same length. All aggregates (mean, std, min, max)
//! are computed over the finite entries; `NaN` inputs propagate to the
//! corresponding output positions, but do not contaminate the
//! aggregates themselves.
//!
//! `center`, `z_score`, and `min_max_scale` dispatch through `pulp` for
//! runtime SIMD acceleration (SSE2 / AVX2 / AVX-512 on x86_64, NEON on
//! aarch64; scalar fallback elsewhere). `box_cox` is scalar — its
//! transcendental kernel (`ln` / `powf`) isn't in pulp's f64 vocabulary.
//!
//! Edge cases:
//!
//! - Empty input → empty output (no error).
//! - Constant input (or all-NaN aggregate) → an all-zero output for
//!   `z_score` and `min_max_scale`. `NaN` positions still propagate.
//! - `box_cox` is the only transform here that can fail: it requires
//!   strictly positive finite values and returns
//!   [`BoxCoxError::NonPositive`] otherwise.

use crate::error::BoxCoxError;

/// Subtract the (finite-entry) mean from every value.
///
/// `NaN` inputs propagate to the same positions in the output. An
/// all-NaN input is treated as having a mean of zero.
pub fn center(y: &[f64]) -> Vec<f64> {
    pulp_impl::center(y)
}

/// Z-score normalisation: `(x - mean) / std` with sample standard
/// deviation (ddof = 1).
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs (and inputs with fewer than two finite
/// entries) produce an all-zero output at finite positions.
pub fn z_score(y: &[f64]) -> Vec<f64> {
    pulp_impl::z_score(y)
}

/// Min-max rescaling into `[0, 1]`: `(x - min) / (max - min)`.
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs produce an all-zero output at finite
/// positions.
pub fn min_max_scale(y: &[f64]) -> Vec<f64> {
    pulp_impl::min_max_scale(y)
}

/// Inverse Box-Cox: given `y = box_cox(x, λ)`, recover `x`.
///
/// ```text
/// x = (1 + λ·y)^(1/λ)   when λ ≠ 0
/// x = exp(y)            when λ = 0
/// ```
///
/// `NaN` entries propagate. Returns [`BoxCoxError::NonInvertible`] when
/// any finite `y` violates `1 + λ·y > 0` (the transform clamps to a
/// half-line that excludes those values).
pub fn inv_box_cox(y: &[f64], lmbda: f64) -> Result<Vec<f64>, BoxCoxError> {
    if lmbda == 0.0 {
        return Ok(y.iter().map(|&v| v.exp()).collect());
    }
    let inv = 1.0 / lmbda;
    let mut out = Vec::with_capacity(y.len());
    for &v in y {
        if v.is_nan() {
            out.push(f64::NAN);
            continue;
        }
        let z = 1.0 + lmbda * v;
        if !v.is_infinite() && z <= 0.0 {
            return Err(BoxCoxError::NonInvertible { value: v, lambda: lmbda });
        }
        out.push(z.powf(inv));
    }
    Ok(out)
}

/// Specification for `box_cox`'s λ parameter — either a fixed numeric
/// value or one of the supported automatic-estimation methods.
///
/// The `box_cox` function accepts anything `impl Into<Lambda>`, and a
/// blanket `From<f64>` conversion is provided, so plain literals work
/// out of the box:
///
/// ```ignore
/// box_cox(&y, 0.5)?;                             // fixed λ
/// box_cox(&y, Lambda::Mle)?;                     // auto, MLE
/// box_cox(&y, Lambda::Pearsonr)?;                // auto, Q-Q robust
/// box_cox(&y, Lambda::Guerrero { period: 12 })?; // auto, R-style
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lambda {
    /// Use the caller-supplied λ verbatim.
    Fixed(f64),
    /// Maximum-likelihood estimate under a Gaussian model for the
    /// transformed series. Matches `scipy.stats.boxcox(x)` and R
    /// `forecast::BoxCox.lambda(method = "loglik")`.
    Mle,
    /// Maximise the Pearson correlation between the *sorted*
    /// transformed values and the corresponding theoretical normal
    /// quantiles. Matches `scipy.stats.boxcox_normmax(x,
    /// method = "pearsonr")`. Rank-based, so more robust to outliers
    /// than `Mle`.
    Pearsonr,
    /// Guerrero (1993) variance-stabilisation criterion: minimise the
    /// coefficient of variation of `σ_b / μ_b^(1−λ)` across consecutive
    /// cycles of length `period`. Matches R
    /// `forecast::BoxCox.lambda(method = "guerrero")`. The canonical
    /// choice for forecasting workflows with seasonal data.
    Guerrero { period: usize },
}

impl From<f64> for Lambda {
    #[inline]
    fn from(v: f64) -> Self {
        Lambda::Fixed(v)
    }
}

/// Result of [`box_cox`]: the transformed series and the λ that was
/// used. The λ is always returned (even when the caller supplied one
/// explicitly) so the same value can be threaded into [`inv_box_cox`]
/// for the back-transformation without bookkeeping.
#[derive(Debug, Clone)]
pub struct BoxCoxOutput {
    /// Box-Cox-transformed series, same length as the input.
    pub transformed: Vec<f64>,
    /// The λ value that was applied — either the caller's fixed input
    /// or the estimator's output.
    pub lambda: f64,
}

/// A Box-Cox transformer that owns its λ.
///
/// Use when you need to apply the *same* transform to multiple series
/// (train/test split, multiple sub-cohorts, etc.) or thread the
/// forward/inverse pair through a forecasting pipeline. For one-shot
/// "transform this single series" use the free [`box_cox`] function
/// instead — it's a thin wrapper around `BoxCox::fit` + `transform`.
///
/// ```ignore
/// // Auto-estimate λ from training data, then apply to test data:
/// let bc = BoxCox::fit(&y_train, Lambda::Guerrero { period: 12 })?;
/// let z_train = bc.transform(&y_train)?;
/// let z_test  = bc.transform(&y_test)?;
/// // …model on z_train, forecast z_hat…
/// let y_hat = bc.inverse_transform(&z_hat)?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BoxCox {
    lambda: f64,
}

impl BoxCox {
    /// Construct from a known λ. No data required.
    #[inline]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Fit a `BoxCox` to `y` using the supplied [`Lambda`] (or a fixed
    /// numeric λ via `From<f64>`). For `Lambda::Fixed(v)` this is
    /// equivalent to [`BoxCox::new(v)`] and doesn't actually look at the
    /// data; for the estimator variants (`Mle`, `Pearsonr`, `Guerrero`)
    /// `y` is consumed to recover λ.
    pub fn fit(y: &[f64], lambda: impl Into<Lambda>) -> Result<Self, BoxCoxError> {
        let lmbda = match lambda.into() {
            Lambda::Fixed(v) => v,
            Lambda::Mle => lambda_mle(y)?,
            Lambda::Pearsonr => lambda_pearsonr(y)?,
            Lambda::Guerrero { period } => lambda_guerrero(y, period)?,
        };
        Ok(Self { lambda: lmbda })
    }

    /// The fitted λ.
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Apply the forward transform with the stored λ.
    pub fn transform(&self, y: &[f64]) -> Result<Vec<f64>, BoxCoxError> {
        box_cox_apply(y, self.lambda)
    }

    /// Invert the transform with the stored λ.
    pub fn inverse_transform(&self, y: &[f64]) -> Result<Vec<f64>, BoxCoxError> {
        inv_box_cox(y, self.lambda)
    }
}

/// Maximum-likelihood λ under the Gaussian transformed-series model
/// (the `scipy.stats.boxcox` objective):
///
/// ```text
/// L(λ) = -n/2 · ln σ²(λ) + (λ − 1) · Σ ln x_i.
/// ```
///
/// Internal helper for [`Lambda::Mle`].
fn lambda_mle(y: &[f64]) -> Result<f64, BoxCoxError> {
    let mut min_finite = f64::INFINITY;
    let mut log_sum = 0.0;
    let mut n_finite = 0usize;
    for &v in y {
        if v.is_finite() {
            if v < min_finite {
                min_finite = v;
            }
            log_sum += v.ln();
            n_finite += 1;
        }
    }
    if min_finite.is_finite() && !(min_finite > 0.0) {
        return Err(BoxCoxError::NonPositive { min: min_finite });
    }
    if n_finite < 2 {
        return Err(BoxCoxError::TooFewObservations {
            n: n_finite,
            min: 2,
        });
    }
    let nf = n_finite as f64;
    let neg_loglik = |lmbda: f64| -> f64 {
        // Transformed values over the finite entries.
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &v in y {
            if !v.is_finite() {
                continue;
            }
            let t = if lmbda == 0.0 {
                v.ln()
            } else {
                (v.powf(lmbda) - 1.0) / lmbda
            };
            sum += t;
            sum_sq += t * t;
        }
        let mean = sum / nf;
        let var = (sum_sq / nf) - mean * mean;
        if var <= 0.0 {
            return f64::INFINITY;
        }
        0.5 * nf * var.ln() - (lmbda - 1.0) * log_sum
    };
    Ok(golden_section_minimize(&neg_loglik, -2.0, 2.0, 1e-8, 200))
}

/// Pearson Q-Q-correlation λ. Internal helper for [`Lambda::Pearsonr`].
fn lambda_pearsonr(y: &[f64]) -> Result<f64, BoxCoxError> {
    let mut min_finite = f64::INFINITY;
    let mut n_finite = 0usize;
    for &v in y {
        if v.is_finite() {
            if v < min_finite {
                min_finite = v;
            }
            n_finite += 1;
        }
    }
    if min_finite.is_finite() && !(min_finite > 0.0) {
        return Err(BoxCoxError::NonPositive { min: min_finite });
    }
    if n_finite < 4 {
        return Err(BoxCoxError::TooFewObservations {
            n: n_finite,
            min: 4,
        });
    }
    let n = n_finite;
    // Theoretical normal quantiles at Blom's plotting positions
    // `(i − 0.375) / (n + 0.25)`. scipy uses the same.
    let q: Vec<f64> = (1..=n)
        .map(|i| inv_phi((i as f64 - 0.375) / (n as f64 + 0.25)))
        .collect();
    let q_mean: f64 = q.iter().sum::<f64>() / n as f64;
    let q_centered: Vec<f64> = q.iter().map(|v| v - q_mean).collect();
    let q_ss_sqrt: f64 = q_centered.iter().map(|v| v * v).sum::<f64>().sqrt();
    if q_ss_sqrt <= 0.0 {
        return Err(BoxCoxError::TooFewObservations { n, min: 4 });
    }
    // Pre-allocate the transformed-and-sorted buffer once.
    let neg_corr = |lmbda: f64| -> f64 {
        let mut z: Vec<f64> = Vec::with_capacity(n);
        for &v in y {
            if !v.is_finite() {
                continue;
            }
            let t = if lmbda == 0.0 {
                v.ln()
            } else {
                (v.powf(lmbda) - 1.0) / lmbda
            };
            if !t.is_finite() {
                return f64::INFINITY;
            }
            z.push(t);
        }
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let z_mean: f64 = z.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut z_ss = 0.0;
        for i in 0..n {
            let zc = z[i] - z_mean;
            num += zc * q_centered[i];
            z_ss += zc * zc;
        }
        let z_ss_sqrt = z_ss.sqrt();
        if z_ss_sqrt <= 0.0 {
            return f64::INFINITY;
        }
        // We minimise; maximising r means minimising −r.
        -(num / (z_ss_sqrt * q_ss_sqrt))
    };
    Ok(golden_section_minimize(&neg_corr, -2.0, 2.0, 1e-8, 200))
}

/// Inverse standard-normal CDF via Acklam's algorithm (~1.15e-9
/// accuracy). Private to this module; the same rational approximation
/// lives in `tsa::arima` for prediction intervals.
fn inv_phi(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Guerrero (1993) variance-stabilisation λ. Internal helper for
/// [`Lambda::Guerrero`].
fn lambda_guerrero(y: &[f64], period: usize) -> Result<f64, BoxCoxError> {
    if period < 2 {
        return Err(BoxCoxError::InvalidPeriod(period));
    }
    let n_blocks = y.len() / period;
    if n_blocks < 2 {
        return Err(BoxCoxError::TooFewObservations {
            n: y.len(),
            min: 2 * period,
        });
    }
    let mut min_finite = f64::INFINITY;
    for &v in &y[..n_blocks * period] {
        if v.is_finite() && v < min_finite {
            min_finite = v;
        }
    }
    if min_finite.is_finite() && !(min_finite > 0.0) {
        return Err(BoxCoxError::NonPositive { min: min_finite });
    }
    // Pre-compute per-block means and sample standard deviations.
    let mut means = Vec::with_capacity(n_blocks);
    let mut sds = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let block = &y[b * period..(b + 1) * period];
        let mean: f64 = block.iter().sum::<f64>() / period as f64;
        let var: f64 = block
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (period as f64 - 1.0);
        means.push(mean);
        sds.push(var.sqrt());
    }
    let n_b = n_blocks as f64;
    let criterion = |lmbda: f64| -> f64 {
        let mut ratios = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let denom = means[b].powf(1.0 - lmbda);
            if denom <= 0.0 || !denom.is_finite() {
                return f64::INFINITY;
            }
            ratios.push(sds[b] / denom);
        }
        let mean_r: f64 = ratios.iter().sum::<f64>() / n_b;
        if mean_r == 0.0 {
            return f64::INFINITY;
        }
        let var_r: f64 = ratios
            .iter()
            .map(|v| (v - mean_r).powi(2))
            .sum::<f64>()
            / (n_b - 1.0);
        var_r.sqrt() / mean_r.abs()
    };
    Ok(golden_section_minimize(&criterion, -1.0, 2.0, 1e-8, 200))
}

/// Golden-section minimum search on a unimodal `f` over `[a, b]`.
fn golden_section_minimize<F: Fn(f64) -> f64>(
    f: &F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    // Resphi = 2 - φ = (3 - √5) / 2.
    const RESPHI: f64 = 0.381_966_011_250_105_2;
    let mut x1 = a + RESPHI * (b - a);
    let mut x2 = b - RESPHI * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + RESPHI * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - RESPHI * (b - a);
            f2 = f(x2);
        }
    }
    if f1 < f2 {
        x1
    } else {
        x2
    }
}

/// Box-Cox power transformation.
///
/// ```text
/// (x^λ − 1) / λ   when λ ≠ 0
/// ln(x)           when λ = 0
/// ```
///
/// The second argument can be either a numeric λ (via the `From<f64>`
/// conversion) or a [`Lambda`] enum variant that selects an automatic
/// estimator:
///
/// ```ignore
/// box_cox(&y, 0.5)?;                                    // fixed
/// box_cox(&y, Lambda::Mle)?;                            // MLE
/// box_cox(&y, Lambda::Pearsonr)?;                       // Q-Q (robust)
/// box_cox(&y, Lambda::Guerrero { period: 12 })?;        // R-style
/// ```
///
/// Returns a [`BoxCoxOutput`] carrying both the transformed series and
/// the λ that was applied. The λ is needed by [`inv_box_cox`] to
/// reverse the transformation, so it's always returned — even when the
/// caller supplied a fixed value (in which case `out.lambda` simply
/// echoes the input).
///
/// Requires every finite input to be strictly positive. `NaN` entries
/// propagate to the output unchanged; `+∞` is treated as finite for
/// propagation purposes but never satisfies the positivity check on
/// its own (only finite values gate the check).
pub fn box_cox(y: &[f64], lambda: impl Into<Lambda>) -> Result<BoxCoxOutput, BoxCoxError> {
    let bc = BoxCox::fit(y, lambda)?;
    let transformed = bc.transform(y)?;
    Ok(BoxCoxOutput {
        transformed,
        lambda: bc.lambda(),
    })
}

/// Apply Box-Cox with a known λ. Internal helper — public callers go
/// through [`box_cox`].
fn box_cox_apply(y: &[f64], lmbda: f64) -> Result<Vec<f64>, BoxCoxError> {
    let mut min_finite = f64::INFINITY;
    for &v in y {
        if v.is_finite() && v < min_finite {
            min_finite = v;
        }
    }
    if min_finite.is_finite() && !(min_finite > 0.0) {
        return Err(BoxCoxError::NonPositive { min: min_finite });
    }

    if lmbda == 0.0 {
        Ok(y.iter().map(|&v| v.ln()).collect())
    } else {
        let inv = 1.0 / lmbda;
        Ok(y.iter().map(|&v| (v.powf(lmbda) - 1.0) * inv).collect())
    }
}

// ============================================================================
// Scalar reference implementations.
//
// Kept private and compiled only under `cfg(test)`: they exist solely as
// an oracle for the pulp parity tests below. External callers always go
// through the public functions, which dispatch to the pulp-backed
// kernels.
// ============================================================================

#[cfg(test)]
mod scalar {
    fn finite_mean(y: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &v in y {
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Sample standard deviation (ddof = 1) over the finite entries.
    /// Returns `0.0` when fewer than two finite entries are present.
    fn finite_std_ddof1(y: &[f64], mean: f64) -> f64 {
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for &v in y {
            if v.is_finite() {
                let d = v - mean;
                sum_sq += d * d;
                count += 1;
            }
        }
        if count < 2 {
            0.0
        } else {
            (sum_sq / (count - 1) as f64).sqrt()
        }
    }

    fn finite_min_max(y: &[f64]) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut any = false;
        for &v in y {
            if v.is_finite() {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
                any = true;
            }
        }
        if any {
            (lo, hi)
        } else {
            (0.0, 0.0)
        }
    }

    pub(super) fn center(y: &[f64]) -> Vec<f64> {
        let mean = finite_mean(y);
        y.iter().map(|&v| v - mean).collect()
    }

    pub(super) fn z_score(y: &[f64]) -> Vec<f64> {
        let mean = finite_mean(y);
        let std = finite_std_ddof1(y, mean);
        if std == 0.0 {
            y.iter().map(|&v| (v - mean) * 0.0).collect()
        } else {
            y.iter().map(|&v| (v - mean) / std).collect()
        }
    }

    pub(super) fn min_max_scale(y: &[f64]) -> Vec<f64> {
        let (lo, hi) = finite_min_max(y);
        let range = hi - lo;
        if range == 0.0 {
            y.iter().map(|&v| (v - lo) * 0.0).collect()
        } else {
            y.iter().map(|&v| (v - lo) / range).collect()
        }
    }
}

// ============================================================================
// SIMD kernels — backed by `pulp` (stable Rust, runtime ISA dispatch).
//
// `pulp::Arch::new()` selects the best SIMD level at runtime (SSE2 /
// AVX2 / AVX-512 on x86_64, NEON on aarch64, scalar fallback elsewhere).
// `S::F64_LANES` is the lane count of the chosen target.
//
// The kernels preserve the scalar contracts: NaN inputs propagate to the
// same output positions, aggregates are computed over the finite entries
// only, and degenerate inputs (empty / constant / fewer than two finite
// values) produce the same zeros-or-NaN pattern as the scalar path.
// ============================================================================

mod pulp_impl {
    use pulp::{Arch, Simd, WithSimd};

    /// Lanewise `is_finite` mask: `abs(v) < +∞` is true for every finite
    /// f64 and false for ±∞ and NaN (NaN < anything is always false).
    #[inline(always)]
    fn finite_mask<S: Simd>(simd: S, v: S::f64s) -> S::m64s {
        simd.less_than_f64s(simd.abs_f64s(v), simd.splat_f64s(f64::INFINITY))
    }

    struct FiniteSumCount<'a> {
        y: &'a [f64],
    }
    impl<'a> WithSimd for FiniteSumCount<'a> {
        type Output = (f64, usize);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let zero = simd.splat_f64s(0.0);
            let one = simd.splat_f64s(1.0);
            let mut sum_v = zero;
            let mut cnt_v = zero;
            for &v in head {
                let m = finite_mask(simd, v);
                sum_v = simd.add_f64s(sum_v, simd.select_f64s(m, v, zero));
                cnt_v = simd.add_f64s(cnt_v, simd.select_f64s(m, one, zero));
            }
            let mut sum = simd.reduce_sum_f64s(sum_v);
            let mut cnt = simd.reduce_sum_f64s(cnt_v) as usize;
            for &v in tail {
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            (sum, cnt)
        }
    }

    struct FiniteSumSq<'a> {
        y: &'a [f64],
        mean: f64,
    }
    impl<'a> WithSimd for FiniteSumSq<'a> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let mean_v = simd.splat_f64s(self.mean);
            let zero = simd.splat_f64s(0.0);
            let mut acc = zero;
            for &v in head {
                let m = finite_mask(simd, v);
                let d = simd.sub_f64s(v, mean_v);
                let dd = simd.mul_f64s(d, d);
                acc = simd.add_f64s(acc, simd.select_f64s(m, dd, zero));
            }
            let mut s = simd.reduce_sum_f64s(acc);
            for &v in tail {
                if v.is_finite() {
                    let d = v - self.mean;
                    s += d * d;
                }
            }
            s
        }
    }

    struct FiniteMinMax<'a> {
        y: &'a [f64],
    }
    impl<'a> WithSimd for FiniteMinMax<'a> {
        type Output = (f64, f64);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let pos_inf = simd.splat_f64s(f64::INFINITY);
            let neg_inf = simd.splat_f64s(f64::NEG_INFINITY);
            let mut lo_v = pos_inf;
            let mut hi_v = neg_inf;
            for &v in head {
                let m = finite_mask(simd, v);
                lo_v = simd.min_f64s(lo_v, simd.select_f64s(m, v, pos_inf));
                hi_v = simd.max_f64s(hi_v, simd.select_f64s(m, v, neg_inf));
            }
            let mut lo = simd.reduce_min_f64s(lo_v);
            let mut hi = simd.reduce_max_f64s(hi_v);
            for &v in tail {
                if v.is_finite() {
                    if v < lo {
                        lo = v;
                    }
                    if v > hi {
                        hi = v;
                    }
                }
            }
            // If no finite value was seen at all, both extremes are still
            // their initialisers — collapse to (0, 0) to match the scalar
            // contract.
            if lo == f64::INFINITY && hi == f64::NEG_INFINITY {
                (0.0, 0.0)
            } else {
                (lo, hi)
            }
        }
    }

    struct AffineInto<'a> {
        y: &'a [f64],
        out: &'a mut [f64],
        c: f64,
        k: f64,
    }
    impl<'a> WithSimd for AffineInto<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self { y, out, c, k } = self;
            let c_v = simd.splat_f64s(c);
            let k_v = simd.splat_f64s(k);
            let (y_head, y_tail) = S::as_simd_f64s(y);
            let (o_head, o_tail) = S::as_mut_simd_f64s(out);
            for (yv, ov) in y_head.iter().zip(o_head.iter_mut()) {
                *ov = simd.mul_f64s(simd.sub_f64s(*yv, c_v), k_v);
            }
            for (yv, ov) in y_tail.iter().zip(o_tail.iter_mut()) {
                *ov = (*yv - c) * k;
            }
        }
    }

    pub(super) fn center(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (sum, cnt) = arch.dispatch(FiniteSumCount { y });
        let mean = if cnt == 0 { 0.0 } else { sum / cnt as f64 };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: mean, k: 1.0 });
        out
    }

    pub(super) fn z_score(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (sum, cnt) = arch.dispatch(FiniteSumCount { y });
        let mean = if cnt == 0 { 0.0 } else { sum / cnt as f64 };
        let std = if cnt < 2 {
            0.0
        } else {
            (arch.dispatch(FiniteSumSq { y, mean }) / (cnt - 1) as f64).sqrt()
        };
        // std == 0 → multiplying by 0 inside `AffineInto` preserves NaN
        // (NaN * 0 = NaN) and zeros every finite position — the same as
        // the scalar `(v - mean) * 0.0` contract.
        let inv = if std == 0.0 { 0.0 } else { 1.0 / std };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: mean, k: inv });
        out
    }

    pub(super) fn min_max_scale(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (lo, hi) = arch.dispatch(FiniteMinMax { y });
        let range = hi - lo;
        let inv = if range == 0.0 { 0.0 } else { 1.0 / range };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: lo, k: inv });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn center_subtracts_mean() {
        let out = center(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn center_empty() {
        assert_eq!(center(&[]), Vec::<f64>::new());
    }

    #[test]
    fn center_propagates_nan() {
        let out = center(&[1.0, f64::NAN, 3.0]);
        // mean over finite = 2.0
        assert_relative_eq!(out[0], -1.0);
        assert!(out[1].is_nan());
        assert_relative_eq!(out[2], 1.0);
    }

    #[test]
    fn z_score_unit_variance() {
        let out = z_score(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // mean=3, std=sqrt(2.5)
        let s = (2.5_f64).sqrt();
        assert_relative_eq!(out[0], -2.0 / s, max_relative = 1e-12);
        assert_relative_eq!(out[4], 2.0 / s, max_relative = 1e-12);
    }

    #[test]
    fn z_score_constant_returns_zeros() {
        assert_eq!(z_score(&[4.0, 4.0, 4.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn min_max_scale_basic() {
        let out = min_max_scale(&[0.0, 5.0, 10.0]);
        assert_eq!(out, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn min_max_scale_constant() {
        assert_eq!(min_max_scale(&[7.0, 7.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn box_cox_lmbda_zero_is_ln() {
        let out = box_cox(&[1.0, std::f64::consts::E], 0.0).unwrap();
        assert_eq!(out.lambda, 0.0);
        assert_relative_eq!(out.transformed[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out.transformed[1], 1.0, max_relative = 1e-12);
    }

    #[test]
    fn box_cox_lmbda_two() {
        // (x^2 - 1) / 2
        let out = box_cox(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert_relative_eq!(out.transformed[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out.transformed[1], 1.5);
        assert_relative_eq!(out.transformed[2], 4.0);
    }

    #[test]
    fn box_cox_rejects_non_positive() {
        let err = box_cox(&[1.0, 0.0, 2.0], 1.0).unwrap_err();
        assert_eq!(err, BoxCoxError::NonPositive { min: 0.0 });
    }

    #[test]
    fn box_cox_propagates_nan() {
        let out = box_cox(&[1.0, f64::NAN, 4.0], 2.0).unwrap();
        assert_eq!(out.transformed[0], 0.0);
        assert!(out.transformed[1].is_nan());
        assert_relative_eq!(out.transformed[2], 7.5);
    }

    #[test]
    fn box_cox_echoes_fixed_lambda() {
        // The returned `lambda` should match whatever the caller passed.
        let out = box_cox(&[1.0, 2.0, 4.0], 0.75).unwrap();
        assert_eq!(out.lambda, 0.75);
    }

    // --- inverse Box-Cox ---

    #[test]
    fn inv_box_cox_lambda_zero_is_exp() {
        let out = inv_box_cox(&[0.0, 1.0, 2.0], 0.0).unwrap();
        assert_relative_eq!(out[0], 1.0, max_relative = 1e-12);
        assert_relative_eq!(out[1], std::f64::consts::E, max_relative = 1e-12);
        assert_relative_eq!(out[2], std::f64::consts::E * std::f64::consts::E, max_relative = 1e-12);
    }

    #[test]
    fn inv_box_cox_lambda_two_roundtrips() {
        let x = vec![1.0, 2.0, 3.5, 7.25];
        let y = box_cox(&x, 2.0).unwrap();
        let back = inv_box_cox(&y.transformed, y.lambda).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn inv_box_cox_lambda_half_roundtrips() {
        let x = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let y = box_cox(&x, 0.5).unwrap();
        let back = inv_box_cox(&y.transformed, y.lambda).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn inv_box_cox_propagates_nan() {
        let out = inv_box_cox(&[0.0, f64::NAN, 1.0], 0.0).unwrap();
        assert_relative_eq!(out[0], 1.0, max_relative = 1e-12);
        assert!(out[1].is_nan());
        assert_relative_eq!(out[2], std::f64::consts::E, max_relative = 1e-12);
    }

    #[test]
    fn inv_box_cox_rejects_out_of_domain() {
        // λ = 1, y = -2 → 1 + 1·(-2) = -1 ≤ 0 → error.
        let err = inv_box_cox(&[-2.0], 1.0).unwrap_err();
        assert!(matches!(err, BoxCoxError::NonInvertible { .. }));
    }

    // --- Lambda estimators (Mle / Pearsonr / Guerrero) ---

    #[test]
    fn lambda_mle_near_one_on_normal_data() {
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let t = (i as f64) * 0.31;
                10.0 + t.sin() + (t * 1.7).cos() * 0.5 + (t * 0.3).sin() * 0.3
            })
            .collect();
        let out = box_cox(&y, Lambda::Mle).unwrap();
        assert!(
            (out.lambda - 1.0).abs() < 0.5,
            "expected λ near 1 for ~normal data, got {}",
            out.lambda
        );
    }

    #[test]
    fn lambda_mle_near_zero_on_lognormal_data() {
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let t = (i as f64) * 0.31;
                (t.sin() + (t * 1.7).cos() * 0.5).exp() * 10.0
            })
            .collect();
        let out = box_cox(&y, Lambda::Mle).unwrap();
        assert!(
            out.lambda.abs() < 0.5,
            "expected λ near 0 for lognormal data, got {}",
            out.lambda
        );
    }

    #[test]
    fn lambda_pearsonr_near_one_on_normal_data() {
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 * 0.31;
                10.0 + t.sin() + (t * 1.7).cos() * 0.5 + (t * 0.3).sin() * 0.3
            })
            .collect();
        let out = box_cox(&y, Lambda::Pearsonr).unwrap();
        assert!((out.lambda - 1.0).abs() < 0.5, "got {}", out.lambda);
    }

    #[test]
    fn lambda_pearsonr_near_zero_on_lognormal_data() {
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 * 0.31;
                (t.sin() + (t * 1.7).cos() * 0.5).exp() * 10.0
            })
            .collect();
        let out = box_cox(&y, Lambda::Pearsonr).unwrap();
        assert!(out.lambda.abs() < 0.5, "got {}", out.lambda);
    }

    #[test]
    fn lambda_pearsonr_rejects_non_positive() {
        let err = box_cox(&[1.0, -2.0, 3.0], Lambda::Pearsonr).unwrap_err();
        assert!(matches!(err, BoxCoxError::NonPositive { .. }));
    }

    #[test]
    fn lambda_pearsonr_close_to_mle_under_outlier() {
        // Both estimators target marginal normality; under a single
        // outlier the MLE pulls harder than the Q-Q-correlation form,
        // but they should still land in the same ballpark.
        let mut y: Vec<f64> = (0..200)
            .map(|i| 10.0 + (i as f64 * 0.31).sin() * 0.5)
            .collect();
        y[100] = 200.0;
        let mle = box_cox(&y, Lambda::Mle).unwrap().lambda;
        let pearson = box_cox(&y, Lambda::Pearsonr).unwrap().lambda;
        assert!(mle.is_finite() && pearson.is_finite());
        assert!(
            (mle - pearson).abs() < 1.5,
            "estimators wildly diverge: mle={mle}, pearson={pearson}"
        );
    }

    #[test]
    fn lambda_mle_rejects_non_positive() {
        let err = box_cox(&[1.0, -2.0, 3.0], Lambda::Mle).unwrap_err();
        assert!(matches!(err, BoxCoxError::NonPositive { .. }));
    }

    #[test]
    fn lambda_guerrero_rejects_invalid_period() {
        let y = vec![1.0; 100];
        let err = box_cox(&y, Lambda::Guerrero { period: 1 }).unwrap_err();
        assert!(matches!(err, BoxCoxError::InvalidPeriod(1)));
    }

    // --- BoxCox struct ---

    #[test]
    fn boxcox_new_stores_lambda() {
        let bc = BoxCox::new(0.5);
        assert_eq!(bc.lambda(), 0.5);
    }

    #[test]
    fn boxcox_fit_with_fixed_lambda_doesnt_touch_data() {
        // Even on an empty slice, Lambda::Fixed should succeed.
        let bc = BoxCox::fit(&[], 0.7).unwrap();
        assert_eq!(bc.lambda(), 0.7);
    }

    #[test]
    fn boxcox_fit_runs_estimator_on_data() {
        // Lognormal data → MLE should pick λ near 0.
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let t = (i as f64) * 0.31;
                (t.sin() + (t * 1.7).cos() * 0.5).exp() * 10.0
            })
            .collect();
        let bc = BoxCox::fit(&y, Lambda::Mle).unwrap();
        assert!(bc.lambda().abs() < 0.5, "got λ = {}", bc.lambda());
    }

    #[test]
    fn boxcox_transform_inverse_roundtrips() {
        let y = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let bc = BoxCox::new(0.5);
        let z = bc.transform(&y).unwrap();
        let back = bc.inverse_transform(&z).unwrap();
        for (a, b) in y.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn boxcox_transform_matches_free_function() {
        // The struct path and the one-shot free function must produce
        // bit-identical transforms for the same λ.
        let y = vec![1.5, 2.5, 4.0, 8.5];
        let from_struct = BoxCox::new(0.75).transform(&y).unwrap();
        let from_free = box_cox(&y, 0.75).unwrap().transformed;
        for (a, b) in from_struct.iter().zip(from_free.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn boxcox_applies_to_multiple_series_with_same_lambda() {
        // The motivating use case: fit λ once, apply to several series.
        let y_train: Vec<f64> = (1..=50).map(|i| (i as f64).sqrt() * 5.0 + 10.0).collect();
        let y_test: Vec<f64> = (51..=60).map(|i| (i as f64).sqrt() * 5.0 + 10.0).collect();
        let bc = BoxCox::fit(&y_train, Lambda::Mle).unwrap();
        let z_train = bc.transform(&y_train).unwrap();
        let z_test = bc.transform(&y_test).unwrap();
        assert_eq!(z_train.len(), y_train.len());
        assert_eq!(z_test.len(), y_test.len());
        // Both calls used the same λ; the inverse on either round-trips.
        let back = bc.inverse_transform(&z_test).unwrap();
        for (a, b) in y_test.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-10);
        }
    }

    #[test]
    fn guerrero_picks_log_for_multiplicative_seasonal_variance() {
        // Series where the cycle-to-cycle standard deviation grows
        // proportionally to the cycle mean (classic "multiplicative
        // seasonality"). Guerrero should recommend λ ≈ 0 (log) to
        // stabilise that.
        let period = 12;
        let n_cycles = 30;
        let mut y = Vec::with_capacity(period * n_cycles);
        for c in 0..n_cycles {
            let level = 10.0 + 0.5 * c as f64;
            for i in 0..period {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
                let seasonal = (phase).sin();
                // σ ∝ μ ⇒ noise scales with level; classic
                // multiplicative-variance case.
                y.push(level * (1.0 + 0.2 * seasonal));
            }
        }
        let out = box_cox(&y, Lambda::Guerrero { period }).unwrap();
        // Guerrero should return small λ — close to 0 or even slightly
        // negative — indicating a log-ish transform.
        assert!(
            out.lambda < 0.5,
            "expected small λ for multiplicative-variance series, got {}",
            out.lambda,
        );
    }

    #[test]
    fn guerrero_picks_one_for_additive_seasonal_variance() {
        // Series with constant-variance noise on top of a stable level
        // — Guerrero should leave it alone (λ ≈ 1).
        let period = 12;
        let n_cycles = 30;
        let mut y = Vec::with_capacity(period * n_cycles);
        for c in 0..n_cycles {
            for i in 0..period {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
                let seasonal = (phase).sin();
                // σ is constant across cycles ⇒ λ = 1 is optimal.
                y.push(10.0 + 0.5 * c as f64 + seasonal);
            }
        }
        let out = box_cox(&y, Lambda::Guerrero { period }).unwrap();
        assert!(
            (out.lambda - 1.0).abs() < 0.5,
            "expected λ near 1 for additive-variance series, got {}",
            out.lambda,
        );
    }

    // --- Pulp vs. scalar parity ---
    //
    // The public functions go through pulp; the private `scalar` module
    // is the oracle. Outputs must agree to ~1e-12 on a mixed-size input
    // that crosses the SIMD lane boundary and includes NaN.

    fn parity_check(scalar_out: &[f64], simd_out: &[f64], ctx: &str) {
        assert_eq!(scalar_out.len(), simd_out.len(), "{ctx}: length mismatch");
        for (i, (a, b)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            if a.is_nan() {
                assert!(b.is_nan(), "{ctx}[{i}]: scalar NaN but simd {b}");
            } else {
                assert!(
                    (a - b).abs() < 1e-12,
                    "{ctx}[{i}]: scalar {a}, simd {b}, |Δ| = {}",
                    (a - b).abs()
                );
            }
        }
    }

    fn make_fixture() -> Vec<f64> {
        // length 11 — guarantees a remainder past any 2/4/8-lane SIMD chunk
        vec![
            1.0, -2.5, 3.25, f64::NAN, 5.5, 0.0, -7.125, 8.75,
            f64::NAN, 11.5, -3.0,
        ]
    }

    #[test]
    fn pulp_center_matches_scalar() {
        let y = make_fixture();
        parity_check(&super::scalar::center(&y), &center(&y), "center");
    }

    #[test]
    fn pulp_z_score_matches_scalar() {
        let y = make_fixture();
        parity_check(&super::scalar::z_score(&y), &z_score(&y), "z_score");

        // Constant path: pulp must also collapse to zeros.
        let constant = vec![4.2; 9];
        parity_check(
            &super::scalar::z_score(&constant),
            &z_score(&constant),
            "z_score constant",
        );
    }

    #[test]
    fn pulp_min_max_matches_scalar() {
        let y = make_fixture();
        parity_check(
            &super::scalar::min_max_scale(&y),
            &min_max_scale(&y),
            "min_max",
        );
    }

    #[test]
    fn pulp_handles_empty_and_short_inputs() {
        assert!(center(&[]).is_empty());
        assert!(z_score(&[]).is_empty());
        assert!(min_max_scale(&[]).is_empty());

        let y = vec![1.0, 2.0, 3.0];
        parity_check(&super::scalar::center(&y), &center(&y), "short center");
        parity_check(&super::scalar::z_score(&y), &z_score(&y), "short z_score");
        parity_check(
            &super::scalar::min_max_scale(&y),
            &min_max_scale(&y),
            "short min_max",
        );
    }
}
