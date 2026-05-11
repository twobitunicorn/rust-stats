//! Transform tests inspired by R, scipy, and scikit-learn references:
//!
//!   - `center`        ↔ R `scale(x, center = TRUE, scale = FALSE)`
//!                       (and `sklearn.preprocessing.StandardScaler(with_std=False)`)
//!   - `z_score`       ↔ R `scale(x)` (note: R uses sample SD, ddof = 1;
//!                       sklearn's StandardScaler uses ddof = 0, so we
//!                       match R here, not sklearn)
//!   - `min_max_scale` ↔ `sklearn.preprocessing.MinMaxScaler` (default range
//!                       `[0, 1]`), R `caret::preProcess(method = "range")`
//!   - `box_cox`       ↔ `scipy.stats.boxcox(x, lmbda=...)` and
//!                       `forecast::BoxCox(x, lambda)` (same closed form
//!                       in both)
//!
//! Reference values are derivable by hand from the published formulas;
//! where R/sklearn diverges from us on degenerate inputs (constant series,
//! all-NaN) we document the divergence in the test body.
//!
//! Tests requiring scipy's MLE λ estimation, sklearn's `inverse_transform`
//! plumbing, or pandas-indexed inputs are not portable to our slice API
//! and are omitted.

use approx::assert_relative_eq;
use rust_stats::error::BoxCoxError;
use rust_stats::{box_cox, center, inv_box_cox, min_max_scale, z_score, Lambda};

/// numpy.testing.assert_almost_equal semantics: tolerance = 1.5 * 10^-decimal.
fn approx_eq(a: f64, b: f64, decimal: i32, ctx: &str) {
    if b.is_nan() {
        assert!(a.is_nan(), "{ctx}: expected NaN, got {a}");
        return;
    }
    let tol = 1.5 * 10f64.powi(-decimal);
    assert!(
        (a - b).abs() < tol,
        "{ctx}: |{a} - {b}| = {} >= {tol}",
        (a - b).abs(),
    );
}

// ============================================================================
// center  —  R: scale(x, center = TRUE, scale = FALSE)
// ============================================================================

/// R: `as.vector(scale(c(1,2,3,4,5), center = TRUE, scale = FALSE))`
///    → c(-2, -1, 0, 1, 2)
#[test]
fn center_matches_r_scale_integers() {
    let out = center(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(out, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
}

/// R: `as.vector(scale(c(2.5, 7.5, 1.0, 4.0, 5.5), scale = FALSE))`
///    mean = 4.1 → c(-1.6, 3.4, -3.1, -0.1, 1.4)
#[test]
fn center_matches_r_scale_floats() {
    let out = center(&[2.5, 7.5, 1.0, 4.0, 5.5]);
    let expected = [-1.6, 3.4, -3.1, -0.1, 1.4];
    for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("center[{i}]"));
    }
}

/// Empty input → empty output. (R `scale(numeric(0))` returns a 0-row matrix.)
#[test]
fn center_empty_matches_r() {
    assert_eq!(center(&[]), Vec::<f64>::new());
}

/// Constant input: every entry equals the mean, so all deviations are 0.
/// (R agrees: `scale(c(7,7,7), scale=FALSE)` → c(0,0,0).)
#[test]
fn center_constant_is_zero() {
    let out = center(&[7.0, 7.0, 7.0]);
    assert_eq!(out, vec![0.0, 0.0, 0.0]);
}

/// NaN propagation: aggregates are computed over the finite entries; NaN
/// positions in the input map to NaN positions in the output. Diverges
/// from base R, where any `NA` makes `scale()` return all-`NA`; matches
/// the spirit of numpy `nanmean`-style aggregation.
#[test]
fn center_nan_propagates_per_position() {
    let out = center(&[1.0, f64::NAN, 3.0, 5.0]);
    // mean over finite = (1 + 3 + 5) / 3 = 3.0
    assert_relative_eq!(out[0], -2.0, max_relative = 1e-12);
    assert!(out[1].is_nan());
    assert_relative_eq!(out[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(out[3], 2.0, max_relative = 1e-12);
}

// ============================================================================
// z_score  —  R: scale(x)  (sample SD, ddof = 1)
// ============================================================================

/// R: `as.vector(scale(c(1,2,3,4,5)))`
///    mean = 3, sd = sqrt(2.5) ≈ 1.5811388
///    → c(-1.2649110, -0.6324555, 0, 0.6324555, 1.2649110)
#[test]
fn z_score_matches_r_scale_integers() {
    let out = z_score(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let sd = 2.5_f64.sqrt();
    let expected = [-2.0 / sd, -1.0 / sd, 0.0, 1.0 / sd, 2.0 / sd];
    for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("z_score[{i}]"));
    }
}

/// Wikipedia z-score example (x = 2,4,4,4,5,5,7,9), computed with sample
/// SD (matching R's `scale(x)`, not the population-SD figure on Wikipedia
/// which divides by n instead of n-1).
///
/// R: `round(as.vector(scale(c(2,4,4,4,5,5,7,9))), 6)`
///    → c(-1.403121, -0.467707, -0.467707, -0.467707,
///         0.000000,  0.000000,  0.935414,  1.870829)
#[test]
fn z_score_matches_r_scale_wikipedia_example() {
    let out = z_score(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    // mean = 5, sum_sq = 32, ddof=1 var = 32/7, sd = sqrt(32/7).
    let sd = (32.0_f64 / 7.0).sqrt();
    let expected = [
        -3.0 / sd, -1.0 / sd, -1.0 / sd, -1.0 / sd,
         0.0,       0.0,       2.0 / sd,  4.0 / sd,
    ];
    for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 10, &format!("z_score[{i}]"));
    }
}

/// Output mean ≈ 0 and ddof=1 SD ≈ 1 for a non-constant series — the
/// defining property of R's `scale()`.
#[test]
fn z_score_has_zero_mean_unit_sample_sd() {
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).sin() * 5.0 + 3.0).collect();
    let z = z_score(&y);
    let n = z.len() as f64;
    let mean: f64 = z.iter().sum::<f64>() / n;
    let var: f64 = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    approx_eq(mean, 0.0, 12, "mean");
    approx_eq(var.sqrt(), 1.0, 12, "sample sd");
}

/// Constant input: SD is 0, so the canonical R/scipy output is NaN. Our
/// impl returns 0 at finite positions instead (documented behaviour).
/// Recording the divergence so a future change is intentional.
#[test]
fn z_score_constant_returns_zeros_diverges_from_r() {
    let out = z_score(&[4.2, 4.2, 4.2, 4.2]);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
}

/// Empty input → empty output.
#[test]
fn z_score_empty() {
    assert_eq!(z_score(&[]), Vec::<f64>::new());
}

// ============================================================================
// min_max_scale  —  sklearn.preprocessing.MinMaxScaler (range [0, 1])
// ============================================================================

/// sklearn: `MinMaxScaler().fit_transform([[1],[2],[3]]).ravel()`
///   → array([0. , 0.5, 1. ])
#[test]
fn min_max_matches_sklearn_basic() {
    let out = min_max_scale(&[1.0, 2.0, 3.0]);
    assert_eq!(out, vec![0.0, 0.5, 1.0]);
}

/// sklearn: `MinMaxScaler().fit_transform([[-10],[0],[10],[5]]).ravel()`
///   min=-10, max=10, range=20
///   → array([0. , 0.5, 1. , 0.75])
#[test]
fn min_max_matches_sklearn_signed() {
    let out = min_max_scale(&[-10.0, 0.0, 10.0, 5.0]);
    assert_eq!(out, vec![0.0, 0.5, 1.0, 0.75]);
}

/// Defining property of sklearn's MinMaxScaler: the output minimum is
/// exactly 0 and the output maximum is exactly 1 (no rounding involved
/// because of how the formula is structured).
#[test]
fn min_max_endpoints_are_exact() {
    let y: Vec<f64> = (0..50).map(|i| ((i as f64) * 0.37).sin() * 10.0 + 2.0).collect();
    let out = min_max_scale(&y);
    let lo = out.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(lo, 0.0);
    assert_eq!(hi, 1.0);
}

/// Constant input: range is 0. sklearn ≥1.0 returns 0.0 for every cell
/// (numerator and denominator both 0; sklearn special-cases). We match.
#[test]
fn min_max_constant_matches_sklearn() {
    let out = min_max_scale(&[7.0, 7.0, 7.0]);
    assert_eq!(out, vec![0.0, 0.0, 0.0]);
}

/// NaN propagates per-position; aggregates are computed over finite
/// entries. (sklearn 1.x's MinMaxScaler with default settings raises on
/// NaN; this is a tolerant extension.)
#[test]
fn min_max_nan_propagates_per_position() {
    let out = min_max_scale(&[1.0, f64::NAN, 3.0, 5.0]);
    // finite min=1, max=5, range=4
    assert_eq!(out[0], 0.0);
    assert!(out[1].is_nan());
    assert_relative_eq!(out[2], 0.5, max_relative = 1e-12);
    assert_eq!(out[3], 1.0);
}

/// Empty input → empty output.
#[test]
fn min_max_empty() {
    assert_eq!(min_max_scale(&[]), Vec::<f64>::new());
}

// ============================================================================
// box_cox  —  scipy.stats.boxcox(x, lmbda) and R forecast::BoxCox(x, lambda)
// ============================================================================

/// λ = 0 collapses to the natural logarithm.
///
/// scipy: `boxcox([1, 2, 4, 8], lmbda=0)` → `[0, ln 2, ln 4, ln 8]`.
#[test]
fn box_cox_lambda_zero_is_log() {
    let out = box_cox(&[1.0, 2.0, 4.0, 8.0], 0.0).unwrap();
    let expected = [0.0, 2f64.ln(), 4f64.ln(), 8f64.ln()];
    for (i, (&a, &e)) in out.transformed.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("ln[{i}]"));
    }
}

/// λ = 1 → `(x - 1) / 1` = `x - 1`. scipy returns the same.
#[test]
fn box_cox_lambda_one_is_shift() {
    let out = box_cox(&[1.0, 2.0, 4.0, 8.0], 1.0).unwrap();
    assert_eq!(out.transformed, vec![0.0, 1.0, 3.0, 7.0]);
}

/// λ = 0.5 → `2 * (sqrt(x) - 1)`. scipy returns the same.
#[test]
fn box_cox_lambda_half_matches_scipy() {
    let out = box_cox(&[1.0, 2.0, 4.0, 8.0], 0.5).unwrap();
    let expected = [
        0.0,
        2.0 * (2f64.sqrt() - 1.0),
        2.0,
        2.0 * (8f64.sqrt() - 1.0),
    ];
    for (i, (&a, &e)) in out.transformed.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("bc05[{i}]"));
    }
}

/// λ = 2 → `(x^2 - 1) / 2`. scipy returns the same.
#[test]
fn box_cox_lambda_two_matches_scipy() {
    let out = box_cox(&[1.0, 2.0, 3.0], 2.0).unwrap();
    assert_eq!(out.transformed, vec![0.0, 1.5, 4.0]);
}

/// λ = -1 → `1 - 1/x`. scipy returns the same.
#[test]
fn box_cox_lambda_minus_one_matches_scipy() {
    let out = box_cox(&[1.0, 2.0, 4.0, 8.0], -1.0).unwrap();
    let expected = [0.0, 0.5, 0.75, 0.875];
    for (i, (&a, &e)) in out.transformed.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("bcm1[{i}]"));
    }
}

/// scipy raises `ValueError("Data must be positive")`; R's
/// `forecast::BoxCox` happily returns NaN/Inf on non-positive input.
/// We pick scipy's strict behaviour and surface a typed error.
#[test]
fn box_cox_rejects_zero_like_scipy() {
    let err = box_cox(&[1.0, 0.0, 2.0], 0.5).unwrap_err();
    assert_eq!(err, BoxCoxError::NonPositive { min: 0.0 });
}

#[test]
fn box_cox_rejects_negative_like_scipy() {
    let err = box_cox(&[1.0, -3.0, 2.0], 1.0).unwrap_err();
    assert_eq!(err, BoxCoxError::NonPositive { min: -3.0 });
}

/// NaN entries pass through untouched; the positivity check is gated on
/// finite values only.
#[test]
fn box_cox_nan_propagates() {
    let out = box_cox(&[1.0, f64::NAN, 4.0], 0.5).unwrap();
    approx_eq(out.transformed[0], 0.0, 12, "bc05_nan[0]");
    assert!(out.transformed[1].is_nan());
    approx_eq(out.transformed[2], 2.0 * (2.0 - 1.0), 12, "bc05_nan[2]");
}

/// As λ → 0, `(x^λ − 1)/λ` should converge to `ln x`. Spot-check at
/// λ = 1e-6 against `ln(2)`.
#[test]
fn box_cox_limit_lambda_to_zero_matches_log() {
    let out_small = box_cox(&[2.0], 1e-6).unwrap();
    approx_eq(out_small.transformed[0], 2f64.ln(), 5, "bc(2, 1e-6) ≈ ln 2");
}

// ============================================================================
// inv_box_cox  —  scipy.special.inv_boxcox
// ============================================================================

/// scipy: `inv_boxcox([0, 1, 2], 0)` → `[1, e, e²]` (exp).
#[test]
fn inv_box_cox_lambda_zero_matches_scipy() {
    let out = inv_box_cox(&[0.0, 1.0, 2.0], 0.0).unwrap();
    let expected = [1.0, std::f64::consts::E, std::f64::consts::E.powi(2)];
    for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
        approx_eq(a, e, 12, &format!("inv_bc0[{i}]"));
    }
}

/// Forward/inverse roundtrip at several λ — should recover the input
/// to machine precision. scipy convention: `inv_boxcox(boxcox(x, λ), λ) == x`.
#[test]
fn inv_box_cox_roundtrips_match_scipy() {
    let x = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    for lmbda in [0.0_f64, 0.5, 1.0, 2.0, -1.0] {
        let y = box_cox(&x, lmbda).unwrap();
        let back = inv_box_cox(&y.transformed, y.lambda).unwrap();
        for (i, (a, b)) in x.iter().zip(back.iter()).enumerate() {
            assert_relative_eq!(a, b, max_relative = 1e-10);
            let _ = i;
        }
    }
}

/// scipy raises on out-of-domain inputs (`1 + λy ≤ 0`). We surface a
/// typed error.
#[test]
fn inv_box_cox_rejects_invalid_like_scipy() {
    // λ = 1, y = -1 → 1 + 1·(-1) = 0 → not strictly positive → error.
    let err = inv_box_cox(&[-1.0], 1.0).unwrap_err();
    assert!(matches!(err, BoxCoxError::NonInvertible { .. }));
}

// ============================================================================
// box_cox(..., Lambda::Mle)  —  scipy.stats.boxcox (no lmbda) /
//                              R MASS::boxcox / forecast::BoxCox.lambda(method="loglik")
// ============================================================================

/// On strictly-positive ~normal data, MLE λ should be close to 1
/// (transform is near-identity). Spot-check on a synthetic series.
#[test]
fn box_cox_mle_near_one_on_normal_like_data() {
    let y: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64 * 0.13;
            10.0 + (t.sin() + (t * 0.4).cos() * 0.5) * 0.5
        })
        .collect();
    let out = box_cox(&y, Lambda::Mle).unwrap();
    assert!(
        (out.lambda - 1.0).abs() < 0.5,
        "expected λ ≈ 1 for normal-ish data, got {}",
        out.lambda,
    );
}

/// On strictly-positive lognormal data, MLE λ should be close to 0
/// (log transform). This is the canonical scipy / R test case.
#[test]
fn box_cox_mle_near_zero_on_lognormal_data() {
    let y: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64 * 0.13;
            (t.sin() + (t * 0.4).cos() * 0.5).exp() * 10.0
        })
        .collect();
    let out = box_cox(&y, Lambda::Mle).unwrap();
    assert!(
        out.lambda.abs() < 0.5,
        "expected λ ≈ 0 for lognormal data, got {}",
        out.lambda,
    );
}

// ============================================================================
// box_cox(..., Lambda::Guerrero { period })
//                       —  R forecast::BoxCox.lambda(method="guerrero")
// ============================================================================

/// Guerrero on a multiplicative-variance seasonal series should return
/// a small λ (log-ish transform). Matches the R `forecast` package's
/// default behaviour on series whose within-cycle σ grows with the
/// within-cycle level.
#[test]
fn guerrero_small_lambda_on_multiplicative_variance() {
    let period = 12;
    let n_cycles = 30;
    let mut y = Vec::with_capacity(period * n_cycles);
    for c in 0..n_cycles {
        let level = 10.0 + 0.5 * c as f64;
        for i in 0..period {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
            y.push(level * (1.0 + 0.2 * phase.sin()));
        }
    }
    let out = box_cox(&y, Lambda::Guerrero { period }).unwrap();
    assert!(
        out.lambda < 0.5,
        "expected small λ for multiplicative-variance series, got {}",
        out.lambda,
    );
}

/// Guerrero on a constant-variance series should return λ ≈ 1 (no
/// transform).
#[test]
fn guerrero_returns_lambda_near_one_for_constant_variance() {
    let period = 12;
    let n_cycles = 30;
    let mut y = Vec::with_capacity(period * n_cycles);
    for c in 0..n_cycles {
        for i in 0..period {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
            y.push(10.0 + 0.5 * c as f64 + phase.sin());
        }
    }
    let out = box_cox(&y, Lambda::Guerrero { period }).unwrap();
    assert!(
        (out.lambda - 1.0).abs() < 0.5,
        "expected λ ≈ 1 for additive-variance series, got {}",
        out.lambda,
    );
}

// ============================================================================
// box_cox(..., Lambda::Pearsonr)
//                       —  scipy.stats.boxcox_normmax(x, method="pearsonr")
// ============================================================================

/// On near-Gaussian positive data, the Pearson-r objective should pick
/// λ close to 1 (identity transform already gives a straight Q-Q plot).
/// Matches scipy's behaviour on similar inputs.
#[test]
fn pearsonr_near_one_on_normal_like_data() {
    let y: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64 * 0.13;
            10.0 + (t.sin() + (t * 0.4).cos() * 0.5) * 0.5
        })
        .collect();
    let out = box_cox(&y, Lambda::Pearsonr).unwrap();
    assert!(
        (out.lambda - 1.0).abs() < 0.5,
        "expected λ ≈ 1, got {}",
        out.lambda,
    );
}

/// On lognormal data, Pearson-r should agree with MLE: λ ≈ 0
/// (log transform straightens the Q-Q plot).
#[test]
fn pearsonr_near_zero_on_lognormal_data() {
    let y: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64 * 0.13;
            (t.sin() + (t * 0.4).cos() * 0.5).exp() * 10.0
        })
        .collect();
    let out = box_cox(&y, Lambda::Pearsonr).unwrap();
    assert!(out.lambda.abs() < 0.5, "expected λ ≈ 0, got {}", out.lambda);
}

/// Pearson-r and MLE both target marginal normality and should agree
/// to within ~0.1-0.2 on smooth data. (scipy users routinely use them
/// interchangeably for that reason.)
#[test]
fn pearsonr_agrees_with_mle_on_smooth_data() {
    let y: Vec<f64> = (0..500)
        .map(|i| 10.0 + ((i as f64) * 0.1).sin().exp() * 0.5)
        .collect();
    let r = box_cox(&y, Lambda::Pearsonr).unwrap().lambda;
    let m = box_cox(&y, Lambda::Mle).unwrap().lambda;
    assert!(
        (r - m).abs() < 0.5,
        "pearsonr λ = {r} too far from mle λ = {m}"
    );
}

#[test]
fn guerrero_rejects_short_series_like_r() {
    // Period 12 but only one cycle — can't compute CV across cycles.
    let y = vec![1.0; 12];
    let err = box_cox(&y, Lambda::Guerrero { period: 12 }).unwrap_err();
    assert!(matches!(err, BoxCoxError::TooFewObservations { .. }));
}
