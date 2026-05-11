//! Per-element transforms over `&[f64]`.
//!
//! Standardisation / scaling helpers that consume a slice and produce a
//! `Vec<f64>` of the same length. All aggregates (mean, std, min, max)
//! are computed over the finite entries; `NaN` inputs propagate to the
//! corresponding output positions, but do not contaminate the
//! aggregates themselves.
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

/// Subtract the (finite-entry) mean from every value.
///
/// `NaN` inputs propagate to the same positions in the output. An
/// all-NaN input is treated as having a mean of zero.
pub fn center(y: &[f64]) -> Vec<f64> {
    let mean = finite_mean(y);
    y.iter().map(|&v| v - mean).collect()
}

/// Z-score normalisation: `(x - mean) / std` with sample standard
/// deviation (ddof = 1).
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs (and inputs with fewer than two finite
/// entries) produce an all-zero output at finite positions.
pub fn z_score(y: &[f64]) -> Vec<f64> {
    let mean = finite_mean(y);
    let std = finite_std_ddof1(y, mean);
    if std == 0.0 {
        // Match the centered * 0.0 behaviour: zeros at finite
        // positions, NaN at NaN positions.
        y.iter().map(|&v| (v - mean) * 0.0).collect()
    } else {
        y.iter().map(|&v| (v - mean) / std).collect()
    }
}

/// Min-max rescaling into `[0, 1]`: `(x - min) / (max - min)`.
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs produce an all-zero output at finite
/// positions.
pub fn min_max_scale(y: &[f64]) -> Vec<f64> {
    let (lo, hi) = finite_min_max(y);
    let range = hi - lo;
    if range == 0.0 {
        y.iter().map(|&v| (v - lo) * 0.0).collect()
    } else {
        y.iter().map(|&v| (v - lo) / range).collect()
    }
}

/// Box-Cox power transformation with a fixed `lmbda`:
///
/// ```text
/// (x^λ − 1) / λ   when λ ≠ 0
/// ln(x)           when λ = 0
/// ```
///
/// Requires every finite input to be strictly positive. `NaN` entries
/// propagate to the output unchanged; `+∞` is treated as finite for
/// propagation purposes but never satisfies the positivity check on its
/// own (only finite values gate the check).
pub fn box_cox(y: &[f64], lmbda: f64) -> Result<Vec<f64>, BoxCoxError> {
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
        assert_relative_eq!(out[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.0, max_relative = 1e-12);
    }

    #[test]
    fn box_cox_lmbda_two() {
        // (x^2 - 1) / 2
        let out = box_cox(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert_relative_eq!(out[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.5);
        assert_relative_eq!(out[2], 4.0);
    }

    #[test]
    fn box_cox_rejects_non_positive() {
        let err = box_cox(&[1.0, 0.0, 2.0], 1.0).unwrap_err();
        assert_eq!(err, BoxCoxError::NonPositive { min: 0.0 });
    }

    #[test]
    fn box_cox_propagates_nan() {
        let out = box_cox(&[1.0, f64::NAN, 4.0], 2.0).unwrap();
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_nan());
        assert_relative_eq!(out[2], 7.5);
    }
}
