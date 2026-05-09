//! Prediction on new observations.

use crate::distributions::t_quantile;
use crate::error::OlsError;
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use faer::{Col, Mat, MatRef, Par};

/// Point prediction: ŷ_new = X̃_new · β̂.
pub(crate) fn predict(res: &OlsResults, x_new: MatRef<'_, f64>) -> Result<Col<f64>, OlsError> {
    let expected = res.p - usize::from(res.has_intercept);
    if x_new.ncols() != expected {
        return Err(OlsError::NewXShapeMismatch { got: x_new.ncols(), expected });
    }
    let x_aug = build_design_matrix(x_new, res.has_intercept);
    let yhat: Col<f64> = x_aug.as_ref() * res.coef.as_ref();
    Ok(yhat)
}

/// Prediction interval (not a confidence interval on the mean).
/// Returns an `n_new × 3` matrix with columns `[fit, lower, upper]`.
/// Uses `ŷ ± t · sqrt(σ̂²(1 + xᵀ(X̃'X̃)⁻¹x))`.
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
    let yhat: Col<f64> = x_aug.as_ref() * res.coef.as_ref();
    let crit = t_quantile(1.0 - alpha / 2.0, res.df_resid() as f64);

    let n_new = x_aug.nrows();
    let p = res.p;

    let mut out: Mat<f64> = Mat::zeros(n_new, 3);
    for i in 0..n_new {
        // x_i in original ordering; permute to pivoted coords.
        let mut z_mat: Mat<f64> = Mat::from_fn(p, 1, |k, _| *x_aug.get(i, res.perm[k]));

        // Solve R' · z = z_mat (in pivoted coords) ⇒ z = R'⁻¹ · x_i_pivoted.
        // Then xᵀ (X̃'X̃)⁻¹ x = xᵀ P (R'R)⁻¹ Pᵀ x = ‖z‖²  (since the permutation
        // preserves the inner product when applied to both sides consistently).
        // faer 0.22 doesn't have solve_upper_triangular_transpose_in_place, so use
        // r.transpose() + solve_lower_triangular_in_place (Task 9 pattern).
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            res.r_factor.as_ref().transpose(),
            z_mat.as_mut(),
            Par::Seq,
        );
        let quad: f64 = (0..p).map(|k| (*z_mat.get(k, 0)).powi(2)).sum();
        let se_pred = (res.sigma2 * (1.0 + quad)).sqrt();
        *out.get_mut(i, 0) = *yhat.get(i);
        *out.get_mut(i, 1) = *yhat.get(i) - crit * se_pred;
        *out.get_mut(i, 2) = *yhat.get(i) + crit * se_pred;
    }
    Ok(out)
}
