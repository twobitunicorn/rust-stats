//! Prediction on new observations.

use rust_stats::distributions::t_quantile;
use crate::error::OlsError;
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use crate::{Matrix, Block};
use faer::linalg::triangular_solve;
use faer::Par;

/// Point prediction: ŷ_new = X̃_new · β̂.
pub(crate) fn predict(res: &OlsResults, x_new: Block<'_, f64>) -> Result<Vec<f64>, OlsError> {
    let expected = res.p - usize::from(res.has_intercept);
    if x_new.ncols() != expected {
        return Err(OlsError::NewXShapeMismatch { got: x_new.ncols(), expected });
    }
    let x_aug = build_design_matrix(x_new, res.has_intercept);
    let n = x_aug.nrows();
    let p = res.p;
    Ok((0..n)
        .map(|i| (0..p).map(|j| x_aug[(i, j)] * res.coef[j]).sum())
        .collect())
}

/// Prediction interval (not a confidence interval on the mean).
/// Returns an `n_new × 3` matrix with columns `[fit, lower, upper]`.
pub(crate) fn predict_interval(
    res: &OlsResults,
    x_new: Block<'_, f64>,
    alpha: f64,
) -> Result<Matrix<f64>, OlsError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(OlsError::InvalidAlpha(alpha));
    }
    let expected = res.p - usize::from(res.has_intercept);
    if x_new.ncols() != expected {
        return Err(OlsError::NewXShapeMismatch { got: x_new.ncols(), expected });
    }

    let x_aug = build_design_matrix(x_new, res.has_intercept);
    let n_new = x_aug.nrows();
    let p = res.p;
    let yhat: Vec<f64> = (0..n_new)
        .map(|i| (0..p).map(|j| x_aug[(i, j)] * res.coef[j]).sum())
        .collect();
    let crit = t_quantile(1.0 - alpha / 2.0, res.df_resid() as f64);

    let mut out: Matrix<f64> = Matrix::zeros(n_new, 3);
    for i in 0..n_new {
        // z = pivoted row i of x_aug, as a p×1 column.
        let mut z: Matrix<f64> = Matrix::zeros(p, 1);
        for k in 0..p {
            z[(k, 0)] = x_aug[(i, res.perm[k])];
        }
        // Replace z with R'⁻¹ z. R' is lower-triangular (= R transposed).
        triangular_solve::solve_lower_triangular_in_place(
            res.r_factor.as_ref().transpose(),
            z.as_mut(),
            Par::Seq,
        );
        let quad: f64 = (0..p).map(|k| z[(k, 0)].powi(2)).sum();
        let se_pred = (res.sigma2 * (1.0 + quad)).sqrt();
        out[(i, 0)] = yhat[i];
        out[(i, 1)] = yhat[i] - crit * se_pred;
        out[(i, 2)] = yhat[i] + crit * se_pred;
    }
    Ok(out)
}
