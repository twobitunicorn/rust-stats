//! Heteroskedasticity-consistent (HC) covariance estimators.

use crate::regression::results::OlsResults;
use crate::Matrix;
use faer::linalg::triangular_solve;
use faer::Par;

/// Compute Cov_HC = (X'X)⁻¹ · M · (X'X)⁻¹  where  M = Σ_i ω_i · x_i · x_i'.
/// `weights[i]` is ω_i.
pub(crate) fn sandwich(res: &OlsResults, weights: &[f64]) -> Matrix<f64> {
    let n = res.n;
    let p = res.p;

    // Build M = X̃' diag(ω) X̃ as Σ ω_i · x_i x_i'.
    let x = &res.x_design;
    let mut m: Matrix<f64> = Matrix::zeros(p, p);
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        for j in 0..p {
            let xij = x[(i, j)];
            for k in 0..p {
                m[(j, k)] += w * xij * x[(i, k)];
            }
        }
    }

    // We need (X̃'X̃)⁻¹ · M · (X̃'X̃)⁻¹. Working in pivoted coordinates:
    //   (X̃'X̃)⁻¹ = P · (R'R)⁻¹ · Pᵀ
    // So permute rows AND cols of M by perm⁻¹: m_p[i,j] = m[perm[i], perm[j]].
    let mut m_p: Matrix<f64> = Matrix::from_fn(p, p, |i, j| m[(res.perm[i], res.perm[j])]);

    // Apply (R'R)⁻¹ on the left of m_p.
    apply_rrt_inverse_on_left(&res.r_factor, &mut m_p);

    // Apply (R'R)⁻¹ on the right of m_p, equivalent to: result' = (R'R)⁻¹ · temp'.
    let mut m_pt: Matrix<f64> = Matrix::from_fn(p, p, |i, j| m_p[(j, i)]);
    apply_rrt_inverse_on_left(&res.r_factor, &mut m_pt);
    let result_p: Matrix<f64> = Matrix::from_fn(p, p, |i, j| m_pt[(j, i)]);

    // Unpermute: out[perm[i], perm[j]] = result_p[i, j].
    let mut out: Matrix<f64> = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            out[(res.perm[i], res.perm[j])] = result_p[(i, j)];
        }
    }
    out
}

/// Apply (R'R)⁻¹ on the left of `rhs` in place: rhs ← (R'R)⁻¹ · rhs.
/// Done as two triangular solves:
///   (1) rhs ← R'⁻¹ · rhs  (R' is lower-triangular)
///   (2) rhs ← R⁻¹ · rhs
fn apply_rrt_inverse_on_left(r: &Matrix<f64>, rhs: &mut Matrix<f64>) {
    triangular_solve::solve_lower_triangular_in_place(
        r.as_ref().transpose(),
        rhs.as_mut(),
        Par::Seq,
    );
    triangular_solve::solve_upper_triangular_in_place(r.as_ref(), rhs.as_mut(), Par::Seq);
}

pub(crate) fn weights_hc0(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| res.residuals[i].powi(2))
        .collect()
}

pub(crate) fn weights_hc1(res: &OlsResults) -> Vec<f64> {
    let scale = res.n as f64 / res.df_resid() as f64;
    (0..res.n)
        .map(|i| res.residuals[i].powi(2) * scale)
        .collect()
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
