//! Heteroskedasticity-consistent (HC) covariance estimators.

use crate::regression::results::OlsResults;
use faer::{Mat, Par};

/// Compute Cov_HC = (X'X)⁻¹ · M · (X'X)⁻¹  where  M = Σ_i ω_i · x_i · x_i'.
/// `weights[i]` is ω_i.
pub(crate) fn sandwich(res: &OlsResults, weights: &[f64]) -> Mat<f64> {
    let n = res.n;
    let p = res.p;

    // Build M = X̃' diag(ω) X̃ as Σ ω_i · x_i x_i'.
    let x = &res.x_design;
    let mut m: Mat<f64> = Mat::zeros(p, p);
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        for j in 0..p {
            let xij = *x.get(i, j);
            for k in 0..p {
                *m.get_mut(j, k) += w * xij * *x.get(i, k);
            }
        }
    }

    // We need (X̃'X̃)⁻¹ · M · (X̃'X̃)⁻¹. Working in pivoted coordinates:
    //   (X̃'X̃)⁻¹ = P · (R'R)⁻¹ · Pᵀ
    // So permute rows AND cols of M by perm⁻¹: m_p[i,j] = m[perm[i], perm[j]].
    let mut m_p: Mat<f64> = Mat::from_fn(p, p, |i, j| *m.get(res.perm[i], res.perm[j]));

    // Apply (R'R)⁻¹ on the left of m_p: temp = (R'R)⁻¹ · m_p
    apply_rrt_inverse_on_left(&res.r_factor, &mut m_p);

    // Apply (R'R)⁻¹ on the right of m_p: result = temp · (R'R)⁻¹
    // This is equivalent to: result' = (R'R)⁻¹ · temp'
    // So transpose, apply on left, transpose back.
    let mut m_pt: Mat<f64> = Mat::from_fn(p, p, |i, j| *m_p.get(j, i));
    apply_rrt_inverse_on_left(&res.r_factor, &mut m_pt);
    let result_p: Mat<f64> = Mat::from_fn(p, p, |i, j| *m_pt.get(j, i));

    // Unpermute: out[perm[i], perm[j]] = result_p[i, j].
    let mut out: Mat<f64> = Mat::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            *out.get_mut(res.perm[i], res.perm[j]) = *result_p.get(i, j);
        }
    }
    out
}

/// Apply (R'R)⁻¹ on the left of `rhs` in place: rhs ← (R'R)⁻¹ · rhs.
/// Done as two triangular solves:
///   (1) rhs ← R'⁻¹ · rhs  (solve R' · A = rhs, where R' is lower-triangular)
///   (2) rhs ← R⁻¹ · rhs   (solve R · A = rhs)
fn apply_rrt_inverse_on_left(r: &Mat<f64>, rhs: &mut Mat<f64>) {
    // R is upper-triangular; R' is lower-triangular.
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        r.as_ref().transpose(),
        rhs.as_mut(),
        Par::Seq,
    );
    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
        r.as_ref(),
        rhs.as_mut(),
        Par::Seq,
    );
}

pub(crate) fn weights_hc0(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| (*res.residuals.get(i)).powi(2))
        .collect()
}

pub(crate) fn weights_hc1(res: &OlsResults) -> Vec<f64> {
    let scale = res.n as f64 / res.df_resid() as f64;
    (0..res.n)
        .map(|i| (*res.residuals.get(i)).powi(2) * scale)
        .collect()
}

pub(crate) fn weights_hc2(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| (*res.residuals.get(i)).powi(2) / (1.0 - *res.leverage.get(i)))
        .collect()
}

pub(crate) fn weights_hc3(res: &OlsResults) -> Vec<f64> {
    (0..res.n)
        .map(|i| {
            let one_minus_h = 1.0 - *res.leverage.get(i);
            (*res.residuals.get(i)).powi(2) / (one_minus_h * one_minus_h)
        })
        .collect()
}
