//! `Ols` builder and the `fit()` entry point.

use crate::error::OlsError;
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use faer::prelude::SolveLstsq;
use faer::{Col, ColRef, Mat, MatRef};
use once_cell::sync::OnceCell;

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
        // ----- 1. Validation -----
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

        // ----- 2. Build X̃ -----
        let x_design = build_design_matrix(self.x, self.intercept);

        // ----- 3. Column-pivoted QR -----
        let qr = x_design.col_piv_qr();

        // Extract R (p×p upper triangular). Since n > p, thin_R() is p×p.
        let r_mat: Mat<f64> = qr.thin_R().to_owned();

        // Extract forward permutation: perm[i] = which column of original X̃
        // is at position i of the permuted matrix (X̃·P^T = Q·R).
        let perm_fwd: Vec<usize> = qr.P().arrays().0.to_vec();

        // ----- 4. Rank detection -----
        // Diagonal entries of R are in decreasing absolute value (CPQR property).
        let tol = (n.max(p) as f64) * f64::EPSILON * (*r_mat.get(0, 0)).abs();
        let rank = (0..p).filter(|&i| (*r_mat.get(i, i)).abs() > tol).count();

        if rank < p {
            return Err(OlsError::RankDeficient { rank, p });
        }

        // ----- 5. Solve for β̂ -----
        // qr.solve_lstsq(y) solves min||X̃β - y||² and returns β̂ of length p,
        // already un-permuted to match original column order of X̃.
        // We pass y as a column matrix (n×1) and get a p×1 result.
        let y_mat: Mat<f64> = Mat::from_fn(n, 1, |i, _| *self.y.get(i));
        let beta_mat = qr.solve_lstsq(y_mat.as_ref());
        // beta_mat is p×1
        let beta: Col<f64> = Col::from_fn(p, |i| *beta_mat.get(i, 0));

        // ----- 6. Fitted values, residuals, RSS, σ̂², TSS -----
        // fitted = X̃ * β̂
        let fitted: Col<f64> = {
            let f = x_design.as_ref() * beta.as_ref();
            f
        };

        let residuals: Col<f64> = {
            let mut r = Col::zeros(n);
            for i in 0..n {
                *r.get_mut(i) = *self.y.get(i) - *fitted.get(i);
            }
            r
        };

        let rss: f64 = residuals.as_ref().squared_norm_l2();
        let sigma2: f64 = rss / (n - p) as f64;

        let tss: f64 = if self.intercept {
            let y_mean: f64 = (0..n).map(|i| *self.y.get(i)).sum::<f64>() / n as f64;
            (0..n).map(|i| { let d = *self.y.get(i) - y_mean; d * d }).sum()
        } else {
            (0..n).map(|i| { let v = *self.y.get(i); v * v }).sum()
        };

        // ----- 7. Leverage h_ii from thin Q -----
        // h_ii = sum_j Q_thin[i,j]^2 (row squared norms of the thin Q)
        let q_thin: Mat<f64> = qr.compute_thin_Q();
        let leverage: Col<f64> = Col::from_fn(n, |i| {
            (0..p).map(|j| { let v = *q_thin.get(i, j); v * v }).sum()
        });

        // ----- 8. Build OlsResults -----
        Ok(OlsResults {
            coef: beta,
            fitted,
            residuals,
            x_design,
            r_factor: r_mat,
            perm: perm_fwd,
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

fn all_finite_col(c: ColRef<'_, f64>) -> bool {
    c.iter().all(|v| v.is_finite())
}

fn all_finite_mat(m: MatRef<'_, f64>) -> bool {
    m.col_iter().all(|col| col.iter().all(|v| v.is_finite()))
}
