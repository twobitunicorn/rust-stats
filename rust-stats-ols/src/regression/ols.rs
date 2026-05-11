//! `Ols` builder and the `fit()` entry point.

use crate::error::OlsError;
use crate::regression::design::build_design_matrix;
use crate::regression::results::OlsResults;
use crate::{Matrix, Block};
use faer::prelude::SolveLstsq;
use once_cell::sync::OnceCell;

/// Ordinary least squares model builder.
///
/// Construct with `Ols::new(y, X)`; an intercept column is auto-prepended
/// at fit time unless `without_intercept` is called.
pub struct Ols<'a> {
    pub(crate) y: &'a [f64],
    pub(crate) x: Block<'a, f64>,
    pub(crate) intercept: bool,
}

impl<'a> Ols<'a> {
    pub fn new(y: &'a [f64], x: Block<'a, f64>) -> Self {
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
        let n_y = self.y.len();
        let n_x = self.x.nrows();
        if n_y != n_x {
            return Err(OlsError::DimensionMismatch { y: n_y, x: n_x });
        }

        let n = n_y;
        let p = self.x.ncols() + usize::from(self.intercept);
        if n <= p {
            return Err(OlsError::InsufficientObservations { n, p });
        }

        if !self.y.iter().all(|v| v.is_finite()) {
            return Err(OlsError::NonFinite);
        }
        for j in 0..self.x.ncols() {
            for i in 0..n {
                if !self.x[(i, j)].is_finite() {
                    return Err(OlsError::NonFinite);
                }
            }
        }

        // ----- 2. Build X̃ -----
        let x_design = build_design_matrix(self.x, self.intercept);

        // ----- 3. Column-pivoted QR -----
        let qr = x_design.col_piv_qr();
        let r_view = qr.thin_R();
        let r_diag: Vec<f64> = (0..p).map(|i| r_view[(i, i)]).collect();
        let perm_fwd: Vec<usize> = qr.P().arrays().0.to_vec();

        // ----- 4. Rank detection -----
        // Diagonal entries of R are in (weakly) decreasing absolute value (CPQR property).
        let tol = (n.max(p) as f64) * f64::EPSILON * r_diag[0].abs();
        let rank = r_diag.iter().filter(|&&d| d.abs() > tol).count();

        if rank < p {
            return Err(OlsError::RankDeficient { rank, p });
        }

        // ----- 5. Solve for β̂ -----
        let y_mat: Matrix<f64> = Matrix::from_fn(n, 1, |i, _| self.y[i]);
        let beta_mat = qr.solve_lstsq(y_mat.as_ref());
        let coef: Vec<f64> = (0..p).map(|i| beta_mat[(i, 0)]).collect();

        // ----- 6. Fitted values, residuals, RSS, σ̂², TSS -----
        let fitted: Vec<f64> = (0..n)
            .map(|i| (0..p).map(|j| x_design[(i, j)] * coef[j]).sum())
            .collect();
        let residuals: Vec<f64> = (0..n).map(|i| self.y[i] - fitted[i]).collect();

        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let sigma2: f64 = rss / (n - p) as f64;

        let tss: f64 = if self.intercept {
            let y_mean: f64 = self.y.iter().sum::<f64>() / n as f64;
            self.y.iter().map(|&v| { let d = v - y_mean; d * d }).sum()
        } else {
            self.y.iter().map(|&v| v * v).sum()
        };

        // ----- 7. Leverage h_ii from thin Q -----
        let q_thin = qr.compute_thin_Q();
        let leverage: Vec<f64> = (0..n)
            .map(|i| (0..p).map(|j| { let v = q_thin[(i, j)]; v * v }).sum())
            .collect();

        // ----- 8. Extract R for storage -----
        let r_factor: Matrix<f64> = qr.thin_R().to_owned();

        // ----- 9. Build OlsResults -----
        Ok(OlsResults {
            coef,
            fitted,
            residuals,
            x_design,
            r_factor,
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
