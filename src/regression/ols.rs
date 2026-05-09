//! `Ols` builder and the `fit()` entry point.

use crate::error::OlsError;
use crate::regression::results::OlsResults;
use faer::{ColRef, MatRef};

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

        // Numerical fit fills in here in Task 6+.
        unimplemented!("numerical fit in Task 6+")
    }
}

fn all_finite_col(c: ColRef<'_, f64>) -> bool {
    c.iter().all(|v| v.is_finite())
}

fn all_finite_mat(m: MatRef<'_, f64>) -> bool {
    m.col_iter().all(|col| col.iter().all(|v| v.is_finite()))
}
