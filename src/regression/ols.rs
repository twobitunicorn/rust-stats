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
        // Filled in by Task 5 onward.
        unimplemented!("fit() implemented in Task 5+")
    }
}
