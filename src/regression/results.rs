//! Fitted-model results object.

use faer::{Col, ColRef, Mat};
use once_cell::sync::OnceCell;

/// Owned result of fitting an OLS model. All accessors are read-only.
#[derive(Debug)]
pub struct OlsResults {
    // Eagerly computed by fit():
    pub(crate) coef: Col<f64>,
    pub(crate) fitted: Col<f64>,
    pub(crate) residuals: Col<f64>,
    pub(crate) x_design: Mat<f64>,    // X̃: includes intercept column if has_intercept
    pub(crate) r_factor: Mat<f64>,    // R from pivoted QR (p×p, upper triangular)
    pub(crate) perm: Vec<usize>,      // column permutation
    pub(crate) leverage: Col<f64>,    // h_ii (diag of hat matrix)
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) rank: usize,
    pub(crate) sigma2: f64,
    pub(crate) rss: f64,
    pub(crate) tss: f64,
    pub(crate) has_intercept: bool,
    pub(crate) names: Option<Vec<String>>,

    // Lazy caches (filled in later tasks):
    pub(crate) cov_unscaled: OnceCell<Mat<f64>>,
    pub(crate) std_err_classical: OnceCell<Col<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

#[derive(Debug)]
pub struct Inference {
    pub std_err: Col<f64>,
    pub t_values: Col<f64>,
    pub p_values: Col<f64>,
}

impl OlsResults {
    pub fn coef(&self) -> ColRef<'_, f64> { self.coef.as_ref() }
    pub fn n_obs(&self) -> usize { self.n }
    pub fn df_resid(&self) -> usize { self.n - self.p }
    pub fn df_model(&self) -> usize {
        if self.has_intercept { self.p - 1 } else { self.p }
    }

    pub fn fitted_values(&self) -> ColRef<'_, f64> { self.fitted.as_ref() }
    pub fn residuals(&self) -> ColRef<'_, f64> { self.residuals.as_ref() }
    pub fn sigma(&self) -> f64 { self.sigma2.sqrt() }

    pub fn r_squared(&self) -> f64 {
        if self.tss == 0.0 { 1.0 } else { 1.0 - self.rss / self.tss }
    }

    pub fn adj_r_squared(&self) -> f64 {
        let n = self.n as f64;
        let dfr = self.df_resid() as f64;
        if dfr == 0.0 || self.tss == 0.0 {
            return self.r_squared();
        }
        1.0 - (1.0 - self.r_squared()) * (n - 1.0) / dfr
    }

    pub fn f_statistic(&self) -> f64 {
        let dfm = self.df_model() as f64;
        let dfr = self.df_resid() as f64;
        ((self.tss - self.rss) / dfm) / (self.rss / dfr)
    }

    pub fn f_pvalue(&self) -> f64 {
        crate::distributions::f_sf(
            self.f_statistic(),
            self.df_model() as f64,
            self.df_resid() as f64,
        )
    }
}
