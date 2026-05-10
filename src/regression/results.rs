//! Fitted-model results object.

use crate::regression::predict::{predict as predict_impl, predict_interval as predict_interval_impl};
use crate::regression::robust::{sandwich, weights_hc0, weights_hc1, weights_hc2, weights_hc3};
use crate::{Matrix, Block};
use faer::linalg::triangular_solve;
use faer::Par;
use once_cell::sync::OnceCell;

/// Owned result of fitting an OLS model. All accessors are read-only.
pub struct OlsResults {
    // Eagerly computed by fit():
    pub(crate) coef: Vec<f64>,
    pub(crate) fitted: Vec<f64>,
    pub(crate) residuals: Vec<f64>,
    pub(crate) x_design: Matrix<f64>,    // X̃: includes intercept column if has_intercept
    pub(crate) r_factor: Matrix<f64>,    // R from pivoted QR (p×p, upper triangular)
    pub(crate) perm: Vec<usize>,      // column permutation
    pub(crate) leverage: Vec<f64>,    // h_ii (diag of hat matrix)
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) rank: usize,
    pub(crate) sigma2: f64,
    pub(crate) rss: f64,
    pub(crate) tss: f64,
    pub(crate) has_intercept: bool,
    pub(crate) names: Option<Vec<String>>,

    // Lazy caches:
    pub(crate) cov_unscaled: OnceCell<Matrix<f64>>,
    pub(crate) std_err_classical: OnceCell<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
}

#[derive(Debug, Clone)]
pub struct Inference {
    pub std_err: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
}

impl OlsResults {
    pub fn coef(&self) -> &[f64] { &self.coef }
    pub fn n_obs(&self) -> usize { self.n }
    pub fn df_resid(&self) -> usize { self.n - self.p }
    pub fn df_model(&self) -> usize {
        if self.has_intercept { self.p - 1 } else { self.p }
    }

    pub fn fitted_values(&self) -> &[f64] { &self.fitted }
    pub fn residuals(&self) -> &[f64] { &self.residuals }
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

    /// Classical (X̃'X̃)⁻¹, computed lazily and cached.
    fn cov_unscaled_inner(&self) -> &Matrix<f64> {
        self.cov_unscaled.get_or_init(|| {
            let p = self.p;
            let mut a: Matrix<f64> = Matrix::identity(p, p);

            // (1) R' a = I  (R' is lower-triangular = R transposed)
            triangular_solve::solve_lower_triangular_in_place(
                self.r_factor.as_ref().transpose(),
                a.as_mut(),
                Par::Seq,
            );
            // (2) R a = a
            triangular_solve::solve_upper_triangular_in_place(
                self.r_factor.as_ref(),
                a.as_mut(),
                Par::Seq,
            );

            // a is now (R'R)⁻¹ in pivoted coordinates. Unpermute.
            let mut out: Matrix<f64> = Matrix::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    out[(self.perm[i], self.perm[j])] = a[(i, j)];
                }
            }
            out
        })
    }

    fn classical_std_err_inner(&self) -> &[f64] {
        self.std_err_classical
            .get_or_init(|| {
                let cov = self.cov_unscaled_inner();
                (0..self.p)
                    .map(|i| (cov[(i, i)] * self.sigma2).sqrt())
                    .collect()
            })
            .as_slice()
    }

    /// Standard errors from the classical (non-robust) covariance estimator.
    pub fn std_err(&self) -> &[f64] {
        self.classical_std_err_inner()
    }

    /// t-statistics from the classical covariance: β̂ᵢ / SE(β̂ᵢ).
    pub fn t_values(&self) -> Vec<f64> {
        let se = self.classical_std_err_inner();
        (0..self.p).map(|i| self.coef[i] / se[i]).collect()
    }

    /// Two-sided p-values from the classical covariance using the t-distribution.
    pub fn p_values(&self) -> Vec<f64> {
        let t = self.t_values();
        let df = self.df_resid() as f64;
        t.iter()
            .map(|&ti| crate::distributions::t_two_sided_pvalue(ti, df))
            .collect()
    }

    /// Confidence intervals as a (p × 2) matrix with columns `[lower, upper]`,
    /// using classical SE and t-distribution critical values.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the open interval (0, 1).
    pub fn conf_int(&self, alpha: f64) -> Matrix<f64> {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1); use conf_int_with for a Result-returning version"
        );
        let crit = crate::distributions::t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64);
        let se = self.classical_std_err_inner();
        Matrix::from_fn(self.p, 2, |i, j| match j {
            0 => self.coef[i] - crit * se[i],
            _ => self.coef[i] + crit * se[i],
        })
    }

    /// Compute SE/t/p for the requested covariance estimator.
    pub fn inference(&self, cov: CovType) -> Inference {
        use crate::distributions::{t_two_sided_pvalue, z_two_sided_pvalue};
        let cov_mat = self.cov_internal(cov);
        let df = self.df_resid() as f64;
        let std_err: Vec<f64> = (0..self.p)
            .map(|i| cov_mat[(i, i)].sqrt())
            .collect();
        let t_values: Vec<f64> = (0..self.p).map(|i| self.coef[i] / std_err[i]).collect();
        let p_values: Vec<f64> = match cov {
            CovType::NonRobust => t_values.iter().map(|&t| t_two_sided_pvalue(t, df)).collect(),
            CovType::HC0 | CovType::HC1 | CovType::HC2 | CovType::HC3 => {
                t_values.iter().map(|&t| z_two_sided_pvalue(t)).collect()
            }
        };
        Inference {
            std_err,
            t_values,
            p_values,
        }
    }

    /// Confidence intervals for the requested covariance, returning Result.
    pub fn conf_int_with(&self, cov: CovType, alpha: f64) -> Result<Matrix<f64>, crate::error::OlsError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(crate::error::OlsError::InvalidAlpha(alpha));
        }
        let inf = self.inference(cov);
        let crit = match cov {
            CovType::NonRobust => {
                crate::distributions::t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64)
            }
            CovType::HC0 | CovType::HC1 | CovType::HC2 | CovType::HC3 => {
                crate::distributions::z_quantile(1.0 - alpha / 2.0)
            }
        };
        Ok(Matrix::from_fn(self.p, 2, |i, j| match j {
            0 => self.coef[i] - crit * inf.std_err[i],
            _ => self.coef[i] + crit * inf.std_err[i],
        }))
    }

    /// HC0 sandwich covariance.
    pub fn cov_hc0(&self) -> Matrix<f64> { sandwich(self, &weights_hc0(self)) }
    pub fn cov_hc1(&self) -> Matrix<f64> { sandwich(self, &weights_hc1(self)) }
    pub fn cov_hc2(&self) -> Matrix<f64> { sandwich(self, &weights_hc2(self)) }
    pub fn cov_hc3(&self) -> Matrix<f64> { sandwich(self, &weights_hc3(self)) }

    /// Coefficient covariance matrix for the requested estimator.
    pub fn cov(&self, cov_type: CovType) -> Matrix<f64> {
        self.cov_internal(cov_type)
    }

    /// Internal helper: same as [`cov`] but kept private so the lazy
    /// `cov_unscaled` cache stays inside this module.
    fn cov_internal(&self, cov_type: CovType) -> Matrix<f64> {
        match cov_type {
            CovType::NonRobust => {
                let unscaled = self.cov_unscaled_inner();
                Matrix::from_fn(self.p, self.p, |i, j| unscaled[(i, j)] * self.sigma2)
            }
            CovType::HC0 => sandwich(self, &weights_hc0(self)),
            CovType::HC1 => sandwich(self, &weights_hc1(self)),
            CovType::HC2 => sandwich(self, &weights_hc2(self)),
            CovType::HC3 => sandwich(self, &weights_hc3(self)),
        }
    }

    /// Point prediction: ŷ_new = X̃_new · β̂.
    pub fn predict(&self, x_new: Block<'_, f64>) -> Result<Vec<f64>, crate::error::OlsError> {
        predict_impl(self, x_new)
    }

    /// Prediction intervals: returns an `n_new × 3` matrix with columns `[fit, lower, upper]`.
    pub fn predict_interval(
        &self,
        x_new: Block<'_, f64>,
        alpha: f64,
    ) -> Result<Matrix<f64>, crate::error::OlsError> {
        predict_interval_impl(self, x_new, alpha)
    }

    /// Set human-readable names for each coefficient (length must equal `p`).
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        assert!(names.len() == self.p,
            "names length {} != p {}", names.len(), self.p);
        self.names = Some(names);
        self
    }

    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }

    pub fn has_intercept(&self) -> bool { self.has_intercept }

    pub fn summary(&self) -> String {
        crate::regression::summary::render(self, CovType::NonRobust)
    }

    pub fn summary_with(&self, cov: CovType) -> String {
        crate::regression::summary::render(self, cov)
    }
}

impl std::fmt::Display for OlsResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary())
    }
}

impl std::fmt::Debug for OlsResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OlsResults")
            .field("n", &self.n)
            .field("p", &self.p)
            .field("rank", &self.rank)
            .field("has_intercept", &self.has_intercept)
            .field("names", &self.names)
            .finish_non_exhaustive()
    }
}
