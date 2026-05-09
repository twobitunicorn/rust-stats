//! Fitted-model results object.

use faer::{Col, ColRef, Mat};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place,
    solve_upper_triangular_in_place,
};
use once_cell::sync::OnceCell;
use crate::regression::robust::{sandwich, weights_hc0, weights_hc1, weights_hc2, weights_hc3};

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

    /// Classical (X̃'X̃)⁻¹, computed lazily and cached.
    ///
    /// Returns the unscaled covariance — multiply by σ² to get classical Cov(β̂).
    /// The QR factorization is of the permuted design X̃·P, so the directly-computed
    /// (R'R)⁻¹ is in permuted coordinates. We apply the forward permutation to
    /// recover (X̃'X̃)⁻¹: `out[perm[i], perm[j]] = a[i, j]`.
    fn cov_unscaled_inner(&self) -> &Mat<f64> {
        self.cov_unscaled.get_or_init(|| {
            let p = self.p;
            // Start with identity; apply two triangular solves to form (R'R)⁻¹.
            // (1) Solve R' · A = I  ⟹  A = R'⁻¹
            //     R is upper-triangular, so R' is lower-triangular.
            //     Use solve_lower_triangular_in_place on R.transpose().
            // (2) Solve R · A = A  ⟹  A = R⁻¹ · R'⁻¹ = (R'R)⁻¹
            let mut a: Mat<f64> = Mat::identity(p, p);

            // Step (1): solve R' a = I  (R.transpose() is lower-triangular)
            solve_lower_triangular_in_place(
                self.r_factor.as_ref().transpose(),
                a.as_mut(),
                faer::Par::Seq,
            );

            // Step (2): solve R a = a
            solve_upper_triangular_in_place(
                self.r_factor.as_ref(),
                a.as_mut(),
                faer::Par::Seq,
            );

            // a is now (R'R)⁻¹ in pivoted coordinates.
            // Unpermute: out[perm[i], perm[j]] = a[i, j].
            let mut out: Mat<f64> = Mat::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    *out.get_mut(self.perm[i], self.perm[j]) = *a.get(i, j);
                }
            }
            out
        })
    }

    fn classical_std_err_inner(&self) -> &Col<f64> {
        self.std_err_classical.get_or_init(|| {
            let cov = self.cov_unscaled_inner();
            Col::from_fn(self.p, |i| (*cov.get(i, i) * self.sigma2).sqrt())
        })
    }

    /// Standard errors from the classical (non-robust) covariance estimator.
    pub fn std_err(&self) -> ColRef<'_, f64> {
        self.classical_std_err_inner().as_ref()
    }

    /// t-statistics from the classical covariance: β̂ᵢ / SE(β̂ᵢ).
    pub fn t_values(&self) -> Col<f64> {
        let beta = self.coef.as_ref();
        let se = self.classical_std_err_inner();
        Col::from_fn(self.p, |i| *beta.get(i) / *se.get(i))
    }

    /// Two-sided p-values from the classical covariance using the t-distribution
    /// with `df_resid` degrees of freedom.
    pub fn p_values(&self) -> Col<f64> {
        let t = self.t_values();
        let df = self.df_resid() as f64;
        Col::from_fn(self.p, |i| crate::distributions::t_two_sided_pvalue(*t.get(i), df))
    }

    /// Confidence intervals as a (p × 2) matrix with columns `[lower, upper]`,
    /// using classical SE and t-distribution critical values.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the open interval (0, 1).
    /// For a `Result`-returning version, use `conf_int_with` (Task 11).
    pub fn conf_int(&self, alpha: f64) -> Mat<f64> {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1); use conf_int_with for a Result-returning version"
        );
        let crit = crate::distributions::t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64);
        let beta = self.coef.as_ref();
        let se = self.classical_std_err_inner();
        Mat::from_fn(self.p, 2, |i, j| match j {
            0 => *beta.get(i) - crit * *se.get(i),
            _ => *beta.get(i) + crit * *se.get(i),
        })
    }

    /// Compute SE/t/p for the requested covariance estimator.
    pub fn inference(&self, cov: CovType) -> Inference {
        use crate::distributions::{t_two_sided_pvalue};
        let cov_mat = self.cov(cov);
        let beta = self.coef.as_ref();
        let df = self.df_resid() as f64;
        let std_err = Col::from_fn(self.p, |i| (*cov_mat.get(i, i)).sqrt());
        let t_values = Col::from_fn(self.p, |i| *beta.get(i) / *std_err.get(i));
        let p_values = Col::from_fn(self.p, |i| t_two_sided_pvalue(*t_values.get(i), df));
        Inference { std_err, t_values, p_values }
    }

    /// Confidence intervals for the requested covariance, returning Result.
    /// `alpha` must be in `(0, 1)` exclusive.
    pub fn conf_int_with(&self, cov: CovType, alpha: f64) -> Result<Mat<f64>, crate::error::OlsError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(crate::error::OlsError::InvalidAlpha(alpha));
        }
        let inf = self.inference(cov);
        let crit = crate::distributions::t_quantile(1.0 - alpha / 2.0, self.df_resid() as f64);
        let beta = self.coef.as_ref();
        Ok(Mat::from_fn(self.p, 2, |i, j| match j {
            0 => *beta.get(i) - crit * *inf.std_err.get(i),
            _ => *beta.get(i) + crit * *inf.std_err.get(i),
        }))
    }

    /// HC0 sandwich covariance: (X'X)⁻¹ · Σ(eᵢ² xᵢxᵢ') · (X'X)⁻¹.
    pub fn cov_hc0(&self) -> Mat<f64> { sandwich(self, &weights_hc0(self)) }

    /// HC1 sandwich covariance: HC0 scaled by n/(n-p).
    pub fn cov_hc1(&self) -> Mat<f64> { sandwich(self, &weights_hc1(self)) }

    /// HC2 sandwich covariance: eᵢ² divided by (1 − hᵢᵢ).
    pub fn cov_hc2(&self) -> Mat<f64> { sandwich(self, &weights_hc2(self)) }

    /// HC3 sandwich covariance: eᵢ² divided by (1 − hᵢᵢ)².
    pub fn cov_hc3(&self) -> Mat<f64> { sandwich(self, &weights_hc3(self)) }

    /// Coefficient covariance matrix for the requested estimator.
    pub fn cov(&self, cov_type: CovType) -> Mat<f64> {
        match cov_type {
            CovType::NonRobust => {
                let unscaled = self.cov_unscaled_inner();
                Mat::from_fn(self.p, self.p, |i, j| *unscaled.get(i, j) * self.sigma2)
            }
            CovType::HC0 => self.cov_hc0(),
            CovType::HC1 => self.cov_hc1(),
            CovType::HC2 => self.cov_hc2(),
            CovType::HC3 => self.cov_hc3(),
        }
    }
}
