//! ARIMA(p, d, q) — non-seasonal AutoRegressive Integrated Moving Average.
//!
//! Fits the model
//!
//! ```text
//! (1 - φ₁B - ⋯ - φ_p B^p) · (1 - B)^d · (y_t - μ)
//!     = (1 + θ₁B + ⋯ + θ_q B^q) · ε_t,    ε_t ~ N(0, σ²)
//! ```
//!
//! by minimising the Conditional Sum of Squares (CSS) objective with an
//! in-tree Nelder-Mead optimiser. To enforce stationarity of the AR
//! polynomial and invertibility of the MA polynomial, parameters are
//! reparameterised through partial autocorrelations (Jones 1980): the
//! optimiser sees an unconstrained ℝ^(p+q) space, and the
//! `tanh`-mapped → PACF → polynomial transformation guarantees the
//! resulting φ / θ satisfy stationarity / invertibility by construction.
//!
//! Starting values are seeded by the Hannan-Rissanen two-step OLS
//! procedure: a high-order AR is fitted by OLS to estimate the
//! innovations, then `y` is regressed on its own lags plus the estimated
//! lagged innovations to recover initial (φ, θ).
//!
//! This module is non-seasonal and does not currently accept exogenous
//! regressors.

use crate::error::ArimaError;

mod kalman;
mod nelder_mead;
mod ols;
mod transform;

/// Estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArimaMethod {
    /// Conditional Sum of Squares (default). Fast and robust; uses the
    /// CSS recursion as the objective.
    Css,
    /// Exact Gaussian MLE via Kalman filter. Asymptotically equivalent
    /// to CSS but slightly more efficient at finite n; the cost is one
    /// state-space build + filter run per likelihood evaluation. Matches
    /// the default in `statsmodels.tsa.statespace.SARIMAX`.
    Mle,
    /// CSS for initial values, then MLE refinement (R's
    /// `arima(method = "CSS-ML")` default). Combines CSS's robustness
    /// with MLE's accuracy.
    CssMle,
}

/// Order and configuration for [`arima`].
#[derive(Debug, Clone, Copy)]
pub struct ArimaOpts {
    /// AR order. `0` ≤ p ≤ 10.
    pub p: u32,
    /// Differencing order. `0` ≤ d ≤ 2.
    pub d: u32,
    /// MA order. `0` ≤ q ≤ 10.
    pub q: u32,
    /// Seasonal AR order `P`. `0` ≤ P ≤ 10. Ignored when
    /// `seasonal_period == 0`.
    pub seasonal_p: u32,
    /// Seasonal differencing order `D`. `0` ≤ D ≤ 2.
    pub seasonal_d: u32,
    /// Seasonal MA order `Q`. `0` ≤ Q ≤ 10.
    pub seasonal_q: u32,
    /// Seasonal period `m`. `0` means no seasonal terms (the
    /// non-seasonal ARIMA(p, d, q) model).
    pub seasonal_period: u32,
    /// Include a constant (intercept) term in the differenced series.
    /// When `d > 0` (or `D > 0`) this is the drift of the integrated
    /// process.
    pub include_constant: bool,
    /// Estimation method.
    pub method: ArimaMethod,
}

impl ArimaOpts {
    /// Build a non-seasonal ARIMA(p, d, q) with the default CSS
    /// estimation method.
    pub fn new(p: u32, d: u32, q: u32) -> Self {
        Self {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            seasonal_period: 0,
            include_constant: true,
            method: ArimaMethod::Css,
        }
    }

    /// Build a seasonal ARIMA(p, d, q)(P, D, Q)[m].
    pub fn seasonal(p: u32, d: u32, q: u32, big_p: u32, big_d: u32, big_q: u32, m: u32) -> Self {
        Self {
            p,
            d,
            q,
            seasonal_p: big_p,
            seasonal_d: big_d,
            seasonal_q: big_q,
            seasonal_period: m,
            include_constant: true,
            method: ArimaMethod::Css,
        }
    }
}

/// Fitted ARIMA model.
#[derive(Debug, Clone)]
pub struct ArimaFit {
    /// AR coefficients `[φ₁, …, φ_p]` (length `p`).
    pub phi: Vec<f64>,
    /// MA coefficients `[θ₁, …, θ_q]` (length `q`). Sign convention:
    /// `y_t = … + ε_t + θ₁ ε_{t−1} + …` (matches R `arima` /
    /// statsmodels `SARIMAX`).
    pub theta: Vec<f64>,
    /// Seasonal AR coefficients `[Φ₁, …, Φ_P]` (length `P`). Empty for
    /// non-seasonal fits.
    pub seasonal_phi: Vec<f64>,
    /// Seasonal MA coefficients `[Θ₁, …, Θ_Q]` (length `Q`). Empty for
    /// non-seasonal fits.
    pub seasonal_theta: Vec<f64>,
    /// Exogenous-regressor coefficients (one per `exog` column, in the
    /// order they were passed). Empty when the model was fitted without
    /// exogenous inputs.
    pub beta: Vec<f64>,
    /// Intercept. For an exog-free fit this is the mean of the
    /// differenced series (`0.0` if `include_constant = false`). For an
    /// exog fit it is the OLS intercept of the level regression
    /// (`β_0`); the inner ARIMA is fitted on the residuals with no
    /// additional constant.
    pub intercept: f64,
    /// Innovation variance σ² = SSE / n_effective.
    pub sigma2: f64,
    /// Gaussian conditional log-likelihood at the fitted parameters.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// In-sample fitted values on the *original* (un-differenced) scale,
    /// same length as the input series. The first `d + max(p, q)`
    /// entries are start-up zeros.
    pub fitted: Vec<f64>,
    /// Residuals `y - fitted`, same length as the input series. The
    /// first `d + max(p, q)` entries are start-up zeros (no model output
    /// yet).
    pub residuals: Vec<f64>,
    /// Number of effective observations used in the CSS objective
    /// (`n - d - max(p, q)`).
    pub n_obs: usize,
    /// Options the model was fitted with.
    pub opts: ArimaOpts,

    // Internal state for forecast continuation. Order matches
    // `last_obs`: oldest → newest.
    last_obs: Vec<f64>,    // last `d` raw observations
    w_tail: Vec<f64>,      // last `max(p, q)` differenced observations
    eps_tail: Vec<f64>,    // last `max(p, q)` residuals (zero before start)
}

const MAX_ORDER: u32 = 10;

/// Fit an ARIMA(p, d, q) model by Conditional Sum of Squares.
pub fn arima(y: &[f64], opts: ArimaOpts) -> Result<ArimaFit, ArimaError> {
    arima_with_exog(y, &[], opts)
}

/// Fit an ARIMAX(p, d, q) model by two-stage Conditional Sum of Squares.
///
/// Stage 1: regress `y` on `[1, exog]` by OLS to recover the level
/// intercept `β₀` and exogenous slopes `β`. Stage 2: fit a centered
/// ARIMA(p, d, q) by CSS on the residuals `y − β₀ − exog·β`.
///
/// This is the simpler of the two standard ARIMAX estimators; R's
/// `forecast::Arima(xreg=)` and statsmodels' SARIMAX would jointly
/// maximise the likelihood instead. The two-stage estimator gives
/// consistent (φ, θ, β) but slightly less efficient β when the
/// residual process has strong autocorrelation.
///
/// `exog` is a slice of regressor columns, each of length `y.len()`.
/// Empty `exog` is allowed and equivalent to [`arima`].
pub fn arima_with_exog(
    y: &[f64],
    exog: &[&[f64]],
    opts: ArimaOpts,
) -> Result<ArimaFit, ArimaError> {
    if opts.p > MAX_ORDER || opts.q > MAX_ORDER || opts.d > 2 {
        return Err(ArimaError::InvalidOrder {
            p: opts.p,
            d: opts.d,
            q: opts.q,
        });
    }
    let n = y.len();
    // Validate exog shape.
    for col in exog {
        if col.len() != n {
            return Err(ArimaError::SeriesTooShort {
                n: col.len(),
                min: n,
                p: opts.p,
                d: opts.d,
                q: opts.q,
            });
        }
    }
    // Stage 1: OLS on [1, exog] if any regressors. Compute residuals.
    let k = exog.len();
    let (beta0, beta, residuals) = if k == 0 {
        (0.0, Vec::new(), y.to_vec())
    } else {
        let cols = 1 + k;
        let mut design = vec![0.0f64; n * cols];
        for i in 0..n {
            design[i * cols] = 1.0;
            for j in 0..k {
                design[i * cols + 1 + j] = exog[j][i];
            }
        }
        let coefs = ols::solve(&design, y, n, cols).ok_or(ArimaError::Singular)?;
        let b0 = coefs[0];
        let b: Vec<f64> = coefs[1..].to_vec();
        let mut r = y.to_vec();
        for i in 0..n {
            r[i] -= b0;
            for j in 0..k {
                r[i] -= b[j] * exog[j][i];
            }
        }
        (b0, b, r)
    };

    // Stage 2: ARIMA on residuals. When exog is present we suppress the
    // inner intercept (the level constant was absorbed by `beta0`).
    let inner_opts = ArimaOpts {
        include_constant: if k == 0 { opts.include_constant } else { false },
        ..opts
    };
    let mut fit = arima_no_exog(&residuals, inner_opts)?;

    // Patch with stage-1 outputs: intercept is β₀; β contains the exog
    // slopes; fitted values get the stage-1 part added back.
    //
    // We deliberately keep `fit.last_obs` pointing at the residual's
    // tail — forecasting integrates the inner ARIMA on the *residual*
    // scale, then adds back (β₀ + β·x_future) to get the y-scale
    // forecasts.
    if k > 0 {
        fit.intercept = beta0;
        fit.beta = beta;
        for i in 0..n {
            let mut adj = beta0;
            for j in 0..k {
                adj += fit.beta[j] * exog[j][i];
            }
            // residual_t = y_t - (β₀ + β·x_t). Inner-ARIMA fitted_t is on
            // the residual scale; original-scale fitted_t = β₀ + β·x_t +
            // inner_fitted_t. Skip the warm-up rows the inner zeroed.
            if fit.fitted[i] != 0.0 || fit.residuals[i] != 0.0 {
                fit.fitted[i] += adj;
            }
        }
    }
    Ok(fit)
}

/// Internal: ARIMA fit with no exog. Pulled out so `arima_with_exog`
/// can reuse the body after stage 1. Handles both non-seasonal
/// ARIMA(p, d, q) and seasonal ARIMA(p, d, q)(P, D, Q)[m] uniformly via
/// polynomial convolution.
fn arima_no_exog(y: &[f64], opts: ArimaOpts) -> Result<ArimaFit, ArimaError> {
    let p = opts.p as usize;
    let q = opts.q as usize;
    let d = opts.d as usize;
    let mm = opts.seasonal_period as usize;
    let has_seasonal = mm > 0;
    let big_p = if has_seasonal { opts.seasonal_p as usize } else { 0 };
    let big_d = if has_seasonal { opts.seasonal_d as usize } else { 0 };
    let big_q = if has_seasonal { opts.seasonal_q as usize } else { 0 };

    let n = y.len();
    let ar_order = p + big_p * mm;
    let ma_order = q + big_q * mm;
    let recursion_order = ar_order.max(ma_order);
    let total_diff = d + big_d * mm;
    let min_n = total_diff + recursion_order + 2;
    if n < min_n {
        return Err(ArimaError::SeriesTooShort {
            n,
            min: min_n,
            p: opts.p,
            d: opts.d,
            q: opts.q,
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ArimaError::NonFinite);
    }

    // 1. Full differencing: seasonal first, then non-seasonal.
    let w = full_difference(y, d, big_d, mm);
    let n_w = w.len();

    // 2. Optionally subtract mean of w.
    let intercept = if opts.include_constant {
        w.iter().sum::<f64>() / n_w as f64
    } else {
        0.0
    };
    let w_centered: Vec<f64> = w.iter().map(|v| v - intercept).collect();

    // 3. Hannan-Rissanen seed for the non-seasonal block. Seasonal seeds
    //    start at zero — the optimiser refines them.
    let (phi_seed, theta_seed) = hannan_rissanen_seed(&w_centered, p, q)?;

    // 4. Pack into unconstrained ℝ^(p + P + q + Q) starting vector
    //    [φ_real, Φ_real, θ_real, Θ_real].
    let pn = p + big_p + q + big_q;
    let mut x0 = vec![0.0f64; pn];
    let mut idx = 0;
    let phi_pacf = transform::ar_poly_to_pacf(&phi_seed);
    for v in &phi_pacf {
        x0[idx] = transform::pacf_to_real(*v);
        idx += 1;
    }
    for _ in 0..big_p {
        x0[idx] = 0.0;
        idx += 1;
    }
    let theta_pacf = transform::ma_poly_to_pacf(&theta_seed);
    for v in &theta_pacf {
        x0[idx] = transform::pacf_to_real(*v);
        idx += 1;
    }
    for _ in 0..big_q {
        x0[idx] = 0.0;
        idx += 1;
    }

    // 5. Optimisation. The unconstrained parameter vector is the same
    //    for both objectives — only the objective itself changes.
    let css_obj = |x: &[f64]| -> f64 {
        let (phi, phi_s, theta, theta_s) = unpack_full(x, p, big_p, q, big_q);
        let total_ar = convolve_ar(&phi, &phi_s, mm);
        let total_ma = convolve_ma(&theta, &theta_s, mm);
        css_sse(&w_centered, &total_ar, &total_ma)
    };
    let mle_obj = |x: &[f64]| -> f64 {
        let (phi, phi_s, theta, theta_s) = unpack_full(x, p, big_p, q, big_q);
        let total_ar = convolve_ar(&phi, &phi_s, mm);
        let total_ma = convolve_ma(&theta, &theta_s, mm);
        kalman::concentrated_neg_loglik(&w_centered, &total_ar, &total_ma)
    };
    let max_iter = 2_000 + 200 * pn;
    let (x_star, _f_star, converged) = match opts.method {
        ArimaMethod::Css => nelder_mead::minimize(&x0, &css_obj, max_iter, 1e-8),
        ArimaMethod::Mle => nelder_mead::minimize(&x0, &mle_obj, max_iter, 1e-8),
        ArimaMethod::CssMle => {
            // CSS first, then MLE refinement from CSS solution.
            let (x_css, _, css_ok) =
                nelder_mead::minimize(&x0, &css_obj, max_iter, 1e-8);
            if !css_ok {
                return Err(ArimaError::OptimizationFailed { iters: max_iter });
            }
            nelder_mead::minimize(&x_css, &mle_obj, max_iter, 1e-8)
        }
    };
    if !converged {
        return Err(ArimaError::OptimizationFailed { iters: max_iter });
    }
    let (phi, phi_s, theta, theta_s) = unpack_full(&x_star, p, big_p, q, big_q);

    // 6. Residuals/fitted on the centered, fully-differenced series.
    //    For CSS we use the SSR over the m_eff effective observations;
    //    for MLE / CSS-ML we use the Kalman-concentrated σ̂² (uses all
    //    observations via the exact initial covariance).
    let total_ar = convolve_ar(&phi, &phi_s, mm);
    let total_ma = convolve_ma(&theta, &theta_s, mm);
    let eps = compute_eps(&w_centered, &total_ar, &total_ma);
    let m_eff = n_w.saturating_sub(recursion_order);
    let sigma2 = match opts.method {
        ArimaMethod::Css => {
            let sse: f64 = eps.iter().skip(recursion_order).map(|e| e * e).sum();
            if m_eff > 0 { sse / m_eff as f64 } else { f64::NAN }
        }
        ArimaMethod::Mle | ArimaMethod::CssMle => {
            kalman::concentrated_sigma2(&w_centered, &total_ar, &total_ma)
        }
    };

    // 7. Lift fitted / residuals back to original scale.
    let fitted_w: Vec<f64> = (0..n_w).map(|t| w_centered[t] - eps[t] + intercept).collect();
    let fitted = full_integrate_in_sample(y, &fitted_w, d, big_d, mm);
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    let warmup = total_diff + recursion_order;
    let mut fitted = fitted;
    let mut residuals = residuals;
    for i in 0..warmup.min(n) {
        fitted[i] = 0.0;
        residuals[i] = 0.0;
    }

    // 8. Information criteria.
    let k = pn as f64 + 1.0 + if opts.include_constant { 1.0 } else { 0.0 };
    let log_lik =
        -0.5 * (m_eff as f64) * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);
    let aic = 2.0 * k - 2.0 * log_lik;
    let bic = (m_eff as f64).ln() * k - 2.0 * log_lik;

    // 9. Tail state for forecasting.
    let take_n = recursion_order.max(1);
    let w_tail: Vec<f64> = w_centered
        .iter()
        .rev()
        .take(take_n)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let eps_tail: Vec<f64> = eps
        .iter()
        .rev()
        .take(take_n)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let last_obs: Vec<f64> = y[n.saturating_sub(total_diff)..].to_vec();

    Ok(ArimaFit {
        phi,
        theta,
        seasonal_phi: phi_s,
        seasonal_theta: theta_s,
        beta: Vec::new(),
        intercept,
        sigma2,
        log_likelihood: log_lik,
        aic,
        bic,
        fitted,
        residuals,
        n_obs: m_eff,
        opts,
        last_obs,
        w_tail,
        eps_tail,
    })
}

/// Forecast result including pointwise prediction intervals.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecasts (length = `steps`).
    pub mean: Vec<f64>,
    /// Forecast-error variance per horizon (length = `steps`).
    pub variance: Vec<f64>,
    /// Lower bound of the (1 − α) prediction interval per horizon.
    pub lower: Vec<f64>,
    /// Upper bound of the (1 − α) prediction interval per horizon.
    pub upper: Vec<f64>,
}

impl ArimaFit {
    /// Multi-step-ahead point forecasts on the *original* scale.
    ///
    /// Future innovations are taken to be zero (their expectation), so
    /// AR/MA recursion proceeds with `ε_{T+h} = 0` for `h ≥ 1`. For a
    /// model with differencing, the recursion runs on the centered
    /// differenced scale and then re-integrates against the last `d`
    /// observed values.
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        // No-exog path: equivalent to forecast_exog with an empty matrix.
        self.forecast_exog_impl(steps, &[])
    }

    /// Multi-step forecast for an ARIMAX model. `exog_future` is one
    /// regressor column per stored `beta` slot; each must have length
    /// `steps`.
    pub fn forecast_exog(&self, exog_future: &[&[f64]]) -> Vec<f64> {
        debug_assert_eq!(exog_future.len(), self.beta.len());
        let steps = exog_future.first().map(|c| c.len()).unwrap_or(0);
        self.forecast_exog_impl(steps, exog_future)
    }

    /// Multi-step forecast with prediction intervals for an ARIMAX model.
    pub fn forecast_intervals_exog(&self, exog_future: &[&[f64]], alpha: f64) -> ForecastResult {
        debug_assert_eq!(exog_future.len(), self.beta.len());
        let steps = exog_future.first().map(|c| c.len()).unwrap_or(0);
        let mean = self.forecast_exog_impl(steps, exog_future);
        self.intervals_for(mean, steps, alpha)
    }

    fn forecast_exog_impl(&self, steps: usize, exog_future: &[&[f64]]) -> Vec<f64> {
        let d = self.opts.d as usize;
        let mm = self.opts.seasonal_period as usize;
        let big_d = if mm > 0 { self.opts.seasonal_d as usize } else { 0 };

        // Build the combined AR / MA polynomials at forecast time.
        let total_ar = convolve_ar(&self.phi, &self.seasonal_phi, mm);
        let total_ma = convolve_ma(&self.theta, &self.seasonal_theta, mm);
        let ar_order = total_ar.len();
        let ma_order = total_ma.len();

        let mut w_tail = self.w_tail.clone();
        let mut eps_tail = self.eps_tail.clone();
        let mut w_forecasts = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut wf = 0.0;
            for i in 0..ar_order {
                let idx = w_tail.len() - 1 - i;
                wf += total_ar[i] * w_tail[idx];
            }
            for i in 0..ma_order {
                let idx = eps_tail.len() - 1 - i;
                wf += total_ma[i] * eps_tail[idx];
            }
            w_forecasts.push(wf);
            if !w_tail.is_empty() {
                w_tail.rotate_left(1);
                let last = w_tail.len() - 1;
                w_tail[last] = wf;
            }
            if !eps_tail.is_empty() {
                eps_tail.rotate_left(1);
                let last = eps_tail.len() - 1;
                eps_tail[last] = 0.0;
            }
        }

        // Re-add the differenced-series mean if any. For an ARIMAX fit
        // the inner ARIMA was centered, so its intercept is 0.
        let inner_intercept = if self.beta.is_empty() { self.intercept } else { 0.0 };
        let w_forecasts: Vec<f64> = w_forecasts.iter().map(|v| v + inner_intercept).collect();
        let mut out = full_integrate_from_tail(&w_forecasts, &self.last_obs, d, big_d, mm);

        // ARIMAX: add β₀ + β·x_future per horizon.
        if !self.beta.is_empty() {
            for h in 0..steps {
                let mut adj = self.intercept;
                for (j, col) in exog_future.iter().enumerate() {
                    adj += self.beta[j] * col[h];
                }
                out[h] += adj;
            }
        }
        out
    }

    fn intervals_for(&self, mean: Vec<f64>, steps: usize, alpha: f64) -> ForecastResult {
        let z = inv_phi(1.0 - alpha / 2.0);
        // ψ-weights of the combined (seasonal × non-seasonal) ARMA.
        let mm = self.opts.seasonal_period as usize;
        let total_ar = convolve_ar(&self.phi, &self.seasonal_phi, mm);
        let total_ma = convolve_ma(&self.theta, &self.seasonal_theta, mm);
        let psi = psi_weights(&total_ar, &total_ma, steps);
        // Both differencing levels broaden the prediction error. The
        // integrating filter (1 − B)^{-d} (1 − B^m)^{-D} applied to the
        // ψ-weights produces the running-sum / seasonal-running-sum
        // sequence whose squared entries give Var(h).
        let d = self.opts.d as usize;
        let big_d = if mm > 0 { self.opts.seasonal_d as usize } else { 0 };
        let psi_star = integrate_psi_seasonal(&psi, d, big_d, mm);
        let mut variance = Vec::with_capacity(steps);
        let mut running = 0.0f64;
        for h in 0..steps {
            running += psi_star[h] * psi_star[h];
            variance.push(self.sigma2 * running);
        }
        let lower: Vec<f64> = mean.iter().zip(&variance).map(|(m, v)| m - z * v.sqrt()).collect();
        let upper: Vec<f64> = mean.iter().zip(&variance).map(|(m, v)| m + z * v.sqrt()).collect();
        ForecastResult { mean, variance, lower, upper }
    }

    /// Multi-step forecasts with Gaussian prediction intervals.
    ///
    /// `alpha` is the tail mass (so 0.05 ↦ 95% intervals). The variance
    /// at horizon `h` is `σ² · Σ_{j=0}^{h−1} (ψ*_j)²`, where the
    /// `ψ*` weights come from the infinite-MA representation of the
    /// (possibly differenced) process; for `d > 0` they are
    /// cumulatively-integrated `d` times.
    ///
    /// Intervals assume Gaussian innovations and treat the fitted
    /// parameters as known (i.e., they capture innovation uncertainty
    /// but not parameter uncertainty — same convention as R `predict.Arima`
    /// and statsmodels' default).
    pub fn forecast_with_intervals(&self, steps: usize, alpha: f64) -> ForecastResult {
        let mean = self.forecast(steps);
        self.intervals_for(mean, steps, alpha)
    }
}

// ----------------------------------------------------------------------
// ψ-weights and inverse normal CDF for prediction intervals
// ----------------------------------------------------------------------

/// Infinite-MA representation: ψ_0 = 1; ψ_k = θ_k + Σ_{j=1..min(k,p)} φ_j ψ_{k-j},
/// with θ_k = 0 for k > q. Length = `len`.
fn psi_weights(phi: &[f64], theta: &[f64], len: usize) -> Vec<f64> {
    let p = phi.len();
    let q = theta.len();
    let mut psi = vec![0.0f64; len.max(1)];
    psi[0] = 1.0;
    for k in 1..len {
        let mut v = if k <= q { theta[k - 1] } else { 0.0 };
        let lim = p.min(k);
        for j in 1..=lim {
            v += phi[j - 1] * psi[k - j];
        }
        psi[k] = v;
    }
    psi
}

/// Inverse standard normal CDF, Acklam's algorithm (accuracy ≈ 1.15e-9).
fn inv_phi(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ----------------------------------------------------------------------
// Differencing / integration
// ----------------------------------------------------------------------

/// Apply the full differencing operator `(1 − B)^d (1 − B^m)^D` to `y`.
/// Seasonal differencing runs first, then non-seasonal.
fn full_difference(y: &[f64], d: usize, big_d: usize, m: usize) -> Vec<f64> {
    let mut cur: Vec<f64> = y.to_vec();
    for _ in 0..big_d {
        if cur.len() <= m {
            return Vec::new();
        }
        cur = (m..cur.len()).map(|i| cur[i] - cur[i - m]).collect();
    }
    for _ in 0..d {
        if cur.is_empty() {
            return cur;
        }
        cur = (1..cur.len()).map(|i| cur[i] - cur[i - 1]).collect();
    }
    cur
}

/// Polynomial multiplication of two coefficient vectors.
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// Expand `(1 − B)^d · (1 − B^m)^D` into its coefficient vector
/// (constant term first; total length `1 + d + D·m`).
fn full_diff_polynomial(d: usize, big_d: usize, m: usize) -> Vec<f64> {
    let mut poly = vec![1.0f64];
    let one_minus_b = vec![1.0f64, -1.0];
    for _ in 0..d {
        poly = poly_mul(&poly, &one_minus_b);
    }
    if big_d > 0 && m > 0 {
        let mut one_minus_bm = vec![0.0f64; m + 1];
        one_minus_bm[0] = 1.0;
        one_minus_bm[m] = -1.0;
        for _ in 0..big_d {
            poly = poly_mul(&poly, &one_minus_bm);
        }
    }
    poly
}

/// In-sample integration: reconstruct y from its full differencing.
/// Given the original `y` (length n) and the centered + differenced
/// fitted values `w` (length n - d - D·m), return a fitted-on-y-scale
/// sequence (length n). The first `d + D·m` entries are populated from
/// `y` itself (we have no model for them).
fn full_integrate_in_sample(
    y: &[f64],
    w: &[f64],
    d: usize,
    big_d: usize,
    m: usize,
) -> Vec<f64> {
    let total_diff = d + big_d * m;
    if total_diff == 0 {
        return w.to_vec();
    }
    let n = y.len();
    let p_coeffs = full_diff_polynomial(d, big_d, m);
    let mut out = y[..total_diff].to_vec();
    for i in 0..w.len() {
        // y_t = w_t − Σ_{k=1..total_diff} p_k · y_{t-k}.
        let mut y_t = w[i];
        for k in 1..=total_diff {
            y_t -= p_coeffs[k] * out[out.len() - k];
        }
        out.push(y_t);
    }
    debug_assert_eq!(out.len(), n);
    out
}

/// Forecast-scale integration: take the last `d + D·m` raw observations
/// of `y` as the tail and append the differenced-scale forecasts,
/// inverting the full differencing operator.
fn full_integrate_from_tail(
    w_forecasts: &[f64],
    y_tail: &[f64],
    d: usize,
    big_d: usize,
    m: usize,
) -> Vec<f64> {
    let total_diff = d + big_d * m;
    if total_diff == 0 {
        return w_forecasts.to_vec();
    }
    debug_assert_eq!(y_tail.len(), total_diff);
    let p_coeffs = full_diff_polynomial(d, big_d, m);
    let mut buf = y_tail.to_vec();
    for &w_t in w_forecasts {
        let mut y_t = w_t;
        for k in 1..=total_diff {
            y_t -= p_coeffs[k] * buf[buf.len() - k];
        }
        buf.push(y_t);
    }
    buf[total_diff..].to_vec()
}

/// Combine non-seasonal AR polynomial `phi` and seasonal AR polynomial
/// `seasonal_phi` (coefficients of `B^m`, `B^(2m)`, …, `B^(P·m)`) into
/// one polynomial of length `p + P·m`. Sign convention follows the
/// non-seasonal `phi` (i.e., the combined polynomial is `1 − Σ c_k B^k`).
fn convolve_ar(phi: &[f64], seasonal_phi: &[f64], m: usize) -> Vec<f64> {
    if seasonal_phi.is_empty() {
        return phi.to_vec();
    }
    let p = phi.len();
    let big_p = seasonal_phi.len();
    let total = p + big_p * m;
    let mut out = vec![0.0f64; total];
    for i in 1..=p {
        out[i - 1] += phi[i - 1];
    }
    for j in 1..=big_p {
        out[j * m - 1] += seasonal_phi[j - 1];
    }
    for i in 1..=p {
        for j in 1..=big_p {
            let k = i + j * m;
            out[k - 1] -= phi[i - 1] * seasonal_phi[j - 1];
        }
    }
    out
}

/// Combine non-seasonal MA polynomial `theta` and seasonal MA polynomial
/// `seasonal_theta` (coefficients of `B^m`, `B^(2m)`, …) into one
/// polynomial of length `q + Q·m`. MA sign convention: combined
/// polynomial is `1 + Σ c_k B^k`.
fn convolve_ma(theta: &[f64], seasonal_theta: &[f64], m: usize) -> Vec<f64> {
    if seasonal_theta.is_empty() {
        return theta.to_vec();
    }
    let q = theta.len();
    let big_q = seasonal_theta.len();
    let total = q + big_q * m;
    let mut out = vec![0.0f64; total];
    for i in 1..=q {
        out[i - 1] += theta[i - 1];
    }
    for j in 1..=big_q {
        out[j * m - 1] += seasonal_theta[j - 1];
    }
    for i in 1..=q {
        for j in 1..=big_q {
            let k = i + j * m;
            out[k - 1] += theta[i - 1] * seasonal_theta[j - 1];
        }
    }
    out
}

/// Integrate ψ-weights through both differencing operators. For
/// non-seasonal `d` we cumulatively sum; for seasonal `D` we cumulatively
/// sum with stride `m`.
fn integrate_psi_seasonal(psi: &[f64], d: usize, big_d: usize, m: usize) -> Vec<f64> {
    let mut cur = psi.to_vec();
    for _ in 0..d {
        let mut running = 0.0f64;
        for v in cur.iter_mut() {
            running += *v;
            *v = running;
        }
    }
    if big_d > 0 && m > 0 {
        for _ in 0..big_d {
            // Stride-m running sum: cur[i] += cur[i - m].
            for i in m..cur.len() {
                cur[i] += cur[i - m];
            }
        }
    }
    cur
}

// ----------------------------------------------------------------------
// CSS objective
// ----------------------------------------------------------------------

/// Compute the innovation sequence ε_t for an ARMA(p, q) on the centered
/// series `w`, given (φ, θ). ε_t = 0 for t < max(p, q).
fn compute_eps(w: &[f64], phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let n = w.len();
    let p = phi.len();
    let q = theta.len();
    let m = p.max(q);
    let mut eps = vec![0.0f64; n];
    for t in m..n {
        let mut e = w[t];
        for i in 0..p {
            e -= phi[i] * w[t - 1 - i];
        }
        for i in 0..q {
            e -= theta[i] * eps[t - 1 - i];
        }
        eps[t] = e;
    }
    eps
}

/// Sum of squared innovations — the CSS objective.
fn css_sse(w: &[f64], phi: &[f64], theta: &[f64]) -> f64 {
    let p = phi.len();
    let q = theta.len();
    let m = p.max(q);
    let eps = compute_eps(w, phi, theta);
    eps.iter().skip(m).map(|e| e * e).sum()
}

/// Unpack the unconstrained vector `x` into the four polynomial blocks
/// via PACF transformation. Layout: `[φ_real…, Φ_real…, θ_real…, Θ_real…]`.
fn unpack_full(
    x: &[f64],
    p: usize,
    big_p: usize,
    q: usize,
    big_q: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut idx = 0;
    let phi_pacf: Vec<f64> = x[idx..idx + p]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    idx += p;
    let phi_s_pacf: Vec<f64> = x[idx..idx + big_p]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    idx += big_p;
    let theta_pacf: Vec<f64> = x[idx..idx + q]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    idx += q;
    let theta_s_pacf: Vec<f64> = x[idx..idx + big_q]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    let phi = transform::pacf_to_ar_poly(&phi_pacf);
    let phi_s = transform::pacf_to_ar_poly(&phi_s_pacf);
    let theta = transform::pacf_to_ma_poly(&theta_pacf);
    let theta_s = transform::pacf_to_ma_poly(&theta_s_pacf);
    (phi, phi_s, theta, theta_s)
}

// ----------------------------------------------------------------------
// Hannan-Rissanen starting values
// ----------------------------------------------------------------------

/// Two-step OLS to get an initial (φ, θ) seed for the optimiser.
///
/// Step 1: fit a long AR(k) by OLS (k ≈ max(p+q, log n)), get residuals ε̂.
/// Step 2: regress w_t on w_{t-1..t-p} and ε̂_{t-1..t-q}.
///
/// Returns (φ_seed, θ_seed). If either step's normal-equations matrix is
/// singular, returns zeros (a perfectly safe starting point — the
/// optimiser will still converge from there).
fn hannan_rissanen_seed(w: &[f64], p: usize, q: usize) -> Result<(Vec<f64>, Vec<f64>), ArimaError> {
    if p == 0 && q == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = w.len();
    let pure_ar = q == 0;

    if pure_ar {
        let phi = ols_ar(w, p)?;
        return Ok((phi, Vec::new()));
    }

    // Step 1: high-order AR for innovation estimates.
    let k = (p + q + 2).max(((n as f64).ln() as usize).max(p + q + 1));
    let k = k.min(n / 2).max(p + q + 1);
    if k >= n {
        return Ok((vec![0.0; p], vec![0.0; q]));
    }
    let ar_long = ols_ar(w, k).unwrap_or_else(|_| vec![0.0; k]);

    // Compute residuals from the long AR.
    let mut eps_hat = vec![0.0f64; n];
    for t in k..n {
        let mut e = w[t];
        for i in 0..k {
            e -= ar_long[i] * w[t - 1 - i];
        }
        eps_hat[t] = e;
    }

    // Step 2: regress w_t on [w_{t-1..t-p}, ε̂_{t-1..t-q}].
    let start = k.max(p).max(q);
    if start >= n {
        return Ok((vec![0.0; p], vec![0.0; q]));
    }
    let rows = n - start;
    let cols = p + q;
    let mut x = vec![0.0f64; rows * cols];
    let mut y = vec![0.0f64; rows];
    for (r, t) in (start..n).enumerate() {
        y[r] = w[t];
        for i in 0..p {
            x[r * cols + i] = w[t - 1 - i];
        }
        for i in 0..q {
            x[r * cols + p + i] = eps_hat[t - 1 - i];
        }
    }
    let beta = ols::solve(&x, &y, rows, cols).unwrap_or_else(|| vec![0.0; cols]);
    let mut phi = beta[..p].to_vec();
    let mut theta = beta[p..p + q].to_vec();

    // Clamp seed to stationary / invertible region if needed by lightly
    // shrinking. The optimiser handles fine-tuning.
    shrink_to_stationary(&mut phi);
    shrink_to_stationary(&mut theta);
    Ok((phi, theta))
}

fn ols_ar(w: &[f64], p: usize) -> Result<Vec<f64>, ArimaError> {
    if p == 0 {
        return Ok(Vec::new());
    }
    let n = w.len();
    if n <= p {
        return Ok(vec![0.0; p]);
    }
    let rows = n - p;
    let mut x = vec![0.0f64; rows * p];
    let mut y = vec![0.0f64; rows];
    for (r, t) in (p..n).enumerate() {
        y[r] = w[t];
        for i in 0..p {
            x[r * p + i] = w[t - 1 - i];
        }
    }
    Ok(ols::solve(&x, &y, rows, p).unwrap_or_else(|| vec![0.0; p]))
}

/// If the PACF representation has any |pacf| ≥ 1, shrink the polynomial
/// coefficients uniformly until stable. Cheap and safe.
fn shrink_to_stationary(c: &mut [f64]) {
    if c.is_empty() {
        return;
    }
    let mut iter = 0;
    while iter < 20 {
        let pacf = transform::ar_poly_to_pacf(c);
        if pacf.iter().all(|&p| p.abs() < 0.999) {
            return;
        }
        for v in c.iter_mut() {
            *v *= 0.8;
        }
        iter += 1;
    }
}

#[cfg(test)]
mod tests;
