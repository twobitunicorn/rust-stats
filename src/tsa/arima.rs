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

mod nelder_mead;
mod ols;
mod transform;

/// Order and configuration for [`arima`].
#[derive(Debug, Clone, Copy)]
pub struct ArimaOpts {
    /// AR order. `0` ≤ p ≤ 10.
    pub p: u32,
    /// Differencing order. `0` ≤ d ≤ 2.
    pub d: u32,
    /// MA order. `0` ≤ q ≤ 10.
    pub q: u32,
    /// Include a constant (intercept) term in the differenced series.
    /// When `d > 0` this is the drift of the integrated process.
    pub include_constant: bool,
}

impl ArimaOpts {
    pub fn new(p: u32, d: u32, q: u32) -> Self {
        Self {
            p,
            d,
            q,
            include_constant: true,
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
    /// Intercept (mean of the differenced series; `0.0` if
    /// `include_constant = false`).
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
    if opts.p > MAX_ORDER || opts.q > MAX_ORDER || opts.d > 2 {
        return Err(ArimaError::InvalidOrder {
            p: opts.p,
            d: opts.d,
            q: opts.q,
        });
    }
    let n = y.len();
    let m = opts.p.max(opts.q) as usize;
    let d = opts.d as usize;
    let min_n = d + m + 2; // need at least a few effective obs
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

    // 1. Difference d times → w of length n - d.
    let w = difference(y, d);
    let n_w = w.len();

    // 2. Optionally subtract mean of w (so the optimiser only sees
    //    centered series). Reattach later as intercept.
    let intercept = if opts.include_constant {
        w.iter().sum::<f64>() / n_w as f64
    } else {
        0.0
    };
    let w_centered: Vec<f64> = w.iter().map(|v| v - intercept).collect();

    let p = opts.p as usize;
    let q = opts.q as usize;

    // 3. Hannan-Rissanen seed: estimate (φ, θ) by two-step OLS.
    let (phi_seed, theta_seed) = hannan_rissanen_seed(&w_centered, p, q)?;

    // 4. Map seed coefficients → unconstrained ℝ^(p+q) starting point
    //    for Nelder-Mead.
    let mut x0 = vec![0.0f64; p + q];
    let phi_pacf = transform::ar_poly_to_pacf(&phi_seed);
    let theta_pacf = transform::ma_poly_to_pacf(&theta_seed);
    for i in 0..p {
        x0[i] = transform::pacf_to_real(phi_pacf[i]);
    }
    for i in 0..q {
        x0[p + i] = transform::pacf_to_real(theta_pacf[i]);
    }

    // 5. Minimise CSS over the unconstrained space.
    let css_obj = |x: &[f64]| -> f64 {
        let (phi, theta) = unpack(x, p, q);
        css_sse(&w_centered, &phi, &theta)
    };
    let (x_star, _f_star, converged) =
        nelder_mead::minimize(&x0, &css_obj, 2_000, 1e-8);
    if !converged {
        return Err(ArimaError::OptimizationFailed { iters: 2_000 });
    }
    let (phi, theta) = unpack(&x_star, p, q);

    // 6. Compute residuals/fitted on the centered, differenced series.
    let eps = compute_eps(&w_centered, &phi, &theta);
    let m_eff = n_w.saturating_sub(m);
    let sse: f64 = eps.iter().skip(m).map(|e| e * e).sum();
    let sigma2 = if m_eff > 0 { sse / m_eff as f64 } else { f64::NAN };

    // 7. Lift fitted / residuals back to original scale (length n).
    //    On the centered scale, fitted_w[t] = w_centered[t] - eps[t].
    //    Add back intercept and integrate d times.
    let fitted_w: Vec<f64> = (0..n_w).map(|t| w_centered[t] - eps[t] + intercept).collect();
    let fitted = integrate(&fitted_w, &y[..d]);
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    // Zero out start-up positions so callers don't read "fitted" from
    // initial conditions.
    let warmup = d + m;
    let mut fitted = fitted;
    let mut residuals = residuals;
    for i in 0..warmup.min(n) {
        fitted[i] = 0.0;
        residuals[i] = 0.0;
    }

    // 8. Information criteria. k = p + q + sigma2 + (1 if intercept).
    let k = (p + q) as f64 + 1.0 + if opts.include_constant { 1.0 } else { 0.0 };
    let log_lik =
        -0.5 * (m_eff as f64) * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);
    let aic = 2.0 * k - 2.0 * log_lik;
    let bic = (m_eff as f64).ln() * k - 2.0 * log_lik;

    // 9. Snapshot tail state for forecasting.
    let take_n = m.max(1);
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
    let last_obs: Vec<f64> = y[n.saturating_sub(d)..].to_vec();

    Ok(ArimaFit {
        phi,
        theta,
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

impl ArimaFit {
    /// Multi-step-ahead point forecasts on the *original* scale.
    ///
    /// Future innovations are taken to be zero (their expectation), so
    /// AR/MA recursion proceeds with `ε_{T+h} = 0` for `h ≥ 1`. For a
    /// model with differencing, the recursion runs on the centered
    /// differenced scale and then re-integrates against the last `d`
    /// observed values.
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        let p = self.opts.p as usize;
        let q = self.opts.q as usize;
        let d = self.opts.d as usize;

        // Forecast the centered differenced series.
        let mut w_tail = self.w_tail.clone();
        let mut eps_tail = self.eps_tail.clone();
        let mut w_forecasts = Vec::with_capacity(steps);

        for _ in 0..steps {
            // AR contribution: sum_{i=1..p} φ_i · w_{t-i}
            let mut wf = 0.0;
            for i in 0..p {
                let idx = w_tail.len() - 1 - i;
                wf += self.phi[i] * w_tail[idx];
            }
            // MA contribution: sum_{i=1..q} θ_i · ε_{t-i}
            for i in 0..q {
                let idx = eps_tail.len() - 1 - i;
                wf += self.theta[i] * eps_tail[idx];
            }
            w_forecasts.push(wf);
            // shift tails
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

        // Re-add intercept and integrate back to the original scale.
        let w_forecasts: Vec<f64> = w_forecasts.iter().map(|v| v + self.intercept).collect();
        integrate_from_last(&w_forecasts, &self.last_obs, d)
    }
}

// ----------------------------------------------------------------------
// Differencing / integration
// ----------------------------------------------------------------------

fn difference(y: &[f64], d: usize) -> Vec<f64> {
    let mut cur: Vec<f64> = y.to_vec();
    for _ in 0..d {
        cur = (1..cur.len()).map(|i| cur[i] - cur[i - 1]).collect();
    }
    cur
}

/// Integrate `w_centered` back to the original scale, given the first
/// `d` observations of the original series as the seed.
fn integrate(w: &[f64], seed: &[f64]) -> Vec<f64> {
    let d = seed.len();
    if d == 0 {
        return w.to_vec();
    }
    // Build the full original-scale series by repeatedly cumulative-summing.
    // After d differences, integrating once each time using the corresponding
    // seed value reconstructs the original.
    let mut cur: Vec<f64> = w.to_vec();
    for k in 0..d {
        // Seed for the k-th integration: the (d-1-k)-th element of `seed`
        // — we integrate the most-differenced result first.
        let seed_val = seed[d - 1 - k];
        let mut next = Vec::with_capacity(cur.len() + 1);
        next.push(seed_val);
        let mut running = seed_val;
        for v in &cur {
            running += v;
            next.push(running);
        }
        cur = next;
    }
    cur
}

/// Integrate forecasts on the differenced scale back to the original
/// scale, using `last_obs` (oldest → newest, length `d`) as the seed.
fn integrate_from_last(w_forecasts: &[f64], last_obs: &[f64], d: usize) -> Vec<f64> {
    if d == 0 {
        return w_forecasts.to_vec();
    }
    let mut cur: Vec<f64> = w_forecasts.to_vec();
    // We need to integrate d times. At each level k (1..=d), the seed is the
    // d-th-difference value just before the forecast starts.
    let mut levels: Vec<Vec<f64>> = Vec::with_capacity(d + 1);
    levels.push(last_obs.to_vec());
    let mut tmp = last_obs.to_vec();
    for _ in 0..d {
        tmp = (1..tmp.len()).map(|i| tmp[i] - tmp[i - 1]).collect();
        levels.push(tmp.clone());
    }
    // levels[k] is the k-th difference of last_obs.
    // Integrate d times: at step k, prepend last value of levels[d-k] and cumsum.
    for k in 0..d {
        let last_val = *levels[d - 1 - k].last().unwrap_or(&0.0);
        let mut next = Vec::with_capacity(cur.len());
        let mut running = last_val;
        for v in &cur {
            running += v;
            next.push(running);
        }
        cur = next;
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

/// Unpack the unconstrained vector `x` into (φ, θ) via the PACF
/// transformation.
fn unpack(x: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let phi_pacf: Vec<f64> = x[..p]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    let theta_pacf: Vec<f64> = x[p..p + q]
        .iter()
        .map(|&v| transform::real_to_pacf(v))
        .collect();
    let phi = transform::pacf_to_ar_poly(&phi_pacf);
    let theta = transform::pacf_to_ma_poly(&theta_pacf);
    (phi, theta)
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
