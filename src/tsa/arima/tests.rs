//! Inline tests for the ARIMA implementation.
//!
//! Strategy: simulate noise-free or low-noise series with known
//! parameters and verify the fitted (φ, θ) recover the truth within a
//! reasonable tolerance (CSS is not MLE, and we use a deterministic
//! pseudo-random generator with finite n, so absolute parity with
//! statsmodels is not the bar — *recovery of truth* is).

use super::*;
use crate::error::ArimaError;

/// xorshift64 — matches `examples/bench_transforms.rs` so test series
/// are reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-300);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn simulate_arma(n: usize, phi: &[f64], theta: &[f64], sigma: f64, seed: u64) -> Vec<f64> {
    // Burn-in to escape transients.
    let burn = 200;
    let total = n + burn;
    let mut rng = Rng::new(seed);
    let mut eps = vec![0.0f64; total];
    let mut y = vec![0.0f64; total];
    let p = phi.len();
    let q = theta.len();
    for t in 0..total {
        eps[t] = sigma * rng.normal();
        let mut yt = eps[t];
        for i in 0..p.min(t) {
            yt += phi[i] * y[t - 1 - i];
        }
        for i in 0..q.min(t) {
            yt += theta[i] * eps[t - 1 - i];
        }
        y[t] = yt;
    }
    y[burn..].to_vec()
}

// ----------------------------------------------------------------------
// Differencing / integration
// ----------------------------------------------------------------------

#[test]
fn diff_then_integrate_roundtrips_d1() {
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2) + 0.5 * i as f64).collect();
    let w = difference(&y, 1);
    let back = integrate(&w, &y[..1]);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-10, "y={a}, back={b}");
    }
}

#[test]
fn diff_then_integrate_roundtrips_d2() {
    let y: Vec<f64> = (0..30).map(|i| (i as f64).sin() * 3.0 + 0.1 * i as f64).collect();
    let w = difference(&y, 2);
    let back = integrate(&w, &y[..2]);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-9, "y={a}, back={b}");
    }
}

#[test]
fn diff_d0_is_identity() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(difference(&y, 0), y);
}

// ----------------------------------------------------------------------
// CSS objective sanity
// ----------------------------------------------------------------------

#[test]
fn css_sse_is_zero_for_perfect_ar_recovery() {
    // y_t = 0.5 y_{t-1} + ε_t with ε ≡ 0 → y collapses to its initial.
    // CSS at φ=0.5 should be 0 by construction.
    let w = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    let sse = css_sse(&w, &[0.5], &[]);
    assert!(sse < 1e-15, "expected 0, got {sse}");
}

// ----------------------------------------------------------------------
// Parameter recovery on simulated series
// ----------------------------------------------------------------------

#[test]
fn recovers_ar1_truth() {
    let phi_true = 0.7;
    let y = simulate_arma(2_000, &[phi_true], &[], 1.0, 0xA1);
    let fit = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    assert!((fit.phi[0] - phi_true).abs() < 0.05, "phi={}", fit.phi[0]);
}

#[test]
fn recovers_ar2_truth() {
    let phi_true = [0.6, -0.2];
    let y = simulate_arma(2_000, &phi_true, &[], 1.0, 0xA2);
    let fit = arima(&y, ArimaOpts::new(2, 0, 0)).unwrap();
    assert!((fit.phi[0] - phi_true[0]).abs() < 0.05, "phi[0]={}", fit.phi[0]);
    assert!((fit.phi[1] - phi_true[1]).abs() < 0.05, "phi[1]={}", fit.phi[1]);
}

#[test]
fn recovers_ma1_truth() {
    let theta_true = 0.5;
    let y = simulate_arma(2_000, &[], &[theta_true], 1.0, 0xA3);
    let fit = arima(&y, ArimaOpts::new(0, 0, 1)).unwrap();
    // MA estimation under CSS is biased on shorter samples; tolerance is wider.
    assert!(
        (fit.theta[0] - theta_true).abs() < 0.10,
        "theta={}",
        fit.theta[0]
    );
}

#[test]
fn recovers_arma11_truth() {
    let phi_true = 0.6;
    let theta_true = 0.3;
    let y = simulate_arma(3_000, &[phi_true], &[theta_true], 1.0, 0xA4);
    let fit = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    assert!(
        (fit.phi[0] - phi_true).abs() < 0.10,
        "phi={}",
        fit.phi[0]
    );
    assert!(
        (fit.theta[0] - theta_true).abs() < 0.10,
        "theta={}",
        fit.theta[0]
    );
}

#[test]
fn recovers_random_walk_with_drift() {
    // ARIMA(0, 1, 0): y_t = y_{t-1} + drift + ε_t.
    let mut rng = Rng::new(0xA5);
    let drift = 0.3;
    let mut y = vec![100.0];
    for _ in 1..1_000 {
        let last = *y.last().unwrap();
        y.push(last + drift + rng.normal());
    }
    let fit = arima(&y, ArimaOpts::new(0, 1, 0)).unwrap();
    assert!(
        (fit.intercept - drift).abs() < 0.10,
        "drift={}",
        fit.intercept
    );
}

// ----------------------------------------------------------------------
// Forecasting
// ----------------------------------------------------------------------

#[test]
fn forecast_returns_requested_length() {
    let y = simulate_arma(200, &[0.5], &[], 1.0, 0xF1);
    let fit = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    let f = fit.forecast(10);
    assert_eq!(f.len(), 10);
    for v in &f {
        assert!(v.is_finite(), "non-finite forecast {v}");
    }
}

#[test]
fn ar1_forecast_decays_to_intercept() {
    // For stationary AR(1) with positive φ, multi-step forecast decays
    // geometrically toward the unconditional mean (≈ intercept).
    let y = simulate_arma(500, &[0.7], &[], 1.0, 0xF2);
    let fit = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    let f = fit.forecast(50);
    let near_end = f[f.len() - 1];
    assert!(
        (near_end - fit.intercept).abs() < 0.10,
        "h=50 forecast {near_end} should be near intercept {}",
        fit.intercept
    );
}

#[test]
fn random_walk_forecast_extends_linearly() {
    // ARIMA(0, 1, 0) with drift → forecast is last value + drift · h.
    let mut rng = Rng::new(0xF3);
    let drift = 0.5;
    let mut y = vec![10.0];
    for _ in 1..500 {
        let last = *y.last().unwrap();
        y.push(last + drift + rng.normal());
    }
    let fit = arima(&y, ArimaOpts::new(0, 1, 0)).unwrap();
    let f = fit.forecast(5);
    let y_last = *y.last().unwrap();
    for (h, fc) in f.iter().enumerate() {
        let expected = y_last + fit.intercept * (h + 1) as f64;
        assert!(
            (fc - expected).abs() < 1e-6,
            "h={h}: {fc} vs expected {expected}"
        );
    }
}

// ----------------------------------------------------------------------
// Error cases
// ----------------------------------------------------------------------

#[test]
fn rejects_invalid_order() {
    let y = vec![1.0; 50];
    let err = arima(&y, ArimaOpts::new(11, 0, 0)).unwrap_err();
    assert!(matches!(err, ArimaError::InvalidOrder { .. }));
    let err = arima(&y, ArimaOpts::new(0, 3, 0)).unwrap_err();
    assert!(matches!(err, ArimaError::InvalidOrder { .. }));
}

#[test]
fn rejects_too_short_series() {
    let y = vec![1.0, 2.0, 3.0];
    let err = arima(&y, ArimaOpts::new(2, 0, 2)).unwrap_err();
    assert!(matches!(err, ArimaError::SeriesTooShort { .. }));
}

#[test]
fn rejects_non_finite() {
    let mut y: Vec<f64> = (0..50).map(|i| i as f64).collect();
    y[10] = f64::NAN;
    let err = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap_err();
    assert_eq!(err, ArimaError::NonFinite);
}

// ----------------------------------------------------------------------
// Prediction intervals
// ----------------------------------------------------------------------

#[test]
fn psi_weights_ar1_match_geometric_decay() {
    // For AR(1) with no MA, ψ_k = φ^k.
    let psi = super::psi_weights(&[0.6], &[], 6);
    let expected = [1.0, 0.6, 0.36, 0.216, 0.1296, 0.07776];
    for (a, e) in psi.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-12, "psi: {a} vs {e}");
    }
}

#[test]
fn psi_weights_ma1_truncate_after_q() {
    // For MA(1), ψ_0 = 1, ψ_1 = θ, ψ_k = 0 for k > 1.
    let psi = super::psi_weights(&[], &[0.4], 5);
    assert!((psi[0] - 1.0).abs() < 1e-12);
    assert!((psi[1] - 0.4).abs() < 1e-12);
    for v in psi.iter().skip(2) {
        assert!(v.abs() < 1e-12, "expected 0, got {v}");
    }
}

#[test]
fn integrate_psi_d1_is_cumsum() {
    let psi = vec![1.0, 0.5, 0.25];
    let starred = super::integrate_psi(&psi, 1);
    assert_eq!(starred, vec![1.0, 1.5, 1.75]);
}

#[test]
fn integrate_psi_d2() {
    let psi = vec![1.0, 0.5, 0.25];
    // After d=1: [1.0, 1.5, 1.75]
    // After d=2: [1.0, 2.5, 4.25]
    let starred = super::integrate_psi(&psi, 2);
    assert_eq!(starred, vec![1.0, 2.5, 4.25]);
}

#[test]
fn inv_phi_known_quantiles() {
    // Standard reference values.
    assert!((super::inv_phi(0.975) - 1.959963984540054).abs() < 1e-7);
    assert!((super::inv_phi(0.95) - 1.6448536269514722).abs() < 1e-7);
    assert!((super::inv_phi(0.5) - 0.0).abs() < 1e-9);
    assert!((super::inv_phi(0.025) - (-1.959963984540054)).abs() < 1e-7);
}

#[test]
fn forecast_intervals_have_correct_shape() {
    let y = simulate_arma(500, &[0.5], &[], 1.0, 0xF4);
    let fit = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    let r = fit.forecast_with_intervals(8, 0.05);
    assert_eq!(r.mean.len(), 8);
    assert_eq!(r.variance.len(), 8);
    assert_eq!(r.lower.len(), 8);
    assert_eq!(r.upper.len(), 8);
    // Variance is monotone non-decreasing in horizon.
    for h in 1..8 {
        assert!(
            r.variance[h] >= r.variance[h - 1] - 1e-12,
            "var[{h}] = {} < var[{}]={}",
            r.variance[h],
            h - 1,
            r.variance[h - 1],
        );
    }
    // Intervals bracket the mean.
    for h in 0..8 {
        assert!(r.lower[h] <= r.mean[h], "lower > mean at h={h}");
        assert!(r.upper[h] >= r.mean[h], "upper < mean at h={h}");
    }
}

#[test]
fn forecast_intervals_widen_for_arima_d1() {
    // ARIMA(0, 1, 0) — random walk — forecast variance grows linearly
    // in h (variance at horizon h = σ² · h).
    let mut rng = Rng::new(0xF5);
    let mut y = vec![0.0];
    for _ in 1..500 {
        let last = *y.last().unwrap();
        y.push(last + rng.normal());
    }
    let mut opts = ArimaOpts::new(0, 1, 0);
    opts.include_constant = false;
    let fit = arima(&y, opts).unwrap();
    let r = fit.forecast_with_intervals(10, 0.05);
    // var[h] should be ≈ sigma2 · (h+1).
    for h in 0..10 {
        let expected = fit.sigma2 * (h + 1) as f64;
        let got = r.variance[h];
        assert!(
            (got - expected).abs() / expected < 1e-9,
            "h={h}: var={got} vs expected {expected}",
        );
    }
}

// ----------------------------------------------------------------------
// Exogenous regressors
// ----------------------------------------------------------------------

#[test]
fn arimax_recovers_beta_when_arma_is_negligible() {
    // y_t = 2 + 3·x1_t - 1.5·x2_t + small AR(1) error.
    let n = 400;
    let mut rng = Rng::new(0xE1);
    let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).cos()).collect();
    let mut e = vec![0.0f64; n];
    for t in 1..n {
        e[t] = 0.3 * e[t - 1] + 0.1 * rng.normal();
    }
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 + 3.0 * x1[i] - 1.5 * x2[i] + e[i])
        .collect();
    let exog: Vec<&[f64]> = vec![&x1, &x2];
    let fit = super::arima_with_exog(&y, &exog, ArimaOpts::new(1, 0, 0)).unwrap();
    assert!(
        (fit.intercept - 2.0).abs() < 0.10,
        "intercept = {}, expected 2.0",
        fit.intercept
    );
    assert!(
        (fit.beta[0] - 3.0).abs() < 0.05,
        "beta[0] = {}, expected 3.0",
        fit.beta[0]
    );
    assert!(
        (fit.beta[1] - (-1.5)).abs() < 0.05,
        "beta[1] = {}, expected -1.5",
        fit.beta[1]
    );
    assert!(
        (fit.phi[0] - 0.3).abs() < 0.15,
        "phi = {}, expected ~0.3",
        fit.phi[0]
    );
}

#[test]
fn arimax_no_exog_matches_arima() {
    // Calling arima_with_exog with empty exog should match arima exactly.
    let y = simulate_arma(300, &[0.5], &[0.2], 1.0, 0xE2);
    let fit1 = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    let fit2 = super::arima_with_exog(&y, &[], ArimaOpts::new(1, 0, 1)).unwrap();
    for (a, b) in fit1.phi.iter().zip(fit2.phi.iter()) {
        assert!((a - b).abs() < 1e-12, "phi differs: {a} vs {b}");
    }
    for (a, b) in fit1.theta.iter().zip(fit2.theta.iter()) {
        assert!((a - b).abs() < 1e-12, "theta differs: {a} vs {b}");
    }
}

#[test]
fn arimax_rejects_mismatched_exog_length() {
    let y = vec![1.0; 50];
    let x_short = vec![1.0; 40];
    let exog: Vec<&[f64]> = vec![&x_short];
    let err = super::arima_with_exog(&y, &exog, ArimaOpts::new(1, 0, 0)).unwrap_err();
    assert!(matches!(err, ArimaError::SeriesTooShort { .. }));
}

#[test]
fn arimax_forecast_uses_future_exog() {
    // Pure regression on a known linear x — forecast should follow.
    let n = 300;
    let mut rng = Rng::new(0xE3);
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = (0..n).map(|i| 5.0 + 2.0 * x[i] + 0.3 * rng.normal()).collect();
    let exog: Vec<&[f64]> = vec![&x];
    let fit = super::arima_with_exog(&y, &exog, ArimaOpts::new(1, 0, 0)).unwrap();
    let x_future: Vec<f64> = (n..n + 5).map(|i| i as f64 * 0.1).collect();
    let f = fit.forecast_exog(&[&x_future]);
    assert_eq!(f.len(), 5);
    // forecast should be ≈ 5 + 2 * x_future_t (plus small AR effect).
    for (h, fc) in f.iter().enumerate() {
        let expected = 5.0 + 2.0 * x_future[h];
        assert!(
            (fc - expected).abs() < 1.0,
            "h={h}: forecast {fc} vs expected {expected}"
        );
    }
}

#[test]
fn arimax_forecast_intervals_have_correct_shape() {
    let n = 200;
    let mut rng = Rng::new(0xE4);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 4.0 * x[i] + rng.normal()).collect();
    let exog: Vec<&[f64]> = vec![&x];
    let fit = super::arima_with_exog(&y, &exog, ArimaOpts::new(1, 0, 0)).unwrap();
    let x_future: Vec<f64> = (0..6).map(|i| ((n + i) as f64 * 0.2).sin()).collect();
    let r = fit.forecast_intervals_exog(&[&x_future], 0.05);
    assert_eq!(r.mean.len(), 6);
    assert_eq!(r.lower.len(), 6);
    assert_eq!(r.upper.len(), 6);
    for h in 0..6 {
        assert!(r.lower[h] <= r.mean[h]);
        assert!(r.upper[h] >= r.mean[h]);
    }
}

#[test]
fn aic_bic_finite() {
    let y = simulate_arma(500, &[0.5], &[0.2], 1.0, 0xC1);
    let fit = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    assert!(fit.aic.is_finite(), "aic={}", fit.aic);
    assert!(fit.bic.is_finite(), "bic={}", fit.bic);
    assert!(fit.bic >= fit.aic - 1e-9, "BIC penalises more than AIC");
}
