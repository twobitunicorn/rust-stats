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

#[test]
fn aic_bic_finite() {
    let y = simulate_arma(500, &[0.5], &[0.2], 1.0, 0xC1);
    let fit = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    assert!(fit.aic.is_finite(), "aic={}", fit.aic);
    assert!(fit.bic.is_finite(), "bic={}", fit.bic);
    assert!(fit.bic >= fit.aic - 1e-9, "BIC penalises more than AIC");
}
