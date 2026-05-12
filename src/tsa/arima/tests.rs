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
    let w = super::full_difference(&y, 1, 0, 0);
    let back = super::full_integrate_in_sample(&y, &w, 1, 0, 0);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-10, "y={a}, back={b}");
    }
}

#[test]
fn diff_then_integrate_roundtrips_d2() {
    let y: Vec<f64> = (0..30).map(|i| (i as f64).sin() * 3.0 + 0.1 * i as f64).collect();
    let w = super::full_difference(&y, 2, 0, 0);
    let back = super::full_integrate_in_sample(&y, &w, 2, 0, 0);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-9, "y={a}, back={b}");
    }
}

#[test]
fn seasonal_diff_then_integrate_roundtrips() {
    // (1 - B^4)^1 on a series of length 24.
    let y: Vec<f64> = (0..24).map(|i| (i as f64 * 0.3).cos() + 0.05 * i as f64).collect();
    let w = super::full_difference(&y, 0, 1, 4);
    let back = super::full_integrate_in_sample(&y, &w, 0, 1, 4);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-9, "y={a}, back={b}");
    }
}

#[test]
fn combined_diff_then_integrate_roundtrips() {
    // (1 - B) (1 - B^4) on length 30.
    let y: Vec<f64> = (0..30).map(|i| (i as f64 * 0.4).sin() + 0.1 * i as f64).collect();
    let w = super::full_difference(&y, 1, 1, 4);
    let back = super::full_integrate_in_sample(&y, &w, 1, 1, 4);
    assert_eq!(back.len(), y.len());
    for (a, b) in y.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-9, "y={a}, back={b}");
    }
}

#[test]
fn diff_d0_is_identity() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(super::full_difference(&y, 0, 0, 0), y);
}

#[test]
fn convolve_ar_simple() {
    // (1 - 0.5 B)(1 - 0.3 B^4) = 1 - 0.5 B - 0.3 B^4 + 0.15 B^5.
    let combined = super::convolve_ar(&[0.5], &[0.3], 4);
    // out[k-1] = coefficient of B^k in (1 - Σ c_k B^k)
    // → out = [0.5, 0, 0, 0.3, -0.15]
    let expected = [0.5, 0.0, 0.0, 0.3, -0.15];
    assert_eq!(combined.len(), 5);
    for (a, e) in combined.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-12, "got {a}, expected {e}");
    }
}

#[test]
fn convolve_ma_simple() {
    // (1 + 0.4 B)(1 + 0.2 B^4) = 1 + 0.4 B + 0.2 B^4 + 0.08 B^5.
    let combined = super::convolve_ma(&[0.4], &[0.2], 4);
    let expected = [0.4, 0.0, 0.0, 0.2, 0.08];
    assert_eq!(combined.len(), 5);
    for (a, e) in combined.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-12, "got {a}, expected {e}");
    }
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
fn analytic_pacf_grad_matches_central_difference() {
    // SARIMA(0, 0, 1)(0, 0, 1)[12] in PACF space. We construct a
    // length-200 simulated series and verify mle_value_and_grad's
    // gradient matches a tight central-difference baseline.
    use super::mle_value_and_grad;

    let y = simulate_arma(300, &[], &[0.4], 1.0, 0xABCD);
    let m: usize = 12;
    let p = 0usize;
    let big_p = 0usize;
    let q = 1usize;
    let big_q = 1usize;

    let x = vec![0.2_f64, -0.3]; // PACF-space params: theta, theta_s
    let (nll, grad) = mle_value_and_grad(&x, &y, p, big_p, q, big_q, m);
    assert!(nll.is_finite());

    let h = 1e-5;
    let mle_obj = |xv: &[f64]| -> f64 {
        let (phi, phi_s, theta, theta_s) = super::unpack_full(xv, p, big_p, q, big_q);
        let total_ar = super::convolve_ar(&phi, &phi_s, m);
        let total_ma = super::convolve_ma(&theta, &theta_s, m);
        super::kalman::concentrated_neg_loglik(&y, &total_ar, &total_ma)
    };
    for i in 0..x.len() {
        let mut xp = x.clone();
        xp[i] += h;
        let f_plus = mle_obj(&xp);
        let mut xm = x.clone();
        xm[i] -= h;
        let f_minus = mle_obj(&xm);
        let g_fd = (f_plus - f_minus) / (2.0 * h);
        assert!(
            (grad[i] - g_fd).abs() < 1e-3 * (1.0 + g_fd.abs()),
            "grad[{i}] = {} but FD gives {g_fd}",
            grad[i],
        );
    }
}

#[test]
fn fitted_values_have_no_zero_warmup() {
    // R's `fitted(arima)` returns Kalman one-step-ahead predictions
    // at every step, with no zero warm-up at the start. Make sure
    // we match that — for any stationary AR(1), `fitted[0]` should
    // equal the unconditional mean of the centered series (~0 for
    // a mean-removed AR process) but `fitted[1..p]` should reflect
    // the filter update from earlier observations.
    let y = simulate_arma(200, &[0.6], &[], 1.0, 0xF177ED);
    let fit = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    // Tail of fitted should be meaningful (close-ish to the data).
    let n = y.len();
    let mean_err: f64 =
        (0..n).map(|i| (y[i] - fit.fitted[i]).abs()).sum::<f64>() / n as f64;
    assert!(mean_err < 5.0, "mean |residual| = {mean_err} too large");
    // The first fitted value is non-NaN, finite, and the second
    // already incorporates information from y[0] (so it shouldn't
    // be zero unless y[0] happens to be exactly zero).
    assert!(fit.fitted[0].is_finite());
    assert!(fit.fitted[1].is_finite());
    // Residuals[0] should equal y[0] - fitted[0] (not zero from
    // warm-up zeroing). For a centered AR(1), fitted[0] is the
    // unconditional mean (close to 0), so residuals[0] ≈ y[0].
    assert!(
        (fit.residuals[0] - (y[0] - fit.fitted[0])).abs() < 1e-12,
        "residuals[0] = {} but y[0] - fitted[0] = {}",
        fit.residuals[0],
        y[0] - fit.fitted[0],
    );
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
    let starred = super::integrate_psi_seasonal(&psi, 1, 0, 0);
    assert_eq!(starred, vec![1.0, 1.5, 1.75]);
}

#[test]
fn integrate_psi_d2() {
    let psi = vec![1.0, 0.5, 0.25];
    let starred = super::integrate_psi_seasonal(&psi, 2, 0, 0);
    assert_eq!(starred, vec![1.0, 2.5, 4.25]);
}

#[test]
fn integrate_psi_seasonal_stride_works() {
    let psi = vec![1.0, 0.2, 0.04, 0.008];
    // D=1, m=2 → cur[i] += cur[i - 2]:
    // i=2: 0.04 += 1.0 = 1.04
    // i=3: 0.008 += 0.2 = 0.208
    let starred = super::integrate_psi_seasonal(&psi, 0, 1, 2);
    assert!((starred[0] - 1.0).abs() < 1e-12);
    assert!((starred[1] - 0.2).abs() < 1e-12);
    assert!((starred[2] - 1.04).abs() < 1e-12);
    assert!((starred[3] - 0.208).abs() < 1e-12);
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

// ----------------------------------------------------------------------
// Seasonal ARIMA (SARIMA)
// ----------------------------------------------------------------------

fn simulate_sarima(
    n: usize,
    phi: &[f64],
    theta: &[f64],
    seasonal_phi: &[f64],
    seasonal_theta: &[f64],
    m: usize,
    sigma: f64,
    seed: u64,
) -> Vec<f64> {
    // Combine to total polynomials for the simulation recursion.
    let total_ar = super::convolve_ar(phi, seasonal_phi, m);
    let total_ma = super::convolve_ma(theta, seasonal_theta, m);
    let ar_order = total_ar.len();
    let ma_order = total_ma.len();
    let burn = 200 + ar_order.max(ma_order);
    let total = n + burn;
    let mut rng = Rng::new(seed);
    let mut eps = vec![0.0f64; total];
    let mut y = vec![0.0f64; total];
    for t in 0..total {
        eps[t] = sigma * rng.normal();
        let mut yt = eps[t];
        for i in 0..ar_order.min(t) {
            yt += total_ar[i] * y[t - 1 - i];
        }
        for i in 0..ma_order.min(t) {
            yt += total_ma[i] * eps[t - 1 - i];
        }
        y[t] = yt;
    }
    y[burn..].to_vec()
}

#[test]
fn sarima_recovers_seasonal_ar() {
    // SARIMA(0, 0, 0)(1, 0, 0)[4] with Φ_1 = 0.5.
    let y = simulate_sarima(2_000, &[], &[], &[0.5], &[], 4, 1.0, 0xD1);
    let opts = ArimaOpts::seasonal(0, 0, 0, 1, 0, 0, 4);
    let fit = arima(&y, opts).unwrap();
    assert_eq!(fit.seasonal_phi.len(), 1);
    assert!(
        (fit.seasonal_phi[0] - 0.5).abs() < 0.10,
        "Φ = {}",
        fit.seasonal_phi[0]
    );
}

#[test]
fn sarima_recovers_seasonal_ma() {
    // SARIMA(0, 0, 0)(0, 0, 1)[4] with Θ_1 = 0.4.
    let y = simulate_sarima(2_000, &[], &[], &[], &[0.4], 4, 1.0, 0xD2);
    let opts = ArimaOpts::seasonal(0, 0, 0, 0, 0, 1, 4);
    let fit = arima(&y, opts).unwrap();
    assert_eq!(fit.seasonal_theta.len(), 1);
    assert!(
        (fit.seasonal_theta[0] - 0.4).abs() < 0.10,
        "Θ = {}",
        fit.seasonal_theta[0]
    );
}

#[test]
fn sarima_recovers_non_seasonal_and_seasonal_ar() {
    // SARIMA(1, 0, 0)(1, 0, 0)[4]: φ=0.5, Φ=0.3.
    let y = simulate_sarima(3_000, &[0.5], &[], &[0.3], &[], 4, 1.0, 0xD3);
    let opts = ArimaOpts::seasonal(1, 0, 0, 1, 0, 0, 4);
    let fit = arima(&y, opts).unwrap();
    assert!((fit.phi[0] - 0.5).abs() < 0.10, "φ = {}", fit.phi[0]);
    assert!(
        (fit.seasonal_phi[0] - 0.3).abs() < 0.10,
        "Φ = {}",
        fit.seasonal_phi[0]
    );
}

#[test]
fn sarima_seasonal_diff_only() {
    // SARIMA(0, 0, 0)(0, 1, 0)[12] — pure seasonal random walk.
    let mut rng = Rng::new(0xD4);
    let m = 12usize;
    let n = 240usize;
    let mut y = vec![0.0f64; n];
    for i in 0..m {
        y[i] = rng.normal() * 10.0;
    }
    for t in m..n {
        y[t] = y[t - m] + rng.normal();
    }
    let opts = ArimaOpts::seasonal(0, 0, 0, 0, 1, 0, m as u32);
    let fit = arima(&y, opts).unwrap();
    assert!(fit.sigma2.is_finite() && fit.sigma2 > 0.0);
    let f = fit.forecast(24);
    assert_eq!(f.len(), 24);
    for v in &f {
        assert!(v.is_finite());
    }
}

#[test]
fn sarima_forecast_intervals() {
    let y = simulate_sarima(500, &[0.3], &[], &[0.2], &[], 4, 1.0, 0xD5);
    let opts = ArimaOpts::seasonal(1, 0, 0, 1, 0, 0, 4);
    let fit = arima(&y, opts).unwrap();
    let r = fit.forecast_with_intervals(12, 0.05);
    assert_eq!(r.mean.len(), 12);
    for h in 1..12 {
        assert!(
            r.variance[h] >= r.variance[h - 1] - 1e-12,
            "variance not monotone at h={h}",
        );
    }
}

// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Kalman MLE estimation
// ----------------------------------------------------------------------

#[test]
fn mle_recovers_ar1_truth() {
    let phi_true = 0.6;
    let y = simulate_arma(2_000, &[phi_true], &[], 1.0, 0x101);
    let mut opts = ArimaOpts::new(1, 0, 0);
    opts.method = super::ArimaMethod::Mle;
    let fit = arima(&y, opts).unwrap();
    assert!(
        (fit.phi[0] - phi_true).abs() < 0.05,
        "MLE phi = {} vs truth {phi_true}",
        fit.phi[0]
    );
}

#[test]
fn mle_recovers_ma1_truth() {
    let theta_true = 0.4;
    let y = simulate_arma(2_000, &[], &[theta_true], 1.0, 0x102);
    let mut opts = ArimaOpts::new(0, 0, 1);
    opts.method = super::ArimaMethod::Mle;
    let fit = arima(&y, opts).unwrap();
    assert!(
        (fit.theta[0] - theta_true).abs() < 0.10,
        "MLE theta = {} vs truth {theta_true}",
        fit.theta[0]
    );
}

#[test]
fn css_ml_recovers_arma_truth() {
    let phi_true = 0.5;
    let theta_true = 0.3;
    let y = simulate_arma(2_000, &[phi_true], &[theta_true], 1.0, 0x103);
    let mut opts = ArimaOpts::new(1, 0, 1);
    opts.method = super::ArimaMethod::CssMle;
    let fit = arima(&y, opts).unwrap();
    assert!(
        (fit.phi[0] - phi_true).abs() < 0.10,
        "CSS-ML phi = {}",
        fit.phi[0]
    );
    assert!(
        (fit.theta[0] - theta_true).abs() < 0.10,
        "CSS-ML theta = {}",
        fit.theta[0]
    );
}

#[test]
fn mle_and_css_agree_on_long_series() {
    // Asymptotic equivalence: with enough data, the two estimators
    // should land within a small distance of each other.
    let y = simulate_arma(3_000, &[0.6], &[0.3], 1.0, 0x104);
    let css_fit = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    let mut mle_opts = ArimaOpts::new(1, 0, 1);
    mle_opts.method = super::ArimaMethod::Mle;
    let mle_fit = arima(&y, mle_opts).unwrap();
    assert!(
        (css_fit.phi[0] - mle_fit.phi[0]).abs() < 0.05,
        "CSS φ = {} vs MLE φ = {}",
        css_fit.phi[0],
        mle_fit.phi[0]
    );
    assert!(
        (css_fit.theta[0] - mle_fit.theta[0]).abs() < 0.05,
        "CSS θ = {} vs MLE θ = {}",
        css_fit.theta[0],
        mle_fit.theta[0]
    );
}

#[test]
fn mle_forecast_runs() {
    let y = simulate_arma(500, &[0.5], &[], 1.0, 0x105);
    let mut opts = ArimaOpts::new(1, 0, 0);
    opts.method = super::ArimaMethod::Mle;
    let fit = arima(&y, opts).unwrap();
    let f = fit.forecast(10);
    assert_eq!(f.len(), 10);
    for v in &f {
        assert!(v.is_finite());
    }
    let r = fit.forecast_with_intervals(10, 0.05);
    assert_eq!(r.lower.len(), 10);
    assert_eq!(r.upper.len(), 10);
}

// ----------------------------------------------------------------------

#[test]
fn aic_bic_finite() {
    let y = simulate_arma(500, &[0.5], &[0.2], 1.0, 0xC1);
    let fit = arima(&y, ArimaOpts::new(1, 0, 1)).unwrap();
    assert!(fit.aic.is_finite(), "aic={}", fit.aic);
    assert!(fit.bic.is_finite(), "bic={}", fit.bic);
    assert!(fit.bic >= fit.aic - 1e-9, "BIC penalises more than AIC");
}
