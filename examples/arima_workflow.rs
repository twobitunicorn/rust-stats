//! End-to-end ARIMA workflow: simulate → fit → diagnose → forecast.
//!
//! Shows the full Box-Jenkins loop for a single series:
//!   1. Fit a model (we'll use ARIMA(1, 1, 1) — the most common shape).
//!   2. Check residual diagnostics with Ljung-Box.
//!   3. Multi-step forecast with 95% prediction intervals.
//!   4. Let `auto_arima` pick orders automatically and compare AICc.
//!
//! Run with:
//!
//!   cargo run --release --example arima_workflow

use rust_stats::{
    arima, auto_arima, ArimaMethod, ArimaOpts, AutoArimaOpts,
};

fn main() {
    // ── Simulate ARIMA(1, 1, 1): random walk with AR/MA noise on the
    //    first differences. φ = 0.5, θ = -0.3, drift ≈ 0.1.
    let n = 400usize;
    let mut rng_state = 0xC0DEu64;
    let mut normal = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u1 = (rng_state as f64 / u64::MAX as f64).max(1e-300);
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u2 = rng_state as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let phi = 0.5;
    let theta = -0.3;
    let mut eps = vec![0.0f64; n];
    let mut diff = vec![0.0f64; n];
    for t in 0..n {
        eps[t] = normal();
        let phi_term = if t >= 1 { phi * diff[t - 1] } else { 0.0 };
        let theta_term = if t >= 1 { theta * eps[t - 1] } else { 0.0 };
        diff[t] = 0.1 + phi_term + theta_term + eps[t];
    }
    // Integrate to get y.
    let mut y = vec![100.0f64];
    for d in &diff[..n - 1] {
        let last = *y.last().unwrap();
        y.push(last + d);
    }

    println!("series: n = {}, y[0] = {:.2}, y[n-1] = {:.2}", n, y[0], y[n - 1]);

    // ── 1. Fit ARIMA(1, 1, 1) with the CSS-ML method. ───────────────
    let mut opts = ArimaOpts::new(1, 1, 1);
    opts.method = ArimaMethod::CssMle;
    let fit = arima(&y, opts).unwrap();
    println!("\nARIMA(1, 1, 1) fit:");
    println!("  φ_1   = {:.3}  (truth: 0.5)", fit.phi[0]);
    println!("  θ_1   = {:.3}  (truth: -0.3)", fit.theta[0]);
    println!("  drift = {:.3}  (truth: 0.1)", fit.intercept);
    println!("  σ²    = {:.3}", fit.sigma2);
    println!("  AIC   = {:.2}", fit.aic);
    println!("  AICc  = {:.2}", fit.aicc);
    println!("  BIC   = {:.2}", fit.bic);

    // ── 2. Residual diagnostics. ────────────────────────────────────
    //    Under a well-specified model, residuals should look like
    //    white noise → Ljung-Box p-value should be high (we *fail to
    //    reject* the null of no autocorrelation).
    let lb = fit.ljung_box(20);
    println!("\nLjung-Box residual test (lags = 20):");
    println!("  Q       = {:.2}", lb.q);
    println!("  df      = {}", lb.df);
    println!("  p-value = {:.4}", lb.p_value);
    println!(
        "  → {}",
        if lb.p_value > 0.05 {
            "residuals look like white noise (model OK)"
        } else {
            "residuals show autocorrelation (model under-specified)"
        }
    );

    // ── 3. Forecast 24 steps with 95% prediction intervals. ─────────
    let horizon = 24;
    let f = fit.forecast_with_intervals(horizon, 0.05);
    println!("\nForecast (h = 1..{horizon}, 95% intervals):");
    println!("  h       mean       [    lower,    upper ]   width");
    for h in 0..horizon {
        let width = f.upper[h] - f.lower[h];
        println!(
            "  {:2}   {:8.2}      [{:8.2}, {:8.2}]    {:7.2}",
            h + 1, f.mean[h], f.lower[h], f.upper[h], width,
        );
    }
    println!("(Interval width grows with √h for IMA/ARIMA — expected.)");

    // ── 4. Compare against auto_arima. ──────────────────────────────
    //    Stepwise selection over (p, d, q) using KPSS-driven d and
    //    AICc as the criterion. May or may not land on (1, 1, 1).
    let auto = auto_arima(&y, AutoArimaOpts::new()).unwrap();
    println!("\nauto_arima picked: ARIMA({}, {}, {})  AICc = {:.2}",
        auto.opts.p, auto.opts.d, auto.opts.q, auto.aicc);
    println!("  ours: ARIMA({}, {}, {})              AICc = {:.2}",
        fit.opts.p, fit.opts.d, fit.opts.q, fit.aicc);
    if (auto.aicc - fit.aicc).abs() < 0.5 {
        println!("  → auto and the hand-picked model are essentially indistinguishable.");
    } else if auto.aicc < fit.aicc {
        println!("  → auto found a slightly better model on AICc.");
    } else {
        println!("  → the hand-picked model beats auto on AICc.");
    }
}
