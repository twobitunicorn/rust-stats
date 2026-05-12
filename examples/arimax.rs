//! ARIMAX example: forecasting with exogenous regressors.
//!
//! The setting: weekly retail sales driven by
//!
//!   y_t = β₀ + β₁ · holiday_t + β₂ · promotion_t + AR(1) residual.
//!
//! `holiday_t` is a 0/1 dummy that flags national holiday weeks;
//! `promotion_t` is a 0/1 dummy for the company's planned discount
//! weeks. Both are *known in advance* — the canonical "future-known
//! regressors" case where ARIMAX shines.
//!
//! We:
//!   1. Simulate two years of weekly history with known coefficients.
//!   2. Fit `arima_with_exog` and check that β₀, β₁, β₂, φ are recovered.
//!   3. Forecast the next 12 weeks given a planned promotion schedule.
//!   4. Compare to the same series fitted *without* exog — naive ARIMA
//!      doesn't know about the holiday/promotion bumps, so its
//!      forecast intervals must be wider.
//!
//! Estimation: joint MLE over (β₀, β, φ, θ) — the same approach R's
//! `arima(xreg=)` and statsmodels' SARIMAX take.
//!
//! Run with:
//!
//!   cargo run --release --example arimax

use rust_stats::{arima, arima_with_exog, ArimaOpts};

fn main() {
    // ── 1. Simulate two years of weekly history. ────────────────────
    let n = 2 * 52usize; // 104 weeks
    let mut rng_state = 0xA12A_F00Du64;
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

    // Truth: β₀ = 100, β_holiday = 35, β_promo = 12, AR(1) φ = 0.4, σ = 3.
    let beta0_true = 100.0;
    let beta_holiday_true = 35.0;
    let beta_promo_true = 12.0;
    let phi_true = 0.4;
    let sigma_true = 3.0;

    // National-holiday weeks: assume one each at weeks 22 (mid-summer)
    // and 47 (Thanksgiving / Black Friday equivalent) in each year.
    let holiday: Vec<f64> = (0..n)
        .map(|i| {
            let w = i % 52;
            if w == 22 || w == 47 { 1.0 } else { 0.0 }
        })
        .collect();

    // Marketing-team promotions: weeks 10, 35, every year.
    let promo: Vec<f64> = (0..n)
        .map(|i| {
            let w = i % 52;
            if w == 10 || w == 35 { 1.0 } else { 0.0 }
        })
        .collect();

    let mut eps = vec![0.0f64; n];
    let mut residual = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for t in 0..n {
        eps[t] = sigma_true * normal();
        if t > 0 {
            residual[t] = phi_true * residual[t - 1] + eps[t];
        } else {
            residual[t] = eps[t];
        }
        y[t] = beta0_true
            + beta_holiday_true * holiday[t]
            + beta_promo_true * promo[t]
            + residual[t];
    }

    println!(
        "series: {} weekly observations, {} holiday weeks, {} promo weeks",
        n,
        holiday.iter().filter(|&&v| v == 1.0).count(),
        promo.iter().filter(|&&v| v == 1.0).count(),
    );

    // ── 2. Fit ARIMAX. The exog matrix is two columns of length n. ──
    let exog: &[&[f64]] = &[&holiday, &promo];
    let fit = arima_with_exog(&y, exog, ArimaOpts::new(1, 0, 0)).unwrap();

    println!("\nFitted ARIMAX(1, 0, 0):");
    println!("  β₀ (intercept)  = {:7.3}     (truth: {:7.3})", fit.intercept, beta0_true);
    println!("  β  (holiday)    = {:7.3}     (truth: {:7.3})", fit.beta[0], beta_holiday_true);
    println!("  β  (promotion)  = {:7.3}     (truth: {:7.3})", fit.beta[1], beta_promo_true);
    println!("  φ_1 (AR)        = {:7.3}     (truth: {:7.3})", fit.phi[0], phi_true);
    println!("  σ²              = {:7.3}     (truth: {:7.3})", fit.sigma2, sigma_true.powi(2));

    // ── 3. Forecast the next 12 weeks. We need future X values: ──
    //    suppose marketing plans a promotion in week +3, and a known
    //    holiday in week +8.
    let h = 12usize;
    let future_holiday: Vec<f64> = (0..h).map(|i| if i == 8 { 1.0 } else { 0.0 }).collect();
    let future_promo: Vec<f64>   = (0..h).map(|i| if i == 3 { 1.0 } else { 0.0 }).collect();
    let f = fit.forecast_intervals_exog(&[&future_holiday, &future_promo], 0.05);

    println!("\nForecast (12 weeks ahead, 95% intervals):");
    println!("  h    hol  promo      mean     [    lower,    upper ]");
    for h_idx in 0..h {
        let hol = future_holiday[h_idx] as i32;
        let pro = future_promo[h_idx] as i32;
        println!(
            "  {:2}    {}    {}     {:7.2}     [{:7.2},  {:7.2}]",
            h_idx + 1, hol, pro,
            f.mean[h_idx], f.lower[h_idx], f.upper[h_idx],
        );
    }
    println!("(Note the bumps at h=4 (promo) and h=9 (holiday) — the model");
    println!(" predicts the level shift cleanly because we tell it the X.)");

    // ── 4. Naive ARIMA without exog: it sees the same y but no X. ───
    //    The holiday/promo effects get absorbed into the residual,
    //    inflating σ² and producing much wider prediction intervals.
    let naive = arima(&y, ArimaOpts::new(1, 0, 0)).unwrap();
    let f_naive = naive.forecast_with_intervals(h, 0.05);

    let arimax_width = f.upper[5] - f.lower[5];
    let naive_width = f_naive.upper[5] - f_naive.lower[5];
    println!(
        "\nNaive ARIMA(1, 0, 0) without exog: σ² = {:.3} (vs ARIMAX σ² = {:.3})",
        naive.sigma2, fit.sigma2,
    );
    println!(
        "  95% interval width at h = 6:  naive = {:.2}, ARIMAX = {:.2}",
        naive_width, arimax_width,
    );
    println!(
        "  → ARIMAX is {:.1}× tighter because the holiday/promo variance",
        naive_width / arimax_width,
    );
    println!("    is *explained* by the regressors instead of inflating σ².");
}
