//! End-to-end Box-Cox + SARIMA forecasting workflow.
//!
//! Pick λ automatically → transform → fit → forecast with intervals →
//! back-transform to the original scale. The synthetic series has
//! multiplicative seasonal variance (σ grows with the level), which is
//! the canonical case Box-Cox is for.
//!
//! Run with:
//!
//!   cargo run --release --example box_cox_workflow

use rust_stats::{
    arima, box_cox, box_cox_lambda_guerrero, inv_box_cox, ArimaOpts,
};

fn main() {
    // ── 1. Build a monthly series with multiplicative variance. ─────
    //    σ grows with the level — Box-Cox stabilises that.
    let m: usize = 12;
    let n_cycles = 15;
    let mut y = Vec::with_capacity(m * n_cycles);
    for cycle in 0..n_cycles {
        let level = 100.0 + 5.0 * cycle as f64;
        for i in 0..m {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / m as f64;
            y.push(level * (1.0 + 0.3 * phase.sin()));
        }
    }
    println!("series length = {}, range = [{:.1}, {:.1}]",
        y.len(),
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── 2. Pick λ that stabilises the within-cycle variance. ────────
    //    Guerrero is the seasonal-aware estimator. For a non-seasonal
    //    series, swap in `box_cox_lambda_mle(&y)` instead.
    let lambda = box_cox_lambda_guerrero(&y, m).unwrap();
    println!("\nGuerrero λ = {:.3}", lambda);

    // ── 3. Transform y → z on a stabilised scale. ───────────────────
    let z = box_cox(&y, lambda).unwrap();

    // ── 4. Fit SARIMA on the transformed series. ────────────────────
    //    For an automated search, swap in:
    //    `auto_arima(&z, AutoArimaOpts::seasonal(m as u32))`.
    let fit = arima(
        &z,
        ArimaOpts::seasonal(1, 1, 1, 0, 1, 1, m as u32),
    )
    .unwrap();
    println!(
        "fit: AICc = {:.2}, σ² = {:.5}, n_obs = {}",
        fit.aicc, fit.sigma2, fit.n_obs,
    );

    // ── 5. Forecast on the transformed scale with 95% intervals. ────
    let horizon = 12;
    let f = fit.forecast_with_intervals(horizon, 0.05);

    // ── 6. Back-transform the mean and the interval bounds. ─────────
    //    inv_box_cox is monotonic, so transforming the lower / upper
    //    quantiles individually gives correctly-calibrated bounds on
    //    the original scale. They will be asymmetric around the
    //    back-transformed mean — that's the right thing (Box-Cox
    //    stretches one tail).
    let mean = inv_box_cox(&f.mean, lambda).unwrap();
    let lower = inv_box_cox(&f.lower, lambda).unwrap();
    let upper = inv_box_cox(&f.upper, lambda).unwrap();

    println!("\nForecast (original scale, 95% prediction intervals):");
    println!("  h     mean          [    lower,    upper ]");
    for h in 0..horizon {
        println!(
            "  {:2}  {:8.2}      [{:8.2}, {:8.2}]",
            h + 1, mean[h], lower[h], upper[h],
        );
    }

    // ── 7. Sanity: forward / inverse should round-trip the input. ───
    let z_back = box_cox(&y, lambda).unwrap();
    let y_back = inv_box_cox(&z_back, lambda).unwrap();
    let max_err = y
        .iter()
        .zip(&y_back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\nForward / inverse round-trip max abs error: {:.2e}", max_err);
}
