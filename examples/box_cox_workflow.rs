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

use rust_stats::{arima, box_cox, inv_box_cox, ArimaOpts, BoxCox, Lambda};

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
    println!(
        "series length = {}, range = [{:.1}, {:.1}]",
        y.len(),
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── 2. API tour: four equivalent ways to box-cox the same data. ─
    //    All four produce identical `transformed` slices for the same
    //    final λ. Pick the form that fits your call site — the struct
    //    is best when you'll apply the same λ to multiple slices or
    //    invert later; the free function is best for one-off use.
    {
        // (a) Free function, *fixed* λ. The classic "I know what I want".
        let out = box_cox(&y, 0.5).unwrap();
        let _z = out.transformed;
        let _back = inv_box_cox(&_z, out.lambda).unwrap();

        // (b) Free function, *estimator*. Convenience for a one-shot
        //     fit + transform; `out.lambda` echoes what was chosen.
        let out = box_cox(&y, Lambda::Mle).unwrap();
        let _z = out.transformed;
        let _back = inv_box_cox(&_z, out.lambda).unwrap();

        // (c) Struct, fixed λ. Useful when you want to apply the same
        //     transform to multiple series without copying λ around.
        let bc = BoxCox::new(0.5);
        let _z = bc.transform(&y).unwrap();
        let _back = bc.inverse_transform(&_z).unwrap();

        // (d) Struct, estimator. The "fit once, transform many, invert
        //     against the forecast" pattern — what the SARIMA workflow
        //     below uses.
        let bc = BoxCox::fit(&y, Lambda::Mle).unwrap();
        let _z = bc.transform(&y).unwrap();
        let _back = bc.inverse_transform(&_z).unwrap();
    }

    // ── 3. Compare the three λ estimators side by side. ─────────────
    //    - Guerrero: minimises CV of σ_b / μ_b^(1-λ) within cycles —
    //      stabilises seasonal variance. Use this for forecasting.
    //    - MLE / loglik: maximises Gaussian log-likelihood of the
    //      transformed series — targets marginal normality.
    //    - Pearson r: maximises the Q-Q correlation of the
    //      transformed values against theoretical normal quantiles —
    //      same goal as MLE but rank-based, so more robust to outliers.
    //
    //    Calling `box_cox` with each `Lambda` variant returns a
    //    `BoxCoxOutput { transformed, lambda }`. We pull `.lambda` off
    //    each one for the comparison table.
    let guerrero = box_cox(&y, Lambda::Guerrero { period: m }).unwrap();
    let mle = box_cox(&y, Lambda::Mle).unwrap();
    let pearson = box_cox(&y, Lambda::Pearsonr).unwrap();
    println!("\nλ comparison:");
    println!("  Guerrero  = {:.3}   (variance stabilisation)", guerrero.lambda);
    println!("  MLE       = {:.3}   (marginal Gaussianity, full likelihood)", mle.lambda);
    println!("  Pearson r = {:.3}   (marginal Gaussianity, rank-based)", pearson.lambda);

    // For a forecasting workflow on a seasonal series, prefer Guerrero.
    // The `BoxCox` struct path encapsulates the chosen λ so we can
    // forward-transform once and invert later without threading the
    // value through every call site.
    let bc = BoxCox::fit(&y, Lambda::Guerrero { period: m }).unwrap();
    println!("\nUsing Guerrero λ = {:.3} for the SARIMA fit.", bc.lambda());

    let z = bc.transform(&y).unwrap();

    // ── 4. Fit SARIMA on the transformed series. ────────────────────
    //    For an automated search, swap in:
    //    `auto_arima(&z, AutoArimaOpts::seasonal(m as u32))`.
    let fit = arima(&z, ArimaOpts::seasonal(1, 1, 1, 0, 1, 1, m as u32)).unwrap();
    println!(
        "fit: AICc = {:.2}, σ² = {:.5}, n_obs = {}",
        fit.aicc, fit.sigma2, fit.n_obs,
    );

    // ── 5. Forecast on the transformed scale with 95% intervals. ────
    let horizon = 12;
    let f = fit.forecast_with_intervals(horizon, 0.05);

    // ── 6. Back-transform the mean and the interval bounds. ─────────
    //    BoxCox::inverse_transform is monotonic, so transforming the
    //    lower / upper quantiles individually gives correctly-
    //    calibrated bounds on the original scale. They will be
    //    asymmetric around the back-transformed mean — that's the right
    //    thing (Box-Cox stretches one tail).
    let mean = bc.inverse_transform(&f.mean).unwrap();
    let lower = bc.inverse_transform(&f.lower).unwrap();
    let upper = bc.inverse_transform(&f.upper).unwrap();

    println!("\nForecast (original scale, 95% prediction intervals):");
    println!("  h     mean          [    lower,    upper ]");
    for h in 0..horizon {
        println!(
            "  {:2}  {:8.2}      [{:8.2}, {:8.2}]",
            h + 1, mean[h], lower[h], upper[h],
        );
    }

    // ── 7. Sanity: forward / inverse should round-trip the input. ───
    //    Two ways to do the same thing:
    //
    //    let out = box_cox(&y, bc.lambda())?;            // one-shot
    //    let y_back = inv_box_cox(&out.transformed, …);
    //
    //    let y_back = bc.inverse_transform(&bc.transform(&y)?)?;  // struct path
    let y_back = bc.inverse_transform(&bc.transform(&y).unwrap()).unwrap();
    let max_err = y
        .iter()
        .zip(&y_back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\nForward / inverse round-trip max abs error: {:.2e}", max_err);
}
