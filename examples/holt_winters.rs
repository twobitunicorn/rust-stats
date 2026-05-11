//! Holt-Winters exponential smoothing across its three specialisations.
//!
//! - **SES** (α only): flat level.
//! - **Holt's linear method** (α + β): level + trend, linear forecast.
//! - **Holt-Winters seasonal** (α + β + γ + m): level + trend + seasonal
//!   index, additive or multiplicative.
//!
//! All three smoothing constants are caller-supplied — this crate
//! doesn't run MLE-style auto-search the way R's `HoltWinters()` does
//! by default.
//!
//! Run with:
//!
//!   cargo run --release --example holt_winters

use rust_stats::{holt_winters, DecomposeMode, HoltWintersOpts};

fn main() {
    // ── A monthly series: linear trend + seasonal cycle + tiny noise.
    let period = 12u32;
    let n = 60usize;
    let mut rng_state = 42u64;
    let mut noise = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64 - 0.5) * 2.0
    };
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let phase = 2.0 * std::f64::consts::PI
                * (i % period as usize) as f64
                / period as f64;
            trend + 10.0 * phase.sin() + 1.5 * noise()
        })
        .collect();
    println!(
        "series: n={n}, period={period}, range=[{:.1}, {:.1}]",
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── 1. Simple exponential smoothing — α only.
    let fit_ses = holt_winters(&y, HoltWintersOpts::new(0.5)).unwrap();

    // ── 2. Holt's linear method — α + β.
    let fit_holt = holt_winters(
        &y,
        HoltWintersOpts {
            beta: 0.2,
            ..HoltWintersOpts::new(0.5)
        },
    )
    .unwrap();

    // ── 3. Full additive Holt-Winters — α + β + γ.
    let fit_add = holt_winters(
        &y,
        HoltWintersOpts {
            alpha: 0.5,
            beta: 0.2,
            gamma: 0.3,
            seasonal_periods: period,
            mode: DecomposeMode::Additive,
        },
    )
    .unwrap();

    // ── 4. Multiplicative variant. Seasonal index multiplies (level +
    //    trend) rather than adding to it. Use when the seasonal swing
    //    scales with the level. Requires strictly-positive y.
    let fit_mul = holt_winters(
        &y,
        HoltWintersOpts {
            alpha: 0.5,
            beta: 0.2,
            gamma: 0.3,
            seasonal_periods: period,
            mode: DecomposeMode::Multiplicative,
        },
    )
    .unwrap();

    // ── In-sample RMSE comparison (skip the first season for warm-up).
    let warmup = period as usize;
    let rmse = |fitted: &[f64]| -> f64 {
        let ss: f64 = y[warmup..]
            .iter()
            .zip(&fitted[warmup..])
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (ss / (n - warmup) as f64).sqrt()
    };
    println!("\nIn-sample RMSE (skipping the first season for warm-up):");
    println!("  SES (α=0.5)                                  {:.3}", rmse(&fit_ses.fitted));
    println!("  Holt linear (α=0.5, β=0.2)                   {:.3}", rmse(&fit_holt.fitted));
    println!("  Holt-Winters additive (α=0.5, β=0.2, γ=0.3)  {:.3}", rmse(&fit_add.fitted));
    println!("  Holt-Winters multiplicative                  {:.3}", rmse(&fit_mul.fitted));

    // ── Final state of the additive HW fit.
    println!("\nFinal state (additive HW):");
    println!("  level      = {:.3}", fit_add.level);
    println!("  trend      = {:.3}", fit_add.trend);
    println!("  seasonal   = [{}]", fit_add.seasonal.iter()
        .map(|v| format!("{:.2}", v))
        .collect::<Vec<_>>()
        .join(", "));

    // ── Forecast the next 12 months from each model.
    let h = period as usize;
    let f_ses  = fit_ses.forecast(h);
    let f_holt = fit_holt.forecast(h);
    let f_add  = fit_add.forecast(h);
    let f_mul  = fit_mul.forecast(h);

    println!("\nForecast (h=1..{h}):");
    println!("  h        SES       Holt       HW-add     HW-mul");
    for i in 0..h {
        println!(
            "  {:2}    {:8.2}   {:8.2}   {:8.2}   {:8.2}",
            i + 1, f_ses[i], f_holt[i], f_add[i], f_mul[i],
        );
    }
}
