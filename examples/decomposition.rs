//! Two ways to decompose a seasonal series into trend + seasonal +
//! residual: classical centered-MA (`seasonal_decompose`) and Cleveland
//! 1990 (`stl`). Additive and multiplicative variants for each.
//!
//! Both return a `Decomposition { trend, seasonal, residual }` of the
//! same length as the input. STL is the modern default — it smooths the
//! cycle-subseries with LOESS so the seasonal pattern can evolve.
//! `seasonal_decompose` is the simpler classical algorithm that
//! statsmodels users will recognise; it pins the seasonal pattern to a
//! repeating cycle.
//!
//! Run with:
//!
//!   cargo run --release --example decomposition

use rust_stats::{
    seasonal_decompose, stl, DecomposeMode, SeasonalDecomposeOpts, SeasonalWindow, StlOpts,
};

fn main() {
    // ── Monthly series: linear trend + sinusoidal seasonal + noise.
    let m = 12usize;
    let n = 8 * m; // 8 years of monthly data
    let mut rng_state = 0xACE5u64;
    let mut noise = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64 - 0.5) * 2.0
    };
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let phase = 2.0 * std::f64::consts::PI * (i % m) as f64 / m as f64;
            trend + 10.0 * phase.sin() + 2.0 * noise()
        })
        .collect();

    println!(
        "series: n = {}, period m = {}, range = [{:.1}, {:.1}]",
        n,
        m,
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── 1. Classical seasonal_decompose (additive). ─────────────────
    //    Centered moving-average trend; seasonal = per-phase deviations.
    //    Matches statsmodels' `seasonal_decompose(model="additive")`
    //    exactly. The first / last `m/2` trend entries are NaN
    //    (window doesn't fit at the boundaries).
    let sd_add = seasonal_decompose(
        &y,
        SeasonalDecomposeOpts {
            period: m as u32,
            mode: DecomposeMode::Additive,
            ..SeasonalDecomposeOpts::new(m as u32)
        },
    )
    .unwrap();
    println!("\nseasonal_decompose (additive):");
    print_components(&y, &sd_add.trend, &sd_add.seasonal, &sd_add.residual, m);

    // ── 2. STL (additive). ──────────────────────────────────────────
    //    LOESS on the cycle subseries → seasonal can evolve over time.
    //    No NaN boundary — STL fits all positions. Default options
    //    (seasonal_window = 7) are the Cleveland 1990 textbook setup.
    let stl_out = stl(&y, StlOpts {
        period: m as u32,
        seasonal_window: SeasonalWindow::Window(7),
        ..StlOpts::new(m as u32)
    })
    .unwrap();
    println!("\nstl (additive, seasonal_window = 7):");
    print_components(&y, &stl_out.trend, &stl_out.seasonal, &stl_out.residual, m);

    // ── 3. Multiplicative on a series where it makes sense. ─────────
    //    Build a series with multiplicative variance: σ ∝ level.
    let y_mult: Vec<f64> = (0..n)
        .map(|i| {
            let level = 100.0 + 0.5 * i as f64;
            let phase = 2.0 * std::f64::consts::PI * (i % m) as f64 / m as f64;
            level * (1.0 + 0.1 * phase.sin())
        })
        .collect();
    let sd_mul = seasonal_decompose(
        &y_mult,
        SeasonalDecomposeOpts {
            period: m as u32,
            mode: DecomposeMode::Multiplicative,
            ..SeasonalDecomposeOpts::new(m as u32)
        },
    )
    .unwrap();
    println!("\nseasonal_decompose (multiplicative on σ∝level series):");
    print_components(&y_mult, &sd_mul.trend, &sd_mul.seasonal, &sd_mul.residual, m);

    // ── 4. Quality check: variance of residual / variance of original
    //    (lower = decomposition explained more of the signal).
    let var = |x: &[f64]| -> f64 {
        let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64
    };
    println!("\nFraction of variance left in the residual:");
    println!("  seasonal_decompose: {:.3}", var(&sd_add.residual) / var(&y));
    println!("  stl:                {:.3}", var(&stl_out.residual) / var(&y));
    println!("(STL typically explains slightly more thanks to LOESS smoothing of the cycle.)");
}

fn print_components(y: &[f64], trend: &[f64], seasonal: &[f64], residual: &[f64], m: usize) {
    println!("  first {} obs:", m);
    println!("    i      y       trend      seasonal   residual");
    for i in 0..m {
        let f = |v: f64| if v.is_finite() { format!("{:8.2}", v) } else { "     NaN".to_string() };
        println!(
            "    {:2}   {:7.2}   {}   {}   {}",
            i, y[i], f(trend[i]), f(seasonal[i]), f(residual[i]),
        );
    }
}
