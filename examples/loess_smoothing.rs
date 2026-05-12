//! LOESS smoothing — local-polynomial regression for noisy series.
//!
//! Two knobs:
//! - `span` (0 < span ≤ 1): fraction of points used in each local fit.
//!   Small → wiggly (low bias, high variance). Large → smooth (high
//!   bias, low variance). 0.25–0.5 is a typical starting range.
//! - `degree` (0, 1, 2): polynomial degree of the local fit. 0 = local
//!   mean. 1 = local line (the standard choice). 2 = local parabola
//!   (handles curvature in the underlying trend better, but slower and
//!   more sensitive to outliers).
//!
//! Run with:
//!
//!   cargo run --release --example loess_smoothing

use rust_stats::loess;

fn main() {
    // ── A noisy curve: sin(2πx/40) + a slow trend + measurement noise.
    let n = 200usize;
    let mut rng_state = 0x10E55u64;
    let mut noise = || {
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
    let truth: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64;
            (2.0 * std::f64::consts::PI * x / 40.0).sin() + 0.01 * x
        })
        .collect();
    let y: Vec<f64> = truth.iter().map(|t| t + 0.3 * noise()).collect();

    // ── Three spans + two degrees.
    let configs = [
        ("narrow  span=0.10 deg=1", 0.10, 1),
        ("medium  span=0.25 deg=1", 0.25, 1),
        ("broad   span=0.50 deg=1", 0.50, 1),
        ("medium  span=0.25 deg=0", 0.25, 0), // local mean
        ("medium  span=0.25 deg=2", 0.25, 2), // local parabola
    ];

    println!("LOESS smoothing on a noisy sin + trend series (n = {n}, σ_noise ≈ 0.3):\n");
    println!("                              RMSE vs truth   RMSE vs y");
    for (label, span, degree) in configs {
        let smooth = loess(&y, span, degree).unwrap();
        let rmse_truth = rmse(&smooth, &truth);
        let rmse_y = rmse(&smooth, &y);
        println!(
            "  {label:30}     {rmse_truth:.3}        {rmse_y:.3}"
        );
    }
    println!("\n(Smaller RMSE-vs-truth is better — that's the signal recovery.");
    println!(" RMSE-vs-y near σ_noise ≈ 0.3 indicates the smoother is recovering");
    println!(" the underlying curve, not memorising the noise.)");

    // ── A "before / after" sample around the peak of the sine curve.
    let smooth = loess(&y, 0.25, 1).unwrap();
    println!("\nSample (median span = 0.25, degree = 1):");
    println!("    i      y         smoothed   |   truth");
    for i in [0usize, 5, 10, 15, 20, 100, 150, 199] {
        println!(
            "  {:3}   {:7.3}    {:7.3}    |  {:7.3}",
            i, y[i], smooth[i], truth[i],
        );
    }
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let ss: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (ss / n as f64).sqrt()
}
