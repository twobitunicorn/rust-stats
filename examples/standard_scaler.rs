//! `StandardScaler` walkthrough: fit on training data, apply the same
//! scaling to held-out test data, train a downstream model on the
//! transformed scale, then invert the result back to original units.
//!
//! The free [`z_score`] function recomputes mean/std on every call, so
//! it can't be re-applied to a different series. The struct path
//! captures the fitted parameters and lets you replay the transform.
//!
//! Run with:
//!
//!   cargo run --release --example standard_scaler

use rust_stats::{z_score, StandardScaler};

fn main() {
    // ── Train and test on the same generating process. ──────────────
    //    Slightly different means / shapes so the train-only fit is
    //    visible in the test output.
    let y_train: Vec<f64> = (1..=30).map(|i| (i as f64) + (i as f64 * 0.4).sin()).collect();
    let y_test:  Vec<f64> = (31..=40).map(|i| (i as f64) + (i as f64 * 0.4).sin()).collect();

    // ── 1. Fit on training data. The struct now owns (mean, std_dev).
    let scaler = StandardScaler::fit(&y_train);
    println!(
        "fit on train: mean = {:.3}, std_dev (ddof=1) = {:.3}",
        scaler.mean(),
        scaler.std_dev(),
    );

    // ── 2. Apply the *same* transform to both train and test. ───────
    let z_train = scaler.transform(&y_train);
    let z_test  = scaler.transform(&y_test);

    println!("\nTrain mean after scaling ≈ 0 (it was fit on this):");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!("  mean(z_train) = {:.6}", mean(&z_train));

    println!("\nTest mean after scaling is NOT zero — the test data is shifted up.");
    println!("  mean(z_test)  = {:.3}", mean(&z_test));

    // ── 3. The free `z_score` function recomputes its own params on
    //    every call, so it cannot enforce "same scaling on test".
    //    Comparing it side-by-side:
    let z_test_free = z_score(&y_test);
    println!(
        "\nz_score(y_test) is centered on the *test's own* mean → mean = {:.6}",
        mean(&z_test_free),
    );
    println!("(That's why we use the struct path for train/test workflows.)");

    // ── 4. Pretend we trained a model on z_train, got predictions
    //    z_hat, then want them in original units.
    let z_hat = vec![-0.1, 0.5, 1.2, 1.8];
    let y_hat = scaler.inverse_transform(&z_hat);
    println!("\nForecast on scaled axis: {:?}", z_hat);
    println!("Back-transformed to original units: {:.3?}", y_hat);

    // ── 5. Round-trip sanity. ───────────────────────────────────────
    let back = scaler.inverse_transform(&z_train);
    let max_err = y_train
        .iter()
        .zip(&back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\nTrain forward / inverse round-trip max abs error: {:.2e}", max_err);
}
