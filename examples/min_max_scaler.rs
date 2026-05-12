//! `MinMaxScaler` walkthrough: fit `[min, max]` on training data, map
//! into `[0, 1]`, then replay the same mapping on held-out test data —
//! crucially, test points outside the training range produce outputs
//! outside `[0, 1]` (sklearn-compatible; the scaler is not a clamp).
//!
//! Run with:
//!
//!   cargo run --release --example min_max_scaler

use rust_stats::{min_max_scale, MinMaxScaler};

fn main() {
    // ── Training data covers some range; test data goes both above
    //    and below it so the out-of-range behaviour is visible.
    let y_train = vec![0.0, 5.0, 10.0, 20.0, 25.0, 30.0];
    let y_test  = vec![-5.0, 15.0, 35.0];

    // ── 1. Fit on training data. ────────────────────────────────────
    let scaler = MinMaxScaler::fit(&y_train);
    println!(
        "fit on train: min = {:.3}, max = {:.3}, range = {:.3}",
        scaler.min(),
        scaler.max(),
        scaler.max() - scaler.min(),
    );

    // ── 2. Replay the train-derived scaling on both sets. ───────────
    let z_train = scaler.transform(&y_train);
    let z_test  = scaler.transform(&y_test);
    println!("\nTrain transformed → exactly [0, 1]:");
    println!("  {:?}", z_train);

    println!("\nTest transformed — values outside [0, 1] flag out-of-range inputs:");
    println!("  y_test = {:?}", y_test);
    println!("  z_test = {:?}   ({} < 0 → below train.min; > 1 → above train.max)",
        z_test, "values");

    // ── 3. Contrast with the free `min_max_scale` function, which
    //    rescales each call against its own min/max — so y_test ends
    //    up in [0, 1] no matter where it lived.
    let z_test_free = min_max_scale(&y_test);
    println!("\nmin_max_scale(y_test) re-fits to the test range → forced into [0, 1]:");
    println!("  {:?}", z_test_free);
    println!("(That's why we use the struct path for train/test workflows.)");

    // ── 4. Back-transform a forecast to original units. ─────────────
    let z_hat = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25];
    let y_hat = scaler.inverse_transform(&z_hat);
    println!("\nForecast on scaled axis: {:?}", z_hat);
    println!("Back-transformed to original units: {:?}", y_hat);

    // ── 5. Round-trip sanity. ───────────────────────────────────────
    let back = scaler.inverse_transform(&z_train);
    let max_err = y_train
        .iter()
        .zip(&back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\nTrain forward / inverse round-trip max abs error: {:.2e}", max_err);
}
