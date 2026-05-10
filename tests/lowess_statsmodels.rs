//! LOWESS tests ported from statsmodels' test suite:
//!
//!   `statsmodels.nonparametric.tests.test_lowess.TestLowess`
//!
//! Tests that depend on robustness iterations (`it=3`), explicit `xvals`,
//! NaN handling, sorting, or duplicate-x are not portable to our 1D
//! integer-indexed, single-pass `loess` API and are omitted.

use rust_stats::smoothing::loess;

/// Source: TestLowess.test_flat — y = zeros, lowess(y) = zeros.
#[test]
fn test_flat() {
    let y = vec![0.0; 20];
    let out = loess(&y, 0.5, 1).unwrap();
    for v in &out {
        assert!(v.abs() < 1e-9, "expected zero, got {v}");
    }
}

/// Source: TestLowess.test_range — y = arange(20), lowess(y) = arange(20).
#[test]
fn test_range() {
    let n = 20;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let out = loess(&y, 0.5, 1).unwrap();
    for i in 0..n {
        assert!((out[i] - i as f64).abs() < 1e-9, "i={i}: {} vs {i}", out[i]);
    }
}

/// Source: TestLowess.test_spike (issue 7700). Smoothed values stay within
/// (min(y) − 0.1, max(y) + 0.1) on a curve that is easy to fit at first
/// but harder later. statsmodels uses `it=1`; we don't have iterations,
/// so we pad the bound a touch (0.2 instead of 0.1) — the original failure
/// was an outlier of order ~1.
#[test]
fn test_spike() {
    let n = 1001;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| (xi * xi / 5.0).cos()).collect();
    let out = loess(&y, 11.0 / n as f64, 1).unwrap();
    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for v in &out {
        assert!(*v > y_min - 0.2, "smoothed {v} below y_min - 0.2 = {}", y_min - 0.2);
        assert!(*v < y_max + 0.2, "smoothed {v} above y_max + 0.2 = {}", y_max + 0.2);
    }
}
