//! Parity tests for `rust_stats::smoothing::loess` (degree=1) against
//! `statsmodels.nonparametric.smoothers_lowess.lowess(it=0)`.
//!
//! statsmodels' LOWESS supports degree=1 only; degree 0 and degree 2 are
//! covered by the analytical tests in `tests/loess.rs`.
//!
//! Observed drift on the bundled goldens is up to ~0.03 absolute (no
//! single-point relative threshold makes sense because the smoothed series
//! crosses zero). RMSE is ~1e-2.

use rust_stats::smoothing::loess;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Golden {
    y: Vec<f64>,
    span: f64,
    degree: u8,
    smoothed: Vec<f64>,
}

fn load(name: &str) -> Golden {
    let path: PathBuf = ["tests", "golden", &format!("loess_{name}.json")].iter().collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {path:?}: {e}; did you run tests/golden/generate.py?")
    });
    serde_json::from_slice(&bytes).expect("invalid golden JSON")
}

fn assert_dataset(name: &str, max_abs: f64, max_rmse: f64) {
    let g = load(name);
    let out = loess(&g.y, g.span, g.degree).expect("loess failed");
    assert_eq!(out.len(), g.smoothed.len());
    let mut sse = 0.0f64;
    for i in 0..out.len() {
        let d = (out[i] - g.smoothed[i]).abs();
        assert!(
            d <= max_abs,
            "{name} drift at i={i}: ours={} ref={} diff={d}",
            out[i],
            g.smoothed[i]
        );
        sse += d * d;
    }
    let rmse = (sse / out.len() as f64).sqrt();
    assert!(rmse <= max_rmse, "{name} RMSE {rmse} > {max_rmse}");
}

#[test] fn smooth_span30() { assert_dataset("smooth_span30", 4e-2, 2e-2); }
#[test] fn smooth_span50() { assert_dataset("smooth_span50", 3e-2, 1e-2); }
#[test] fn noisy_span30()  { assert_dataset("noisy_span30",  4e-2, 2e-2); }
#[test] fn noisy_span50()  { assert_dataset("noisy_span50",  3e-2, 1e-2); }
