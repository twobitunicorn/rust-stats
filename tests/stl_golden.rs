//! Parity tests for `rust_stats::tsa::stl` against
//! `statsmodels.tsa.seasonal.STL(robust=False).fit(inner_iter=2, outer_iter=0)`.
//!
//! Both implement Cleveland 1990 STL but differ in low-level numerics
//! (LOESS internals, low-pass filter details, boundary handling). The
//! parity tolerances below track observed drift on AirPassengers and a
//! quarterly synthetic series:
//!
//!   trend:    ~1 unit absolute (≤ 0.4% relative on series with mean ~300)
//!   seasonal: up to ~3 units absolute on additive AirPassengers; ~1% in
//!             multiplicative / log space
//!   residual: dominated by trend+seasonal drift, similar magnitude
//!
//! The reconstruction identity (y = T+S+R or y = T*S*R) is checked tightly
//! and exposes any internal bug that does not show in component drift.

use approx::assert_relative_eq;
use rust_stats::tsa::{stl, DecomposeMode, SeasonalWindow, StlOpts};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Golden {
    y: Vec<f64>,
    period: u32,
    seasonal_window: u32,
    mode: String,
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    residual: Vec<f64>,
}

fn load(name: &str) -> Golden {
    let path: PathBuf = ["tests", "golden", &format!("stl_{name}.json")].iter().collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {path:?}: {e}; did you run tests/golden/generate.py?")
    });
    serde_json::from_slice(&bytes).expect("invalid golden JSON")
}

fn decompose_mode(name: &str) -> DecomposeMode {
    match name {
        "additive" => DecomposeMode::Additive,
        "multiplicative" => DecomposeMode::Multiplicative,
        other => panic!("unknown mode {other}"),
    }
}

struct Tol {
    /// Per-component absolute tolerance.
    trend_abs:    f64,
    seasonal_abs: f64,
    residual_abs: f64,
}

fn assert_dataset(name: &str, tol: Tol) {
    let g = load(name);
    let mut opts = StlOpts::new(g.period);
    opts.seasonal_window = SeasonalWindow::Window(g.seasonal_window);
    opts.inner_iters = 2;
    opts.mode = decompose_mode(&g.mode);
    let d = stl(&g.y, opts).expect("stl failed");

    let n = g.y.len();
    assert_eq!(d.trend.len(), n);
    assert_eq!(d.seasonal.len(), n);
    assert_eq!(d.residual.len(), n);

    // Component-wise drift against the statsmodels reference.
    for i in 0..n {
        assert!(
            (d.trend[i] - g.trend[i]).abs() <= tol.trend_abs,
            "trend drift at i={i}: ours={} ref={} diff={}", d.trend[i], g.trend[i], (d.trend[i] - g.trend[i]).abs()
        );
        assert!(
            (d.seasonal[i] - g.seasonal[i]).abs() <= tol.seasonal_abs,
            "seasonal drift at i={i}: ours={} ref={} diff={}", d.seasonal[i], g.seasonal[i], (d.seasonal[i] - g.seasonal[i]).abs()
        );
        assert!(
            (d.residual[i] - g.residual[i]).abs() <= tol.residual_abs,
            "residual drift at i={i}: ours={} ref={} diff={}", d.residual[i], g.residual[i], (d.residual[i] - g.residual[i]).abs()
        );
    }

    // Reconstruction identity — independent of statsmodels — checked tightly.
    let mode = decompose_mode(&g.mode);
    for i in 0..n {
        let recon = match mode {
            DecomposeMode::Additive       => d.trend[i] + d.seasonal[i] + d.residual[i],
            DecomposeMode::Multiplicative => d.trend[i] * d.seasonal[i] * d.residual[i],
        };
        assert_relative_eq!(recon, g.y[i], max_relative = 1e-10, epsilon = 1e-10);
    }
}

#[test]
fn quarterly_additive() {
    assert_dataset("quarterly_additive", Tol { trend_abs: 0.2, seasonal_abs: 0.2, residual_abs: 0.2 });
}

#[test]
fn quarterly_multiplicative() {
    assert_dataset("quarterly_multiplicative", Tol { trend_abs: 0.2, seasonal_abs: 5e-3, residual_abs: 5e-3 });
}

#[test]
fn airpassengers_additive() {
    assert_dataset("airpassengers_additive", Tol { trend_abs: 1.5, seasonal_abs: 4.0, residual_abs: 4.0 });
}

#[test]
fn airpassengers_multiplicative() {
    assert_dataset("airpassengers_multiplicative", Tol { trend_abs: 1.5, seasonal_abs: 2e-2, residual_abs: 2e-2 });
}
