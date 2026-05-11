//! Run all goldens through rust-stats and report worst-case + RMSE drift
//! against the statsmodels reference. Used to set parity-test tolerances.

use rust_stats::smoothing::loess;
use rust_stats::tsa::{
    seasonal_decompose, stl, DecomposeMode, SeasonalDecomposeOpts, SeasonalWindow, StlOpts,
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct SdGolden {
    y: Vec<f64>,
    period: u32,
    mode: String,
    trend: Vec<Option<f64>>,
    seasonal: Vec<Option<f64>>,
    residual: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct StlGolden {
    y: Vec<f64>,
    period: u32,
    seasonal_window: u32,
    mode: String,
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    residual: Vec<f64>,
}

#[derive(Deserialize)]
struct LoessGolden {
    y: Vec<f64>,
    span: f64,
    degree: u8,
    smoothed: Vec<f64>,
}

fn load<T: for<'de> Deserialize<'de>>(prefix: &str, name: &str) -> T {
    let path: PathBuf = ["tests", "golden", &format!("{prefix}_{name}.json")]
        .iter()
        .collect();
    let bytes = std::fs::read(&path).expect("read failed");
    serde_json::from_slice(&bytes).expect("invalid JSON")
}

fn mode_of(s: &str) -> DecomposeMode {
    match s {
        "additive" => DecomposeMode::Additive,
        "multiplicative" => DecomposeMode::Multiplicative,
        _ => panic!(),
    }
}

fn drift_stats(label: &str, a: &[f64], b: &[f64]) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sse = 0.0f64;
    let mut n_skipped = 0;
    for i in 0..a.len() {
        if !a[i].is_finite() || !b[i].is_finite() {
            n_skipped += 1;
            continue;
        }
        let d = (a[i] - b[i]).abs();
        max_abs = max_abs.max(d);
        let r = if b[i].abs() > 1e-12 { d / b[i].abs() } else { d };
        max_rel = max_rel.max(r);
        sse += d * d;
    }
    let rmse = (sse / a.len() as f64).sqrt();
    println!(
        "    {label:>10}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  rmse={rmse:.3e}  skipped={n_skipped}"
    );
}

fn drift_stats_optref(label: &str, ours: &[f64], ref_: &[Option<f64>]) {
    let ref_vec: Vec<f64> = ref_
        .iter()
        .map(|o| o.unwrap_or(f64::NAN))
        .collect();
    drift_stats(label, ours, &ref_vec);
}

fn check_seasonal_decompose(name: &str) {
    let g: SdGolden = load("seasonal_decompose", name);
    let mut opts = SeasonalDecomposeOpts::new(g.period);
    opts.mode = mode_of(&g.mode);
    let d = seasonal_decompose(&g.y, opts).unwrap();
    println!("seasonal_decompose / {name} (mode={}, n={}):", g.mode, g.y.len());
    drift_stats_optref("trend",    &d.trend,    &g.trend);
    drift_stats_optref("seasonal", &d.seasonal, &g.seasonal);
    drift_stats_optref("residual", &d.residual, &g.residual);
}

fn check_stl(name: &str) {
    let g: StlGolden = load("stl", name);
    let mut opts = StlOpts::new(g.period);
    opts.seasonal_window = SeasonalWindow::Window(g.seasonal_window);
    opts.inner_iters = 2;
    opts.mode = mode_of(&g.mode);
    let d = stl(&g.y, opts).unwrap();
    println!("stl / {name} (mode={}, n={}, seasonal={}):", g.mode, g.y.len(), g.seasonal_window);
    drift_stats("trend",    &d.trend,    &g.trend);
    drift_stats("seasonal", &d.seasonal, &g.seasonal);
    drift_stats("residual", &d.residual, &g.residual);
}

fn check_loess(name: &str) {
    let g: LoessGolden = load("loess", name);
    let out = loess(&g.y, g.span, g.degree).unwrap();
    println!("loess / {name} (span={}, degree={}, n={}):", g.span, g.degree, g.y.len());
    drift_stats("smoothed", &out, &g.smoothed);
}

#[test]
#[ignore = "diagnostic only — run with: cargo test --test parity_diagnostics -- --ignored --nocapture"]
fn report_drift() {
    for name in [
        "quarterly_additive",
        "quarterly_multiplicative",
        "airpassengers_additive",
        "airpassengers_multiplicative",
    ] {
        check_seasonal_decompose(name);
    }
    println!();
    for name in [
        "quarterly_additive",
        "quarterly_multiplicative",
        "airpassengers_additive",
        "airpassengers_multiplicative",
    ] {
        check_stl(name);
    }
    println!();
    for name in ["smooth_span30", "smooth_span50", "noisy_span30", "noisy_span50"] {
        check_loess(name);
    }
}
