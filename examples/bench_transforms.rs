//! Benchmark rust-stats transforms and Holt-Winters on synthetic data.
//! Workload sizes are chosen to mirror R / statsmodels / scipy / sklearn
//! defaults so the output can be compared side-by-side against:
//!
//!   - `tests/golden/bench_r.R`
//!   - `tests/golden/bench_statsmodels.py`
//!
//! Run with (scalar only):
//!
//!   cargo run --release --example bench_transforms
//!
//! Or with the SIMD kernels (backed by `pulp`, stable Rust, runtime ISA
//! dispatch):
//!
//!   cargo run --release --features simd --example bench_transforms
//!
//! With `--features simd`, each per-element transform is reported twice —
//! `(scalar)` and `(simd)` — so the speedup is visible inline. Box-Cox
//! and Holt-Winters stay scalar (powf / ln aren't part of `pulp`'s f64
//! vocabulary, and Holt-Winters' recurrence has a hard data dependency
//! between steps).

use rust_stats::transforms::{box_cox, center, min_max_scale, z_score};
use rust_stats::tsa::{holt_winters, DecomposeMode, HoltWintersOpts};
use std::time::Instant;

// xorshift64 RNG — same one used by examples/bench.rs.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-300);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn gaussian(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal()).collect()
}

/// Strictly positive series — required by `box_cox` and the
/// multiplicative branch of Holt-Winters.
fn positive_series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal().exp() + 0.5).collect()
}

fn series_with_seasonality(n: usize, period: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let trend = 10.0 + 0.05 * i as f64;
            let phase = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            let seasonal = 3.0 * phase.sin() + 1.5 * (2.0 * phase).cos();
            trend + seasonal + rng.normal() * 0.5
        })
        .collect()
}

fn time_iters<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn report(label: &str, n: usize, extra: &str, secs: f64) {
    println!("{label:<28} n={n:<8} {extra:<20} {:>10.3} ms", secs * 1e3);
}

// ──────────────────────────────────────────────────────────────────────
// center  ↔  R `scale(x, scale = FALSE)`,
//            sklearn `StandardScaler(with_std=False)`
// ──────────────────────────────────────────────────────────────────────

fn bench_center() {
    for &(n, iters) in &[(10_000usize, 200), (100_000, 100), (1_000_000, 30)] {
        let y = gaussian(n, 0xC1);
        let secs = time_iters(iters, || {
            let _ = center(&y);
        });
        report("center (scalar)", n, "", secs);

        #[cfg(feature = "simd")]
        {
            let secs = time_iters(iters, || {
                let _ = rust_stats::transforms::center_simd(&y);
            });
            report("center (simd)", n, "", secs);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// z_score  ↔  R `scale(x)` (ddof = 1),
//             sklearn `StandardScaler` (ddof = 0; not strictly equal)
// ──────────────────────────────────────────────────────────────────────

fn bench_z_score() {
    for &(n, iters) in &[(10_000usize, 200), (100_000, 100), (1_000_000, 30)] {
        let y = gaussian(n, 0xC2);
        let secs = time_iters(iters, || {
            let _ = z_score(&y);
        });
        report("z_score (scalar)", n, "", secs);

        #[cfg(feature = "simd")]
        {
            let secs = time_iters(iters, || {
                let _ = rust_stats::transforms::z_score_simd(&y);
            });
            report("z_score (simd)", n, "", secs);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// min_max_scale  ↔  sklearn `MinMaxScaler`,
//                   R `caret::preProcess(method = "range")`
// ──────────────────────────────────────────────────────────────────────

fn bench_min_max() {
    for &(n, iters) in &[(10_000usize, 200), (100_000, 100), (1_000_000, 30)] {
        let y = gaussian(n, 0xC3);
        let secs = time_iters(iters, || {
            let _ = min_max_scale(&y);
        });
        report("min_max_scale (scalar)", n, "", secs);

        #[cfg(feature = "simd")]
        {
            let secs = time_iters(iters, || {
                let _ = rust_stats::transforms::min_max_scale_simd(&y);
            });
            report("min_max_scale (simd)", n, "", secs);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// box_cox  ↔  scipy.stats.boxcox(x, lmbda=),
//             R `forecast::BoxCox(x, lambda=)`
//
// We bench three λ values that exercise different fast paths:
//   λ = 0.0  → `ln(x)` per element (slowest in scipy)
//   λ = 0.5  → `x^0.5` per element (the "sqrt-like" common choice)
//   λ = 2.0  → `x^2`   per element (integer power; LLVM may simplify)
// ──────────────────────────────────────────────────────────────────────

fn bench_box_cox() {
    for &(n, iters) in &[(10_000usize, 100), (100_000, 30), (1_000_000, 5)] {
        let y = positive_series(n, 0xC4);
        for &lmbda in &[0.0_f64, 0.5, 2.0] {
            let secs = time_iters(iters, || {
                let _ = box_cox(&y, lmbda).unwrap();
            });
            report("box_cox (scalar)", n, &format!("lambda={lmbda}"), secs);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// holt_winters  ↔  statsmodels `ExponentialSmoothing(...).fit(optimized=False)`,
//                  R `stats::HoltWinters(...)`
//
// Workload sizes mirror the STL/seasonal-decompose benches: 144 monthly,
// 720 daily, 2880 hourly. We exercise SES, Holt's linear, and full
// triple-Holt (additive + multiplicative).
// ──────────────────────────────────────────────────────────────────────

fn bench_holt_winters() {
    for &(n, period, iters) in &[(144usize, 12usize, 200), (720, 12, 100), (2_880, 24, 30)] {
        let y = series_with_seasonality(n, period, 0xC5);
        let y_pos = positive_series(n, 0xC5 ^ 0xFF);

        // SES — α only.
        let secs = time_iters(iters, || {
            let _ = holt_winters(&y, HoltWintersOpts::new(0.5)).unwrap();
        });
        report("hw SES", n, &format!("period={period}"), secs);

        // Holt's linear — α + β.
        let secs = time_iters(iters, || {
            let opts = HoltWintersOpts {
                beta: 0.1,
                ..HoltWintersOpts::new(0.5)
            };
            let _ = holt_winters(&y, opts).unwrap();
        });
        report("hw Holt linear", n, &format!("period={period}"), secs);

        // Additive Holt-Winters — α + β + γ.
        let secs = time_iters(iters, || {
            let opts = HoltWintersOpts {
                alpha: 0.5,
                beta: 0.1,
                gamma: 0.2,
                seasonal_periods: period as u32,
                mode: DecomposeMode::Additive,
            };
            let _ = holt_winters(&y, opts).unwrap();
        });
        report("hw additive", n, &format!("period={period}"), secs);

        // Multiplicative Holt-Winters — requires strictly positive y.
        let secs = time_iters(iters, || {
            let opts = HoltWintersOpts {
                alpha: 0.5,
                beta: 0.1,
                gamma: 0.2,
                seasonal_periods: period as u32,
                mode: DecomposeMode::Multiplicative,
            };
            let _ = holt_winters(&y_pos, opts).unwrap();
        });
        report("hw multiplicative", n, &format!("period={period}"), secs);
    }
}

fn main() {
    println!("# rust-stats transforms + Holt-Winters benchmark");
    #[cfg(feature = "simd")]
    println!("# (simd feature enabled — pulp runtime ISA dispatch active)");
    #[cfg(not(feature = "simd"))]
    println!("# (simd feature off — scalar only)");
    println!();
    bench_center();
    println!();
    bench_z_score();
    println!();
    bench_min_max();
    println!();
    bench_box_cox();
    println!();
    bench_holt_winters();
}
