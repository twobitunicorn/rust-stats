//! Benchmark rust-stats on synthetic datasets and print median wall-clock
//! per call. Pair with `tests/golden/bench_statsmodels.py` for parity.
//!
//! Run with:
//!   cargo run --release --example bench

use rust_stats::smoothing::loess;
use rust_stats::tsa::{seasonal_decompose, stl, DecomposeMode, SeasonalDecomposeOpts, StlOpts};
use rust_stats::{Matrix, Ols};
use std::time::Instant;

/// xorshift64 RNG so we don't pull in a crate just for benchmark inputs.
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
    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-300);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn ols_inputs(n: usize, p: usize, seed: u64) -> (Vec<f64>, Matrix<f64>) {
    let mut rng = Rng::new(seed);
    let x_data: Vec<f64> = (0..n * p).map(|_| rng.normal()).collect();
    let x = Matrix::from_fn(n, p, |i, j| x_data[i * p + j]);
    let beta: Vec<f64> = (0..p).map(|j| 0.5 + j as f64 * 0.1).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = 1.0;
            for j in 0..p {
                acc += beta[j] * x_data[i * p + j];
            }
            acc + rng.normal() * 0.5
        })
        .collect();
    (y, x)
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
    // Warmup — run once, ignore.
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
    println!(
        "{label:<22} n={n:<6} {extra:<20} {:>10.3} ms",
        secs * 1e3
    );
}

fn bench_ols() {
    for &(n, p, iters) in &[(100usize, 5usize, 200), (1_000, 10, 100), (10_000, 20, 30)] {
        let (y, x) = ols_inputs(n, p, 0xC0FFEE);
        let secs = time_iters(iters, || {
            let res = Ols::new(&y, x.as_ref()).fit().unwrap();
            // Force inference + coef materialisation so we measure a realistic call.
            let _ = res.inference(rust_stats::CovType::NonRobust);
            let _ = res.r_squared();
        });
        report("ols + nonrobust inf", n, &format!("p={p}"), secs);
    }
    for &(n, p, iters) in &[(1_000usize, 10usize, 100), (10_000, 20, 30)] {
        let (y, x) = ols_inputs(n, p, 0xC0FFEE);
        let secs = time_iters(iters, || {
            let res = Ols::new(&y, x.as_ref()).fit().unwrap();
            let _ = res.inference(rust_stats::CovType::HC3);
        });
        report("ols + HC3 inf", n, &format!("p={p}"), secs);
    }
}

fn bench_loess() {
    for &(n, span, iters) in &[(100usize, 0.3, 100), (1_000, 0.3, 30), (5_000, 0.3, 10)] {
        let mut rng = Rng::new(0xBEEF);
        let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let secs = time_iters(iters, || {
            let _ = loess(&y, span, 1).unwrap();
        });
        report("loess (deg=1)", n, &format!("span={span}"), secs);
    }
}

fn bench_stl() {
    for &(n, period, iters) in &[(144usize, 12usize, 50), (720, 12, 20), (2_880, 24, 10)] {
        let y = series_with_seasonality(n, period, 0xCAFE);
        let secs = time_iters(iters, || {
            let _ = stl(&y, StlOpts::new(period as u32)).unwrap();
        });
        report("stl", n, &format!("period={period}"), secs);
    }
}

fn bench_seasonal_decompose() {
    for &(n, period, iters) in &[(144usize, 12usize, 200), (720, 12, 100), (2_880, 24, 50)] {
        let y = series_with_seasonality(n, period, 0xCAFE);
        let secs_add = time_iters(iters, || {
            let mut opts = SeasonalDecomposeOpts::new(period as u32);
            opts.mode = DecomposeMode::Additive;
            let _ = seasonal_decompose(&y, opts).unwrap();
        });
        report("seasonal_decompose +", n, &format!("period={period}"), secs_add);
        let secs_mul = time_iters(iters, || {
            let mut opts = SeasonalDecomposeOpts::new(period as u32);
            opts.mode = DecomposeMode::Multiplicative;
            let _ = seasonal_decompose(&y, opts).unwrap();
        });
        report("seasonal_decompose *", n, &format!("period={period}"), secs_mul);
    }
}

fn main() {
    println!("# rust-stats benchmark");
    println!();
    bench_ols();
    println!();
    bench_loess();
    println!();
    bench_stl();
    println!();
    bench_seasonal_decompose();
}
