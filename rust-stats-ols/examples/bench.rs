//! Benchmark rust-stats-ols on synthetic datasets. Pair with
//! tests/golden/bench_statsmodels.py.
//!
//! Run with:
//!   cargo run --release -p rust-stats-ols --example bench

use rust_stats_ols::{CovType, Matrix, Ols};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
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
    println!("{label:<22} n={n:<6} {extra:<20} {:>10.3} ms", secs * 1e3);
}

fn main() {
    println!("# rust-stats-ols benchmark");
    println!();

    for &(n, p, iters) in &[(100usize, 5usize, 200), (1_000, 10, 100), (10_000, 20, 30)] {
        let (y, x) = ols_inputs(n, p, 0xC0FFEE);
        let secs = time_iters(iters, || {
            let res = Ols::new(&y, x.as_ref()).fit().unwrap();
            let _ = res.inference(CovType::NonRobust);
            let _ = res.r_squared();
        });
        report("ols + nonrobust inf", n, &format!("p={p}"), secs);
    }

    for &(n, p, iters) in &[(1_000usize, 10usize, 100), (10_000, 20, 30)] {
        let (y, x) = ols_inputs(n, p, 0xC0FFEE);
        let secs = time_iters(iters, || {
            let res = Ols::new(&y, x.as_ref()).fit().unwrap();
            let _ = res.inference(CovType::HC3);
        });
        report("ols + HC3 inf", n, &format!("p={p}"), secs);
    }
}
