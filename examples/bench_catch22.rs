//! Benchmark rust-stats catch24 on synthetic Gaussian data and print
//! median wall-clock per call. Pair with
//! `tests/golden/bench_catch22_pycatch22.py` for parity.
//!
//! Run with:
//!   cargo run --release --example bench_catch22

use rust_stats::catch22::catch24;
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

fn gaussian(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal()).collect()
}

fn time_iters<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // Warmup.
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

fn main() {
    println!("\ncatch24 (22 catch22 features + DN_Mean + DN_Spread_Std)\n");
    println!("{:>8}  {:>10}", "n", "ms/call");
    println!("{}", "-".repeat(22));

    for &(n, iters) in &[
        (200_usize,   200),
        (1_000,        50),
        (5_000,        20),
        (20_000,       10),
        (50_000,        5),
        (100_000,       3),
    ] {
        let y = gaussian(n, 0xCAFEBABE);
        let secs = time_iters(iters, || {
            let _ = catch24(&y);
        });
        println!("{:>8}  {:>10.3}", n, secs * 1000.0);
    }
}
