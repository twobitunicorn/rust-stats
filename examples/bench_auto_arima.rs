//! Benchmark rust-stats `auto_arima` against pmdarima.auto_arima.
//!
//! Both perform a Hyndman-Khandakar stepwise search over (p, d, q)
//! (P, D, Q). The fundamental difference is the per-candidate fit
//! cost: rust-stats uses CSS by default (no Kalman filter); pmdarima
//! drives statsmodels' SARIMAX (Kalman + L-BFGS) for every candidate.
//!
//! Run with:
//!
//!   cargo run --release --example bench_auto_arima
//!
//! Pair with `tests/golden/bench_auto_arima_pmdarima.py`.

use rust_stats::{auto_arima, ArimaMethod, AutoArimaOpts};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s) }
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

fn simulate_arma(n: usize, phi: &[f64], theta: &[f64], sigma: f64, seed: u64) -> Vec<f64> {
    let burn = 200;
    let total = n + burn;
    let mut rng = Rng::new(seed);
    let mut eps = vec![0.0; total];
    let mut y = vec![0.0; total];
    let p = phi.len();
    let q = theta.len();
    for t in 0..total {
        eps[t] = sigma * rng.normal();
        let mut yt = eps[t];
        for i in 0..p.min(t) { yt += phi[i] * y[t - 1 - i]; }
        for i in 0..q.min(t) { yt += theta[i] * eps[t - 1 - i]; }
        y[t] = yt;
    }
    y[burn..].to_vec()
}

fn integrate_once(y: &[f64], start: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    let mut r = start;
    for v in y { r += v; out.push(r); }
    out
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

fn report(label: &str, n: usize, secs: f64) {
    println!("{label:<40} n={n:<6}             {:>10.2} ms", secs * 1e3);
}

fn bench_nonseasonal(method: ArimaMethod, label: &str) {
    println!("\nNon-seasonal, {}:", label);
    for &(n, iters) in &[(144usize, 10), (720, 5), (2880, 2)] {
        let arma = simulate_arma(n, &[0.5], &[-0.3], 1.0, 0xAA1);
        let y = integrate_once(&arma, 100.0);
        let mut opts = AutoArimaOpts::new();
        opts.method = method;
        let secs = time_iters(iters, || {
            let _ = auto_arima(&y, opts).unwrap();
        });
        report(&format!("auto_arima ({label})"), n, secs);
    }
}

fn bench_seasonal(method: ArimaMethod, label: &str) {
    println!("\nSeasonal airline [m=12], {}:", label);
    for &(n, iters) in &[(144usize, 3), (288, 2)] {
        let arma = simulate_arma(n, &[], &[-0.4], 1.0, 0xAA6);
        let mut y: Vec<f64> = arma
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let trend = 0.05 * i as f64;
                let phase = 2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0;
                v + trend + 3.0 * phase.sin() + 100.0
            })
            .collect();
        for k in 1..n { y[k] += y[k - 1] * 0.001; }
        let mut opts = AutoArimaOpts::seasonal(12);
        opts.method = method;
        let secs = time_iters(iters, || {
            let _ = auto_arima(&y, opts).unwrap();
        });
        report(&format!("auto_arima [m=12] ({label})"), n, secs);
    }
}

fn main() {
    println!("# rust-stats auto_arima benchmark");
    // CSS = default; fast per candidate. The realistic "I just want a
    // quick auto-fit" path.
    bench_nonseasonal(ArimaMethod::Css, "CSS");
    bench_seasonal(ArimaMethod::Css, "CSS");
    // MLE = same objective as pmdarima per candidate. Apples-to-apples.
    bench_nonseasonal(ArimaMethod::Mle, "MLE — same objective as pmdarima");
    bench_seasonal(ArimaMethod::Mle, "MLE — same objective as pmdarima");
}
