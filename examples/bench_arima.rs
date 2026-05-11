//! Benchmark rust-stats' ARIMA across workloads and estimation methods.
//! Pair with `tests/golden/bench_arima_statsmodels.py` for parity.
//!
//! Run with:
//!
//!   cargo run --release --example bench_arima
//!
//! Workload sizes mirror the existing TSA bench triad (144 monthly,
//! 720 daily, 2880 hourly). Three methods are exercised per workload:
//! CSS (rust-stats default), MLE (Kalman likelihood), CSS-ML (CSS for
//! initial values then MLE refinement — R's default).

use rust_stats::tsa::{arima, ArimaMethod, ArimaOpts};
use std::time::Instant;

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

/// Simulate ARMA(p, q) with the given (phi, theta) and σ. Burns 200
/// observations to remove start-up transients.
fn simulate_arma(n: usize, phi: &[f64], theta: &[f64], sigma: f64, seed: u64) -> Vec<f64> {
    let burn = 200;
    let total = n + burn;
    let mut rng = Rng::new(seed);
    let mut eps = vec![0.0f64; total];
    let mut y = vec![0.0f64; total];
    let p = phi.len();
    let q = theta.len();
    for t in 0..total {
        eps[t] = sigma * rng.normal();
        let mut yt = eps[t];
        for i in 0..p.min(t) {
            yt += phi[i] * y[t - 1 - i];
        }
        for i in 0..q.min(t) {
            yt += theta[i] * eps[t - 1 - i];
        }
        y[t] = yt;
    }
    y[burn..].to_vec()
}

/// Cumulatively integrate an ARMA series once → ARIMA(p, 1, q).
fn integrate_once(y: &[f64], start: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    let mut running = start;
    for v in y {
        running += v;
        out.push(running);
    }
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

fn report(label: &str, n: usize, extra: &str, secs: f64) {
    println!("{label:<32} n={n:<6} {extra:<20} {:>10.2} ms", secs * 1e3);
}

fn method_label(m: ArimaMethod) -> &'static str {
    match m {
        ArimaMethod::Css => "CSS",
        ArimaMethod::Mle => "MLE",
        ArimaMethod::CssMle => "CSS-ML",
    }
}

fn bench_one(label: &str, y: &[f64], base_opts: ArimaOpts, iters: usize) {
    for method in [ArimaMethod::Css, ArimaMethod::Mle, ArimaMethod::CssMle] {
        let mut opts = base_opts;
        opts.method = method;
        let secs = time_iters(iters, || {
            let _ = arima(y, opts).unwrap();
        });
        report(
            &format!("{label} ({})", method_label(method)),
            y.len(),
            "",
            secs,
        );
    }
}

fn bench_ar1() {
    for &(n, iters) in &[(144usize, 50), (720, 20), (2880, 5)] {
        let y = simulate_arma(n, &[0.6], &[], 1.0, 0xA1);
        bench_one("ARIMA(1,0,0)", &y, ArimaOpts::new(1, 0, 0), iters);
    }
}

fn bench_ma1() {
    for &(n, iters) in &[(144usize, 50), (720, 20), (2880, 5)] {
        let y = simulate_arma(n, &[], &[0.5], 1.0, 0xA2);
        bench_one("ARIMA(0,0,1)", &y, ArimaOpts::new(0, 0, 1), iters);
    }
}

fn bench_arma11() {
    for &(n, iters) in &[(144usize, 30), (720, 15), (2880, 5)] {
        let y = simulate_arma(n, &[0.5], &[0.3], 1.0, 0xA3);
        bench_one("ARIMA(1,0,1)", &y, ArimaOpts::new(1, 0, 1), iters);
    }
}

fn bench_ima11() {
    for &(n, iters) in &[(144usize, 30), (720, 15), (2880, 5)] {
        let arma = simulate_arma(n, &[], &[-0.4], 1.0, 0xA4);
        let y = integrate_once(&arma, 100.0);
        bench_one("ARIMA(0,1,1)", &y, ArimaOpts::new(0, 1, 1), iters);
    }
}

fn bench_arima111() {
    for &(n, iters) in &[(144usize, 20), (720, 10), (2880, 3)] {
        let arma = simulate_arma(n, &[0.5], &[-0.3], 1.0, 0xA5);
        let y = integrate_once(&arma, 100.0);
        bench_one("ARIMA(1,1,1)", &y, ArimaOpts::new(1, 1, 1), iters);
    }
}

fn bench_sarima_airline() {
    // The "airline model": SARIMA(0, 1, 1)(0, 1, 1)[12] — Box-Jenkins'
    // workhorse for monthly seasonal series.
    for &(n, iters) in &[(144usize, 10), (288, 5)] {
        // Simulate a roughly-airline-shaped seasonal series.
        let arma = simulate_arma(n, &[], &[-0.4], 1.0, 0xA6);
        let mut y: Vec<f64> = arma.iter().enumerate().map(|(i, v)| {
            let trend = 0.05 * i as f64;
            let phase = 2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0;
            let seasonal = 3.0 * phase.sin();
            v + trend + seasonal + 100.0
        }).collect();
        // Lightly integrate to put first differences in a sensible scale.
        for i in 1..n { y[i] += y[i - 1] * 0.001; }
        bench_one(
            "SARIMA(0,1,1)(0,1,1)[12]",
            &y,
            ArimaOpts::seasonal(0, 1, 1, 0, 1, 1, 12),
            iters,
        );
    }
}

fn main() {
    println!("# rust-stats ARIMA benchmark");
    println!("# CSS = Conditional Sum of Squares");
    println!("# MLE = Kalman-filter Gaussian MLE");
    println!("# CSS-ML = CSS seed + MLE refinement (R's `arima` default)");
    println!();
    bench_ar1();
    println!();
    bench_ma1();
    println!();
    bench_arma11();
    println!();
    bench_ima11();
    println!();
    bench_arima111();
    println!();
    bench_sarima_airline();
}
