//! Per-fit benchmark for `arima_with_exog` — joint MLE for ARIMAX.
//!
//! Each cell fits a single model on a synthetic series with `k` known
//! exogenous regressors and ARIMA-correlated residuals. Sizes mirror
//! the existing `bench_arima.rs` triad (144, 720, 2880).

use rust_stats::{arima_with_exog, ArimaMethod, ArimaOpts};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Self(s)
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
    println!("{label:<28} n={n:<6} {extra:<14} {:>10.2} ms", secs * 1e3);
}

/// Simulate y = β₀ + β·x + ARIMA(p, 0, q) error.
fn sim_arimax(
    n: usize,
    beta0: f64,
    beta: &[f64],
    phi: &[f64],
    theta: &[f64],
    sigma: f64,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = beta.len();
    let mut rng = Rng::new(seed);
    let mut exog: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    // Add a deterministic structure to one regressor so it's identifiable
    if k > 0 {
        for i in 0..n {
            exog[0][i] += (i as f64 * 0.1).sin();
        }
    }
    let mut eps = vec![0.0; n];
    let mut e = vec![0.0; n];
    let p = phi.len();
    let q = theta.len();
    for t in 0..n {
        eps[t] = sigma * rng.normal();
        let mut et = eps[t];
        for i in 0..p.min(t) {
            et += phi[i] * e[t - 1 - i];
        }
        for i in 0..q.min(t) {
            et += theta[i] * eps[t - 1 - i];
        }
        e[t] = et;
    }
    let mut y = vec![0.0; n];
    for t in 0..n {
        y[t] = beta0 + e[t];
        for j in 0..k {
            y[t] += beta[j] * exog[j][t];
        }
    }
    (y, exog)
}

fn bench_one(label: &str, y: &[f64], exog: &[Vec<f64>], order: (u32, u32, u32), iters: usize) {
    let exog_ref: Vec<&[f64]> = exog.iter().map(|v| v.as_slice()).collect();
    for method in [ArimaMethod::Css, ArimaMethod::Mle, ArimaMethod::CssMle] {
        let mut opts = ArimaOpts::new(order.0, order.1, order.2);
        opts.method = method;
        let secs = time_iters(iters, || {
            let _ = arima_with_exog(y, &exog_ref, opts).unwrap();
        });
        let m = match method {
            ArimaMethod::Css => "CSS",
            ArimaMethod::Mle => "MLE",
            ArimaMethod::CssMle => "CSS-ML",
        };
        report(&format!("{label} ({m})"), y.len(), &format!("k={}", exog.len()), secs);
    }
}

fn main() {
    println!("# rust-stats ARIMAX joint-MLE benchmark");
    println!("# Each fit: arima_with_exog(y, exog, ArimaOpts(p, d, q))\n");

    // ARIMAX(1, 0, 0) with k = 2 regressors. Mirrors the workload in
    // examples/arimax.rs but at three different sizes.
    for &(n, iters) in &[(144usize, 30), (720, 10), (2_880, 3)] {
        let (y, exog) = sim_arimax(n, 100.0, &[3.0, -1.5], &[0.4], &[], 1.0, 0xA12A_F00D);
        bench_one("ARIMAX(1,0,0)", &y, &exog, (1, 0, 0), iters);
    }
    println!();

    // ARIMAX(1, 1, 1) with k = 1. Common forecasting shape: differencing
    // + one regressor.
    for &(n, iters) in &[(144usize, 20), (720, 5), (2_880, 2)] {
        let (y, exog) = sim_arimax(n, 50.0, &[2.0], &[0.5], &[-0.3], 1.0, 0xA1AB1A);
        bench_one("ARIMAX(1,1,1)", &y, &exog, (1, 1, 1), iters);
    }
    println!();

    // SARIMAX airline with k = 1 (e.g., a holiday dummy). The seasonal
    // case is where joint MLE matters most — the inner L-BFGS path
    // works on the joined parameter vector.
    for &(n, iters) in &[(144usize, 5), (288, 2)] {
        let (y, exog) = sim_arimax(n, 100.0, &[5.0], &[], &[-0.4], 1.0, 0xA12A_BBBB);
        let exog_ref: Vec<&[f64]> = exog.iter().map(|v| v.as_slice()).collect();
        for method in [ArimaMethod::Css, ArimaMethod::Mle, ArimaMethod::CssMle] {
            let mut opts = ArimaOpts::seasonal(0, 1, 1, 0, 1, 1, 12);
            opts.method = method;
            let secs = time_iters(iters, || {
                let _ = arima_with_exog(&y, &exog_ref, opts).unwrap();
            });
            let m = match method {
                ArimaMethod::Css => "CSS",
                ArimaMethod::Mle => "MLE",
                ArimaMethod::CssMle => "CSS-ML",
            };
            report(
                &format!("SARIMAX(0,1,1)(0,1,1)[12] ({m})"),
                n,
                "k=1",
                secs,
            );
        }
    }
}
