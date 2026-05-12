//! Scaling sweep: how long does ARIMA(1, 1, 1) take as n grows?
//!
//! Sweeps n = 10⁴, 10⁵, 10⁶, 10⁷ for each estimation method. The MLE
//! / CSS-ML rows are skipped past n = 10⁶ because they'd take too long
//! to be useful as a benchmark.
//!
//! Run with:
//!
//!   cargo run --release --example bench_scaling
//!
//! Pair with `tests/golden/bench_scaling_r.R` and
//! `tests/golden/bench_scaling_statsmodels.py`.

use rust_stats::{arima, ArimaMethod, ArimaOpts};
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

/// Simulate ARIMA(1, 1, 1) of length `n`: φ=0.5, θ=-0.3, drift=0.1.
fn simulate_arima_111(n: usize) -> Vec<f64> {
    let phi = 0.5;
    let theta = -0.3;
    let mut rng = Rng::new(0x5CA1ED);
    let mut eps = vec![0.0f64; n];
    let mut diff = vec![0.0f64; n];
    for t in 0..n {
        eps[t] = rng.normal();
        let phi_term = if t >= 1 { phi * diff[t - 1] } else { 0.0 };
        let theta_term = if t >= 1 { theta * eps[t - 1] } else { 0.0 };
        diff[t] = 0.1 + phi_term + theta_term + eps[t];
    }
    let mut y = vec![0.0f64; n];
    y[0] = 100.0;
    for t in 1..n {
        y[t] = y[t - 1] + diff[t];
    }
    y
}

fn time_one<F: FnOnce()>(label: &str, n: usize, f: F) {
    let start = Instant::now();
    f();
    let secs = start.elapsed().as_secs_f64();
    let rate_us_per_pt = secs * 1e6 / n as f64;
    println!(
        "  {label:<10}  n={n:>10}    {:>10.3} s    {rate_us_per_pt:>7.2} µs/pt",
        secs
    );
}

fn main() {
    println!("# scaling sweep: ARIMA(1, 1, 1) — one fit per cell, no warmup\n");
    println!("  method      n              time          throughput");

    for &n in &[10_000usize, 100_000, 1_000_000, 10_000_000] {
        let y = simulate_arima_111(n);

        // CSS — the fast default. Should scale near-linearly.
        time_one("CSS", n, || {
            let mut opts = ArimaOpts::new(1, 1, 1);
            opts.method = ArimaMethod::Css;
            arima(&y, opts).unwrap();
        });

        // CSS-ML / MLE — matches R's `arima` default and statsmodels'
        // SARIMAX default respectively. Skip past 10⁶ because the
        // Kalman filter pass × Nelder-Mead iters gets expensive.
        if n <= 1_000_000 {
            time_one("CSS-ML", n, || {
                let mut opts = ArimaOpts::new(1, 1, 1);
                opts.method = ArimaMethod::CssMle;
                arima(&y, opts).unwrap();
            });
            time_one("MLE", n, || {
                let mut opts = ArimaOpts::new(1, 1, 1);
                opts.method = ArimaMethod::Mle;
                arima(&y, opts).unwrap();
            });
        } else {
            println!("  (CSS-ML / MLE skipped at n={n} — Kalman MLE is too slow to be useful here)");
        }
        println!();
    }
}
