//! Benchmark rust-stats on synthetic datasets and print median wall-clock
//! per call. Pair with `tests/golden/bench_statsmodels.py` for parity.
//!
//! Run with:
//!   cargo run --release --example bench
//!   cargo run --release --features arrow --example bench   # adds batched section

use rust_stats::smoothing::loess;
use rust_stats::tsa::{seasonal_decompose, stl, DecomposeMode, SeasonalDecomposeOpts, StlOpts};
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
    println!("{label:<22} n={n:<6} {extra:<20} {:>10.3} ms", secs * 1e3);
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

#[cfg(feature = "arrow")]
fn bench_batched() {
    use arrow::array::{Array, Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use rust_stats::arrow_compat;
    use std::sync::Arc;

    fn make_batch(n: usize, p: usize, period: usize) -> RecordBatch {
        let mut fields = Vec::with_capacity(p);
        let mut cols: Vec<Arc<dyn Array>> = Vec::with_capacity(p);
        for j in 0..p {
            let s = series_with_seasonality(n, period, 0xABCD ^ (j as u64));
            fields.push(Field::new(format!("c{j}"), DataType::Float64, true));
            cols.push(Arc::new(Float64Array::from(s)));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    for &(n, p, period, iters) in
        &[(1_000usize, 50usize, 12usize, 20), (720, 50, 12, 30), (2_880, 50, 24, 10)]
    {
        let batch = make_batch(n, p, period);
        let secs = time_iters(iters, || {
            let _ = arrow_compat::stl_batch(&batch, StlOpts::new(period as u32)).unwrap();
        });
        report("stl_batch (rayon)", n, &format!("p={p} period={period}"), secs);
    }

    for &(n, p, period, iters) in
        &[(1_000usize, 50usize, 12usize, 20), (720, 50, 12, 30), (2_880, 50, 24, 10)]
    {
        let batch = make_batch(n, p, period);
        let secs = time_iters(iters, || {
            let _ = arrow_compat::seasonal_decompose_batch(
                &batch,
                SeasonalDecomposeOpts::new(period as u32),
            )
            .unwrap();
        });
        report("seasonal_decompose_batch", n, &format!("p={p} period={period}"), secs);
    }

    for &(n, p, iters) in &[(1_000usize, 50usize, 20), (5_000, 50, 5)] {
        let batch = make_batch(n, p, 12);
        let secs = time_iters(iters, || {
            let _ = arrow_compat::loess_batch(&batch, 0.3, 1).unwrap();
        });
        report("loess_batch (rayon)", n, &format!("p={p} span=0.3"), secs);
    }
}

#[cfg(not(feature = "arrow"))]
fn bench_batched() {
    println!("(skipping batched bench — rebuild with --features arrow)");
}

fn main() {
    println!("# rust-stats benchmark");
    println!();
    bench_loess();
    println!();
    bench_stl();
    println!();
    bench_seasonal_decompose();
    println!();
    bench_batched();
}
