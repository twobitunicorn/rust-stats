//! Extract catch22 features from four canonical synthetic signals
//! (white noise, sine wave, AR(1), random walk) and print the
//! 22-feature fingerprints side-by-side.
//!
//! The point of catch22 is that very different time-series classes
//! produce very different feature vectors — flip through the columns
//! below and you'll see, for example:
//!
//! - `CO_f1ecac` (ACF 1/e crossing) is small for white noise (~1) and
//!   large for the sine wave / random walk (slow-decaying ACF).
//! - `SB_BinaryStats_mean_longstretch1` (longest run above the mean)
//!   is short for noise, long for the sine wave.
//! - `SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1` (DFA exponent proxy)
//!   separates the random walk (~Hurst 0.5+) from white noise.
//!
//! Run with:
//!
//!   cargo run --release --example catch22_features

use rust_stats::catch22::{catch24_named, CATCH22_NAMES};

/// Tiny xorshift64 RNG so the example has no external dep.
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
    /// Box–Muller standard normal.
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-300);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn white_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal()).collect()
}

fn sine(n: usize, period: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin())
        .collect()
}

/// AR(1) with coefficient phi and unit-variance Gaussian innovations.
fn ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    let mut x = 0.0;
    for _ in 0..n {
        x = phi * x + rng.normal();
        out.push(x);
    }
    out
}

fn random_walk(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x += rng.normal();
            x
        })
        .collect()
}

fn main() {
    let n = 500;

    // Build the four signals.
    let signals: [(&str, Vec<f64>); 4] = [
        ("white_noise", white_noise(n, 0xC0FFEE)),
        ("sine_p20",    sine(n, 20.0)),
        ("ar1_phi0.8",  ar1(n, 0.8, 0xC0FFEE)),
        ("random_walk", random_walk(n, 0xC0FFEE)),
    ];

    // Compute catch24 (catch22 + DN_Mean + DN_Spread_Std) for each.
    let panels: Vec<[(&'static str, f64); 24]> =
        signals.iter().map(|(_, y)| catch24_named(y)).collect();

    // Header.
    print!("{:>44}", "feature");
    for (name, _) in &signals {
        print!("  {:>14}", name);
    }
    println!();
    println!("{}", "-".repeat(44 + 16 * signals.len()));

    // 22 catch22 features.
    for (i, fname) in CATCH22_NAMES.iter().enumerate() {
        print!("{:>44}", fname);
        for panel in &panels {
            print!("  {:>14.5}", panel[i].1);
        }
        println!();
    }

    // catch24 extras (raw mean / sample std).
    println!("{}", "-".repeat(44 + 16 * signals.len()));
    for i in 22..24 {
        print!("{:>44}", panels[0][i].0);
        for panel in &panels {
            print!("  {:>14.5}", panel[i].1);
        }
        println!();
    }

    println!(
        "\nLegend: each column is one length-{n} synthetic series; each row \
         is one of the 22 catch22 features (Lubba et al. 2019), plus the \
         two catch24 extras at the bottom."
    );
}
