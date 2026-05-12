//! Stationarity diagnostics — KPSS test and the `ndiffs` / `nsdiffs`
//! helpers that `auto_arima` uses internally to pick `d` and `D`.
//!
//! The KPSS null is *stationarity* — we reject (and difference) when
//! the statistic exceeds the 5% critical value. Three series here:
//! white noise (stationary; should not reject), a random walk
//! (non-stationary; should reject), and the random walk after one
//! difference (should not reject — differencing fixed it).
//!
//! Run with:
//!
//!   cargo run --release --example stationarity

use rust_stats::tsa::stationarity::{
    kpss, ndiffs, nsdiffs, seasonal_strength, KpssRegression,
};

fn main() {
    // ── Build three series with a deterministic noise generator. ────
    let mut state = 0xCAFEu64;
    let mut normal = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state as f64 / u64::MAX as f64).max(1e-300);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = state as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let n = 1_000usize;
    let white_noise: Vec<f64> = (0..n).map(|_| 5.0 + normal()).collect();
    let mut random_walk = vec![0.0f64; n];
    for t in 1..n {
        random_walk[t] = random_walk[t - 1] + normal();
    }
    let diff_rw: Vec<f64> = (1..n)
        .map(|i| random_walk[i] - random_walk[i - 1])
        .collect();

    // ── KPSS on each. ───────────────────────────────────────────────
    println!("KPSS test (level stationarity; α = 0.05):\n");
    println!("                            statistic    crit(5%)   p ≈    reject?");
    for (name, y) in [
        ("white noise",      white_noise.as_slice()),
        ("random walk",      random_walk.as_slice()),
        ("Δ(random walk)",   diff_rw.as_slice()),
    ] {
        let r = kpss(y, KpssRegression::Constant);
        println!(
            "  {name:<25}   {:7.4}     {:6.3}   {:5.3}    {}",
            r.statistic,
            r.critical_5pct,
            r.p_value,
            if r.reject_stationarity { "yes (difference)" } else { "no" },
        );
    }
    println!("\n→ The random walk clearly needs differencing.");
    println!("  (KPSS at finite n can be twitchy: the once-differenced series may");
    println!("   sit near the critical value for some seeds. That's why `ndiffs`");
    println!("   re-tests after each difference rather than committing to d=1.)");

    // ── ndiffs picks d automatically by iterated KPSS. ──────────────
    println!("\nndiffs picks `d` by iterating KPSS until stationary (cap = 2):");
    for (name, y) in [
        ("white noise",     white_noise.as_slice()),
        ("random walk",     random_walk.as_slice()),
        ("Δ(random walk)",  diff_rw.as_slice()),
    ] {
        println!("  {name:<25} → d = {}", ndiffs(y, 2));
    }

    // ── Seasonal differencing via `seasonal_strength` / `nsdiffs`. ──
    //    Build a strongly-seasonal monthly series; show that the
    //    Hyndman strength measure ranks the seasonal component above
    //    the 0.64 threshold (so `nsdiffs` returns D = 1).
    let m = 12usize;
    let seasonal: Vec<f64> = (0..240)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * (i % m) as f64 / m as f64;
            10.0 + 0.05 * i as f64 + 3.0 * phase.sin() + 0.3 * normal()
        })
        .collect();

    let strength = seasonal_strength(&seasonal, m as u32);
    let big_d = nsdiffs(&seasonal, m as u32, 1);

    println!("\nSeasonal diagnostics on a monthly seasonal series:");
    println!("  seasonal_strength = {:.3}  (threshold: 0.64)", strength);
    println!("  nsdiffs(..., max=1) → D = {}", big_d);
    println!("  → auto_arima would apply one seasonal difference (1 − B^12).");

    // Compare with a non-seasonal noise series.
    let noisy: Vec<f64> = (0..240).map(|_| 10.0 + normal()).collect();
    let strength_noise = seasonal_strength(&noisy, m as u32);
    println!(
        "\nFor a plain white-noise series (no real seasonality): strength = {:.3} → D = {}",
        strength_noise,
        nsdiffs(&noisy, m as u32, 1),
    );
}
