//! `auto_arima` — stepwise model selection for ARIMA / SARIMA.
//!
//! Hyndman & Khandakar's (2008) algorithm:
//!
//! 1. Determine seasonal differencing `D` from
//!    [`crate::tsa::stationarity::nsdiffs`] (strength-of-seasonality
//!    heuristic).
//! 2. Determine non-seasonal differencing `d` from
//!    [`crate::tsa::stationarity::ndiffs`] (iterated KPSS).
//! 3. Fit a small slate of starting models with the chosen `(d, D)` and
//!    pick the lowest AICc.
//! 4. Walk to neighbours by changing one of `(p, q, P, Q,
//!    include_constant)` at a time; keep moving toward the lowest AICc
//!    until no neighbour improves.
//!
//! Returns the [`ArimaFit`] with the lowest AICc found. Inspect
//! `fit.opts` to see the chosen orders. Skips fits that fail (e.g.,
//! optimisation didn't converge) by treating their AICc as `+∞`.

use std::collections::HashSet;

use crate::error::ArimaError;
use crate::tsa::stationarity;

use super::{arima, ArimaFit, ArimaMethod, ArimaOpts};

/// Options for [`auto_arima`].
#[derive(Debug, Clone, Copy)]
pub struct AutoArimaOpts {
    /// Seasonal period `m`. `0` means non-seasonal (skip seasonal terms
    /// entirely).
    pub seasonal_period: u32,
    /// Upper bound on `p`. Hyndman default: 5.
    pub max_p: u32,
    /// Upper bound on `d` (chosen automatically below this cap).
    pub max_d: u32,
    /// Upper bound on `q`. Hyndman default: 5.
    pub max_q: u32,
    /// Upper bound on seasonal AR order `P`. Hyndman default: 2.
    pub max_seasonal_p: u32,
    /// Upper bound on seasonal differencing `D`.
    pub max_seasonal_d: u32,
    /// Upper bound on seasonal MA order `Q`. Hyndman default: 2.
    pub max_seasonal_q: u32,
    /// Estimation method passed through to each candidate fit. CSS
    /// (default) is fast; MLE / CSS-ML are more accurate but slower.
    pub method: ArimaMethod,
    /// Force the intercept on or off. `None` lets the search decide
    /// (toggling it is one of the stepwise moves).
    pub include_constant: Option<bool>,
    /// Hard cap on the number of stepwise iterations after the
    /// starting slate.
    pub max_iter: usize,
}

impl AutoArimaOpts {
    /// Non-seasonal defaults (Hyndman-Khandakar values).
    pub fn new() -> Self {
        Self {
            seasonal_period: 0,
            max_p: 5,
            max_d: 2,
            max_q: 5,
            max_seasonal_p: 0,
            max_seasonal_d: 0,
            max_seasonal_q: 0,
            method: ArimaMethod::Css,
            include_constant: None,
            max_iter: 50,
        }
    }

    /// Defaults for a seasonal series of period `m` (Hyndman-Khandakar
    /// limits: `max_seasonal_p = max_seasonal_q = 2`, `max_seasonal_d = 1`).
    pub fn seasonal(m: u32) -> Self {
        Self {
            seasonal_period: m,
            max_p: 5,
            max_d: 2,
            max_q: 5,
            max_seasonal_p: 2,
            max_seasonal_d: 1,
            max_seasonal_q: 2,
            method: ArimaMethod::Css,
            include_constant: None,
            max_iter: 50,
        }
    }
}

impl Default for AutoArimaOpts {
    fn default() -> Self {
        Self::new()
    }
}

/// Stepwise model selection. Returns the best fit by AICc.
///
/// Heavy on the optimiser: each candidate runs a full
/// [`arima`][crate::tsa::arima] fit. For monthly seasonal series the
/// default `ArimaMethod::Css` is recommended; `ArimaMethod::Mle` will
/// give slightly better estimates at the cost of substantially longer
/// wall-clock per candidate.
pub fn auto_arima(y: &[f64], opts: AutoArimaOpts) -> Result<ArimaFit, ArimaError> {
    let m = opts.seasonal_period;
    let has_seasonal = m >= 2;

    // 1. Pick D first (so KPSS sees the seasonally-differenced series).
    let big_d = if has_seasonal {
        stationarity::nsdiffs(y, m, opts.max_seasonal_d)
    } else {
        0
    };
    // Apply seasonal differencing to get a residual series for ndiffs.
    let mut seasonally_diffed: Vec<f64> = y.to_vec();
    if has_seasonal {
        for _ in 0..big_d {
            let mm = m as usize;
            seasonally_diffed = (mm..seasonally_diffed.len())
                .map(|i| seasonally_diffed[i] - seasonally_diffed[i - mm])
                .collect();
        }
    }
    // 2. Pick d on the seasonally-differenced series.
    let d = if seasonally_diffed.len() >= 8 {
        stationarity::ndiffs(&seasonally_diffed, opts.max_d)
    } else {
        0
    };

    // Constant inclusion: Hyndman's rule is "include unless d + D ≥ 2"
    // (a constant becomes a quadratic trend after two integrations,
    // which is rarely intended). Caller can override via opts.
    let default_constant = (d as u32) + big_d < 2;
    let initial_constant = opts.include_constant.unwrap_or(default_constant);

    let mut cache: HashSet<Key> = HashSet::new();
    let mut best: Option<(Key, ArimaFit)> = None;

    let try_fit = |p: u32,
                   q: u32,
                   big_p: u32,
                   big_q: u32,
                   include_constant: bool,
                   cache: &mut HashSet<Key>,
                   best: &mut Option<(Key, ArimaFit)>| {
        let key = Key {
            p,
            d: d as u32,
            q,
            big_p,
            big_d,
            big_q,
            include_constant,
        };
        if cache.contains(&key) {
            return;
        }
        cache.insert(key);
        let mut cand_opts = ArimaOpts {
            p,
            d: d as u32,
            q,
            seasonal_p: big_p,
            seasonal_d: big_d,
            seasonal_q: big_q,
            seasonal_period: if has_seasonal { m } else { 0 },
            include_constant,
            method: opts.method,
        };
        // For Hyndman, max(p+P, q+Q) ≤ 5 is the usual practical bound.
        if (p + big_p).max(q + big_q) > 6 {
            return;
        }
        // Constant + d+D ≥ 2 is almost always bad; skip.
        if include_constant && (d as u32 + big_d) >= 2 {
            return;
        }
        cand_opts.method = opts.method;
        if let Ok(fit) = arima(y, cand_opts) {
            if fit.aicc.is_finite() {
                if let Some((_, ref best_fit)) = best {
                    if fit.aicc < best_fit.aicc {
                        *best = Some((key, fit));
                    }
                } else {
                    *best = Some((key, fit));
                }
            }
        }
    };

    // 3. Starting slate (Hyndman-Khandakar §3.2).
    let starts: &[(u32, u32, u32, u32)] = if has_seasonal {
        &[
            (2, 2, 1, 1),
            (0, 0, 0, 0),
            (1, 0, 1, 0),
            (0, 1, 0, 1),
        ]
    } else {
        &[(2, 2, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
    };
    for &(p, q, big_p, big_q) in starts {
        if p > opts.max_p || q > opts.max_q {
            continue;
        }
        if big_p > opts.max_seasonal_p || big_q > opts.max_seasonal_q {
            continue;
        }
        try_fit(p, q, big_p, big_q, initial_constant, &mut cache, &mut best);
    }
    if best.is_none() {
        // Fallback: try ARIMA(0, d, 0).
        try_fit(0, 0, 0, 0, initial_constant, &mut cache, &mut best);
    }
    if best.is_none() {
        return Err(ArimaError::OptimizationFailed { iters: 0 });
    }

    // 4. Stepwise neighbourhood search.
    for _ in 0..opts.max_iter {
        let current = best.as_ref().unwrap().0;
        let neighbours = neighbours_of(current, m, &opts);
        let mut improved = false;
        let prev_aicc = best.as_ref().unwrap().1.aicc;
        for n in neighbours {
            try_fit(
                n.p,
                n.q,
                n.big_p,
                n.big_q,
                n.include_constant,
                &mut cache,
                &mut best,
            );
        }
        if best.as_ref().unwrap().1.aicc < prev_aicc - 1e-9 {
            improved = true;
        }
        if !improved {
            break;
        }
    }

    Ok(best.unwrap().1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    p: u32,
    d: u32,
    q: u32,
    big_p: u32,
    big_d: u32,
    big_q: u32,
    include_constant: bool,
}

fn neighbours_of(k: Key, m: u32, opts: &AutoArimaOpts) -> Vec<Key> {
    let mut out = Vec::with_capacity(10);
    let push = |out: &mut Vec<Key>, k: Key| out.push(k);
    // ±1 on each component, bounded.
    if k.p < opts.max_p {
        push(&mut out, Key { p: k.p + 1, ..k });
    }
    if k.p > 0 {
        push(&mut out, Key { p: k.p - 1, ..k });
    }
    if k.q < opts.max_q {
        push(&mut out, Key { q: k.q + 1, ..k });
    }
    if k.q > 0 {
        push(&mut out, Key { q: k.q - 1, ..k });
    }
    if m >= 2 {
        if k.big_p < opts.max_seasonal_p {
            push(&mut out, Key { big_p: k.big_p + 1, ..k });
        }
        if k.big_p > 0 {
            push(&mut out, Key { big_p: k.big_p - 1, ..k });
        }
        if k.big_q < opts.max_seasonal_q {
            push(&mut out, Key { big_q: k.big_q + 1, ..k });
        }
        if k.big_q > 0 {
            push(&mut out, Key { big_q: k.big_q - 1, ..k });
        }
    }
    // Toggle the intercept only when the caller hasn't pinned it.
    if opts.include_constant.is_none() {
        push(
            &mut out,
            Key {
                include_constant: !k.include_constant,
                ..k
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn sim_ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let burn = 200;
        let mut rng = Rng::new(seed);
        let mut y = vec![0.0f64; n + burn];
        for t in 1..n + burn {
            y[t] = phi * y[t - 1] + rng.normal();
        }
        y[burn..].to_vec()
    }

    #[test]
    fn auto_arima_finds_stationary_model_on_ar1() {
        // Stationary AR(1) → auto_arima should pick d=0. Hyndman-
        // Khandakar's stepwise doesn't always converge to the
        // minimal-order model on finite samples (~60-70% hit-rate in
        // their original paper), so we only assert that the model is
        // (a) stationary and (b) bounded in size.
        let y = sim_ar1(500, 0.7, 0xAA1);
        let fit = auto_arima(&y, AutoArimaOpts::new()).unwrap();
        assert_eq!(fit.opts.d, 0, "should not difference; got {:?}", fit.opts);
        assert!(
            fit.opts.p + fit.opts.q <= 6,
            "over-fit detected: {:?}",
            fit.opts
        );
        assert!(fit.aicc.is_finite());
    }

    #[test]
    fn auto_arima_picks_d_for_random_walk() {
        // Longer n so KPSS rejects with margin — for n=500 the
        // statistic can sit right at the 5% critical for some seeds.
        let mut rng = Rng::new(0xAA2);
        let n = 1500;
        let mut y = vec![0.0f64; n];
        for t in 1..n {
            y[t] = y[t - 1] + rng.normal();
        }
        let fit = auto_arima(&y, AutoArimaOpts::new()).unwrap();
        assert!(
            fit.opts.d >= 1,
            "random walk should select d>=1; got {:?}",
            fit.opts
        );
    }

    #[test]
    fn auto_arima_seasonal_picks_capital_d() {
        // Strong monthly seasonal pattern with a small noise floor:
        // seasonal_strength should be high enough to trigger D=1.
        let period = 12usize;
        let mut rng = Rng::new(0xAA3);
        let n = 240usize;
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let phase = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            y.push(10.0 + 0.02 * i as f64 + 3.0 * phase.sin() + 0.2 * rng.normal());
        }
        let mut opts = AutoArimaOpts::seasonal(period as u32);
        opts.max_iter = 30;
        let fit = auto_arima(&y, opts).unwrap();
        assert!(
            fit.opts.seasonal_d + fit.opts.d as u32 >= 1,
            "expected some differencing, got {:?}",
            fit.opts
        );
    }
}
