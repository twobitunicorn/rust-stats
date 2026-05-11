//! Stationarity tests for time-series differencing decisions.
//!
//! - [`kpss`] — Kwiatkowski-Phillips-Schmidt-Shin test for *level* or
//!   *trend* stationarity. Null hypothesis: the series is stationary;
//!   reject (i.e., difference) when the statistic exceeds the critical
//!   value. Asymptotic critical values are tabulated from KPSS 1992.
//! - [`seasonal_strength`] — Hyndman's strength-of-seasonality measure
//!   from a classical additive decomposition. `auto_arima` uses it to
//!   decide whether seasonal differencing is needed; the rule of thumb
//!   is `D = 1` when strength > 0.64.

use crate::tsa::seasonal;

/// Result of a KPSS test.
#[derive(Debug, Clone, Copy)]
pub struct KpssResult {
    /// KPSS statistic.
    pub statistic: f64,
    /// Truncation lag `L` used for the Newey-West long-run variance.
    pub lag_used: usize,
    /// Asymptotic critical value at α = 0.05.
    pub critical_5pct: f64,
    /// Linear-interpolated p-value against the (0.01, 0.025, 0.05, 0.10)
    /// critical-value grid. Clamped to [0.01, 0.10]; "0.10" means
    /// `p ≥ 0.10` (cannot reject) and "0.01" means `p ≤ 0.01`.
    pub p_value: f64,
    /// True iff we reject stationarity at α = 0.05 (`statistic > critical_5pct`).
    /// `auto_arima` interprets this as "needs differencing."
    pub reject_stationarity: bool,
}

/// Choice of deterministic regressor to detrend against before the
/// stationarity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssRegression {
    /// Test for stationarity around a constant (default — appropriate
    /// for selecting `d` in ARIMA).
    Constant,
    /// Test for stationarity around a linear trend (use when the
    /// series has a deterministic linear trend you want to exclude).
    ConstantTrend,
}

/// KPSS stationarity test.
///
/// Returns the test statistic, asymptotic 5% critical value, and an
/// interpolated p-value. The null hypothesis is *stationarity*; reject
/// (and difference) when `statistic > critical_5pct`, or equivalently
/// `p_value < 0.05`.
pub fn kpss(y: &[f64], regression: KpssRegression) -> KpssResult {
    let n = y.len();
    debug_assert!(n >= 3, "KPSS needs at least 3 observations");

    // 1. Residuals from the deterministic regression.
    let residuals: Vec<f64> = match regression {
        KpssRegression::Constant => {
            let mean: f64 = y.iter().sum::<f64>() / n as f64;
            y.iter().map(|v| v - mean).collect()
        }
        KpssRegression::ConstantTrend => {
            let nf = n as f64;
            let sum_t = (n - 1) as f64 * nf / 2.0;
            let sum_t2 = (n - 1) as f64 * nf * (2.0 * (n - 1) as f64 + 1.0) / 6.0;
            let sum_y: f64 = y.iter().sum();
            let sum_ty: f64 = y.iter().enumerate().map(|(t, v)| t as f64 * v).sum();
            // OLS for (a + b·t): solve 2x2 normal equations.
            let det = nf * sum_t2 - sum_t * sum_t;
            let a = (sum_t2 * sum_y - sum_t * sum_ty) / det;
            let b = (nf * sum_ty - sum_t * sum_y) / det;
            (0..n).map(|t| y[t] - a - b * t as f64).collect()
        }
    };

    // 2. Partial sums of residuals.
    let partial: Vec<f64> = residuals
        .iter()
        .scan(0.0f64, |acc, &v| {
            *acc += v;
            Some(*acc)
        })
        .collect();
    let s_sq: f64 = partial.iter().map(|s| s * s).sum();

    // 3. Long-run variance via Newey-West (Bartlett kernel). Truncation
    //    lag from Schwert's rule (statsmodels' "auto" default):
    //    L = floor(12 · (n/100)^(1/4)).
    let lag = ((12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize).max(1);
    let lag = lag.min(n.saturating_sub(1));
    let mut s2 = residuals.iter().map(|v| v * v).sum::<f64>() / n as f64;
    for k in 1..=lag {
        let mut acov = 0.0;
        for t in k..n {
            acov += residuals[t] * residuals[t - k];
        }
        acov /= n as f64;
        let w = 1.0 - k as f64 / (lag + 1) as f64;
        s2 += 2.0 * w * acov;
    }

    let statistic = s_sq / ((n as f64).powi(2) * s2);

    // 4. Critical values (KPSS 1992 Table 1, asymptotic).
    //    For "Constant" (level stationarity) and "ConstantTrend" (trend
    //    stationarity). Columns: 1%, 2.5%, 5%, 10%.
    let crits = match regression {
        KpssRegression::Constant => [0.739, 0.574, 0.463, 0.347],
        KpssRegression::ConstantTrend => [0.216, 0.176, 0.146, 0.119],
    };
    let p_value = interp_p_value(statistic, &crits);
    let critical_5pct = crits[2];

    KpssResult {
        statistic,
        lag_used: lag,
        critical_5pct,
        p_value,
        reject_stationarity: statistic > critical_5pct,
    }
}

/// Linear interpolation of `statistic` against the descending critical-
/// value grid `(0.01, 0.025, 0.05, 0.10)` → `p_value`. Clamped to the
/// grid bounds: statistics above the 1% critical map to `0.01`, below
/// the 10% critical map to `0.10`.
fn interp_p_value(statistic: f64, crits: &[f64; 4]) -> f64 {
    let probs = [0.01_f64, 0.025, 0.05, 0.10];
    if statistic >= crits[0] {
        return 0.01;
    }
    if statistic <= crits[3] {
        return 0.10;
    }
    for i in 0..3 {
        if statistic >= crits[i + 1] {
            let t = (crits[i] - statistic) / (crits[i] - crits[i + 1]);
            return probs[i] + t * (probs[i + 1] - probs[i]);
        }
    }
    0.10
}

/// Hyndman's strength-of-seasonality measure (Forecasting: Principles
/// and Practice, §6.7). Decomposes `y` into trend + seasonal + remainder
/// via classical additive decomposition, then returns
///
/// ```text
/// max(0, 1 − Var(remainder) / Var(remainder + seasonal)).
/// ```
///
/// A value near 1 indicates strong seasonality. `auto_arima` typically
/// applies seasonal differencing (`D = 1`) when this is above ~0.64.
///
/// Returns `0.0` (no seasonality) when `period < 2` or the series is
/// too short for a classical decomposition.
pub fn seasonal_strength(y: &[f64], period: u32) -> f64 {
    if period < 2 {
        return 0.0;
    }
    let opts = seasonal::SeasonalDecomposeOpts::new(period);
    let dec = match seasonal::seasonal_decompose(y, opts) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    let mut sum_r = 0.0;
    let mut sum_r2 = 0.0;
    let mut sum_rs = 0.0;
    let mut sum_rs2 = 0.0;
    let mut n_finite = 0usize;
    for (r, s) in dec.residual.iter().zip(dec.seasonal.iter()) {
        if r.is_finite() && s.is_finite() {
            sum_r += r;
            sum_r2 += r * r;
            let rs = r + s;
            sum_rs += rs;
            sum_rs2 += rs * rs;
            n_finite += 1;
        }
    }
    if n_finite < 2 {
        return 0.0;
    }
    let nf = n_finite as f64;
    let var_r = sum_r2 / nf - (sum_r / nf).powi(2);
    let var_rs = sum_rs2 / nf - (sum_rs / nf).powi(2);
    if var_rs <= 0.0 {
        return 0.0;
    }
    (1.0 - var_r / var_rs).max(0.0).min(1.0)
}

/// Choose ordinary differencing order `d` by iterating KPSS until the
/// series is judged stationary, capped at `max_d`. Matches the
/// `auto_arima` / `forecast::ndiffs(test="kpss")` default.
pub fn ndiffs(y: &[f64], max_d: u32) -> u32 {
    let mut cur: Vec<f64> = y.to_vec();
    let mut d = 0u32;
    while d < max_d {
        if cur.len() < 8 {
            return d;
        }
        let r = kpss(&cur, KpssRegression::Constant);
        if !r.reject_stationarity {
            return d;
        }
        cur = (1..cur.len()).map(|i| cur[i] - cur[i - 1]).collect();
        d += 1;
    }
    d
}

/// Choose seasonal differencing order `D` from the seasonal-strength
/// heuristic, capped at `max_big_d`. Matches
/// `forecast::nsdiffs(test="seas")`.
pub fn nsdiffs(y: &[f64], period: u32, max_big_d: u32) -> u32 {
    if period < 2 || max_big_d == 0 {
        return 0;
    }
    let mut cur: Vec<f64> = y.to_vec();
    let mut big_d = 0u32;
    while big_d < max_big_d {
        if cur.len() < 2 * period as usize {
            return big_d;
        }
        let strength = seasonal_strength(&cur, period);
        if strength <= 0.64 {
            return big_d;
        }
        let m = period as usize;
        cur = (m..cur.len()).map(|i| cur[i] - cur[i - m]).collect();
        big_d += 1;
    }
    big_d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kpss_flags_random_walk() {
        // Random walk is non-stationary; KPSS should reject.
        let mut s = 1u64;
        let n = 500usize;
        let mut y = vec![0.0f64; n];
        for t in 1..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = (s as f64 / u64::MAX as f64).max(1e-300);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = s as f64 / u64::MAX as f64;
            let eps = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            y[t] = y[t - 1] + eps;
        }
        let r = kpss(&y, KpssRegression::Constant);
        assert!(
            r.reject_stationarity,
            "expected to reject stationarity for RW, got stat={} crit={}",
            r.statistic,
            r.critical_5pct
        );
    }

    #[test]
    fn kpss_accepts_white_noise() {
        // White noise around a constant is stationary; KPSS should not reject.
        let mut s = 1u64;
        let n = 500usize;
        let mut y = vec![0.0f64; n];
        for v in y.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = (s as f64 / u64::MAX as f64).max(1e-300);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = s as f64 / u64::MAX as f64;
            *v = 5.0 + (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let r = kpss(&y, KpssRegression::Constant);
        assert!(
            !r.reject_stationarity,
            "white noise flagged: stat={} crit={}",
            r.statistic,
            r.critical_5pct
        );
    }

    #[test]
    fn ndiffs_random_walk_returns_one() {
        let mut s = 1u64;
        let n = 500usize;
        let mut y = vec![0.0f64; n];
        for t in 1..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = (s as f64 / u64::MAX as f64).max(1e-300);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = s as f64 / u64::MAX as f64;
            let eps = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            y[t] = y[t - 1] + eps;
        }
        assert_eq!(ndiffs(&y, 2), 1);
    }

    #[test]
    fn seasonal_strength_high_for_seasonal_series() {
        let period = 12;
        let n_cycles = 30;
        let mut y = Vec::with_capacity(period * n_cycles);
        for c in 0..n_cycles {
            for i in 0..period {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
                // Strong seasonal signal, weak noise.
                y.push(10.0 + 0.05 * c as f64 + 3.0 * phase.sin());
            }
        }
        let s = seasonal_strength(&y, period as u32);
        assert!(s > 0.6, "expected strong seasonality (>0.6), got {s}");
    }

    #[test]
    fn seasonal_strength_low_for_noise() {
        let mut s_rng = 1u64;
        let n = 240usize;
        let y: Vec<f64> = (0..n)
            .map(|_| {
                s_rng ^= s_rng << 13;
                s_rng ^= s_rng >> 7;
                s_rng ^= s_rng << 17;
                let u1 = (s_rng as f64 / u64::MAX as f64).max(1e-300);
                s_rng ^= s_rng << 13;
                s_rng ^= s_rng >> 7;
                s_rng ^= s_rng << 17;
                let u2 = s_rng as f64 / u64::MAX as f64;
                10.0 + (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let strength = seasonal_strength(&y, 12);
        assert!(strength < 0.4, "expected weak seasonality (<0.4), got {strength}");
    }
}
