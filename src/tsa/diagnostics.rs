//! Residual diagnostics for time-series models.
//!
//! Currently: the **Ljung-Box** test for serial correlation in the
//! residuals of a fitted ARMA / ARIMA / SARIMA. Under the null that the
//! residuals are uncorrelated white noise, the statistic
//!
//! ```text
//! Q = n · (n + 2) · Σ_{k=1..h} ρ̂_k² / (n − k)
//! ```
//!
//! is asymptotically χ²(h − m), where `n` is the residual count, `h` is
//! the lag cutoff, and `m` is the number of fitted ARMA parameters
//! (`p + q` for non-seasonal, `p + q + P + Q` for seasonal). Reject the
//! null (residuals show autocorrelation) when `Q` exceeds the χ² critical
//! value or, equivalently, when `p_value < α`.

/// Result of a Ljung-Box test.
#[derive(Debug, Clone, Copy)]
pub struct LjungBox {
    /// Q statistic.
    pub q: f64,
    /// Degrees of freedom = `lags − fitted_params`. Always at least 1
    /// (we clamp internally to keep the χ² approximation usable).
    pub df: usize,
    /// Approximate two-sided p-value via the Wilson-Hilferty
    /// transformation of `Q ∼ χ²(df)`. Reject the null of white-noise
    /// residuals when `p_value < α` (typically `α = 0.05`).
    pub p_value: f64,
}

/// Ljung-Box test on a residual series.
///
/// - `lags` is the cutoff `h`. Box-Jenkins-style guidance: `min(10, n/5)`
///   for non-seasonal residuals, `2·m` for seasonal.
/// - `fitted_params` is the number of ARMA-side parameters consumed by
///   the original fit (`p + q + P + Q`) — these reduce the χ² degrees
///   of freedom. Pass `0` if you're testing a raw series rather than
///   model residuals.
pub fn ljung_box(residuals: &[f64], lags: usize, fitted_params: usize) -> LjungBox {
    let n = residuals.len();
    let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|r| r - mean).collect();
    let denom: f64 = centered.iter().map(|v| v * v).sum();

    let mut q = 0.0;
    for k in 1..=lags {
        if k >= n {
            break;
        }
        let mut num = 0.0;
        for t in k..n {
            num += centered[t] * centered[t - k];
        }
        let rho_k = if denom > 0.0 { num / denom } else { 0.0 };
        q += rho_k * rho_k / (n - k) as f64;
    }
    q *= (n as f64) * (n as f64 + 2.0);

    let df = lags.saturating_sub(fitted_params).max(1);
    let p_value = chi2_survival(q, df as f64);
    LjungBox { q, df, p_value }
}

// ----------------------------------------------------------------------
// χ² survival function via Wilson-Hilferty + Φ approximation.
// Accurate to a few decimals in the relevant 0.001 < p < 0.999 range,
// which is plenty for Ljung-Box-style hypothesis testing.
// ----------------------------------------------------------------------

/// `P[X > x]` where `X ∼ χ²(df)`. Returns 1 for `x ≤ 0`.
pub(crate) fn chi2_survival(x: f64, df: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    // Wilson-Hilferty: Y = (X/df)^(1/3) is approximately
    // N(μ = 1 − 2/(9·df), σ² = 2/(9·df)).
    let two_ninths_df = 2.0 / (9.0 * df);
    let z = ((x / df).cbrt() - (1.0 - two_ninths_df)) / two_ninths_df.sqrt();
    1.0 - phi_cdf(z)
}

/// Forward standard-normal CDF `Φ(z)` via the Abramowitz-Stegun 7.1.26
/// rational approximation of `erf` (max absolute error ≈ 1.5e-7).
fn phi_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // A&S 7.1.26; valid for x ≥ 0. For x < 0 use erf(-x) = -erf(x).
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        // erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) = -erf(1).
        assert!(erf(0.0).abs() < 1e-7);
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-6);
        assert!((erf(-1.0) + 0.8427007929497149).abs() < 1e-6);
        // erf(infinity) → 1.
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chi2_survival_known_values() {
        // χ²(1) median ≈ 0.4549, so survival at 0.4549 should be ≈ 0.5.
        let s = chi2_survival(0.4549, 1.0);
        assert!((s - 0.5).abs() < 0.05, "got {s}");
        // 95th percentile of χ²(1) is 3.841; survival should be ≈ 0.05.
        let s = chi2_survival(3.841, 1.0);
        assert!((s - 0.05).abs() < 0.01, "got {s}");
        // 99th percentile of χ²(10) is 23.21; survival should be ≈ 0.01.
        let s = chi2_survival(23.21, 10.0);
        assert!((s - 0.01).abs() < 0.01, "got {s}");
    }

    #[test]
    fn ljung_box_white_noise_not_rejected() {
        // A truly white-noise series should not be flagged.
        let mut s = 1u64;
        let n = 500usize;
        let mut e = vec![0.0f64; n];
        for v in e.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = (s as f64 / u64::MAX as f64).max(1e-300);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = s as f64 / u64::MAX as f64;
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let r = ljung_box(&e, 10, 0);
        assert!(
            r.p_value > 0.01,
            "white-noise residuals flagged: Q={} p={}",
            r.q,
            r.p_value
        );
    }

    #[test]
    fn ljung_box_autocorrelated_rejected() {
        // AR(1) series should be flagged on its own — strong serial
        // correlation at lag 1.
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
            y[t] = 0.7 * y[t - 1] + eps;
        }
        let r = ljung_box(&y, 10, 0);
        assert!(
            r.p_value < 0.01,
            "autocorrelated series not flagged: Q={} p={}",
            r.q,
            r.p_value
        );
    }
}
