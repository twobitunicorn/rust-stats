//! Holt-Winters exponential smoothing.
//!
//! In-sample one-step-ahead fitted values for the additive and
//! multiplicative Holt-Winters models, with all smoothing parameters
//! supplied by the caller (no MLE / grid-search optimisation).
//!
//! Reduces to:
//!
//! - Single exponential smoothing (SES) when `β = 0` and `γ = 0`.
//! - Holt's linear method when `β > 0` and `γ = 0`.
//! - Full Holt-Winters seasonal smoothing when `β > 0`, `γ > 0`, and
//!   `seasonal_periods ≥ 2`.
//!
//! Initialisation follows the standard textbook recipe: the level and
//! trend are seeded from the means of the first one (or two) seasonal
//! periods, and the initial seasonal indices from the deviations of the
//! first period from that level (additive: `y - mean`; multiplicative:
//! `y / mean`). Multiplicative mode requires strictly positive `y`.

use crate::error::HoltWintersError;
use crate::tsa::seasonal::DecomposeMode;

/// Options for [`holt_winters`].
#[derive(Debug, Clone)]
pub struct HoltWintersOpts {
    /// Level smoothing in `[0, 1]`.
    pub alpha: f64,
    /// Trend smoothing in `[0, 1]`. `0` disables the trend term.
    pub beta: f64,
    /// Seasonal smoothing in `[0, 1]`. `0` disables the seasonal term.
    pub gamma: f64,
    /// Length of one seasonal cycle. Required for seasonal smoothing
    /// (`gamma > 0`); ignored otherwise. The series must contain at
    /// least `2 * seasonal_periods` samples when seasonal smoothing is
    /// active.
    pub seasonal_periods: u32,
    /// Seasonal combination mode (additive or multiplicative).
    pub mode: DecomposeMode,
}

impl HoltWintersOpts {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            beta: 0.0,
            gamma: 0.0,
            seasonal_periods: 0,
            mode: DecomposeMode::Additive,
        }
    }
}

/// In-sample Holt-Winters fitted values. Output has the same length as
/// the input.
///
/// Errors when any smoothing parameter is outside `[0, 1]`, when the
/// input contains non-finite values, when the input is too short for
/// the requested model (seasonal mode requires `n >= 2 * m`; Holt's
/// linear method requires `n >= 2`), or when multiplicative seasonal
/// mode is requested with a non-positive value present.
pub fn holt_winters(y: &[f64], opts: HoltWintersOpts) -> Result<Vec<f64>, HoltWintersError> {
    let HoltWintersOpts {
        alpha,
        beta,
        gamma,
        seasonal_periods,
        mode,
    } = opts;

    if !(0.0..=1.0).contains(&alpha) {
        return Err(HoltWintersError::InvalidAlpha(alpha));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(HoltWintersError::InvalidBeta(beta));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(HoltWintersError::InvalidGamma(gamma));
    }

    let multiplicative = matches!(mode, DecomposeMode::Multiplicative);
    let m = seasonal_periods as usize;
    let has_seasonal = gamma > 0.0 && m >= 2;
    let has_trend = beta > 0.0;

    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(HoltWintersError::NonFinite);
    }
    if multiplicative {
        let min = y.iter().copied().fold(f64::INFINITY, f64::min);
        if min <= 0.0 {
            return Err(HoltWintersError::NonPositiveForMultiplicative { min });
        }
    }

    // ---- initialisation ----
    let mut level: f64;
    let mut trend: f64;
    let mut s_buf: Vec<f64>;
    if has_seasonal {
        if n < 2 * m {
            return Err(HoltWintersError::SeriesTooShort { n, min: 2 * m });
        }
        let mean_first: f64 = y[..m].iter().sum::<f64>() / m as f64;
        let mean_second: f64 = y[m..2 * m].iter().sum::<f64>() / m as f64;
        level = mean_first;
        trend = if has_trend {
            (mean_second - mean_first) / m as f64
        } else {
            0.0
        };
        s_buf = if multiplicative {
            y[..m].iter().map(|&v| v / mean_first).collect()
        } else {
            y[..m].iter().map(|&v| v - mean_first).collect()
        };
    } else if has_trend {
        if n < 2 {
            return Err(HoltWintersError::SeriesTooShort { n, min: 2 });
        }
        level = y[0];
        trend = y[1] - y[0];
        s_buf = Vec::new();
    } else {
        level = y[0];
        trend = 0.0;
        s_buf = Vec::new();
    }

    // ---- recursion ----
    // For each t, emit ŷ_t (one-step forecast from previous state),
    // then update level/trend/seasonal.
    let mut fitted: Vec<f64> = Vec::with_capacity(n);
    for t in 0..n {
        let s_idx = if has_seasonal { t % m } else { 0 };
        let prev_s = if has_seasonal {
            s_buf[s_idx]
        } else if multiplicative {
            1.0
        } else {
            0.0
        };

        let yhat = if has_seasonal {
            if multiplicative {
                (level + trend) * prev_s
            } else {
                level + trend + prev_s
            }
        } else {
            level + trend
        };
        fitted.push(yhat);

        let y_t = y[t];
        let new_level = if has_seasonal {
            if multiplicative {
                alpha * (y_t / prev_s) + (1.0 - alpha) * (level + trend)
            } else {
                alpha * (y_t - prev_s) + (1.0 - alpha) * (level + trend)
            }
        } else {
            alpha * y_t + (1.0 - alpha) * (level + trend)
        };
        let new_trend = if has_trend {
            beta * (new_level - level) + (1.0 - beta) * trend
        } else {
            0.0
        };
        if has_seasonal {
            let new_s = if multiplicative {
                gamma * (y_t / new_level) + (1.0 - gamma) * prev_s
            } else {
                gamma * (y_t - new_level) + (1.0 - gamma) * prev_s
            };
            s_buf[s_idx] = new_s;
        }
        level = new_level;
        trend = new_trend;
    }

    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ses_first_fitted_is_y0() {
        // alpha-only smoothing (SES) initialises level = y[0], so the
        // first one-step forecast equals y[0].
        let out = holt_winters(
            &[1.0, 2.0, 3.0, 4.0],
            HoltWintersOpts::new(0.5),
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn holt_linear_runs() {
        let opts = HoltWintersOpts {
            beta: 0.3,
            ..HoltWintersOpts::new(0.5)
        };
        let out = holt_winters(&[1.0, 2.0, 3.0, 4.0, 5.0], opts).unwrap();
        assert_eq!(out.len(), 5);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn seasonal_too_short() {
        let opts = HoltWintersOpts {
            beta: 0.1,
            gamma: 0.3,
            seasonal_periods: 4,
            ..HoltWintersOpts::new(0.5)
        };
        let err = holt_winters(&[1.0, 2.0, 3.0, 4.0, 5.0], opts).unwrap_err();
        assert_eq!(err, HoltWintersError::SeriesTooShort { n: 5, min: 8 });
    }

    #[test]
    fn invalid_alpha_errors() {
        let err = holt_winters(&[1.0, 2.0], HoltWintersOpts::new(1.5)).unwrap_err();
        assert_eq!(err, HoltWintersError::InvalidAlpha(1.5));
    }

    #[test]
    fn multiplicative_rejects_non_positive() {
        let opts = HoltWintersOpts {
            beta: 0.1,
            gamma: 0.3,
            seasonal_periods: 2,
            mode: DecomposeMode::Multiplicative,
            ..HoltWintersOpts::new(0.5)
        };
        let err = holt_winters(&[1.0, 0.0, 2.0, 3.0], opts).unwrap_err();
        assert!(matches!(
            err,
            HoltWintersError::NonPositiveForMultiplicative { .. }
        ));
    }

    #[test]
    fn rejects_nan() {
        let err = holt_winters(
            &[1.0, f64::NAN, 3.0],
            HoltWintersOpts::new(0.5),
        )
        .unwrap_err();
        assert_eq!(err, HoltWintersError::NonFinite);
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = holt_winters(&[], HoltWintersOpts::new(0.5)).unwrap();
        assert!(out.is_empty());
    }
}
