//! Classical (moving-average) seasonal-trend decomposition.
//!
//! Trend: centered moving average of length `period`.
//! Seasonal: per-phase mean of detrended values, centred so the seasonal
//! pattern sums to zero (additive) or products to one (multiplicative).
//! Residual: `y - trend - seasonal` (additive) or `y / (trend * seasonal)`
//! (multiplicative).
//!
//! The first/last `period/2` positions of `trend` and `residual` are NaN
//! (the centred moving-average edge band).

use crate::error::SeasonalDecomposeError;
use crate::tsa::seasonal::{DecomposeMode, Decomposition, SeasonalDecomposeOpts};

pub fn seasonal_decompose(
    y: &[f64],
    opts: SeasonalDecomposeOpts,
) -> Result<Decomposition, SeasonalDecomposeError> {
    if opts.period < 2 {
        return Err(SeasonalDecomposeError::InvalidPeriod(opts.period));
    }
    let period = opts.period as usize;

    if y.is_empty() {
        return Err(SeasonalDecomposeError::SeriesTooShort {
            n: 0,
            min: 2 * period,
        });
    }
    let raw: Vec<f64> = y.to_vec();
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(SeasonalDecomposeError::NonFinite);
    }
    let n = raw.len();
    if n < 2 * period {
        return Err(SeasonalDecomposeError::SeriesTooShort {
            n,
            min: 2 * period,
        });
    }

    let multiplicative = matches!(opts.mode, DecomposeMode::Multiplicative);
    if multiplicative {
        let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
        if min <= 0.0 {
            return Err(SeasonalDecomposeError::NonPositiveForMultiplicative { min });
        }
    }

    // Work in log-space for multiplicative mode.
    let work: Vec<f64> = if multiplicative {
        raw.iter().map(|v| v.ln()).collect()
    } else {
        raw
    };

    let trend = centered_ma(&work, period);

    let detrended: Vec<f64> = work
        .iter()
        .zip(trend.iter())
        .map(|(yi, ti)| if ti.is_nan() { f64::NAN } else { yi - ti })
        .collect();

    let mut phase_sums = vec![0.0f64; period];
    let mut phase_counts = vec![0usize; period];
    for (i, &d) in detrended.iter().enumerate() {
        if !d.is_nan() {
            phase_sums[i % period] += d;
            phase_counts[i % period] += 1;
        }
    }
    let phase_means: Vec<f64> = (0..period)
        .map(|k| {
            if phase_counts[k] > 0 {
                phase_sums[k] / phase_counts[k] as f64
            } else {
                0.0
            }
        })
        .collect();
    let pattern_mean: f64 = phase_means.iter().sum::<f64>() / period as f64;
    let centered_pattern: Vec<f64> = phase_means.iter().map(|m| m - pattern_mean).collect();

    let seasonal: Vec<f64> = (0..n).map(|i| centered_pattern[i % period]).collect();

    let residual: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_nan() {
                f64::NAN
            } else {
                work[i] - trend[i] - seasonal[i]
            }
        })
        .collect();

    let (trend, seasonal, residual) = if multiplicative {
        (
            trend.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            seasonal.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            residual.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
        )
    } else {
        (trend, seasonal, residual)
    };

    let _ = n;
    Ok(Decomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Centered moving average of length `window`. For odd window: standard
/// `(2k+1)`-MA at index i averages `y[i-k..=i+k]`. For even window m: a
/// `(m, 2)`-MA — equivalent to taking the m-MA twice and averaging — which
/// weights the endpoints by `1/(2m)` and the m-1 middle points by `1/m`.
/// Returns NaN at the first/last `m/2` positions where the centred window
/// doesn't fit.
fn centered_ma(y: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window {
        return out;
    }
    if window % 2 == 1 {
        let half = window / 2;
        let inv = 1.0 / window as f64;
        for i in half..(n - half) {
            let sum: f64 = y[i - half..=i + half].iter().sum();
            out[i] = sum * inv;
        }
    } else {
        let half = window / 2;
        let inv = 1.0 / (2 * window) as f64;
        for i in half..(n - half) {
            let mut sum = y[i - half] + y[i + half];
            for j in (i - half + 1)..(i + half) {
                sum += 2.0 * y[j];
            }
            out[i] = sum * inv;
        }
    }
    out
}
