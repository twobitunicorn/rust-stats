//! Classical (moving-average) seasonal-trend decomposition.
//!
//! Trend: centered moving average of length `period`.
//! Seasonal (additive):       per-phase mean of `y - trend`, centred to sum
//!                            to zero across one period.
//! Seasonal (multiplicative): per-phase mean of `y / trend`, normalised so
//!                            the pattern's arithmetic mean across one
//!                            period is one.
//! Residual: `y - trend - seasonal` (additive) or `y / (trend * seasonal)`
//! (multiplicative).
//!
//! Matches `statsmodels.tsa.seasonal.seasonal_decompose` for both modes.
//!
//! The first/last `period/2` positions of `trend` and `residual` are NaN
//! (the centred moving-average edge band).

use crate::error::SeasonalDecomposeError;
use crate::tsa::seasonal::{
    interpolate_missing, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
};

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
    let mut missing_mask: Option<Vec<bool>> = None;
    let raw: Vec<f64> = match opts.missing {
        Missing::Error => {
            if y.iter().any(|v| !v.is_finite()) {
                return Err(SeasonalDecomposeError::NonFinite);
            }
            y.to_vec()
        }
        Missing::Interpolate => {
            let any_missing = y.iter().any(|v| !v.is_finite());
            if any_missing {
                let mask: Vec<bool> = y.iter().map(|v| !v.is_finite()).collect();
                let filled = interpolate_missing(y)
                    .ok_or(SeasonalDecomposeError::NonFinite)?;
                missing_mask = Some(mask);
                filled
            } else {
                y.to_vec()
            }
        }
    };
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

    let trend = centered_ma(&raw, period);

    // Detrend in the appropriate space.
    let detrended: Vec<f64> = raw
        .iter()
        .zip(trend.iter())
        .map(|(yi, ti)| {
            if ti.is_nan() {
                f64::NAN
            } else if multiplicative {
                yi / ti
            } else {
                yi - ti
            }
        })
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
            } else if multiplicative {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let pattern_mean: f64 = phase_means.iter().sum::<f64>() / period as f64;
    // Additive: subtract so the pattern sums to zero. Multiplicative:
    // divide so the pattern's arithmetic mean is one.
    let centered_pattern: Vec<f64> = if multiplicative {
        phase_means.iter().map(|m| m / pattern_mean).collect()
    } else {
        phase_means.iter().map(|m| m - pattern_mean).collect()
    };

    let seasonal: Vec<f64> = (0..n).map(|i| centered_pattern[i % period]).collect();

    let mut residual: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_nan() {
                f64::NAN
            } else if multiplicative {
                raw[i] / (trend[i] * seasonal[i])
            } else {
                raw[i] - trend[i] - seasonal[i]
            }
        })
        .collect();

    if let Some(mask) = missing_mask {
        for (i, &missing) in mask.iter().enumerate() {
            if missing {
                residual[i] = f64::NAN;
            }
        }
    }

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
