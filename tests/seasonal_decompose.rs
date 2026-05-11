//! Unit tests for `rust_stats::tsa::seasonal::seasonal_decompose`.

use approx::assert_relative_eq;
use rust_stats::error::SeasonalDecomposeError;
use rust_stats::tsa::{seasonal_decompose, DecomposeMode, Missing, SeasonalDecomposeOpts};

#[test]
fn linear_trend_recovered_in_inner_band() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let d = seasonal_decompose(&y, SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        assert_relative_eq!(d.trend[i], i as f64, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], 0.0, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn seasonal_pattern_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let pattern_mean = pattern.iter().sum::<f64>() / 4.0;
    let n = pattern.len() * 6;
    let half = (period as usize) / 2;
    let y: Vec<f64> = (0..n).map(|i| pattern[i % 4]).collect();
    let d = seasonal_decompose(&y, SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        assert_relative_eq!(d.trend[i], pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], pattern[i % 4] - pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn additive_reconstruction_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = 24usize;
    let half = (period as usize) / 2;
    let y: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let d = seasonal_decompose(&y, SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let half = (period as usize) / 2;
    let y: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let n = y.len();
    let d = seasonal_decompose(
        &y,
        SeasonalDecomposeOpts {
            mode: DecomposeMode::Multiplicative,
            ..SeasonalDecomposeOpts::new(period)
        },
    )
    .unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y[i], max_relative = 1e-9);
    }
}

#[test]
fn edges_are_nan() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let d = seasonal_decompose(&y, SeasonalDecomposeOpts::new(period)).unwrap();
    for i in 0..half {
        assert!(d.trend[i].is_nan(), "trend[{}] not NaN", i);
        assert!(d.residual[i].is_nan(), "residual[{}] not NaN", i);
    }
    for i in (n - half)..n {
        assert!(d.trend[i].is_nan(), "trend[{}] not NaN", i);
        assert!(d.residual[i].is_nan(), "residual[{}] not NaN", i);
    }
}

#[test]
fn validation_paths() {
    let y = vec![1.0; 24];
    assert!(matches!(
        seasonal_decompose(&y, SeasonalDecomposeOpts::new(1)),
        Err(SeasonalDecomposeError::InvalidPeriod(1))
    ));
    let short = vec![1.0, 2.0, 3.0];
    assert!(matches!(
        seasonal_decompose(&short, SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::SeriesTooShort { .. })
    ));
    let bad = [1.0, 2.0, 0.0, 1.5].repeat(6);
    assert!(matches!(
        seasonal_decompose(
            &bad,
            SeasonalDecomposeOpts {
                mode: DecomposeMode::Multiplicative,
                ..SeasonalDecomposeOpts::new(4)
            }
        ),
        Err(SeasonalDecomposeError::NonPositiveForMultiplicative { .. })
    ));
    let mut v = vec![1.0; 24];
    v[5] = f64::NAN;
    assert!(matches!(
        seasonal_decompose(&v, SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::NonFinite)
    ));
}

// ── Missing handling ─────────────────────────────────────────────────────

fn airpassengers_like(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            100.0 + 0.5 * i as f64 + 30.0 * phase.sin()
        })
        .collect()
}

#[test]
fn missing_interpolate_fills_residual_nan() {
    let period = 12;
    let n = 144;
    let mut y = airpassengers_like(n, period);
    y[50] = f64::NAN;
    y[51] = f64::NAN;
    y[100] = f64::INFINITY;

    let opts = SeasonalDecomposeOpts {
        missing: Missing::Interpolate,
        ..SeasonalDecomposeOpts::new(period as u32)
    };
    let d = seasonal_decompose(&y, opts).unwrap();
    let half = period / 2;

    for i in 0..n {
        let in_edge = i < half || i >= n - half;
        // Trend NaN at edges (centred MA), finite elsewhere.
        if in_edge {
            assert!(d.trend[i].is_nan(), "trend at edge i={i} should be NaN");
        } else {
            assert!(d.trend[i].is_finite(), "trend at i={i} should be finite, got {}", d.trend[i]);
        }
        // Seasonal always finite.
        assert!(d.seasonal[i].is_finite(), "seasonal at i={i} should be finite");
        // Residual NaN at edges AND at originally-missing positions; finite elsewhere.
        let was_missing = !y[i].is_finite();
        if in_edge || was_missing {
            assert!(d.residual[i].is_nan(), "residual at i={i} should be NaN (edge={in_edge}, missing={was_missing})");
        } else {
            assert!(d.residual[i].is_finite(), "residual at i={i} should be finite");
        }
    }
}

#[test]
fn missing_error_still_default() {
    let mut y = airpassengers_like(48, 4);
    y[10] = f64::NAN;
    assert!(matches!(
        seasonal_decompose(&y, SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::NonFinite)
    ));
}

#[test]
fn missing_interpolate_all_nan_errors() {
    let y = vec![f64::NAN; 48];
    let opts = SeasonalDecomposeOpts {
        missing: Missing::Interpolate,
        ..SeasonalDecomposeOpts::new(4)
    };
    assert!(matches!(
        seasonal_decompose(&y, opts),
        Err(SeasonalDecomposeError::NonFinite)
    ));
}
