//! Unit tests for `rust_stats::tsa::seasonal::seasonal_decompose`.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::error::SeasonalDecomposeError;
use rust_stats::tsa::{seasonal_decompose, DecomposeMode, SeasonalDecomposeOpts};

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn linear_trend_recovered_in_inner_band() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
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
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
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
    let y_vec: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let y = col_from(y_vec.clone());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y_vec[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_in_inner_band() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let half = (period as usize) / 2;
    let y_vec: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let n = y_vec.len();
    let y = col_from(y_vec.clone());
    let d = seasonal_decompose(
        y.as_ref(),
        SeasonalDecomposeOpts {
            mode: DecomposeMode::Multiplicative,
            ..SeasonalDecomposeOpts::new(period)
        },
    )
    .unwrap();
    for i in half..(n - half) {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y_vec[i], max_relative = 1e-9);
    }
}

#[test]
fn edges_are_nan() {
    let period = 4u32;
    let n = 24usize;
    let half = (period as usize) / 2;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(period)).unwrap();
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
    let y = col_from(vec![1.0; 24]);
    assert!(matches!(
        seasonal_decompose(y.as_ref(), SeasonalDecomposeOpts::new(1)),
        Err(SeasonalDecomposeError::InvalidPeriod(1))
    ));
    let short = col_from(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        seasonal_decompose(short.as_ref(), SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::SeriesTooShort { .. })
    ));
    let bad = col_from(vec![1.0, 2.0, 0.0, 1.5].repeat(6));
    assert!(matches!(
        seasonal_decompose(
            bad.as_ref(),
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
        seasonal_decompose(col_from(v).as_ref(), SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::NonFinite)
    ));
}
