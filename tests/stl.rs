//! Unit tests for `rust_stats::tsa::seasonal::stl`.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::error::StlError;
use rust_stats::tsa::{stl, DecomposeMode, StlOpts};

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn pure_linear_trend_recovered_everywhere() {
    let n = 24usize;
    let period = 4u32;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        assert_relative_eq!(d.trend[i], i as f64, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], 0.0, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn pure_seasonal_pattern_recovered_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let pattern_mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % pattern.len()]).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        assert_relative_eq!(d.trend[i], pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.seasonal[i], pattern[i % 4] - pattern_mean, epsilon = 1e-9);
        assert_relative_eq!(d.residual[i], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn additive_reconstruction_exact_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = 24usize;
    let y_vec: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let y = col_from(y_vec.clone());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    for i in 0..n {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y_vec[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_exact_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let y_vec: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let y = col_from(y_vec.clone());
    let d = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(period)
        },
    )
    .unwrap();
    for i in 0..y_vec.len() {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y_vec[i], max_relative = 1e-9);
    }
}

#[test]
fn additive_seasonal_pattern_sums_to_zero() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = stl(y.as_ref(), StlOpts::new(period)).unwrap();
    let inner: f64 = (8..12).map(|i| d.seasonal[i]).sum();
    assert_relative_eq!(inner, 0.0, epsilon = 1e-9);
}

#[test]
fn multiplicative_seasonal_pattern_products_to_one() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let n = pattern.len() * 6;
    let y = col_from((0..n).map(|i| pattern[i % 4]).collect());
    let d = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(period)
        },
    )
    .unwrap();
    let prod: f64 = (8..12).map(|i| d.seasonal[i]).product();
    assert_relative_eq!(prod, 1.0, max_relative = 1e-9);
}

#[test]
fn validation_paths() {
    let y = col_from(vec![1.0; 24]);
    assert!(matches!(
        stl(y.as_ref(), StlOpts::new(1)),
        Err(StlError::InvalidPeriod(1))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                seasonal_window: 8,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidSeasonalWindow(8))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                trend_window: Some(10),
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidTrendWindow(10))
    ));
    assert!(matches!(
        stl(
            y.as_ref(),
            StlOpts {
                inner_iters: 0,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidInnerIters)
    ));
    let short = col_from(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        stl(short.as_ref(), StlOpts::new(4)),
        Err(StlError::SeriesTooShort { .. })
    ));
}

#[test]
fn multiplicative_rejects_non_positive() {
    let y = col_from(vec![1.0, 2.0, 0.0, 1.5].repeat(6));
    let err = stl(
        y.as_ref(),
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(4)
        },
    )
    .unwrap_err();
    assert!(matches!(err, StlError::NonPositiveForMultiplicative { .. }));
}

#[test]
fn rejects_non_finite() {
    let mut v = vec![1.0; 24];
    v[5] = f64::NAN;
    let y = col_from(v);
    assert!(matches!(
        stl(y.as_ref(), StlOpts::new(4)),
        Err(StlError::NonFinite)
    ));
}
