//! Unit tests for `rust_stats::tsa::seasonal::stl`.

use approx::assert_relative_eq;
use rust_stats::error::StlError;
use rust_stats::tsa::{stl, DecomposeMode, StlOpts};

#[test]
fn pure_linear_trend_recovered_everywhere() {
    let n = 24usize;
    let period = 4u32;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let d = stl(&y, StlOpts::new(period)).unwrap();
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
    let y: Vec<f64> = (0..n).map(|i| pattern[i % pattern.len()]).collect();
    let d = stl(&y, StlOpts::new(period)).unwrap();
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
    let y: Vec<f64> = (0..n).map(|i| i as f64 + pattern[i % 4]).collect();
    let d = stl(&y, StlOpts::new(period)).unwrap();
    for i in 0..n {
        let recon = d.trend[i] + d.seasonal[i] + d.residual[i];
        assert_relative_eq!(recon, y[i], epsilon = 1e-9);
    }
}

#[test]
fn multiplicative_reconstruction_exact_everywhere() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let y: Vec<f64> = (0..24)
        .map(|i| (1.0 + 0.05 * i as f64) * pattern[i % 4])
        .collect();
    let d = stl(
        &y,
        StlOpts {
            mode: DecomposeMode::Multiplicative,
            ..StlOpts::new(period)
        },
    )
    .unwrap();
    for i in 0..y.len() {
        let recon = d.trend[i] * d.seasonal[i] * d.residual[i];
        assert_relative_eq!(recon, y[i], max_relative = 1e-9);
    }
}

#[test]
fn additive_seasonal_pattern_sums_to_zero() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 3.0, 2.0];
    let n = pattern.len() * 6;
    let y: Vec<f64> = (0..n).map(|i| pattern[i % 4]).collect();
    let d = stl(&y, StlOpts::new(period)).unwrap();
    let inner: f64 = (8..12).map(|i| d.seasonal[i]).sum();
    assert_relative_eq!(inner, 0.0, epsilon = 1e-9);
}

#[test]
fn multiplicative_seasonal_pattern_products_to_one() {
    let period = 4u32;
    let pattern = [1.0, 2.0, 0.5, 1.5];
    let n = pattern.len() * 6;
    let y: Vec<f64> = (0..n).map(|i| pattern[i % 4]).collect();
    let d = stl(
        &y,
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
    let y = vec![1.0; 24];
    assert!(matches!(
        stl(&y, StlOpts::new(1)),
        Err(StlError::InvalidPeriod(1))
    ));
    assert!(matches!(
        stl(
            &y,
            StlOpts {
                seasonal_window: 8,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidSeasonalWindow(8))
    ));
    assert!(matches!(
        stl(
            &y,
            StlOpts {
                trend_window: Some(10),
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidTrendWindow(10))
    ));
    assert!(matches!(
        stl(
            &y,
            StlOpts {
                inner_iters: 0,
                ..StlOpts::new(4)
            }
        ),
        Err(StlError::InvalidInnerIters)
    ));
    let short = vec![1.0, 2.0, 3.0];
    assert!(matches!(
        stl(&short, StlOpts::new(4)),
        Err(StlError::SeriesTooShort { .. })
    ));
}

#[test]
fn multiplicative_rejects_non_positive() {
    let y = [1.0, 2.0, 0.0, 1.5].repeat(6);
    let err = stl(
        &y,
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
    assert!(matches!(
        stl(&v, StlOpts::new(4)),
        Err(StlError::NonFinite)
    ));
}

// ── Jump-parameter tests ─────────────────────────────────────────────────

fn airpassengers_like(n: usize, period: usize) -> Vec<f64> {
    let mut state = 1u64;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (state as f64 / u64::MAX as f64) - 0.5;
            let phase = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            100.0 + 0.5 * i as f64 + 30.0 * phase.sin() + 5.0 * noise
        })
        .collect()
}

#[test]
fn jump_one_matches_default() {
    let y = airpassengers_like(144, 12);
    let default = stl(&y, StlOpts::new(12)).unwrap();
    let explicit_one = stl(
        &y,
        StlOpts {
            seasonal_jump: 1,
            trend_jump:    1,
            low_pass_jump: 1,
            ..StlOpts::new(12)
        },
    )
    .unwrap();
    for i in 0..y.len() {
        assert_relative_eq!(default.trend[i],    explicit_one.trend[i],    epsilon = 1e-12);
        assert_relative_eq!(default.seasonal[i], explicit_one.seasonal[i], epsilon = 1e-12);
        assert_relative_eq!(default.residual[i], explicit_one.residual[i], epsilon = 1e-12);
    }
}

#[test]
fn jump_two_close_to_exact_and_reconstructs() {
    let y = airpassengers_like(144, 12);
    let exact = stl(&y, StlOpts::new(12)).unwrap();
    let jumped = stl(
        &y,
        StlOpts {
            seasonal_jump: 2,
            trend_jump:    2,
            low_pass_jump: 2,
            ..StlOpts::new(12)
        },
    )
    .unwrap();
    // Reconstruction identity still holds (T + S + R = y) regardless of jumps.
    for i in 0..y.len() {
        assert_relative_eq!(
            jumped.trend[i] + jumped.seasonal[i] + jumped.residual[i],
            y[i],
            epsilon = 1e-10
        );
    }
    // Jumped components should be close to exact — the linear interpolation
    // introduces error proportional to the curvature of the LOESS surface;
    // on a smooth seasonal pattern with period=12 the drift is small.
    for i in 0..y.len() {
        assert!(
            (jumped.trend[i] - exact.trend[i]).abs() < 3.0,
            "trend drift at i={i}: {} vs {}", jumped.trend[i], exact.trend[i]
        );
        assert!(
            (jumped.seasonal[i] - exact.seasonal[i]).abs() < 3.0,
            "seasonal drift at i={i}: {} vs {}", jumped.seasonal[i], exact.seasonal[i]
        );
    }
}

#[test]
fn zero_jumps_error() {
    let y = airpassengers_like(48, 4);
    assert!(matches!(
        stl(&y, StlOpts { seasonal_jump: 0, ..StlOpts::new(4) }),
        Err(StlError::InvalidJump { which: "seasonal" })
    ));
    assert!(matches!(
        stl(&y, StlOpts { trend_jump: 0, ..StlOpts::new(4) }),
        Err(StlError::InvalidJump { which: "trend" })
    ));
    assert!(matches!(
        stl(&y, StlOpts { low_pass_jump: 0, ..StlOpts::new(4) }),
        Err(StlError::InvalidJump { which: "low_pass" })
    ));
}
