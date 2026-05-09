//! Unit tests for `rust_stats::smoothing::loess`.

use approx::assert_relative_eq;
use faer::Col;
use rust_stats::smoothing::{loess, loess_at};
use rust_stats::error::LoessError;

fn col_from(v: Vec<f64>) -> Col<f64> {
    Col::<f64>::from_fn(v.len(), |i| v[i])
}

#[test]
fn constant_signal_returns_constant() {
    let y = col_from(vec![3.0; 20]);
    let out = loess(y.as_ref(), 0.5, 1).unwrap();
    for i in 0..out.nrows() {
        assert_relative_eq!(out[i], 3.0, epsilon = 1e-9);
    }
}

#[test]
fn linear_signal_exact_recovery_degree_one() {
    let n = 50;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let out = loess(y.as_ref(), 0.5, 1).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], i as f64, epsilon = 1e-9);
    }
}

#[test]
fn quadratic_signal_exact_recovery_degree_two() {
    let n = 30;
    let y = col_from((0..n).map(|i| (i as f64).powi(2)).collect());
    let out = loess(y.as_ref(), 0.5, 2).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], (i as f64).powi(2), epsilon = 1e-9, max_relative = 1e-9);
    }
}

#[test]
fn wider_span_smooths_more() {
    // Deterministic noisy linear series — wider span reduces residual variance.
    let n = 300;
    let y: Vec<f64> = {
        let mut state: u64 = 1;
        (0..n)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as i32 as f64) / (1u64 << 31) as f64;
                i as f64 + noise
            })
            .collect()
    };
    let y_col = col_from(y);

    let narrow = loess(y_col.as_ref(), 0.05, 1).unwrap();
    let wide = loess(y_col.as_ref(), 0.5, 1).unwrap();
    let narrow_var: f64 = (0..n).map(|i| (narrow[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    let wide_var: f64 = (0..n).map(|i| (wide[i] - i as f64).powi(2)).sum::<f64>() / n as f64;
    assert!(wide_var < narrow_var, "wide_var={} not < narrow_var={}", wide_var, narrow_var);
}

#[test]
fn step_function_smooths_with_bounded_overshoot() {
    let n = 100;
    let half = n / 2;
    let mut v = vec![0.0; half];
    v.extend(vec![1.0; n - half]);
    let y = col_from(v);
    let out = loess(y.as_ref(), 0.2, 1).unwrap();
    assert!(out[0] < 0.05);
    assert!(out[n - 1] > 0.95);
    for i in 0..n {
        assert!((-0.1..=1.1).contains(&out[i]), "overshoot at {}: {}", i, out[i]);
    }
}

#[test]
fn constant_signal_preserved_with_degree_two() {
    let n = 50;
    let y = col_from(vec![4.2; n]);
    let out = loess(y.as_ref(), 0.4, 2).unwrap();
    for i in 0..n {
        assert_relative_eq!(out[i], 4.2, epsilon = 1e-9);
    }
}

#[test]
fn short_series_falls_back_gracefully() {
    let y = col_from(vec![1.0, 2.0, 3.0]);
    let out = loess(y.as_ref(), 1.0, 1).unwrap();
    assert!(out.as_ref().iter().all(|v| v.is_finite()));
    assert_relative_eq!(out[0], 1.0, epsilon = 1e-9);
    assert_relative_eq!(out[1], 2.0, epsilon = 1e-9);
    assert_relative_eq!(out[2], 3.0, epsilon = 1e-9);
}

#[test]
fn boundary_recovery_exact_on_linear_input() {
    let n = 100;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let out = loess(y.as_ref(), 0.3, 1).unwrap();
    assert_relative_eq!(out[0], 0.0, epsilon = 1e-9);
    assert_relative_eq!(out[n - 1], (n - 1) as f64, epsilon = 1e-9);
}

#[test]
fn loess_at_extrapolates_past_boundary() {
    let n = 50;
    let y = col_from((0..n).map(|i| i as f64).collect());
    let v = loess_at(y.as_ref(), -1.0, 0.3, 1).unwrap();
    assert_relative_eq!(v, -1.0, epsilon = 1e-6);
    let v2 = loess_at(y.as_ref(), n as f64, 0.3, 1).unwrap();
    assert_relative_eq!(v2, n as f64, epsilon = 1e-6);
}

#[test]
fn validation_rejects_bad_span_and_degree() {
    let y = col_from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(loess(y.as_ref(), 0.0, 1), Err(LoessError::InvalidSpan(0.0)));
    assert_eq!(loess(y.as_ref(), 1.5, 1), Err(LoessError::InvalidSpan(1.5)));
    assert_eq!(loess(y.as_ref(), 0.5, 3), Err(LoessError::InvalidDegree(3)));
}

#[test]
fn rejects_non_finite_input() {
    let y = col_from(vec![1.0, f64::NAN, 3.0]);
    assert_eq!(loess(y.as_ref(), 0.5, 1), Err(LoessError::NonFinite));
}
