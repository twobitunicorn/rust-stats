//! Holt-Winters tests inspired by R and statsmodels:
//!
//!   - R: `stats::HoltWinters(x, alpha, beta, gamma, seasonal = "additive"|"multiplicative")`
//!   - statsmodels: `statsmodels.tsa.holtwinters.SimpleExpSmoothing`,
//!                  `Holt`, and `ExponentialSmoothing`
//!                  with `.fit(smoothing_level=α, smoothing_trend=β,
//!                              smoothing_seasonal=γ, optimized=False)`.
//!
//! Exact-value parity with either reference is not portable: R's
//! `HoltWinters` seeds level/trend via least-squares regression on the
//! first two seasons, while statsmodels uses heuristic initial states
//! that have changed between releases. Our implementation uses the
//! textbook recipe (level = y[0], trend = y[1] - y[0], seasonal indices
//! from the first-period deviations). The tests below therefore focus on
//! the *invariant* behaviours that any correct Holt-Winters implementation
//! must reproduce — naive/no-update extremes, perfect recovery on
//! noise-free seasonal/linear data, and error conditions documented by
//! both upstream APIs.
//!
//! Tests that depend on MLE-fitted (α, β, γ), boxcox preprocessing,
//! damped trend, or `forecast()` extrapolation beyond the sample are not
//! ported.

use approx::assert_relative_eq;
use rust_stats::error::HoltWintersError;
use rust_stats::tsa::{holt_winters, DecomposeMode, HoltWintersOpts};

// ============================================================================
// Simple exponential smoothing  —  statsmodels SimpleExpSmoothing
// ============================================================================

/// statsmodels: with `initial_level=y[0]` and `optimized=False`,
/// `SimpleExpSmoothing(y).fit(smoothing_level=0.0).fittedvalues` is a
/// constant series equal to `y[0]` (no update ever moves level).
#[test]
fn ses_alpha_zero_freezes_at_initial() {
    let y = [3.0, 7.0, 5.0, 9.0, 8.0];
    let fit = holt_winters(&y, HoltWintersOpts::new(0.0)).unwrap();
    assert_eq!(fit.fitted, vec![3.0; 5]);
}

/// statsmodels: with `smoothing_level=1.0`, SES reduces to the naive
/// (random-walk) one-step forecast: `ŷ[t] = y[t-1]` for `t ≥ 1`, and
/// `ŷ[0] = y[0]` because the initial level is seeded to `y[0]`.
#[test]
fn ses_alpha_one_is_naive_forecast() {
    let y = [3.0, 7.0, 5.0, 9.0, 8.0];
    let fit = holt_winters(&y, HoltWintersOpts::new(1.0)).unwrap();
    assert_eq!(fit.fitted, vec![3.0, 3.0, 7.0, 5.0, 9.0]);
}

/// Hand-computed SES recursion (α = 0.5, y = [1, 2, 3, 4]):
///
/// ```text
/// level₀ = 1
/// ŷ₀ = 1.00 ; level₁ = 0.5·1 + 0.5·1 = 1.00
/// ŷ₁ = 1.00 ; level₂ = 0.5·2 + 0.5·1 = 1.50
/// ŷ₂ = 1.50 ; level₃ = 0.5·3 + 0.5·1.5 = 2.25
/// ŷ₃ = 2.25
/// ```
///
/// Matches `SimpleExpSmoothing([1,2,3,4]).fit(smoothing_level=0.5,
/// initial_level=1.0, optimized=False).fittedvalues`.
#[test]
fn ses_alpha_half_matches_hand_computation() {
    let fit = holt_winters(&[1.0, 2.0, 3.0, 4.0], HoltWintersOpts::new(0.5)).unwrap();
    for (i, (&a, &e)) in fit.fitted.iter().zip(&[1.0, 1.0, 1.5, 2.25]).enumerate() {
        assert_relative_eq!(a, e, max_relative = 1e-12, epsilon = 1e-12);
        let _ = i;
    }
}

/// On a constant series, SES (and any Holt-Winters specialisation) must
/// return the same constant — there is no error to absorb.
#[test]
fn ses_constant_series_is_constant() {
    let fit = holt_winters(&[4.2; 10], HoltWintersOpts::new(0.3)).unwrap();
    for v in fit.fitted {
        assert_relative_eq!(v, 4.2, max_relative = 1e-12);
    }
}

// ============================================================================
// Holt's linear method  —  statsmodels Holt; R HoltWinters(gamma = FALSE)
// ============================================================================

/// With `α = β = 1` and a perfectly linear series, after the two-step
/// initialisation transient the fitted values should match `y[t]` exactly:
/// the level absorbs `y[t-1]` and the trend recovers the per-step slope.
#[test]
fn holt_alpha_beta_one_recovers_linear_after_transient() {
    let y: Vec<f64> = (10..20).map(|i| i as f64).collect();
    let opts = HoltWintersOpts {
        alpha: 1.0,
        beta: 1.0,
        ..HoltWintersOpts::new(1.0)
    };
    let fit = holt_winters(&y, opts).unwrap();
    // Transient at t=0 (yhat = level₀ + trend₀ = 11) and t=1 (yhat = 10).
    // From t = 2 onward, fitted == y exactly.
    for t in 2..y.len() {
        assert_relative_eq!(fit.fitted[t], y[t], max_relative = 1e-12);
    }
}

/// statsmodels behaviour: with `smoothing_trend=0.0` the trend term is
/// pinned to its initial value forever (no β update). Our implementation
/// treats β = 0 as "no trend term"; the fitted values must still be
/// finite and match SES at α — i.e., dropping β does not crash.
#[test]
fn holt_beta_zero_reduces_to_ses() {
    let y = [3.0, 7.0, 5.0, 9.0, 8.0];
    let ses = holt_winters(&y, HoltWintersOpts::new(0.5)).unwrap();
    let holt = holt_winters(
        &y,
        HoltWintersOpts {
            beta: 0.0,
            ..HoltWintersOpts::new(0.5)
        },
    )
    .unwrap();
    for (a, b) in ses.fitted.iter().zip(holt.fitted.iter()) {
        assert_relative_eq!(*a, *b, max_relative = 1e-12);
    }
}

// ============================================================================
// Seasonal Holt-Winters  —  statsmodels ExponentialSmoothing; R HoltWinters
// ============================================================================

/// Additive seasonal with a perfectly seasonal noise-free series should
/// reproduce the input exactly: the initial seasonal indices already
/// encode the [10, 20] pattern, and every update is a no-op fixed point.
///
/// Hand trace (m = 2, α = γ = 0.5, β = 0):
/// `level₀ = 15, s_buf = [-5, +5]` →
/// `ŷ₀ = 15 + (-5) = 10 = y₀`, `ŷ₁ = 15 + 5 = 20 = y₁`, etc.
#[test]
fn additive_seasonal_recovers_perfect_seasonal_series() {
    let y = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0];
    let opts = HoltWintersOpts {
        alpha: 0.5,
        beta: 0.0,
        gamma: 0.5,
        seasonal_periods: 2,
        mode: DecomposeMode::Additive,
    };
    let fit = holt_winters(&y, opts).unwrap();
    for (i, (&a, &e)) in fit.fitted.iter().zip(y.iter()).enumerate() {
        assert_relative_eq!(a, e, max_relative = 1e-12, epsilon = 1e-12);
        let _ = i;
    }
}

/// Multiplicative seasonal with the same noise-free series and
/// `mode = Multiplicative`. Initial seasonals are `[2/3, 4/3]`; one-step
/// fitted values are `level * s` = `15 * 2/3 = 10` and `15 * 4/3 = 20`,
/// matching `y` exactly across the whole series.
#[test]
fn multiplicative_seasonal_recovers_perfect_seasonal_series() {
    let y = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0];
    let opts = HoltWintersOpts {
        alpha: 0.5,
        beta: 0.0,
        gamma: 0.5,
        seasonal_periods: 2,
        mode: DecomposeMode::Multiplicative,
    };
    let fit = holt_winters(&y, opts).unwrap();
    for (&a, &e) in fit.fitted.iter().zip(y.iter()) {
        assert_relative_eq!(a, e, max_relative = 1e-12, epsilon = 1e-12);
    }
}

/// Constant series with full additive Holt-Winters: every state is at
/// equilibrium (level = const, trend = 0, seasonals = 0), so fitted
/// values must equal the constant. R and statsmodels agree.
#[test]
fn seasonal_constant_series_is_constant() {
    let y = [5.0; 12];
    let opts = HoltWintersOpts {
        alpha: 0.5,
        beta: 0.3,
        gamma: 0.4,
        seasonal_periods: 4,
        mode: DecomposeMode::Additive,
    };
    let fit = holt_winters(&y, opts).unwrap();
    for v in fit.fitted {
        assert_relative_eq!(v, 5.0, max_relative = 1e-12);
    }
}

// ============================================================================
// Error conditions  —  matches statsmodels validation and R argument checks
// ============================================================================

/// statsmodels and R both reject smoothing parameters outside [0, 1].
#[test]
fn alpha_out_of_range_errors() {
    let err = holt_winters(&[1.0, 2.0], HoltWintersOpts::new(1.5)).unwrap_err();
    assert_eq!(err, HoltWintersError::InvalidAlpha(1.5));
    let err = holt_winters(&[1.0, 2.0], HoltWintersOpts::new(-0.1)).unwrap_err();
    assert_eq!(err, HoltWintersError::InvalidAlpha(-0.1));
}

#[test]
fn beta_out_of_range_errors() {
    let opts = HoltWintersOpts {
        beta: 2.0,
        ..HoltWintersOpts::new(0.5)
    };
    let err = holt_winters(&[1.0, 2.0], opts).unwrap_err();
    assert_eq!(err, HoltWintersError::InvalidBeta(2.0));
}

#[test]
fn gamma_out_of_range_errors() {
    let opts = HoltWintersOpts {
        gamma: -0.5,
        seasonal_periods: 2,
        ..HoltWintersOpts::new(0.5)
    };
    let err = holt_winters(&[1.0, 2.0, 3.0, 4.0], opts).unwrap_err();
    assert_eq!(err, HoltWintersError::InvalidGamma(-0.5));
}

/// statsmodels raises `ValueError("Cannot fit ... with fewer than 2 *
/// seasonal_periods observations")`. We surface a typed error with the
/// same threshold.
#[test]
fn seasonal_series_too_short_errors() {
    let opts = HoltWintersOpts {
        beta: 0.1,
        gamma: 0.3,
        seasonal_periods: 4,
        ..HoltWintersOpts::new(0.5)
    };
    let err = holt_winters(&[1.0, 2.0, 3.0, 4.0, 5.0], opts).unwrap_err();
    assert_eq!(err, HoltWintersError::SeriesTooShort { n: 5, min: 8 });
}

/// Holt's linear method needs at least two observations to seed the
/// trend (we use `y[1] - y[0]`). statsmodels' `Holt` likewise raises on
/// `n < 2`.
#[test]
fn holt_linear_needs_two_observations() {
    let opts = HoltWintersOpts {
        beta: 0.3,
        ..HoltWintersOpts::new(0.5)
    };
    let err = holt_winters(&[1.0], opts).unwrap_err();
    assert_eq!(err, HoltWintersError::SeriesTooShort { n: 1, min: 2 });
}

/// statsmodels' multiplicative seasonal model rejects non-positive
/// values (you cannot divide by zero seasonals): `ValueError`.
#[test]
fn multiplicative_rejects_zero_and_negative() {
    let opts = HoltWintersOpts {
        beta: 0.1,
        gamma: 0.3,
        seasonal_periods: 2,
        mode: DecomposeMode::Multiplicative,
        ..HoltWintersOpts::new(0.5)
    };
    let err = holt_winters(&[1.0, 0.0, 2.0, 3.0], opts.clone()).unwrap_err();
    assert!(matches!(
        err,
        HoltWintersError::NonPositiveForMultiplicative { .. }
    ));

    let err = holt_winters(&[1.0, -2.0, 2.0, 3.0], opts).unwrap_err();
    assert!(matches!(
        err,
        HoltWintersError::NonPositiveForMultiplicative { .. }
    ));
}

/// R's `HoltWinters` errors on `NA` input (after `na.action`); statsmodels
/// rejects non-finite. We do the same.
#[test]
fn rejects_non_finite_input() {
    let err = holt_winters(&[1.0, f64::NAN, 3.0], HoltWintersOpts::new(0.5)).unwrap_err();
    assert_eq!(err, HoltWintersError::NonFinite);

    let err = holt_winters(
        &[1.0, f64::INFINITY, 3.0],
        HoltWintersOpts::new(0.5),
    )
    .unwrap_err();
    assert_eq!(err, HoltWintersError::NonFinite);
}

/// Empty input returns empty output without erroring — consistent with
/// our other slice transforms and with `ExponentialSmoothing` accepting
/// length-0 numpy arrays.
#[test]
fn empty_input_returns_empty() {
    let fit = holt_winters(&[], HoltWintersOpts::new(0.5)).unwrap();
    assert!(fit.fitted.is_empty());

    let fit = holt_winters(
        &[],
        HoltWintersOpts {
            beta: 0.3,
            gamma: 0.3,
            seasonal_periods: 4,
            ..HoltWintersOpts::new(0.5)
        },
    )
    .unwrap();
    assert!(fit.fitted.is_empty());
}

/// Fitted-value length always equals input length — a hard invariant of
/// both R and statsmodels' in-sample fits.
#[test]
fn fitted_length_matches_input_length() {
    let y: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) + 5.0 * ((i as f64) * 0.5).sin()).collect();
    let opts = HoltWintersOpts {
        alpha: 0.3,
        beta: 0.1,
        gamma: 0.2,
        seasonal_periods: 4,
        mode: DecomposeMode::Additive,
    };
    let fit = holt_winters(&y, opts).unwrap();
    assert_eq!(fit.fitted.len(), y.len());
    for v in &fit.fitted {
        assert!(v.is_finite(), "fitted value is not finite: {v}");
    }
}
