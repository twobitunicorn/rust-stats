//! STL tests ported from statsmodels' test suite:
//!
//!   `statsmodels.tsa.stl.tests.test_stl`
//!
//! Tests that depend on the upstream `stl_co2.csv` / `stl_test_results.csv`
//! fixtures, robust outer iterations, jump parameters, or pandas indices
//! are not portable and are omitted. The CSV-based regression checks are
//! covered by our generated parity tests in `tests/stl_golden.rs`.

use rust_stats::error::StlError;
use rust_stats::tsa::{stl, StlOpts};

fn dummy_series(n: usize, period: usize) -> Vec<f64> {
    // Cleveland-style: trend + sin seasonal + noise-free.
    (0..n)
        .map(|i| {
            let t = i as f64;
            let ph = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            10.0 + 0.05 * t + 3.0 * ph.sin()
        })
        .collect()
}

/// Source: test_parameter_checks_period.
#[test]
fn test_parameter_checks_period() {
    let y = dummy_series(120, 12);
    // statsmodels: ValueError on period=1.
    assert!(matches!(
        stl(&y, StlOpts::new(1)),
        Err(StlError::InvalidPeriod(1))
    ));
    // period=0 is also invalid.
    assert!(matches!(
        stl(&y, StlOpts::new(0)),
        Err(StlError::InvalidPeriod(0))
    ));
    // (Negative / non-integer / multi-column endog are not representable
    //  in our typed API and therefore not ported.)
}

/// Source: test_parameter_checks_seasonal — seasonal must be odd and >= 7
/// (statsmodels says >= 3; our constructor tightens to >= 7 because the
/// implementation isn't valid below that).
#[test]
fn test_parameter_checks_seasonal() {
    let y = dummy_series(120, 12);
    let mut opts = StlOpts::new(12);
    opts.seasonal_window = 2;
    assert!(matches!(
        stl(&y, opts),
        Err(StlError::InvalidSeasonalWindow(2))
    ));

    let mut opts = StlOpts::new(12);
    opts.seasonal_window = 8; // even
    assert!(matches!(
        stl(&y, opts),
        Err(StlError::InvalidSeasonalWindow(8))
    ));

    let mut opts = StlOpts::new(12);
    opts.seasonal_window = 5; // odd but < 7
    assert!(matches!(
        stl(&y, opts),
        Err(StlError::InvalidSeasonalWindow(5))
    ));
}

/// Source: test_parameter_checks_trend — trend must be odd. (Our API uses
/// `Option<u32>` for trend_window so we can only test the explicit-even
/// case; statsmodels also rejects trend <= period, which we don't enforce
/// because the LOESS span clamps to series length anyway.)
#[test]
fn test_parameter_checks_trend() {
    let y = dummy_series(120, 12);
    let mut opts = StlOpts::new(12);
    opts.trend_window = Some(14); // even
    assert!(matches!(
        stl(&y, opts),
        Err(StlError::InvalidTrendWindow(14))
    ));
}

/// Source: test_default_trend (GH 6686). The default trend window must be
/// the smallest odd integer >= ceil(1.5 * period / (1 - 1.5 / seasonal)).
/// We don't expose the computed window directly, so we verify behaviour:
/// the auto-default must produce the same decomposition as supplying that
/// integer explicitly.
#[test]
fn test_default_trend() {
    let y = dummy_series(120, 12);

    for &(period, seasonal) in &[(12u32, 17u32), (12, 7)] {
        let raw = 1.5 * period as f64 / (1.0 - 1.5 / seasonal as f64);
        let mut expected = raw.ceil() as u32;
        if expected % 2 == 0 {
            expected += 1;
        }

        let opts_default = StlOpts {
            period,
            seasonal_window: seasonal,
            trend_window: None,
            ..StlOpts::new(period)
        };
        let opts_explicit = StlOpts {
            period,
            seasonal_window: seasonal,
            trend_window: Some(expected),
            ..StlOpts::new(period)
        };

        let d_default  = stl(&y, opts_default).unwrap();
        let d_explicit = stl(&y, opts_explicit).unwrap();

        for i in 0..y.len() {
            let dt = (d_default.trend[i] - d_explicit.trend[i]).abs();
            let ds = (d_default.seasonal[i] - d_explicit.seasonal[i]).abs();
            assert!(dt < 1e-12, "trend mismatch at i={i}: {dt}");
            assert!(ds < 1e-12, "seasonal mismatch at i={i}: {ds}");
        }
    }
}
