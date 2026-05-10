//! seasonal_decompose tests ported from statsmodels' test suite:
//!
//!   `statsmodels.tsa.tests.test_seasonal.TestDecompose`
//!   `statsmodels.tsa.tests.test_seasonal.test_seasonal_decompose_too_short`
//!
//! The hard-coded SEASONAL/TREND/RANDOM constants below are taken
//! verbatim from upstream and carry the same provenance. Tests that
//! exercise `two_sided=False`, `extrapolate_trend`, multi-column input,
//! pandas indices, or custom filters are not portable to our API.

use rust_stats::error::SeasonalDecomposeError;
use rust_stats::tsa::{seasonal_decompose, DecomposeMode, SeasonalDecomposeOpts};

const DATA: [f64; 32] = [
    -50.0, 175.0, 149.0, 214.0, 247.0, 237.0, 225.0, 329.0, 729.0, 809.0,
    530.0, 489.0, 540.0, 457.0, 195.0, 176.0, 337.0, 239.0, 128.0, 102.0,
    232.0, 429.0, 3.0, 98.0, 43.0, -141.0, -77.0, -13.0, 125.0, 361.0,
    -45.0, 184.0,
];

const SEASONAL: [f64; 32] = [
    62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38, -60.25, 62.46,
    86.17, -88.38, -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17,
    -88.38, -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
    -60.25, 62.46, 86.17, -88.38, -60.25,
];

const TREND: [f64; 32] = [
    f64::NAN, f64::NAN, 159.12, 204.00, 221.25, 245.12, 319.75, 451.50,
    561.12, 619.25, 615.62, 548.00, 462.12, 381.12, 316.62, 264.00, 228.38,
    210.75, 188.38, 199.00, 207.12, 191.00, 166.88, 72.00, -9.25, -33.12,
    -36.75, 36.25, 103.00, 131.62, f64::NAN, f64::NAN,
];

const RANDOM: [f64; 32] = [
    f64::NAN, f64::NAN, 78.254, 70.254, -36.710, -94.299, -6.371, -62.246,
    105.415, 103.576, 2.754, 1.254, 15.415, -10.299, -33.246, -27.746,
    46.165, -57.924, 28.004, -36.746, -37.585, 151.826, -75.496, 86.254,
    -10.210, -194.049, 48.129, 11.004, -40.460, 143.201, f64::NAN, f64::NAN,
];

const MULT_SEASONAL: [f64; 32] = [
    1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815,
    1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538,
    0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716,
    0.6931, 1.0815, 1.5538, 0.6716, 0.6931,
];

const MULT_TREND: [f64; 32] = [
    f64::NAN, f64::NAN, 171.62, 204.00, 221.25, 245.12, 319.75, 451.50,
    561.12, 619.25, 615.62, 548.00, 462.12, 381.12, 316.62, 264.00, 228.38,
    210.75, 188.38, 199.00, 207.12, 191.00, 166.88, 107.25, 80.50, 79.12,
    78.75, 116.50, 140.00, 157.38, f64::NAN, f64::NAN,
];

const MULT_RANDOM: [f64; 32] = [
    f64::NAN, f64::NAN, 1.29263, 1.51360, 1.03223, 0.62226, 1.04771, 1.05139,
    1.20124, 0.84080, 1.28182, 1.28752, 1.08043, 0.77172, 0.91697, 0.96191,
    1.36441, 0.72986, 1.01171, 0.73956, 1.03566, 1.44556, 0.02677, 1.31843,
    0.49390, 1.14688, 1.45582, 0.16101, 0.82555, 1.47633, f64::NAN, f64::NAN,
];

fn approx(a: f64, b: f64, decimal: i32, ctx: &str) {
    // Match numpy.testing.assert_almost_equal: tolerance = 1.5 * 10^-decimal.
    if b.is_nan() {
        assert!(a.is_nan(), "{ctx}: expected NaN, got {a}");
        return;
    }
    let tol = 1.5 * 10f64.powi(-decimal);
    assert!(
        (a - b).abs() < tol,
        "{ctx}: differs beyond {decimal} decimals: got {a}, expected {b}",
    );
}

/// Source: TestDecompose.test_ndarray (the even-length additive case).
#[test]
fn test_additive_even() {
    let opts = SeasonalDecomposeOpts::new(4);
    let d = seasonal_decompose(&DATA, opts).unwrap();
    for i in 0..32 {
        approx(d.seasonal[i], SEASONAL[i], 2, &format!("seasonal[{i}]"));
        approx(d.trend[i],    TREND[i],    2, &format!("trend[{i}]"));
        approx(d.residual[i], RANDOM[i],   3, &format!("resid[{i}]"));
    }
}

/// Source: TestDecompose.test_ndarray (the odd-length additive case).
#[test]
fn test_additive_odd() {
    let opts = SeasonalDecomposeOpts::new(4);
    let d = seasonal_decompose(&DATA[..31], opts).unwrap();
    let seasonal = [
        68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18,
        69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02,
        -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66,
        -54.54, 68.18, 69.02, -82.66,
    ];
    let trend = [
        f64::NAN, f64::NAN, 159.12, 204.00, 221.25, 245.12, 319.75, 451.50,
        561.12, 619.25, 615.62, 548.00, 462.12, 381.12, 316.62, 264.00,
        228.38, 210.75, 188.38, 199.00, 207.12, 191.00, 166.88, 72.00,
        -9.25, -33.12, -36.75, 36.25, 103.00, f64::NAN, f64::NAN,
    ];
    let random = [
        f64::NAN, f64::NAN, 72.538, 64.538, -42.426, -77.150, -12.087,
        -67.962, 99.699, 120.725, -2.962, -4.462, 9.699, 6.850, -38.962,
        -33.462, 40.449, -40.775, 22.288, -42.462, -43.301, 168.975,
        -81.212, 80.538, -15.926, -176.900, 42.413, 5.288, -46.176, f64::NAN,
        f64::NAN,
    ];
    for i in 0..31 {
        approx(d.seasonal[i], seasonal[i], 2, &format!("seasonal[{i}]"));
        approx(d.trend[i],    trend[i],    2, &format!("trend[{i}]"));
        approx(d.residual[i], random[i],   3, &format!("resid[{i}]"));
    }
}

/// Source: TestDecompose.test_ndarray (the multiplicative case on |DATA|).
#[test]
fn test_multiplicative() {
    let y: Vec<f64> = DATA.iter().map(|v| v.abs()).collect();
    let mut opts = SeasonalDecomposeOpts::new(4);
    opts.mode = DecomposeMode::Multiplicative;
    let d = seasonal_decompose(&y, opts).unwrap();
    for i in 0..32 {
        approx(d.seasonal[i], MULT_SEASONAL[i], 4, &format!("seasonal[{i}]"));
        approx(d.trend[i],    MULT_TREND[i],    2, &format!("trend[{i}]"));
        approx(d.residual[i], MULT_RANDOM[i],   4, &format!("resid[{i}]"));
    }
}

/// Source: test_seasonal_decompose_too_short.
#[test]
fn test_too_short() {
    let y4: Vec<f64> = (0..4).map(|i| (i as f64 / 4.0 * 2.0 * std::f64::consts::PI).sin()).collect();
    assert!(matches!(
        seasonal_decompose(&y4, SeasonalDecomposeOpts::new(4)),
        Err(SeasonalDecomposeError::SeriesTooShort { .. })
    ));
    let y12: Vec<f64> = (0..12).map(|i| (i as f64 / 12.0 * 2.0 * std::f64::consts::PI).sin()).collect();
    assert!(matches!(
        seasonal_decompose(&y12, SeasonalDecomposeOpts::new(12)),
        Err(SeasonalDecomposeError::SeriesTooShort { .. })
    ));
}
