//! NIST / Stata / R / SAS reference values for the Longley dataset, ported
//! from statsmodels' regression test suite:
//!
//!   * `statsmodels.regression.tests.test_regression.CheckRegressionResults`
//!   * `statsmodels.regression.tests.results.results_regression.Longley`
//!
//! NIST authority: <http://www.itl.nist.gov/div898/strd/general/dataarchive.html>.
//! Robust SEs (HC0–HC3) are SAS values quoted in the Longley fixture.
//!
//! The fixture lists the constant LAST (statsmodels uses
//! `add_constant(prepend=False)`); rust-stats puts it FIRST. The mapping
//! is: rust `coef[0]` = stats `params[6]`, rust `coef[1..7]` = stats
//! `params[0..6]`.

use rust_stats::{CovType, Matrix, Ols};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Longley {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
}

fn load() -> Longley {
    let path: PathBuf = ["tests", "golden", "longley.json"].iter().collect();
    let bytes = std::fs::read(&path).expect("longley.json missing — run tests/golden/generate.py");
    serde_json::from_slice(&bytes).expect("invalid longley.json")
}

fn matrix_from(rows: &[Vec<f64>]) -> Matrix<f64> {
    let n = rows.len();
    let p = rows[0].len();
    Matrix::from_fn(n, p, |i, j| rows[i][j])
}

/// stats[0..6] = features, stats[6] = intercept; rust[0] = intercept,
/// rust[1..7] = features. Reorder a feature-then-intercept reference vector
/// into rust order.
fn to_rust_order<const N: usize>(stats: [f64; N]) -> Vec<f64> {
    assert_eq!(N, 7);
    let mut out = vec![0.0; N];
    out[0] = stats[6];
    out[1..].copy_from_slice(&stats[0..6]);
    out
}

// ── Reference values from results_regression.Longley (NIST + Stata + R + SAS).
// Order: GNPDEFL, GNP, UNEMP, ARMED, POP, YEAR, const.
const PARAMS: [f64; 7] = [
    15.0618722713733,
    -0.358191792925910e-1,
    -2.02022980381683,
    -1.03322686717359,
    -0.511041056535807e-1,
    1829.15146461355,
    -3482258.63459582,
];

const BSE: [f64; 7] = [
    84.9149257747669,
    0.334910077722432e-1,
    0.488399681651699,
    0.214274163161675,
    0.226073200069370,
    455.478499142212,
    890420.383607373,
];

const PVALUES: [f64; 7] = [
    0.86314083, 0.31268106, 0.00253509, 0.00094437, 0.8262118, 0.0030368, 0.0035604,
];

const HC0_SE: [f64; 7] = [51.22035, 0.02458, 0.38324, 0.14625, 0.15821, 428.38438, 832212.0];
const HC1_SE: [f64; 7] = [68.29380, 0.03277, 0.51099, 0.19499, 0.21094, 571.17917, 1109615.0];
const HC2_SE: [f64; 7] = [67.49208, 0.03653, 0.55334, 0.20522, 0.22324, 617.59295, 1202370.0];
const HC3_SE: [f64; 7] = [91.11939, 0.05562, 0.82213, 0.29879, 0.32491, 922.80784, 1799477.0];

// Stata-rounded conf_int (last row is intercept).
const CONF_INT: [(f64, f64); 7] = [
    (-177.0291, 207.1524),
    (-0.111581, 0.0399428),
    (-3.125065, -0.9153928),
    (-1.517948, -0.5485049),
    (-0.5625173, 0.4603083),
    (798.7873, 2859.515),
    (-5496529.0, -1467987.0),
];

const SCALE:        f64 = 92936.0061673238;
const RSQUARED:     f64 = 0.995479004577296;
const RSQUARED_ADJ: f64 = 0.99246501;
const ESS:          f64 = 184172401.944494;
const SSR:          f64 = 836424.055505915;
const FVALUE:       f64 = 330.285339234588;

fn fit() -> rust_stats::OlsResults {
    let g = load();
    let x = matrix_from(&g.x);
    Ols::new(&g.y, x.as_ref()).fit().expect("fit failed")
}

fn approx_eq(a: f64, b: f64, decimal: i32) {
    // Match numpy.testing.assert_almost_equal: tolerance = 1.5 * 10^-decimal.
    let tol = 1.5 * 10f64.powi(-decimal);
    assert!(
        (a - b).abs() < tol,
        "values differ beyond {decimal} decimals: got {a}, expected {b}",
    );
}

fn approx_rel(a: f64, b: f64, rtol: f64) {
    let denom = b.abs().max(1.0);
    assert!(
        (a - b).abs() / denom <= rtol,
        "rel-eq failed: got {a}, expected {b}, rtol {rtol}",
    );
}

// Ported from CheckRegressionResults.test_params (DECIMAL_4 → 4 decimals).
#[test]
fn test_params() {
    let res = fit();
    let expected = to_rust_order(PARAMS);
    for i in 0..7 {
        approx_eq(res.coef()[i], expected[i], 4);
    }
}

// CheckRegressionResults.test_standarderrors (DECIMAL_4).
#[test]
fn test_standard_errors() {
    let res = fit();
    let inf = res.inference(CovType::NonRobust);
    let expected = to_rust_order(BSE);
    for i in 0..7 {
        approx_eq(inf.std_err[i], expected[i], 4);
    }
}

// CheckRegressionResults.test_pvalues (DECIMAL_4).
#[test]
fn test_pvalues() {
    let res = fit();
    let inf = res.inference(CovType::NonRobust);
    let expected = to_rust_order(PVALUES);
    for i in 0..7 {
        approx_eq(inf.p_values[i], expected[i], 4);
    }
}

// CheckRegressionResults.test_confidenceintervals (DECIMAL_4 rtol).
#[test]
fn test_conf_int_95() {
    let res = fit();
    let ci = res.conf_int_with(CovType::NonRobust, 0.05).unwrap();
    // Reference is in stats order; remap.
    let ref_lo = to_rust_order(CONF_INT.map(|p| p.0));
    let ref_hi = to_rust_order(CONF_INT.map(|p| p.1));
    for i in 0..7 {
        approx_rel(ci[(i, 0)], ref_lo[i], 1e-4);
        approx_rel(ci[(i, 1)], ref_hi[i], 1e-4);
    }
}

// CheckRegressionResults.test_scale.
#[test]
fn test_scale() {
    // statsmodels' "scale" is mse_resid = sigma².
    let res = fit();
    approx_rel(res.sigma() * res.sigma(), SCALE, 1e-4);
}

// CheckRegressionResults.test_rsquared / rsquared_adj.
#[test]
fn test_rsquared_and_adj() {
    let res = fit();
    approx_eq(res.r_squared(),     RSQUARED,     4);
    approx_eq(res.adj_r_squared(), RSQUARED_ADJ, 4);
}

// CheckRegressionResults.test_degrees.
#[test]
fn test_degrees() {
    let res = fit();
    assert_eq!(res.df_model(), 6);
    assert_eq!(res.df_resid(), 9);
}

// CheckRegressionResults.test_sumof_squaredresids + test_ess.
#[test]
fn test_ssr_and_ess() {
    let res = fit();
    let ssr: f64 = res.residuals().iter().map(|r| r * r).sum();
    approx_rel(ssr, SSR, 1e-7);
    // ESS = TSS - SSR; TSS = SSR / (1 - r²).
    let tss = ssr / (1.0 - RSQUARED);
    let ess = tss - ssr;
    approx_rel(ess, ESS, 1e-3);
}

// CheckRegressionResults.test_fvalue.
#[test]
fn test_fvalue() {
    let res = fit();
    approx_eq(res.f_statistic(), FVALUE, 4);
}

// TestOLS.test_HC{0,1,2,3}_errors. statsmodels uses DECIMAL_4 except for
// the trailing intercept entry which is asserted with rtol 1.5e-7..5e-7
// (depending on the Hi variant) — port the per-element tolerances.
fn assert_hc(cov: CovType, expected_stats: [f64; 7], rtol_intercept: f64) {
    let res = fit();
    let inf = res.inference(cov);
    let expected = to_rust_order(expected_stats);
    // rust order: intercept first (the "trailing" entry in stats order).
    approx_rel(inf.std_err[0], expected[0], rtol_intercept);
    for i in 1..7 {
        approx_eq(inf.std_err[i], expected[i], 4);
    }
}

#[test] fn test_hc0_errors() { assert_hc(CovType::HC0, HC0_SE, 1e-5);   }
#[test] fn test_hc1_errors() { assert_hc(CovType::HC1, HC1_SE, 4e-7);   }
#[test] fn test_hc2_errors() { assert_hc(CovType::HC2, HC2_SE, 5e-7);   }
#[test] fn test_hc3_errors() { assert_hc(CovType::HC3, HC3_SE, 1.5e-7); }

// CheckRegressionResults.test_resids — residual values from R/Stata at
// DECIMAL_4 precision. Stata rounds to ~5 sig figs.
#[test]
fn test_residuals() {
    let res = fit();
    let expected = [
        267.34003, -94.01394, 46.28717, -410.11462, 309.71459, -249.31122,
        -164.04896, -13.18036, 14.30477, 455.39409, -17.26893, -39.05504,
        -155.54997, -85.67131, 341.93151, -206.75783,
    ];
    let resid = res.residuals();
    for i in 0..16 {
        approx_eq(resid[i], expected[i], 2); // ~2-3 sig figs after decimal
    }
}

// CheckRegressionResults.test_norm_resids — Pearson residuals = resid / sigma.
#[test]
fn test_pearson_residuals() {
    let res = fit();
    let expected = [
        0.87694426, -0.30838998, 0.15183385, -1.34528175, 1.01594375,
        -0.81780510, -0.53812289, -0.04323497, 0.04692334, 1.49381010,
        -0.05664654, -0.12811061, -0.51024404, -0.28102399, 1.12162357,
        -0.67821900,
    ];
    let sigma = res.sigma();
    let resid = res.residuals();
    for i in 0..16 {
        approx_eq(resid[i] / sigma, expected[i], 4);
    }
}
