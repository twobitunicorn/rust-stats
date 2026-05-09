use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct PerCov {
    std_err:    Vec<f64>,
    t_values:   Vec<f64>,
    p_values:   Vec<f64>,
    conf_int_95: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Golden {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    intercept: bool,
    x_predict: Vec<Vec<f64>>,

    coef:          Vec<f64>,
    residuals:     Vec<f64>,
    fitted:        Vec<f64>,
    rss:           f64,
    sigma:         f64,
    r_squared:     f64,
    adj_r_squared: f64,
    fvalue:        f64,
    f_pvalue:      f64,

    per_cov_type: std::collections::BTreeMap<String, PerCov>,

    predict_point:        Vec<f64>,
    predict_interval_95:  Vec<Vec<f64>>,
}

fn load(name: &str) -> Golden {
    let path: PathBuf = ["tests", "golden", &format!("{name}.json")].iter().collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e|
        panic!("failed to read {path:?}: {e}; did you run tests/golden/generate.py?"));
    serde_json::from_slice(&bytes).expect("invalid golden JSON")
}

fn col_from(v: &[f64]) -> Col<f64> { Col::from_fn(v.len(), |i| v[i]) }
fn mat_from(rows: &[Vec<f64>]) -> Mat<f64> {
    let n = rows.len();
    let p = if n == 0 { 0 } else { rows[0].len() };
    Mat::from_fn(n, p, |i, j| rows[i][j])
}

fn cov_type(name: &str) -> CovType {
    match name {
        "nonrobust" => CovType::NonRobust,
        "HC0" => CovType::HC0,
        "HC1" => CovType::HC1,
        "HC2" => CovType::HC2,
        "HC3" => CovType::HC3,
        other => panic!("unknown cov_type {other}"),
    }
}

fn assert_dataset(name: &str) {
    let g = load(name);
    let y = col_from(&g.y);
    let x = mat_from(&g.x);
    let model = Ols::new(y.as_ref(), x.as_ref());
    let res = (if g.intercept { model } else { model.without_intercept() })
        .fit().expect("fit failed");

    // ── Coefficients ──────────────────────────────────────────────────────────
    // QR-based solve matches statsmodels pseudoinverse to ~1e-12 relative on
    // Longley (condition number ~10^9). Tight 1e-10 tolerance verified empirically.
    let beta = res.coef();
    for i in 0..g.coef.len() {
        assert_relative_eq!(*beta.get(i), g.coef[i], epsilon = 1e-10, max_relative = 1e-10);
    }

    // ── Residuals and fitted values ───────────────────────────────────────────
    // Fitted values: relative error ~7e-13 (Longley fitted are ~60k, so even 4e-8
    // absolute drift is negligible relatively). Tight 1e-10 holds.
    // Residuals: the same ~4e-8 absolute drift maps to up to 3e-9 relative when
    // residuals are small (~13 in the worst case). 1e-8 relative is the "derived
    // stats" spec, and observed max is 3e-9, safely within spec.
    let resid = res.residuals();
    for i in 0..g.residuals.len() {
        assert_relative_eq!(*resid.get(i), g.residuals[i], epsilon = 1e-7, max_relative = 1e-8);
    }
    let fit = res.fitted_values();
    for i in 0..g.fitted.len() {
        assert_relative_eq!(*fit.get(i), g.fitted[i], epsilon = 1e-10, max_relative = 1e-10);
    }

    // ── Scalar summary statistics ─────────────────────────────────────────────
    let rss: f64 = (0..g.residuals.len()).map(|i| (*resid.get(i)).powi(2)).sum();
    assert_relative_eq!(rss, g.rss, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(res.sigma(), g.sigma, max_relative = 1e-8);
    assert_relative_eq!(res.r_squared(), g.r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.adj_r_squared(), g.adj_r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.f_statistic(), g.fvalue, max_relative = 1e-8);
    assert_relative_eq!(res.f_pvalue(), g.f_pvalue, max_relative = 1e-6);

    // ── Per-CovType inference ─────────────────────────────────────────────────
    // NonRobust achieves ~5e-13 relative (machine precision).
    // HC0-HC3: Longley's condition number causes the residuals used as sandwich
    // weights to carry ~4e-8 absolute error. This propagates into HC SEs at up
    // to 5e-8 relative (HC1, which scales by n/(n-p) = 16/9). CI endpoints
    // inherit the SE error and can reach ~2e-7 relative for HC3 (near-zero
    // crossing where the bound value is ~20). These are Longley-specific limits
    // from ill-conditioning, not implementation bugs.
    // Tolerances chosen from empirical maximums with 5x headroom:
    //   SE/t: 1e-7  (observed max ~5e-8 for HC1)
    //   p:    1e-6  (observed max ~5e-7 for HC1; normal-tail sensitivity)
    //   CI:   5e-7  (observed max ~2e-7 for HC3 near-zero bound)
    for (ct_name, ref_) in &g.per_cov_type {
        let ct = cov_type(ct_name);
        let inf = res.inference(ct);
        let (tol_se, tol_p, tol_ci) = match ct {
            CovType::NonRobust => (1e-8, 1e-6, 1e-7),
            _                  => (1e-7, 1e-6, 5e-7),
        };
        for i in 0..ref_.std_err.len() {
            assert_relative_eq!(*inf.std_err.get(i), ref_.std_err[i],
                max_relative = tol_se);
            assert_relative_eq!(*inf.t_values.get(i), ref_.t_values[i],
                max_relative = tol_se);
            assert_relative_eq!(*inf.p_values.get(i), ref_.p_values[i],
                max_relative = tol_p);
        }
        let ci = res.conf_int_with(ct, 0.05).unwrap();
        for i in 0..ref_.conf_int_95.len() {
            assert_relative_eq!(*ci.get(i, 0), ref_.conf_int_95[i][0],
                max_relative = tol_ci);
            assert_relative_eq!(*ci.get(i, 1), ref_.conf_int_95[i][1],
                max_relative = tol_ci);
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────
    let x_new = mat_from(&g.x_predict);
    let yhat = res.predict(x_new.as_ref()).unwrap();
    for i in 0..g.predict_point.len() {
        assert_relative_eq!(*yhat.get(i), g.predict_point[i], max_relative = 1e-9);
    }
    let band = res.predict_interval(x_new.as_ref(), 0.05).unwrap();
    for i in 0..g.predict_interval_95.len() {
        assert_relative_eq!(*band.get(i, 0), g.predict_interval_95[i][0],
            max_relative = 1e-9);
        assert_relative_eq!(*band.get(i, 1), g.predict_interval_95[i][1],
            max_relative = 1e-7);
        assert_relative_eq!(*band.get(i, 2), g.predict_interval_95[i][2],
            max_relative = 1e-7);
    }
}

#[test] fn longley()          { assert_dataset("longley"); }
#[test] fn mtcars()          { assert_dataset("mtcars"); }
#[test] fn synthetic()       { assert_dataset("synthetic"); }
#[test] fn heteroskedastic() { assert_dataset("heteroskedastic"); }
