use approx::assert_relative_eq;
use rust_stats::{CovType, Matrix, Ols};
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

fn matrix_from(rows: &[Vec<f64>]) -> Matrix<f64> {
    let n = rows.len();
    let p = if n == 0 { 0 } else { rows[0].len() };
    Matrix::from_fn(n, p, |i, j| rows[i][j])
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
    let x = matrix_from(&g.x);
    let model = Ols::new(&g.y, x.as_ref());
    let res = (if g.intercept { model } else { model.without_intercept() })
        .fit().expect("fit failed");

    // ── Coefficients ──────────────────────────────────────────────────────────
    let beta = res.coef();
    for i in 0..g.coef.len() {
        assert_relative_eq!(beta[i], g.coef[i], epsilon = 1e-10, max_relative = 1e-10);
    }

    // ── Residuals and fitted values ───────────────────────────────────────────
    let resid = res.residuals();
    for i in 0..g.residuals.len() {
        assert_relative_eq!(resid[i], g.residuals[i], epsilon = 1e-7, max_relative = 1e-8);
    }
    let fit = res.fitted_values();
    for i in 0..g.fitted.len() {
        assert_relative_eq!(fit[i], g.fitted[i], epsilon = 1e-10, max_relative = 1e-10);
    }

    // ── Scalar summary statistics ─────────────────────────────────────────────
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    assert_relative_eq!(rss, g.rss, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(res.sigma(), g.sigma, max_relative = 1e-8);
    assert_relative_eq!(res.r_squared(), g.r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.adj_r_squared(), g.adj_r_squared, max_relative = 1e-8);
    assert_relative_eq!(res.f_statistic(), g.fvalue, max_relative = 1e-8);
    assert_relative_eq!(res.f_pvalue(), g.f_pvalue, max_relative = 1e-6);

    // ── Per-CovType inference ─────────────────────────────────────────────────
    for (ct_name, ref_) in &g.per_cov_type {
        let ct = cov_type(ct_name);
        let inf = res.inference(ct);
        let (tol_se, tol_p, tol_ci) = match ct {
            CovType::NonRobust => (1e-8, 1e-6, 1e-7),
            _                  => (1e-7, 1e-6, 1e-6),
        };
        for i in 0..ref_.std_err.len() {
            assert_relative_eq!(inf.std_err[i], ref_.std_err[i],
                max_relative = tol_se);
            assert_relative_eq!(inf.t_values[i], ref_.t_values[i],
                max_relative = tol_se);
            assert_relative_eq!(inf.p_values[i], ref_.p_values[i],
                max_relative = tol_p);
        }
        let ci = res.conf_int_with(ct, 0.05).unwrap();
        for i in 0..ref_.conf_int_95.len() {
            assert_relative_eq!(ci[(i, 0)], ref_.conf_int_95[i][0],
                max_relative = tol_ci);
            assert_relative_eq!(ci[(i, 1)], ref_.conf_int_95[i][1],
                max_relative = tol_ci);
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────
    let x_new = matrix_from(&g.x_predict);
    let yhat = res.predict(x_new.as_ref()).unwrap();
    for i in 0..g.predict_point.len() {
        assert_relative_eq!(yhat[i], g.predict_point[i], max_relative = 1e-9);
    }
    let band = res.predict_interval(x_new.as_ref(), 0.05).unwrap();
    for i in 0..g.predict_interval_95.len() {
        assert_relative_eq!(band[(i, 0)], g.predict_interval_95[i][0],
            max_relative = 1e-9);
        assert_relative_eq!(band[(i, 1)], g.predict_interval_95[i][1],
            max_relative = 1e-7);
        assert_relative_eq!(band[(i, 2)], g.predict_interval_95[i][2],
            max_relative = 1e-7);
    }
}

#[test] fn longley()          { assert_dataset("longley"); }
#[test] fn mtcars()          { assert_dataset("mtcars"); }
#[test] fn synthetic()       { assert_dataset("synthetic"); }
#[test] fn heteroskedastic() { assert_dataset("heteroskedastic"); }
