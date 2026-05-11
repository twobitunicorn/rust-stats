use rust_stats_ols::{CovType, Matrix, Ols};

fn fit() -> rust_stats_ols::OlsResults {
    let n = 16;
    let x = Matrix::from_fn(n, 2, |i, j| {
        if j == 0 { (i as f64) * 0.13 } else { ((i as f64) * 0.5).sin() }
    });
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.5 * x[(i, 0)] + 0.1 * ((i as f64).sin()))
        .collect();
    Ols::new(&y, x.as_ref()).fit().unwrap()
        .with_names(vec!["const".to_string(), "x1".to_string(), "x2".to_string()])
}

#[test]
fn summary_contains_required_headers() {
    let s = fit().summary();
    for needle in [
        "OLS Regression Results",
        "Dep. Variable",
        "R-squared",
        "Adj. R-squared",
        "F-statistic",
        "No. Observations",
        "Df Residuals",
        "Df Model",
        "Covariance Type:",
        "coef",
        "std err",
        "P>|t|",
    ] {
        assert!(s.contains(needle), "summary missing {needle:?}\n---\n{s}");
    }
}

#[test]
fn summary_lists_each_coefficient_name() {
    let s = fit().summary();
    for name in ["const", "x1", "x2"] {
        assert!(s.contains(name), "summary missing coef name {name}");
    }
}

#[test]
fn summary_with_changes_covariance_label() {
    let s = fit().summary_with(CovType::HC1);
    assert!(s.contains("HC1"), "expected covariance label HC1\n---\n{s}");
}

#[test]
fn display_is_summary() {
    let res = fit();
    let s_disp = format!("{res}");
    let s_summ = res.summary();
    assert_eq!(s_disp, s_summ);
}
