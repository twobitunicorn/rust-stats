use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols, OlsError};

fn fit() -> rust_stats::OlsResults {
    let n = 25;
    // Two linearly independent columns: linear and quadratic terms.
    // With an auto-prepended intercept this gives a rank-3 design.
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        let t = (i as f64) * 0.1;
        if j == 0 { t } else { t * t + 0.5 }
    });
    let y: Col<f64> = Col::from_fn(n, |i| 1.0 + 2.0 * (i as f64) * 0.1
        + 0.05 * ((i as f64).cos()));
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn inference_nonrobust_matches_direct_accessors() {
    let res = fit();
    let inf = res.inference(CovType::NonRobust);
    let se = res.std_err();
    let t = res.t_values();
    let p = res.p_values();
    for i in 0..res.coef().nrows() {
        assert_relative_eq!(*inf.std_err.get(i),  *se.get(i), epsilon = 1e-12);
        assert_relative_eq!(*inf.t_values.get(i), *t.get(i),  epsilon = 1e-12);
        assert_relative_eq!(*inf.p_values.get(i), *p.get(i),  epsilon = 1e-12);
    }
}

#[test]
fn inference_hc1_differs_from_nonrobust() {
    let res = fit();
    let nr  = res.inference(CovType::NonRobust);
    let hc1 = res.inference(CovType::HC1);
    let mut any_diff = false;
    for i in 0..res.coef().nrows() {
        if (*nr.std_err.get(i) - *hc1.std_err.get(i)).abs() > 1e-8 {
            any_diff = true;
        }
    }
    assert!(any_diff, "HC1 SEs should differ from classical on this dataset");
}

#[test]
fn conf_int_with_invalid_alpha_returns_error() {
    let res = fit();
    let err = res.conf_int_with(CovType::NonRobust, 1.5).unwrap_err();
    assert_eq!(err, OlsError::InvalidAlpha(1.5));
}
