use approx::assert_relative_eq;
use rust_stats::{CovType, Matrix, Ols, OlsError};

fn fit() -> rust_stats::OlsResults {
    let n = 25;
    let x = Matrix::from_fn(n, 2, |i, j| {
        let t = (i as f64) * 0.1;
        if j == 0 { t } else { t * t + 0.5 }
    });
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * (i as f64) * 0.1 + 0.05 * ((i as f64).cos()))
        .collect();
    Ols::new(&y, x.as_ref()).fit().unwrap()
}

#[test]
fn inference_nonrobust_matches_direct_accessors() {
    let res = fit();
    let inf = res.inference(CovType::NonRobust);
    let se = res.std_err();
    let t = res.t_values();
    let p = res.p_values();
    for i in 0..res.coef().len() {
        assert_relative_eq!(inf.std_err[i],  se[i], epsilon = 1e-12);
        assert_relative_eq!(inf.t_values[i], t[i],  epsilon = 1e-12);
        assert_relative_eq!(inf.p_values[i], p[i],  epsilon = 1e-12);
    }
}

#[test]
fn inference_hc1_differs_from_nonrobust() {
    let res = fit();
    let nr  = res.inference(CovType::NonRobust);
    let hc1 = res.inference(CovType::HC1);
    let mut any_diff = false;
    for i in 0..res.coef().len() {
        if (nr.std_err[i] - hc1.std_err[i]).abs() > 1e-8 {
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
