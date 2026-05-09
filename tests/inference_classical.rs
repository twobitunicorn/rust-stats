use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::{CovType, Ols};

/// Use a noisy 5x1 problem with intercept for non-degenerate inference.
fn noisy_small() -> rust_stats::OlsResults {
    let y: Col<f64> = Col::from_fn(5, |i| (i as f64 + 1.0) + 0.1 * (i as f64 - 2.0));
    let x: Mat<f64> = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn std_err_positive_finite() {
    let res = noisy_small();
    let se = res.std_err();
    assert!(se.get(0).is_finite() && *se.get(0) > 0.0);
    assert!(se.get(1).is_finite() && *se.get(1) > 0.0);
}

#[test]
fn t_value_equals_coef_over_std_err() {
    let res = noisy_small();
    let beta = res.coef();
    let se = res.std_err();
    let t = res.t_values();
    assert_relative_eq!(*t.get(0), *beta.get(0) / *se.get(0), epsilon = 1e-12);
    assert_relative_eq!(*t.get(1), *beta.get(1) / *se.get(1), epsilon = 1e-12);
}

#[test]
fn p_value_in_zero_one() {
    let res = noisy_small();
    let p = res.p_values();
    for i in 0..res.coef().nrows() {
        assert!(*p.get(i) >= 0.0 && *p.get(i) <= 1.0);
    }
}

#[test]
fn conf_int_brackets_coefficient() {
    let res = noisy_small();
    let beta = res.coef();
    let ci = res.conf_int(0.05);
    for i in 0..beta.nrows() {
        assert!(*ci.get(i, 0) <= *beta.get(i));
        assert!(*ci.get(i, 1) >= *beta.get(i));
    }
}

#[test]
fn cov_nonrobust_diagonal_matches_std_err_squared() {
    let res = noisy_small();
    let cov = res.cov(CovType::NonRobust);
    let se = res.std_err();
    for i in 0..res.coef().nrows() {
        assert_relative_eq!(*cov.get(i, i), *se.get(i) * *se.get(i), epsilon = 1e-12);
    }
}

#[test]
fn invalid_alpha_panics_in_conf_int_wrapper() {
    let res = noisy_small();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        res.conf_int(0.0);
    }));
    assert!(result.is_err());
}
