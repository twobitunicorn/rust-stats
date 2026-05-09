use approx::assert_relative_eq;
use faer::{Col, Mat};
use rust_stats::Ols;

/// Synthetic: y = 2 + 3*x1 - 1*x2 exactly, no noise, with intercept.
#[test]
fn recovers_known_coefficients_exactly() {
    let n = 50;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        if j == 0 { i as f64 * 0.1 } else { (i as f64 * 0.05).sin() }
    });
    let y: Col<f64> = Col::from_fn(n, |i| {
        2.0 + 3.0 * (*x.get(i, 0)) - 1.0 * (*x.get(i, 1))
    });
    let res = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap();
    let beta = res.coef();
    assert_relative_eq!(*beta.get(0),  2.0, epsilon = 1e-10);
    assert_relative_eq!(*beta.get(1),  3.0, epsilon = 1e-10);
    assert_relative_eq!(*beta.get(2), -1.0, epsilon = 1e-10);
    assert_eq!(res.n_obs(), n);
    assert_eq!(res.df_resid(), n - 3);
    assert_eq!(res.df_model(), 2);
}

#[test]
fn without_intercept_recovers_known_coefficients() {
    let n = 30;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        if j == 0 { (i + 1) as f64 } else { (i as f64).cos() }
    });
    let y: Col<f64> = Col::from_fn(n, |i| 0.5 * (*x.get(i, 0)) + 1.5 * (*x.get(i, 1)));
    let res = Ols::new(y.as_ref(), x.as_ref())
        .without_intercept()
        .fit()
        .unwrap();
    let beta = res.coef();
    assert_relative_eq!(*beta.get(0), 0.5, epsilon = 1e-10);
    assert_relative_eq!(*beta.get(1), 1.5, epsilon = 1e-10);
}
