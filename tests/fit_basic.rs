use approx::assert_relative_eq;
use rust_stats::{Matrix, Ols};

/// Synthetic: y = 2 + 3*x1 - 1*x2 exactly, no noise, with intercept.
#[test]
fn recovers_known_coefficients_exactly() {
    let n = 50;
    let x = Matrix::from_fn(n, 2, |i, j| {
        if j == 0 { i as f64 * 0.1 } else { (i as f64 * 0.05).sin() }
    });
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 + 3.0 * x[(i, 0)] - 1.0 * x[(i, 1)])
        .collect();
    let res = Ols::new(&y, x.as_ref()).fit().unwrap();
    let beta = res.coef();
    assert_relative_eq!(beta[0],  2.0, epsilon = 1e-10);
    assert_relative_eq!(beta[1],  3.0, epsilon = 1e-10);
    assert_relative_eq!(beta[2], -1.0, epsilon = 1e-10);
    assert_eq!(res.n_obs(), n);
    assert_eq!(res.df_resid(), n - 3);
    assert_eq!(res.df_model(), 2);
}

#[test]
fn without_intercept_recovers_known_coefficients() {
    let n = 30;
    let x = Matrix::from_fn(n, 2, |i, j| {
        if j == 0 { (i + 1) as f64 } else { (i as f64).cos() }
    });
    let y: Vec<f64> = (0..n)
        .map(|i| 0.5 * x[(i, 0)] + 1.5 * x[(i, 1)])
        .collect();
    let res = Ols::new(&y, x.as_ref())
        .without_intercept()
        .fit()
        .unwrap();
    let beta = res.coef();
    assert_relative_eq!(beta[0], 0.5, epsilon = 1e-10);
    assert_relative_eq!(beta[1], 1.5, epsilon = 1e-10);
}
