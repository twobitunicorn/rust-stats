use faer::{Col, Mat};
use rust_stats::Ols;

#[test]
fn builder_constructs_with_intercept_by_default() {
    let y: Col<f64> = Col::from_fn(3, |i| i as f64);
    let x: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);

    let ols = Ols::new(y.as_ref(), x.as_ref());
    assert!(ols.has_intercept());
}

#[test]
fn without_intercept_disables_intercept() {
    let y: Col<f64> = Col::from_fn(3, |_| 1.0);
    let x: Mat<f64> = Mat::from_fn(3, 2, |_, _| 0.0);

    let ols = Ols::new(y.as_ref(), x.as_ref()).without_intercept();
    assert!(!ols.has_intercept());
}
