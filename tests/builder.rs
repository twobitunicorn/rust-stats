use rust_stats::{Matrix, Ols};

#[test]
fn builder_constructs_with_intercept_by_default() {
    let y: Vec<f64> = (0..3).map(|i| i as f64).collect();
    let x = Matrix::from_fn(3, 2, |i, j| (i + j) as f64);

    let ols = Ols::new(&y, x.as_ref());
    assert!(ols.has_intercept());
}

#[test]
fn without_intercept_disables_intercept() {
    let y: Vec<f64> = vec![1.0; 3];
    let x = Matrix::from_fn(3, 2, |_, _| 0.0);

    let ols = Ols::new(&y, x.as_ref()).without_intercept();
    assert!(!ols.has_intercept());
}
