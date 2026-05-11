use approx::assert_relative_eq;
use rust_stats_ols::{Matrix, Ols, OlsError};

fn fit_simple() -> (rust_stats_ols::OlsResults, Matrix<f64>) {
    let n = 20;
    let x = Matrix::from_fn(n, 1, |i, _| i as f64 * 0.1);
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * (i as f64) * 0.1).collect();
    let res = Ols::new(&y, x.as_ref()).fit().unwrap();
    let x_new = Matrix::from_fn(3, 1, |i, _| i as f64);
    (res, x_new)
}

#[test]
fn predict_matches_known_function() {
    let (res, x_new) = fit_simple();
    let yhat = res.predict(x_new.as_ref()).unwrap();
    for i in 0..3 {
        assert_relative_eq!(yhat[i], 1.0 + 2.0 * (i as f64), epsilon = 1e-10);
    }
}

#[test]
fn predict_rejects_wrong_column_count() {
    let (res, _) = fit_simple();
    let bad = Matrix::from_fn(2, 5, |_, _| 0.0);
    let err = res.predict(bad.as_ref()).unwrap_err();
    assert_eq!(err, OlsError::NewXShapeMismatch { got: 5, expected: 1 });
}

#[test]
fn predict_interval_brackets_point_estimate() {
    let (res, x_new) = fit_simple();
    let band = res.predict_interval(x_new.as_ref(), 0.05).unwrap();
    for i in 0..3 {
        let fit = band[(i, 0)];
        let lo = band[(i, 1)];
        let hi = band[(i, 2)];
        assert!(lo < fit, "lower bound must be below fit");
        assert!(hi > fit, "upper bound must be above fit");
    }
}

#[test]
fn predict_interval_rejects_invalid_alpha() {
    let (res, x_new) = fit_simple();
    let err = res.predict_interval(x_new.as_ref(), 0.0).unwrap_err();
    assert_eq!(err, OlsError::InvalidAlpha(0.0));
}
