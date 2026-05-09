use faer::{Col, Mat};
use rust_stats::Ols;

fn fit() -> rust_stats::OlsResults {
    // Use a full-rank design so fit() succeeds.
    // p = 3 (intercept + 2 predictors).
    let n = 10;
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, j| {
        if j == 0 { i as f64 * 0.5 } else { (i as f64 * 0.3).sin() }
    });
    let y: Col<f64> = Col::from_fn(n, |i| 1.0 + 2.0 * (*x.get(i, 0)) - (*x.get(i, 1)));
    Ols::new(y.as_ref(), x.as_ref()).fit().unwrap()
}

#[test]
fn names_default_is_none() {
    let res = fit();
    assert!(res.names().is_none());
}

#[test]
fn with_names_stores_them() {
    let res = fit().with_names(vec![
        "const".to_string(), "age".to_string(), "income".to_string(),
    ]);
    let names = res.names().unwrap();
    assert_eq!(names, ["const", "age", "income"]);
}

#[test]
#[should_panic(expected = "names length")]
fn with_names_wrong_length_panics() {
    let _ = fit().with_names(vec!["only_one".to_string()]);
}
