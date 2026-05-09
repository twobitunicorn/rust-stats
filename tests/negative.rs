use faer::{Col, Mat};
use rust_stats::{Ols, OlsError};

#[test]
fn fit_rejects_rank_deficient_x() {
    let n = 10;
    // Two identical columns ⇒ rank-deficient even with intercept.
    let x: Mat<f64> = Mat::from_fn(n, 2, |i, _| i as f64);
    let y: Col<f64> = Col::from_fn(n, |i| i as f64);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    match err {
        OlsError::RankDeficient { rank, p } => {
            assert!(rank < p);
            assert_eq!(p, 3);
        }
        other => panic!("expected RankDeficient, got {:?}", other),
    }
}

#[test]
fn error_variants_display_correctly() {
    let cases: Vec<(OlsError, &str)> = vec![
        (
            OlsError::DimensionMismatch { y: 10, x: 8 },
            "dimension mismatch: y has 10 rows but X has 8",
        ),
        (
            OlsError::InsufficientObservations { n: 3, p: 5 },
            "not enough observations: n=3 must exceed p=5",
        ),
        (
            OlsError::RankDeficient { rank: 2, p: 3 },
            "rank deficient design matrix: rank 2 < p 3",
        ),
        (OlsError::NonFinite, "input contains non-finite values"),
        (
            OlsError::NewXShapeMismatch { got: 4, expected: 3 },
            "predict X has 4 columns, expected 3",
        ),
        (
            OlsError::InvalidAlpha(1.5),
            "invalid alpha 1.5: must be in (0, 1)",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(format!("{}", err), expected);
    }
}

#[test]
fn fit_rejects_mismatched_y_x_rows() {
    let y: Col<f64> = Col::from_fn(5, |_| 0.0);
    let x: Mat<f64> = Mat::from_fn(4, 2, |_, _| 1.0);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::DimensionMismatch { y: 5, x: 4 });
}

#[test]
fn fit_rejects_insufficient_observations() {
    // n=2, intercept=true, so p=3 ⇒ n <= p
    let y: Col<f64> = Col::from_fn(2, |_| 1.0);
    let x: Mat<f64> = Mat::from_fn(2, 2, |_, _| 1.0);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::InsufficientObservations { n: 2, p: 3 });
}

#[test]
fn fit_rejects_non_finite_in_y() {
    let y_data = vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0];
    let y: Col<f64> = Col::from_fn(5, |i| y_data[i]);
    let x: Mat<f64> = Mat::from_fn(5, 2, |i, j| (i + j) as f64);
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::NonFinite);
}

#[test]
fn fit_rejects_non_finite_in_x() {
    let y: Col<f64> = Col::from_fn(5, |i| i as f64);
    let x: Mat<f64> = Mat::from_fn(5, 2, |i, j| {
        if i == 2 && j == 1 { f64::INFINITY } else { 1.0 }
    });
    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    assert_eq!(err, OlsError::NonFinite);
}

#[test]
fn rank_deficient_golden_dataset_errors() {
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Rd { y: Vec<f64>, x: Vec<Vec<f64>> }

    let path: PathBuf = ["tests", "golden", "rank_deficient.json"].iter().collect();
    let bytes = std::fs::read(path).unwrap();
    let rd: Rd = serde_json::from_slice(&bytes).unwrap();

    let y = Col::from_fn(rd.y.len(), |i| rd.y[i]);
    let n = rd.x.len();
    let p = rd.x[0].len();
    let x = Mat::from_fn(n, p, |i, j| rd.x[i][j]);

    let err = Ols::new(y.as_ref(), x.as_ref()).fit().unwrap_err();
    match err {
        OlsError::RankDeficient { .. } => {}
        other => panic!("expected RankDeficient, got {:?}", other),
    }
}
