use approx::assert_relative_eq;
use faer::Mat;
use rust_stats::regression::design::build_design_matrix;

#[test]
fn with_intercept_prepends_column_of_ones() {
    let x: Mat<f64> = Mat::from_fn(4, 2, |i, j| (i * 10 + j) as f64);
    let xt = build_design_matrix(x.as_ref(), true);
    assert_eq!(xt.nrows(), 4);
    assert_eq!(xt.ncols(), 3);
    for i in 0..4 {
        assert_relative_eq!(read(xt.as_ref(), i, 0), 1.0);
        assert_relative_eq!(read(xt.as_ref(), i, 1), (i * 10) as f64);
        assert_relative_eq!(read(xt.as_ref(), i, 2), (i * 10 + 1) as f64);
    }
}

#[test]
fn without_intercept_copies_x_unchanged() {
    let x: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
    let xt = build_design_matrix(x.as_ref(), false);
    assert_eq!(xt.nrows(), 3);
    assert_eq!(xt.ncols(), 2);
    for i in 0..3 {
        for j in 0..2 {
            assert_relative_eq!(read(xt.as_ref(), i, j), (i + j) as f64);
        }
    }
}

fn read(m: faer::MatRef<'_, f64>, i: usize, j: usize) -> f64 {
    *m.get(i, j)
}
