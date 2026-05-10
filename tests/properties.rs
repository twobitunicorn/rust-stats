use approx::assert_abs_diff_eq;
use rust_stats::{Matrix, Ols};

/// Residuals must be orthogonal to every column of X̃ (including the intercept
/// column when present).
#[test]
fn residuals_orthogonal_to_design_with_intercept() {
    let n = 40;
    let x = Matrix::from_fn(n, 3, |i, j| {
        let t = i as f64;
        match j {
            0 => t.sin(),
            1 => (2.0_f64.sqrt() * t).cos(),
            _ => (3.0_f64.sqrt() * t).sin() * t,
        }
    });
    let y: Vec<f64> = (0..n).map(|i| (i as f64).cos() + 0.5 * (i as f64)).collect();
    let res = Ols::new(&y, x.as_ref()).fit().unwrap();
    let e = res.residuals();
    let sum_e: f64 = e.iter().sum();
    assert_abs_diff_eq!(sum_e, 0.0, epsilon = 1e-10);
    for j in 0..x.ncols() {
        let dot: f64 = (0..n).map(|i| e[i] * x[(i, j)]).sum();
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn residuals_orthogonal_to_design_without_intercept() {
    let n = 30;
    let x = Matrix::from_fn(n, 2, |i, j| (i as f64) + (j as f64) * 0.7);
    let y: Vec<f64> = (0..n)
        .map(|i| 0.3 * (i as f64) + 0.05 * ((i as f64).sin()))
        .collect();
    let res = Ols::new(&y, x.as_ref()).without_intercept().fit().unwrap();
    let e = res.residuals();
    for j in 0..x.ncols() {
        let dot: f64 = (0..n).map(|i| e[i] * x[(i, j)]).sum();
        assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn permuting_columns_of_x_preserves_predictions_and_r_squared() {
    let n = 40;
    let x = Matrix::from_fn(n, 3, |i, j| {
        let t = i as f64;
        match j {
            0 => t.sin(),
            1 => (2.0_f64.sqrt() * t).cos(),
            _ => (3.0_f64.sqrt() * t).sin() * t,
        }
    });
    let y: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    let r1 = Ols::new(&y, x.as_ref()).fit().unwrap();

    // Swap columns 0 and 2.
    let x2 = Matrix::from_fn(n, 3, |i, j| match j {
        0 => x[(i, 2)],
        2 => x[(i, 0)],
        _ => x[(i, j)],
    });
    let r2 = Ols::new(&y, x2.as_ref()).fit().unwrap();

    assert_abs_diff_eq!(r1.r_squared(), r2.r_squared(), epsilon = 1e-12);
    let f1 = r1.fitted_values();
    let f2 = r2.fitted_values();
    for i in 0..n {
        assert_abs_diff_eq!(f1[i], f2[i], epsilon = 1e-10);
    }
}
