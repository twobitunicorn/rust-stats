//! Tiny OLS solver for the Hannan-Rissanen starting-values step.
//!
//! Builds the normal equations `XᵀX β = Xᵀy` and solves them by Cholesky
//! factorisation. Adequate for ARMA seeding (small `p+q`, well-conditioned
//! Toeplitz-flavoured systems); we don't need a general-purpose LAPACK
//! port here.

/// Solve `X β = y` (rows × cols, row-major) in the least-squares sense.
/// Returns `None` if the normal-equations matrix is singular.
pub fn solve(x: &[f64], y: &[f64], rows: usize, cols: usize) -> Option<Vec<f64>> {
    if cols == 0 {
        return Some(Vec::new());
    }
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(y.len(), rows);

    // Build XᵀX (cols × cols, symmetric) and Xᵀy (cols).
    let mut xtx = vec![0.0f64; cols * cols];
    let mut xty = vec![0.0f64; cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let yi = y[r];
        for i in 0..cols {
            xty[i] += row[i] * yi;
            for j in i..cols {
                xtx[i * cols + j] += row[i] * row[j];
            }
        }
    }
    // Mirror lower triangle.
    for i in 0..cols {
        for j in 0..i {
            xtx[i * cols + j] = xtx[j * cols + i];
        }
    }

    cholesky_solve(&mut xtx, &mut xty, cols)
}

/// In-place Cholesky factorisation of a symmetric positive-definite
/// `n×n` matrix `a` (row-major), followed by a forward/backward
/// substitution against the RHS `b`. Returns the solution, or `None` if
/// `a` is not strictly positive definite.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    // Factor: a = L Lᵀ. Store L in the lower triangle of `a`.
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                a[i * n + j] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    // Forward solve L · z = b.
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * z[k];
        }
        z[i] = sum / a[i * n + i];
    }
    // Backward solve Lᵀ · x = z.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * x[k];
        }
        x[i] = sum / a[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod ols_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn solves_a_known_linear_system() {
        // y = 2 + 3·x. Add intercept column.
        let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let rows = 10;
        let cols = 2;
        let mut x = vec![0.0; rows * cols];
        let mut y = vec![0.0; rows];
        for r in 0..rows {
            x[r * cols] = 1.0;
            x[r * cols + 1] = x_data[r];
            y[r] = 2.0 + 3.0 * x_data[r];
        }
        let beta = solve(&x, &y, rows, cols).unwrap();
        assert_relative_eq!(beta[0], 2.0, max_relative = 1e-10);
        assert_relative_eq!(beta[1], 3.0, max_relative = 1e-10);
    }

    #[test]
    fn returns_none_for_singular() {
        // Duplicate columns → XᵀX singular.
        let rows = 5;
        let cols = 2;
        let x: Vec<f64> = (0..rows * cols).map(|i| (i / cols) as f64).collect();
        let y: Vec<f64> = (0..rows).map(|i| i as f64).collect();
        assert!(solve(&x, &y, rows, cols).is_none());
    }
}
