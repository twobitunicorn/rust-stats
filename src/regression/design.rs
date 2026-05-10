//! Design-matrix construction.

use faer::{Mat, MatRef};

/// Build the augmented design matrix `X̃` from `x`. If `intercept`, prepends
/// a column of ones; otherwise returns an owned copy of `x`.
pub fn build_design_matrix(x: MatRef<'_, f64>, intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let p_in = x.ncols();
    let p_out = p_in + usize::from(intercept);
    Mat::from_fn(n, p_out, |i, j| {
        if intercept {
            if j == 0 {
                1.0
            } else {
                x[(i, j - 1)]
            }
        } else {
            x[(i, j)]
        }
    })
}
