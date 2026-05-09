//! Locally estimated scatterplot smoothing (LOESS).
//!
//! For each output index `i`, takes the `span * n` nearest indices in the
//! input, weights them with the tricube `w(d) = (1 - |d|^3)^3` of the
//! distance from `i` (normalised by the furthest window point), and fits a
//! weighted polynomial of `degree` to those points. The fitted value at
//! index `i` (the intercept of the centred fit) is the smoothed output.
//!
//! `degree = 1` is the classic LOWESS smoother; `degree = 0` reduces to a
//! tricube-weighted moving average; `degree = 2` is Cleveland's default.
//! No robustness iterations — this is single-pass (non-robust) LOESS.

use crate::error::LoessError;
use faer::{Col, ColRef};
use rayon::prelude::*;

/// Smooth the input series at every integer position `0..n`.
///
/// `span` is the fraction of points used in each local fit (0 < span <= 1);
/// `degree` is the polynomial degree (0, 1, or 2).
pub fn loess(
    y: ColRef<'_, f64>,
    span: f64,
    degree: u8,
) -> Result<Col<f64>, LoessError> {
    validate_loess_args(y, span, degree)?;
    let vec: Vec<f64> = y.iter().copied().collect();
    let slice = vec.as_slice();
    let n = slice.len();
    if n == 0 {
        return Ok(Col::zeros(0));
    }
    let degree_us = degree as usize;
    let window = ((span * n as f64).ceil() as usize)
        .max(degree_us + 2)
        .min(n);
    let smoothed = loess_compute(slice, window, degree_us);
    Ok(Col::<f64>::from_fn(smoothed.len(), |i| smoothed[i]))
}

/// Fitted LOESS value at a single (possibly fractional) query point `xq`.
/// `xq` may be outside `[0, n-1]` — the window snaps to the nearest
/// boundary slice, giving LOESS extrapolation by extension of the
/// boundary fit. Used by STL's cycle-subseries one-period extrapolation.
pub fn loess_at(
    y: ColRef<'_, f64>,
    xq: f64,
    span: f64,
    degree: u8,
) -> Result<f64, LoessError> {
    validate_loess_args(y, span, degree)?;
    let vec: Vec<f64> = y.iter().copied().collect();
    let slice = vec.as_slice();
    let n = slice.len();
    if n == 0 {
        return Err(LoessError::Empty);
    }
    let degree_us = degree as usize;
    let window = ((span * n as f64).ceil() as usize)
        .max(degree_us + 2)
        .min(n);
    Ok(local_poly_fit_at_xf64(slice, xq, window, degree_us))
}

fn validate_loess_args(
    y: ColRef<'_, f64>,
    span: f64,
    degree: u8,
) -> Result<(), LoessError> {
    if !(span > 0.0 && span <= 1.0) {
        return Err(LoessError::InvalidSpan(span));
    }
    if degree > 2 {
        return Err(LoessError::InvalidDegree(degree));
    }
    if y.nrows() == 0 {
        return Err(LoessError::Empty);
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(LoessError::NonFinite);
    }
    Ok(())
}

/// Window of size `k` (clipped to `n`) centred around the integer floor of
/// `xq`. `xq` may be outside `[0, n-1]`, in which case the window snaps to
/// the nearest boundary slice.
pub(crate) fn loess_window_f(n: usize, xq: f64, k: usize) -> (usize, usize) {
    if k >= n {
        return (0, n);
    }
    let half = k / 2;
    let xq_clamp = xq.max(0.0).min((n - 1) as f64);
    let lo_unclamped = (xq_clamp - half as f64).floor();
    let lo = (lo_unclamped.max(0.0) as usize).min(n - k);
    (lo, lo + k)
}

/// Solve an `n x n` linear system `mat * x = rhs` via Gaussian elimination
/// with partial pivoting. Returns `None` if the matrix is singular.
pub(crate) fn gauss_solve_n(
    n: usize,
    mut mat: Vec<f64>,
    mut rhs: Vec<f64>,
) -> Option<Vec<f64>> {
    for i in 0..n {
        let mut pivot = i;
        let mut best = mat[i * n + i].abs();
        for r in (i + 1)..n {
            if mat[r * n + i].abs() > best {
                best = mat[r * n + i].abs();
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != i {
            for c in i..n {
                mat.swap(i * n + c, pivot * n + c);
            }
            rhs.swap(i, pivot);
        }
        for j in (i + 1)..n {
            let factor = mat[j * n + i] / mat[i * n + i];
            rhs[j] -= factor * rhs[i];
            for c in i..n {
                mat[j * n + c] -= factor * mat[i * n + c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..n {
            s -= mat[i * n + j] * x[j];
        }
        x[i] = s / mat[i * n + i];
    }
    Some(x)
}

/// Local polynomial fit at (possibly fractional) position `xq`, using the
/// `k` closest indices in `y` and tricube distance weights. Returns the
/// fitted value at `xq` (the intercept of the centred fit). Falls back to
/// a weighted mean if the polynomial system is singular.
pub(crate) fn local_poly_fit_at_xf64(
    y: &[f64],
    xq: f64,
    k: usize,
    degree: usize,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return f64::NAN;
    }
    let (lo, hi) = loess_window_f(n, xq, k);

    // Furthest distance from xq to any window point. Bumped by 1 so the
    // boundary point doesn't get exactly zero weight, which preserves
    // (degree+1) effective points and keeps the normal-equations matrix
    // non-singular for centred windows.
    let max_dist = {
        let left = (xq - lo as f64).abs();
        let right = ((hi - 1) as f64 - xq).abs();
        left.max(right).max(1.0) + 1.0
    };

    let m = degree + 1;
    let nearest_idx = || -> usize { xq.round().max(0.0).min((n - 1) as f64) as usize };

    if degree == 0 {
        let mut wsum = 0.0;
        let mut wysum = 0.0;
        for i in lo..hi {
            let d = (i as f64 - xq).abs() / max_dist;
            let w = if d >= 1.0 {
                0.0
            } else {
                let u = 1.0 - d * d * d;
                u * u * u
            };
            wsum += w;
            wysum += w * y[i];
        }
        return if wsum > 0.0 { wysum / wsum } else { y[nearest_idx()] };
    }

    let mut xtwx = vec![0.0; m * m];
    let mut xtwy = vec![0.0; m];
    let p_len = 2 * m - 1;
    let mut powers = vec![0.0; p_len];

    let mut wsum = 0.0;
    for i in lo..hi {
        let dx = i as f64 - xq;
        let abs_d = dx.abs() / max_dist;
        let w = if abs_d >= 1.0 {
            0.0
        } else {
            let u = 1.0 - abs_d * abs_d * abs_d;
            u * u * u
        };
        if w == 0.0 {
            continue;
        }
        wsum += w;

        powers[0] = 1.0;
        for r in 1..p_len {
            powers[r] = powers[r - 1] * dx;
        }
        let yi = y[i];
        for r in 0..m {
            xtwy[r] += w * powers[r] * yi;
            for c in 0..m {
                xtwx[r * m + c] += w * powers[r + c];
            }
        }
    }

    if wsum == 0.0 {
        return y[nearest_idx()];
    }

    match gauss_solve_n(m, xtwx, xtwy) {
        Some(coefs) => coefs[0],
        None => {
            let mut wsum = 0.0;
            let mut wysum = 0.0;
            for i in lo..hi {
                let d = (i as f64 - xq).abs() / max_dist;
                let w = if d >= 1.0 {
                    0.0
                } else {
                    let u = 1.0 - d * d * d;
                    u * u * u
                };
                wsum += w;
                wysum += w * y[i];
            }
            if wsum > 0.0 {
                wysum / wsum
            } else {
                y[nearest_idx()]
            }
        }
    }
}

pub(crate) fn local_poly_fit_at(y: &[f64], xq: usize, k: usize, degree: usize) -> f64 {
    local_poly_fit_at_xf64(y, xq as f64, k, degree)
}

/// LOESS smoother that takes an integer window directly. Used by `loess`
/// (which converts a fractional span first) and STL (which uses integer
/// windows throughout). Parallelises via rayon when n >= 256.
pub(crate) fn loess_compute(y: &[f64], window: usize, degree: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let k = window.max(degree + 2).min(n);
    if n >= 256 {
        (0..n)
            .into_par_iter()
            .map(|i| local_poly_fit_at(y, i, k, degree))
            .collect()
    } else {
        (0..n).map(|i| local_poly_fit_at(y, i, k, degree)).collect()
    }
}
