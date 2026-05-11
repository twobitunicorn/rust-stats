//! Per-element transforms over `&[f64]`.
//!
//! Standardisation / scaling helpers that consume a slice and produce a
//! `Vec<f64>` of the same length. All aggregates (mean, std, min, max)
//! are computed over the finite entries; `NaN` inputs propagate to the
//! corresponding output positions, but do not contaminate the
//! aggregates themselves.
//!
//! Edge cases:
//!
//! - Empty input → empty output (no error).
//! - Constant input (or all-NaN aggregate) → an all-zero output for
//!   `z_score` and `min_max_scale`. `NaN` positions still propagate.
//! - `box_cox` is the only transform here that can fail: it requires
//!   strictly positive finite values and returns
//!   [`BoxCoxError::NonPositive`] otherwise.

use crate::error::BoxCoxError;

fn finite_mean(y: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in y {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Sample standard deviation (ddof = 1) over the finite entries.
/// Returns `0.0` when fewer than two finite entries are present.
fn finite_std_ddof1(y: &[f64], mean: f64) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for &v in y {
        if v.is_finite() {
            let d = v - mean;
            sum_sq += d * d;
            count += 1;
        }
    }
    if count < 2 {
        0.0
    } else {
        (sum_sq / (count - 1) as f64).sqrt()
    }
}

fn finite_min_max(y: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    for &v in y {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
            any = true;
        }
    }
    if any {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

/// Subtract the (finite-entry) mean from every value.
///
/// `NaN` inputs propagate to the same positions in the output. An
/// all-NaN input is treated as having a mean of zero.
pub fn center(y: &[f64]) -> Vec<f64> {
    let mean = finite_mean(y);
    y.iter().map(|&v| v - mean).collect()
}

/// Z-score normalisation: `(x - mean) / std` with sample standard
/// deviation (ddof = 1).
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs (and inputs with fewer than two finite
/// entries) produce an all-zero output at finite positions.
pub fn z_score(y: &[f64]) -> Vec<f64> {
    let mean = finite_mean(y);
    let std = finite_std_ddof1(y, mean);
    if std == 0.0 {
        // Match the centered * 0.0 behaviour: zeros at finite
        // positions, NaN at NaN positions.
        y.iter().map(|&v| (v - mean) * 0.0).collect()
    } else {
        y.iter().map(|&v| (v - mean) / std).collect()
    }
}

/// Min-max rescaling into `[0, 1]`: `(x - min) / (max - min)`.
///
/// Aggregates are computed over the finite entries; `NaN` inputs
/// propagate. Constant inputs produce an all-zero output at finite
/// positions.
pub fn min_max_scale(y: &[f64]) -> Vec<f64> {
    let (lo, hi) = finite_min_max(y);
    let range = hi - lo;
    if range == 0.0 {
        y.iter().map(|&v| (v - lo) * 0.0).collect()
    } else {
        y.iter().map(|&v| (v - lo) / range).collect()
    }
}

/// Box-Cox power transformation with a fixed `lmbda`:
///
/// ```text
/// (x^λ − 1) / λ   when λ ≠ 0
/// ln(x)           when λ = 0
/// ```
///
/// Requires every finite input to be strictly positive. `NaN` entries
/// propagate to the output unchanged; `+∞` is treated as finite for
/// propagation purposes but never satisfies the positivity check on its
/// own (only finite values gate the check).
pub fn box_cox(y: &[f64], lmbda: f64) -> Result<Vec<f64>, BoxCoxError> {
    let mut min_finite = f64::INFINITY;
    for &v in y {
        if v.is_finite() && v < min_finite {
            min_finite = v;
        }
    }
    if min_finite.is_finite() && !(min_finite > 0.0) {
        return Err(BoxCoxError::NonPositive { min: min_finite });
    }

    if lmbda == 0.0 {
        Ok(y.iter().map(|&v| v.ln()).collect())
    } else {
        let inv = 1.0 / lmbda;
        Ok(y.iter().map(|&v| (v.powf(lmbda) - 1.0) * inv).collect())
    }
}

// ============================================================================
// SIMD kernels — backed by `pulp` (stable Rust, runtime ISA dispatch).
//
// `pulp::Arch::new()` selects the best SIMD level at runtime (SSE2 /
// AVX2 / AVX-512 on x86_64, NEON on aarch64, scalar fallback elsewhere).
// `S::F64_LANES` is the lane count of the chosen target.
//
// All kernels preserve the scalar contracts: NaN inputs propagate to the
// same output positions, aggregates are computed over the finite entries
// only, and degenerate inputs (empty / constant / fewer than two finite
// values) produce the same zeros-or-NaN pattern as the scalar path.
//
// Transcendentals (`ln`, `pow`) are not part of pulp's f64 vocabulary,
// so `box_cox` is not vectorised here.
// ============================================================================

#[cfg(feature = "simd")]
mod simd_impl {
    use pulp::{Arch, Simd, WithSimd};

    /// Lanewise `is_finite` mask: `abs(v) < +∞` is true for every finite
    /// f64 and false for ±∞ and NaN (NaN < anything is always false).
    #[inline(always)]
    fn finite_mask<S: Simd>(simd: S, v: S::f64s) -> S::m64s {
        simd.less_than_f64s(simd.abs_f64s(v), simd.splat_f64s(f64::INFINITY))
    }

    struct FiniteSumCount<'a> {
        y: &'a [f64],
    }
    impl<'a> WithSimd for FiniteSumCount<'a> {
        type Output = (f64, usize);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let zero = simd.splat_f64s(0.0);
            let one = simd.splat_f64s(1.0);
            let mut sum_v = zero;
            let mut cnt_v = zero;
            for &v in head {
                let m = finite_mask(simd, v);
                sum_v = simd.add_f64s(sum_v, simd.select_f64s(m, v, zero));
                cnt_v = simd.add_f64s(cnt_v, simd.select_f64s(m, one, zero));
            }
            let mut sum = simd.reduce_sum_f64s(sum_v);
            let mut cnt = simd.reduce_sum_f64s(cnt_v) as usize;
            for &v in tail {
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            (sum, cnt)
        }
    }

    struct FiniteSumSq<'a> {
        y: &'a [f64],
        mean: f64,
    }
    impl<'a> WithSimd for FiniteSumSq<'a> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let mean_v = simd.splat_f64s(self.mean);
            let zero = simd.splat_f64s(0.0);
            let mut acc = zero;
            for &v in head {
                let m = finite_mask(simd, v);
                let d = simd.sub_f64s(v, mean_v);
                let dd = simd.mul_f64s(d, d);
                acc = simd.add_f64s(acc, simd.select_f64s(m, dd, zero));
            }
            let mut s = simd.reduce_sum_f64s(acc);
            for &v in tail {
                if v.is_finite() {
                    let d = v - self.mean;
                    s += d * d;
                }
            }
            s
        }
    }

    struct FiniteMinMax<'a> {
        y: &'a [f64],
    }
    impl<'a> WithSimd for FiniteMinMax<'a> {
        type Output = (f64, f64);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.y);
            let pos_inf = simd.splat_f64s(f64::INFINITY);
            let neg_inf = simd.splat_f64s(f64::NEG_INFINITY);
            let mut lo_v = pos_inf;
            let mut hi_v = neg_inf;
            for &v in head {
                let m = finite_mask(simd, v);
                lo_v = simd.min_f64s(lo_v, simd.select_f64s(m, v, pos_inf));
                hi_v = simd.max_f64s(hi_v, simd.select_f64s(m, v, neg_inf));
            }
            let mut lo = simd.reduce_min_f64s(lo_v);
            let mut hi = simd.reduce_max_f64s(hi_v);
            for &v in tail {
                if v.is_finite() {
                    if v < lo {
                        lo = v;
                    }
                    if v > hi {
                        hi = v;
                    }
                }
            }
            // If no finite value was seen at all, both extremes are still
            // their initialisers — collapse to (0, 0) to match the scalar
            // contract.
            if lo == f64::INFINITY && hi == f64::NEG_INFINITY {
                (0.0, 0.0)
            } else {
                (lo, hi)
            }
        }
    }

    struct AffineInto<'a> {
        y: &'a [f64],
        out: &'a mut [f64],
        c: f64,
        k: f64,
    }
    impl<'a> WithSimd for AffineInto<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self { y, out, c, k } = self;
            let c_v = simd.splat_f64s(c);
            let k_v = simd.splat_f64s(k);
            let (y_head, y_tail) = S::as_simd_f64s(y);
            let (o_head, o_tail) = S::as_mut_simd_f64s(out);
            for (yv, ov) in y_head.iter().zip(o_head.iter_mut()) {
                *ov = simd.mul_f64s(simd.sub_f64s(*yv, c_v), k_v);
            }
            for (yv, ov) in y_tail.iter().zip(o_tail.iter_mut()) {
                *ov = (*yv - c) * k;
            }
        }
    }

    pub(super) fn center(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (sum, cnt) = arch.dispatch(FiniteSumCount { y });
        let mean = if cnt == 0 { 0.0 } else { sum / cnt as f64 };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: mean, k: 1.0 });
        out
    }

    pub(super) fn z_score(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (sum, cnt) = arch.dispatch(FiniteSumCount { y });
        let mean = if cnt == 0 { 0.0 } else { sum / cnt as f64 };
        let std = if cnt < 2 {
            0.0
        } else {
            (arch.dispatch(FiniteSumSq { y, mean }) / (cnt - 1) as f64).sqrt()
        };
        // std == 0 → multiplying by 0 inside `AffineInto` preserves NaN
        // (NaN * 0 = NaN) and zeros every finite position — the same as
        // the scalar `(v - mean) * 0.0` contract.
        let inv = if std == 0.0 { 0.0 } else { 1.0 / std };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: mean, k: inv });
        out
    }

    pub(super) fn min_max_scale(y: &[f64]) -> Vec<f64> {
        let arch = Arch::new();
        let (lo, hi) = arch.dispatch(FiniteMinMax { y });
        let range = hi - lo;
        let inv = if range == 0.0 { 0.0 } else { 1.0 / range };
        let mut out = vec![0.0; y.len()];
        arch.dispatch(AffineInto { y, out: &mut out, c: lo, k: inv });
        out
    }
}

/// SIMD-accelerated [`center`]. Requires the `simd` feature (nightly).
/// Numerically identical to the scalar implementation up to FP
/// associativity in the SIMD reduction.
#[cfg(feature = "simd")]
pub fn center_simd(y: &[f64]) -> Vec<f64> {
    simd_impl::center(y)
}

/// SIMD-accelerated [`z_score`]. Requires the `simd` feature (nightly).
#[cfg(feature = "simd")]
pub fn z_score_simd(y: &[f64]) -> Vec<f64> {
    simd_impl::z_score(y)
}

/// SIMD-accelerated [`min_max_scale`]. Requires the `simd` feature
/// (nightly).
#[cfg(feature = "simd")]
pub fn min_max_scale_simd(y: &[f64]) -> Vec<f64> {
    simd_impl::min_max_scale(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn center_subtracts_mean() {
        let out = center(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn center_empty() {
        assert_eq!(center(&[]), Vec::<f64>::new());
    }

    #[test]
    fn center_propagates_nan() {
        let out = center(&[1.0, f64::NAN, 3.0]);
        // mean over finite = 2.0
        assert_relative_eq!(out[0], -1.0);
        assert!(out[1].is_nan());
        assert_relative_eq!(out[2], 1.0);
    }

    #[test]
    fn z_score_unit_variance() {
        let out = z_score(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // mean=3, std=sqrt(2.5)
        let s = (2.5_f64).sqrt();
        assert_relative_eq!(out[0], -2.0 / s, max_relative = 1e-12);
        assert_relative_eq!(out[4], 2.0 / s, max_relative = 1e-12);
    }

    #[test]
    fn z_score_constant_returns_zeros() {
        assert_eq!(z_score(&[4.0, 4.0, 4.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn min_max_scale_basic() {
        let out = min_max_scale(&[0.0, 5.0, 10.0]);
        assert_eq!(out, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn min_max_scale_constant() {
        assert_eq!(min_max_scale(&[7.0, 7.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn box_cox_lmbda_zero_is_ln() {
        let out = box_cox(&[1.0, std::f64::consts::E], 0.0).unwrap();
        assert_relative_eq!(out[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.0, max_relative = 1e-12);
    }

    #[test]
    fn box_cox_lmbda_two() {
        // (x^2 - 1) / 2
        let out = box_cox(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert_relative_eq!(out[0], 0.0, max_relative = 1e-12, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.5);
        assert_relative_eq!(out[2], 4.0);
    }

    #[test]
    fn box_cox_rejects_non_positive() {
        let err = box_cox(&[1.0, 0.0, 2.0], 1.0).unwrap_err();
        assert_eq!(err, BoxCoxError::NonPositive { min: 0.0 });
    }

    #[test]
    fn box_cox_propagates_nan() {
        let out = box_cox(&[1.0, f64::NAN, 4.0], 2.0).unwrap();
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_nan());
        assert_relative_eq!(out[2], 7.5);
    }

    // --- SIMD parity ---
    //
    // When the `simd` feature is on, the `_simd` kernels must produce
    // bit-close output to the scalar kernels. We test on a mixed-size
    // input that crosses the 4-lane boundary and includes NaN to
    // exercise the masked finite reductions.

    #[cfg(feature = "simd")]
    fn parity_check(scalar: &[f64], simd: &[f64], ctx: &str) {
        assert_eq!(scalar.len(), simd.len(), "{ctx}: length mismatch");
        for (i, (a, b)) in scalar.iter().zip(simd.iter()).enumerate() {
            if a.is_nan() {
                assert!(b.is_nan(), "{ctx}[{i}]: scalar NaN but simd {b}");
            } else {
                assert!(
                    (a - b).abs() < 1e-12,
                    "{ctx}[{i}]: scalar {a}, simd {b}, |Δ| = {}",
                    (a - b).abs()
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    fn make_fixture() -> Vec<f64> {
        // length 11 — guarantees a chunks_exact(4) remainder of 3
        vec![
            1.0, -2.5, 3.25, f64::NAN, 5.5, 0.0, -7.125, 8.75,
            f64::NAN, 11.5, -3.0,
        ]
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_center_matches_scalar() {
        let y = make_fixture();
        parity_check(&center(&y), &super::center_simd(&y), "center");
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_z_score_matches_scalar() {
        let y = make_fixture();
        parity_check(&z_score(&y), &super::z_score_simd(&y), "z_score");

        // Constant path: simd must also collapse to zeros.
        let constant = vec![4.2; 9];
        parity_check(
            &z_score(&constant),
            &super::z_score_simd(&constant),
            "z_score constant",
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_min_max_matches_scalar() {
        let y = make_fixture();
        parity_check(
            &min_max_scale(&y),
            &super::min_max_scale_simd(&y),
            "min_max",
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_handles_empty_and_short_inputs() {
        // empty
        assert!(super::center_simd(&[]).is_empty());
        assert!(super::z_score_simd(&[]).is_empty());
        assert!(super::min_max_scale_simd(&[]).is_empty());
        // shorter than one SIMD lane group
        let y = vec![1.0, 2.0, 3.0];
        parity_check(&center(&y), &super::center_simd(&y), "short center");
        parity_check(&z_score(&y), &super::z_score_simd(&y), "short z_score");
        parity_check(
            &min_max_scale(&y),
            &super::min_max_scale_simd(&y),
            "short min_max",
        );
    }
}
