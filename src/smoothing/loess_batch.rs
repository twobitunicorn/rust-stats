//! Cross-column SIMD batched LOESS, dispatched at runtime via `pulp`.
//!
//! For each output index `i`, the tricube weights and the moments
//! `Σw·dx^r` depend only on the integer x-grid — they are shared across
//! all columns. The per-column part is just `Σw·dx^r · y[j]`, which
//! vectorises cleanly across columns. The 2×2 normal-equation solve
//! (degree=1) becomes one SIMD divide instead of one scalar divide per
//! column.
//!
//! The same kernel runs on every `pulp::Simd` impl: the scalar fallback
//! (`f64s = f64`) compiles to ordinary scalar code, while AVX2 / AVX-512
//! / Neon impls compile to their respective SIMD instructions. One
//! source, runtime-dispatched by `pulp::Arch::new()`.
//!
//! degree=0 and degree=1 use the SIMD kernel; degree=2 (rare) is handled
//! by the per-column scalar path in the caller.
//!
//! Correctness is guarded by the column-by-column equivalence test in
//! `tests/arrow_compat.rs` — every batched output must match the scalar
//! `loess()` to ~1e-12.

use bytemuck::Pod;
use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;

use crate::error::LoessError;

/// Smooth every column in `cols` and write the result into `out`. The two
/// must have matching shape: same number of columns, every output column
/// is the same length as the corresponding input column.
///
/// Returns `Err` if `degree > 1` (caller is expected to route those to
/// the per-column scalar path).
pub(crate) fn loess_batch_simd(
    cols: &[&[f64]],
    span: f64,
    degree: u8,
    out: &mut [Vec<f64>],
) -> Result<(), LoessError> {
    if degree > 1 {
        // Degree=2 is not implemented in the SIMD path. The caller is
        // expected to fall back to per-column scalar LOESS.
        return Err(LoessError::InvalidDegree(degree));
    }
    if cols.is_empty() {
        return Ok(());
    }
    let n = cols[0].len();
    if n == 0 {
        return Err(LoessError::Empty);
    }
    if !(span > 0.0 && span <= 1.0) {
        return Err(LoessError::InvalidSpan(span));
    }
    for c in cols {
        if c.len() != n {
            // Defensive: callers must validate lengths.
            return Err(LoessError::Empty);
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err(LoessError::NonFinite);
        }
    }

    let window = ((span * n as f64).ceil() as usize)
        .max((degree as usize) + 2)
        .min(n);

    Arch::new().dispatch(BatchKernel {
        cols,
        n,
        window,
        degree,
        out,
    });
    Ok(())
}

struct BatchKernel<'a> {
    cols:   &'a [&'a [f64]],
    n:      usize,
    window: usize,
    degree: u8,
    out:    &'a mut [Vec<f64>],
}

impl<'a> WithSimd for BatchKernel<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let BatchKernel { cols, n, window, degree, out } = self;
        let p = cols.len();
        let lanes = S::F64_LANES;
        let p_padded = p.div_ceil(lanes) * lanes;
        let n_chunks = p_padded / lanes;

        // Row-major transpose so that for each row k the L=lanes columns
        // of a chunk are contiguous and can be loaded as one S::f64s.
        // Trailing lanes (p_padded - p) are zero-padded.
        let mut packed = vec![0.0f64; n * p_padded];
        for j in 0..p {
            let col = cols[j];
            for k in 0..n {
                packed[k * p_padded + j] = col[k];
            }
        }

        // Each chunk gets its own n*lanes output buffer; they're
        // disjoint so we can process them in parallel.
        let chunk_outputs: Vec<Vec<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let c0 = chunk_idx * lanes;
                let mut out_chunk = vec![0.0f64; n * lanes];
                for i in 0..n {
                    let alpha = fit_point::<S>(
                        simd, &packed, n, p_padded, c0, i, window, degree,
                    );
                    let dst = &mut out_chunk[i * lanes..(i + 1) * lanes];
                    dst.copy_from_slice(bytemuck::cast_slice(core::slice::from_ref(&alpha)));
                }
                out_chunk
            })
            .collect();

        // Scatter chunk outputs back into the caller's per-column Vecs.
        for chunk_idx in 0..n_chunks {
            let c0 = chunk_idx * lanes;
            let chunk = &chunk_outputs[chunk_idx];
            for lane in 0..lanes {
                let j = c0 + lane;
                if j >= p {
                    break; // trailing padding — discard
                }
                let dst = &mut out[j];
                for i in 0..n {
                    dst[i] = chunk[i * lanes + lane];
                }
            }
        }
    }
}

/// Fit one output point across L columns and return the result as an
/// `S::f64s`. Trailing-lane values for padded columns are arbitrary
/// (they're discarded by the untranspose).
#[inline(always)]
fn fit_point<S: Simd>(
    simd:    S,
    packed:  &[f64],
    n:       usize,
    p_padded: usize,
    c0:      usize,
    i:       usize,
    window:  usize,
    degree:  u8,
) -> S::f64s
where
    S::f64s: Pod,
{
    let lanes = S::F64_LANES;

    // Centred window around i, clipped to [0, n].
    let half = window / 2;
    let lo_unclamped = (i as isize) - (half as isize);
    let lo = (lo_unclamped.max(0) as usize).min(n.saturating_sub(window));
    let hi = (lo + window).min(n);

    // Distance normaliser. Matches `local_poly_fit_at_xf64` — the +1.0
    // bump keeps the boundary point from getting exactly zero weight.
    let left  = (i as f64 - lo as f64).abs();
    let right = ((hi - 1) as f64 - i as f64).abs();
    let max_dist = left.max(right).max(1.0) + 1.0;

    // Shared scalar accumulators: moments of (w, dx) over the window.
    let mut wsum  = 0.0f64;
    let mut swdx  = 0.0f64;
    let mut swdx2 = 0.0f64;

    // Per-chunk SIMD accumulators.
    let mut wy:    S::f64s = simd.splat_f64s(0.0);
    let mut swdxy: S::f64s = simd.splat_f64s(0.0);

    for k in lo..hi {
        let dx = k as f64 - i as f64;
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
        wsum  += w;
        swdx  += w * dx;
        swdx2 += w * dx * dx;

        // Load the row's L column values (aligned by construction).
        let yvec_slice = &packed[k * p_padded + c0..k * p_padded + c0 + lanes];
        let yvec: S::f64s = bytemuck::pod_read_unaligned(bytemuck::cast_slice(yvec_slice));

        wy    = simd.add_f64s(wy,    simd.mul_f64s(simd.splat_f64s(w),       yvec));
        if degree >= 1 {
            swdxy = simd.add_f64s(swdxy, simd.mul_f64s(simd.splat_f64s(w * dx), yvec));
        }
    }

    if wsum == 0.0 {
        // All weights zero — fall back to value at the nearest index
        // (mirrors the scalar path).
        let nearest = i.min(n - 1);
        let nyvec_slice = &packed[nearest * p_padded + c0..nearest * p_padded + c0 + lanes];
        return bytemuck::pod_read_unaligned(bytemuck::cast_slice(nyvec_slice));
    }

    if degree == 0 {
        // Weighted mean: wy / wsum.
        return simd.div_f64s(wy, simd.splat_f64s(wsum));
    }

    // degree == 1: solve the 2×2 normal equations per lane.
    //   det = wsum*swdx2 − swdx²
    //   alpha = (swdx2*wy − swdx*swdxy) / det
    let det = wsum * swdx2 - swdx * swdx;
    if det.abs() < 1e-12 {
        // Singular — fall back to weighted mean.
        return simd.div_f64s(wy, simd.splat_f64s(wsum));
    }
    let swdx2_v = simd.splat_f64s(swdx2);
    let swdx_v  = simd.splat_f64s(swdx);
    let det_v   = simd.splat_f64s(det);
    let num     = simd.sub_f64s(simd.mul_f64s(swdx2_v, wy), simd.mul_f64s(swdx_v, swdxy));
    simd.div_f64s(num, det_v)
}
