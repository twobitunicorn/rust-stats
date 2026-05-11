//! Batched LOESS for multi-column inputs.
//!
//! With the `simd` feature on (nightly only), this module dispatches to
//! a `std::simd` `f64x4` kernel that vectorises across columns. Without
//! it, the fallback is plain `rayon`-over-columns scalar LOESS. Both
//! implementations expose the same `loess_batch_simd` entry point so
//! `arrow_compat::loess_batch` doesn't know which is compiled.
//!
//! Correctness is guarded by the column-by-column equivalence test in
//! `tests/arrow_compat.rs` — batched output must match the scalar
//! `loess()` to ~1e-12.

#[cfg(not(feature = "simd"))]
use crate::error::LoessError;

#[cfg(not(feature = "simd"))]
pub(crate) fn loess_batch_simd(
    cols: &[&[f64]],
    span: f64,
    degree: u8,
    out: &mut [Vec<f64>],
) -> Result<(), LoessError> {
    use rayon::prelude::*;

    if cols.is_empty() {
        return Ok(());
    }
    let n = cols[0].len();
    if n == 0 {
        return Err(LoessError::Empty);
    }
    for c in cols {
        if c.len() != n {
            return Err(LoessError::Empty);
        }
    }

    let results: Result<Vec<Vec<f64>>, LoessError> = cols
        .par_iter()
        .map(|c| crate::smoothing::loess::loess(c, span, degree))
        .collect();
    let results = results?;
    for (slot, src) in out.iter_mut().zip(results) {
        *slot = src;
    }
    Ok(())
}

#[cfg(feature = "simd")]
mod simd_kernel {
    use crate::error::LoessError;
    use crate::smoothing::loess::loess as scalar_loess;
    use rayon::prelude::*;
    use std::simd::f64x4;

    /// Width of one SIMD lane group. f64x4 is the sweet spot: full
    /// width on AVX2 / scalar; on Neon LLVM lowers cleanly to two f64x2
    /// instructions with the same code.
    const LANES: usize = 4;
    type V = f64x4;

    pub(crate) fn loess_batch_simd(
        cols: &[&[f64]],
        span: f64,
        degree: u8,
        out: &mut [Vec<f64>],
    ) -> Result<(), LoessError> {
        if degree > 1 {
            // Degree 2 is rare; fall back to per-column scalar LOESS.
            let results: Result<Vec<Vec<f64>>, LoessError> =
                cols.par_iter().map(|c| scalar_loess(c, span, degree)).collect();
            let results = results?;
            for (slot, src) in out.iter_mut().zip(results) {
                *slot = src;
            }
            return Ok(());
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
                return Err(LoessError::Empty);
            }
            if c.iter().any(|v| !v.is_finite()) {
                return Err(LoessError::NonFinite);
            }
        }

        let window = ((span * n as f64).ceil() as usize)
            .max((degree as usize) + 2)
            .min(n);

        let p = cols.len();
        let p_padded = p.div_ceil(LANES) * LANES;
        let n_chunks = p_padded / LANES;

        // Row-major transpose so each row's L=LANES columns are
        // contiguous and can be loaded as one SIMD vector.
        let mut packed = vec![0.0f64; n * p_padded];
        for j in 0..p {
            let col = cols[j];
            for k in 0..n {
                packed[k * p_padded + j] = col[k];
            }
        }

        // Independent output buffers per chunk → safe under rayon.
        let chunk_outputs: Vec<Vec<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let c0 = chunk_idx * LANES;
                let mut out_chunk = vec![0.0f64; n * LANES];
                for i in 0..n {
                    let alpha = fit_point(&packed, n, p_padded, c0, i, window, degree);
                    let dst = &mut out_chunk[i * LANES..(i + 1) * LANES];
                    dst.copy_from_slice(alpha.as_array());
                }
                out_chunk
            })
            .collect();

        // Scatter back into the caller's per-column Vecs.
        for chunk_idx in 0..n_chunks {
            let c0 = chunk_idx * LANES;
            let chunk = &chunk_outputs[chunk_idx];
            for lane in 0..LANES {
                let j = c0 + lane;
                if j >= p {
                    break;
                }
                let dst = &mut out[j];
                for i in 0..n {
                    dst[i] = chunk[i * LANES + lane];
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn fit_point(
        packed: &[f64],
        n: usize,
        p_padded: usize,
        c0: usize,
        i: usize,
        window: usize,
        degree: u8,
    ) -> V {
        let half = window / 2;
        let lo_unclamped = (i as isize) - (half as isize);
        let lo = (lo_unclamped.max(0) as usize).min(n.saturating_sub(window));
        let hi = (lo + window).min(n);

        let left = (i as f64 - lo as f64).abs();
        let right = ((hi - 1) as f64 - i as f64).abs();
        let max_dist = left.max(right).max(1.0) + 1.0;

        let mut wsum = 0.0f64;
        let mut swdx = 0.0f64;
        let mut swdx2 = 0.0f64;
        let mut wy: V = V::splat(0.0);
        let mut swdxy: V = V::splat(0.0);

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
            wsum += w;
            swdx += w * dx;
            swdx2 += w * dx * dx;

            let slice = &packed[k * p_padded + c0..k * p_padded + c0 + LANES];
            let yvec: V = V::from_slice(slice);
            wy = wy + V::splat(w) * yvec;
            if degree >= 1 {
                swdxy = swdxy + V::splat(w * dx) * yvec;
            }
        }

        if wsum == 0.0 {
            let nearest = i.min(n - 1);
            let slice = &packed[nearest * p_padded + c0..nearest * p_padded + c0 + LANES];
            return V::from_slice(slice);
        }
        if degree == 0 {
            return wy / V::splat(wsum);
        }
        let det = wsum * swdx2 - swdx * swdx;
        if det.abs() < 1e-12 {
            return wy / V::splat(wsum);
        }
        (V::splat(swdx2) * wy - V::splat(swdx) * swdxy) / V::splat(det)
    }
}

#[cfg(feature = "simd")]
pub(crate) use simd_kernel::loess_batch_simd;
