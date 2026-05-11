//! Batched LOESS for multi-column inputs.
//!
//! With the `simd` feature on (stable Rust, backed by `pulp`), this
//! module dispatches to a kernel that vectorises across columns at the
//! best SIMD width available at runtime (SSE2 / AVX2 / AVX-512 / NEON).
//! Without the feature, the fallback is plain `rayon`-over-columns
//! scalar LOESS. Both implementations expose the same
//! `loess_batch_simd` entry point so `arrow_compat::loess_batch` doesn't
//! know which is compiled.
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
    use pulp::{Arch, Simd, WithSimd};
    use rayon::prelude::*;

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

        Arch::new().dispatch(LoessBatchInner { cols, span, degree, out })
    }

    /// Load exactly `S::F64_LANES` consecutive `f64`s as one SIMD vector.
    /// Cleaner than `partial_load_f64s` for full-width loads: the latter
    /// uses a masked load.
    #[inline(always)]
    fn load_f64s<S: Simd>(slice: &[f64]) -> S::f64s {
        debug_assert_eq!(slice.len(), S::F64_LANES);
        let (head, _tail) = S::as_simd_f64s(slice);
        head[0]
    }

    struct LoessBatchInner<'a> {
        cols: &'a [&'a [f64]],
        span: f64,
        degree: u8,
        out: &'a mut [Vec<f64>],
    }

    impl<'a> WithSimd for LoessBatchInner<'a> {
        type Output = Result<(), LoessError>;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { cols, span, degree, out } = self;
            // F64_LANES is a compile-time constant for the chosen ISA
            // (e.g. 4 on AVX2, 8 on AVX-512, 2 on NEON).
            let lanes = S::F64_LANES;
            let n = cols[0].len();

            let window = ((span * n as f64).ceil() as usize)
                .max((degree as usize) + 2)
                .min(n);

            let p = cols.len();
            let p_padded = p.div_ceil(lanes) * lanes;
            let n_chunks = p_padded / lanes;

            // Row-major transpose so each row's `lanes` columns are
            // contiguous and can be loaded as one SIMD vector.
            let mut packed = vec![0.0f64; n * p_padded];
            for j in 0..p {
                let col = cols[j];
                for k in 0..n {
                    packed[k * p_padded + j] = col[k];
                }
            }

            // Independent output buffers per chunk → safe under rayon.
            // `simd: S` is `Copy + Send + Sync + 'static`, so it can be
            // captured by the parallel closure.
            let chunk_outputs: Vec<Vec<f64>> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let c0 = chunk_idx * lanes;
                    let mut out_chunk = vec![0.0f64; n * lanes];
                    for i in 0..n {
                        let alpha = fit_point(
                            simd, &packed, n, p_padded, c0, i, window, degree,
                        );
                        let dst = &mut out_chunk[i * lanes..(i + 1) * lanes];
                        let (head, _tail) = S::as_mut_simd_f64s(dst);
                        head[0] = alpha;
                    }
                    out_chunk
                })
                .collect();

            // Scatter back into the caller's per-column Vecs.
            for chunk_idx in 0..n_chunks {
                let c0 = chunk_idx * lanes;
                let chunk = &chunk_outputs[chunk_idx];
                for lane in 0..lanes {
                    let j = c0 + lane;
                    if j >= p {
                        break;
                    }
                    let dst = &mut out[j];
                    for i in 0..n {
                        dst[i] = chunk[i * lanes + lane];
                    }
                }
            }
            Ok(())
        }
    }

    #[inline(always)]
    fn fit_point<S: Simd>(
        simd: S,
        packed: &[f64],
        n: usize,
        p_padded: usize,
        c0: usize,
        i: usize,
        window: usize,
        degree: u8,
    ) -> S::f64s {
        let lanes = S::F64_LANES;
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
        let mut wy = simd.splat_f64s(0.0);
        let mut swdxy = simd.splat_f64s(0.0);

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

            let slice = &packed[k * p_padded + c0..k * p_padded + c0 + lanes];
            let yvec = load_f64s::<S>(slice);
            let wv = simd.splat_f64s(w);
            wy = simd.add_f64s(wy, simd.mul_f64s(wv, yvec));
            if degree >= 1 {
                let wdv = simd.splat_f64s(w * dx);
                swdxy = simd.add_f64s(swdxy, simd.mul_f64s(wdv, yvec));
            }
        }

        if wsum == 0.0 {
            let nearest = i.min(n - 1);
            let slice = &packed[nearest * p_padded + c0..nearest * p_padded + c0 + lanes];
            return load_f64s::<S>(slice);
        }
        if degree == 0 {
            return simd.div_f64s(wy, simd.splat_f64s(wsum));
        }
        let det = wsum * swdx2 - swdx * swdx;
        if det.abs() < 1e-12 {
            return simd.div_f64s(wy, simd.splat_f64s(wsum));
        }
        // (swdx2 * wy − swdx * swdxy) / det
        let numer = simd.sub_f64s(
            simd.mul_f64s(simd.splat_f64s(swdx2), wy),
            simd.mul_f64s(simd.splat_f64s(swdx), swdxy),
        );
        simd.div_f64s(numer, simd.splat_f64s(det))
    }
}

#[cfg(feature = "simd")]
pub(crate) use simd_kernel::loess_batch_simd;
