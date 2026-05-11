//! Cleveland 1990 STL — seasonal-trend decomposition by LOESS.
//!
//! Inner loop:
//!   1. Detrend                 `D = Y − T`
//!   2. Cycle-subseries LOESS   one-period extrapolation each end → `C` of length n+2*period
//!   3. Low-pass filter         `MA(period) → MA(period) → MA(3) → LOESS` → `L` of length n
//!   4. Seasonal                `S = C[period..period+n] − L`
//!   5. Deseasonalize           `Y − S`
//!   6. Trend LOESS             `T = LOESS(Y − S)`
//! repeated `inner_iters` times. No outer robustness loop.
//!
//! Multiplicative mode: log-transform → additive STL → exp components.

use crate::error::StlError;
use crate::smoothing::loess::{
    local_poly_fit_at_xf64_weighted, loess_compute_with_jump,
    loess_compute_with_jump_weighted,
};
use crate::tsa::seasonal::{interpolate_missing, DecomposeMode, Decomposition, Missing, StlOpts};

/// Cleveland 1990 STL.
///
/// Returns a `Decomposition` whose three columns reconstruct `y` exactly
/// (additive: `y = T + S + R`; multiplicative: `y = T * S * R`).
/// LOESS-based — no NaN edges.
///
/// All tunable parameters live on `StlOpts` — use `StlOpts::new(period)`
/// for Cleveland defaults and override fields with struct-update syntax.
pub fn stl(y: &[f64], opts: StlOpts) -> Result<Decomposition, StlError> {
    if opts.period < 2 {
        return Err(StlError::InvalidPeriod(opts.period));
    }
    let period = opts.period as usize;

    let n_s = opts.seasonal_window as usize;
    if n_s < 7 || n_s % 2 == 0 {
        return Err(StlError::InvalidSeasonalWindow(opts.seasonal_window));
    }

    let n_l = if period % 2 == 0 { period + 1 } else { period };

    let n_t = match opts.trend_window {
        // Cleveland 1990 §3.4: trend smoother span defaults to the smallest
        // odd integer >= 1.5 * period / (1 - 1.5 / seasonal_window).
        None => next_odd_ceil(1.5 * period as f64 / (1.0 - 1.5 / n_s as f64)),
        Some(t) => {
            if t % 2 == 0 {
                return Err(StlError::InvalidTrendWindow(t));
            }
            t as usize
        }
    };

    let n_i = opts.inner_iters as usize;
    if n_i == 0 {
        return Err(StlError::InvalidInnerIters);
    }

    if opts.seasonal_jump == 0 {
        return Err(StlError::InvalidJump { which: "seasonal" });
    }
    if opts.trend_jump == 0 {
        return Err(StlError::InvalidJump { which: "trend" });
    }
    if opts.low_pass_jump == 0 {
        return Err(StlError::InvalidJump { which: "low_pass" });
    }
    let seasonal_jump = opts.seasonal_jump as usize;
    let trend_jump    = opts.trend_jump    as usize;
    let low_pass_jump = opts.low_pass_jump as usize;

    if y.is_empty() {
        return Err(StlError::SeriesTooShort {
            n: 0,
            min: 2 * period,
        });
    }

    // Track originally-missing positions so we can NaN out the residual
    // there on output. None when no imputation happened.
    let mut missing_mask: Option<Vec<bool>> = None;
    let raw: Vec<f64> = match opts.missing {
        Missing::Error => {
            if y.iter().any(|v| !v.is_finite()) {
                return Err(StlError::NonFinite);
            }
            y.to_vec()
        }
        Missing::Interpolate => {
            let any_missing = y.iter().any(|v| !v.is_finite());
            if any_missing {
                let mask: Vec<bool> = y.iter().map(|v| !v.is_finite()).collect();
                let filled = interpolate_missing(y).ok_or(StlError::NonFinite)?;
                missing_mask = Some(mask);
                filled
            } else {
                y.to_vec()
            }
        }
    };
    let n = raw.len();

    if n < 2 * period {
        return Err(StlError::SeriesTooShort {
            n,
            min: 2 * period,
        });
    }

    let multiplicative = matches!(opts.mode, DecomposeMode::Multiplicative);
    if multiplicative {
        let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
        if min <= 0.0 {
            return Err(StlError::NonPositiveForMultiplicative { min });
        }
    }

    let work: Vec<f64> = if multiplicative {
        raw.iter().map(|v| v.ln()).collect()
    } else {
        raw
    };

    // Outer loop: outer_iters + 1 inner passes total. First pass uses
    // all-1 weights (= the non-robust case when outer_iters == 0). After
    // each pass, recompute robustness weights from residuals and fold
    // them into the next inner pass.
    let n_o = opts.outer_iters as usize;
    let mut weights: Vec<f64> = vec![1.0; n];
    let mut trend: Vec<f64>;
    let mut seasonal: Vec<f64>;
    let mut residual: Vec<f64>;
    let mut pass = 0;
    loop {
        let w_ref: Option<&[f64]> = if pass == 0 { None } else { Some(&weights) };
        let (t, s) = stl_inner_loop(
            &work, period, n_s, n_l, n_t, n_i,
            seasonal_jump, trend_jump, low_pass_jump,
            w_ref,
        );
        trend = t;
        seasonal = s;
        residual = (0..n).map(|i| work[i] - trend[i] - seasonal[i]).collect();
        if pass == n_o {
            break;
        }
        weights = robust_weights(&residual);
        pass += 1;
    }

    let (trend, seasonal, mut residual) = if multiplicative {
        (
            trend.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            seasonal.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            residual.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
        )
    } else {
        (trend, seasonal, residual)
    };

    if let Some(mask) = missing_mask {
        for (i, &missing) in mask.iter().enumerate() {
            if missing {
                residual[i] = f64::NAN;
            }
        }
    }

    let _ = n;
    Ok(Decomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Smallest odd integer >= x.
fn next_odd_ceil(x: f64) -> usize {
    let n = x.ceil() as usize;
    if n % 2 == 0 {
        (n + 1).max(1)
    } else {
        n.max(1)
    }
}

/// Valid (non-padded) moving average. Input length n, output length
/// `n - window + 1`. Output[k] is the mean of input[k..k+window].
fn valid_ma(y: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    if window == 0 || n < window {
        return Vec::new();
    }
    let out_n = n - window + 1;
    let mut out = Vec::with_capacity(out_n);
    let inv = 1.0 / window as f64;
    let mut sum: f64 = y[..window].iter().sum();
    out.push(sum * inv);
    for i in window..n {
        sum += y[i] - y[i - window];
        out.push(sum * inv);
    }
    out
}

/// Cycle-subseries smoothing — Step 2 of STL. Within-subseries LOESS
/// fits can use Cleveland's `jump` approximation and accept per-point
/// robustness weights (for the outer loop). `weights = None` is exact
/// and reproduces the non-robust path bitwise.
#[allow(clippy::too_many_arguments)]
fn cycle_subseries_smooth(
    d: &[f64],
    period: usize,
    span: usize,
    degree: usize,
    jump: usize,
    weights: Option<&[f64]>,
) -> Vec<f64> {
    let n = d.len();
    let mut c = vec![0.0; n + 2 * period];

    for phase in 0..period {
        let subs: Vec<f64> = (phase..n).step_by(period).map(|i| d[i]).collect();
        let sub_n = subs.len();
        if sub_n == 0 {
            continue;
        }
        let k = span.max(degree + 2).min(sub_n);

        // Sub-series of robustness weights (when supplied).
        let sub_weights_storage: Option<Vec<f64>> = weights.map(|w| {
            (phase..n).step_by(period).map(|i| w[i]).collect()
        });
        let sub_w_ref: Option<&[f64]> = sub_weights_storage.as_deref();

        let fit_one = |xq: f64| local_poly_fit_at_xf64_weighted(&subs, xq, k, degree, sub_w_ref);

        c[phase] = fit_one(-1.0);

        if jump <= 1 || sub_n <= 2 {
            for j in 0..sub_n {
                let orig = phase + j * period;
                c[period + orig] = fit_one(j as f64);
            }
        } else {
            let mut fit_at: Vec<usize> = (0..sub_n).step_by(jump).collect();
            if *fit_at.last().unwrap() != sub_n - 1 {
                fit_at.push(sub_n - 1);
            }
            let fit_vals: Vec<f64> = fit_at.iter().map(|&j| fit_one(j as f64)).collect();
            for w in 0..(fit_at.len() - 1) {
                let j0 = fit_at[w];
                let j1 = fit_at[w + 1];
                let y0 = fit_vals[w];
                let y1 = fit_vals[w + 1];
                let span_j = (j1 - j0) as f64;
                c[period + (phase + j0 * period)] = y0;
                if j1 > j0 + 1 {
                    for j in (j0 + 1)..j1 {
                        let alpha = (j - j0) as f64 / span_j;
                        c[period + (phase + j * period)] = y0 + alpha * (y1 - y0);
                    }
                }
            }
            let last_j = *fit_at.last().unwrap();
            c[period + (phase + last_j * period)] = *fit_vals.last().unwrap();
        }

        let after = phase + sub_n * period;
        c[period + after] = fit_one(sub_n as f64);
    }
    c
}

/// Low-pass filter — Step 3 of STL.
fn low_pass_filter(
    c: &[f64],
    period: usize,
    span: usize,
    degree: usize,
    jump: usize,
) -> Vec<f64> {
    let ma1 = valid_ma(c, period);
    let ma2 = valid_ma(&ma1, period);
    let ma3 = valid_ma(&ma2, 3);
    loess_compute_with_jump(&ma3, span, degree, jump)
}

/// One inner loop pass repeated `n_i` times. Returns `(trend, seasonal)`.
/// Robustness weights, when supplied, are folded into the cycle-subseries
/// LOESS (Step 2) and the trend LOESS (Step 6) per Cleveland 1990 §3.5.
/// The low-pass MAs are unweighted.
#[allow(clippy::too_many_arguments)]
fn stl_inner_loop(
    y: &[f64],
    period: usize,
    n_s: usize,
    n_l: usize,
    n_t: usize,
    n_i: usize,
    seasonal_jump: usize,
    trend_jump:    usize,
    low_pass_jump: usize,
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut trend = vec![0.0f64; n];
    let mut seasonal = vec![0.0f64; n];

    for _ in 0..n_i {
        let detrended: Vec<f64> = (0..n).map(|i| y[i] - trend[i]).collect();
        let c = cycle_subseries_smooth(&detrended, period, n_s, 1, seasonal_jump, weights);
        let l = low_pass_filter(&c, period, n_l, 1, low_pass_jump);
        seasonal = (0..n).map(|i| c[period + i] - l[i]).collect();
        let deseasonalized: Vec<f64> = (0..n).map(|i| y[i] - seasonal[i]).collect();
        trend = loess_compute_with_jump_weighted(&deseasonalized, n_t, 1, trend_jump, weights);
    }
    (trend, seasonal)
}

/// Bisquare (Tukey biweight) robustness weights from residuals.
/// `ρ_i = (1 - (R_i / h)²)²` for `|R_i / h| < 1`, else `0`;
/// `h = 6 · median(|R|)`. If `h` is essentially zero (perfect fit),
/// returns all-1 weights so the next inner pass is unweighted.
fn robust_weights(residuals: &[f64]) -> Vec<f64> {
    let n = residuals.len();
    let mut abs_r: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n == 0 {
        0.0
    } else if n % 2 == 1 {
        abs_r[n / 2]
    } else {
        0.5 * (abs_r[n / 2 - 1] + abs_r[n / 2])
    };
    let h = 6.0 * median;
    if h < 1e-12 {
        return vec![1.0; n];
    }
    residuals
        .iter()
        .map(|&r| {
            let u = (r / h).abs();
            if u >= 1.0 {
                0.0
            } else {
                let v = 1.0 - u * u;
                v * v
            }
        })
        .collect()
}
