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
use crate::smoothing::loess::{local_poly_fit_at_xf64, loess_compute};
use crate::tsa::seasonal::{DecomposeMode, Decomposition, StlOpts};

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

    if y.is_empty() {
        return Err(StlError::SeriesTooShort {
            n: 0,
            min: 2 * period,
        });
    }

    let raw: Vec<f64> = y.to_vec();
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(StlError::NonFinite);
    }
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

    let (trend, seasonal) = stl_inner_loop(&work, period, n_s, n_l, n_t, n_i);

    let residual: Vec<f64> = (0..n)
        .map(|i| work[i] - trend[i] - seasonal[i])
        .collect();

    let (trend, seasonal, residual) = if multiplicative {
        (
            trend.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            seasonal.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
            residual.into_iter().map(|v| v.exp()).collect::<Vec<_>>(),
        )
    } else {
        (trend, seasonal, residual)
    };

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

/// Cycle-subseries smoothing — Step 2 of STL.
fn cycle_subseries_smooth(d: &[f64], period: usize, span: usize, degree: usize) -> Vec<f64> {
    let n = d.len();
    let mut c = vec![0.0; n + 2 * period];

    for phase in 0..period {
        let subs: Vec<f64> = (phase..n).step_by(period).map(|i| d[i]).collect();
        let sub_n = subs.len();
        if sub_n == 0 {
            continue;
        }
        let k = span.max(degree + 2).min(sub_n);

        c[phase] = local_poly_fit_at_xf64(&subs, -1.0, k, degree);

        for j in 0..sub_n {
            let orig = phase + j * period;
            c[period + orig] = local_poly_fit_at_xf64(&subs, j as f64, k, degree);
        }

        let after = phase + sub_n * period;
        c[period + after] = local_poly_fit_at_xf64(&subs, sub_n as f64, k, degree);
    }
    c
}

/// Low-pass filter — Step 3 of STL.
fn low_pass_filter(c: &[f64], period: usize, span: usize, degree: usize) -> Vec<f64> {
    let ma1 = valid_ma(c, period);
    let ma2 = valid_ma(&ma1, period);
    let ma3 = valid_ma(&ma2, 3);
    loess_compute(&ma3, span, degree)
}

/// One inner loop pass repeated `n_i` times. Returns `(trend, seasonal)`.
fn stl_inner_loop(
    y: &[f64],
    period: usize,
    n_s: usize,
    n_l: usize,
    n_t: usize,
    n_i: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut trend = vec![0.0f64; n];
    let mut seasonal = vec![0.0f64; n];

    for _ in 0..n_i {
        let detrended: Vec<f64> = (0..n).map(|i| y[i] - trend[i]).collect();
        let c = cycle_subseries_smooth(&detrended, period, n_s, 1);
        let l = low_pass_filter(&c, period, n_l, 1);
        seasonal = (0..n).map(|i| c[period + i] - l[i]).collect();
        let deseasonalized: Vec<f64> = (0..n).map(|i| y[i] - seasonal[i]).collect();
        trend = loess_compute(&deseasonalized, n_t, 1);
    }
    (trend, seasonal)
}
