//! Seasonal-trend decomposition: Cleveland 1990 STL and the classical
//! moving-average `seasonal_decompose`. Both produce identically-shaped
//! `Decomposition` output.

pub mod decompose;
pub mod stl;

pub use decompose::seasonal_decompose;
pub use stl::stl;

/// Output of a seasonal-trend decomposition. The components reconstruct
/// the input where defined:
///   additive:        `y[i] = trend[i] + seasonal[i] + residual[i]`
///   multiplicative:  `y[i] = trend[i] * seasonal[i] * residual[i]`
///
/// STL produces finite values everywhere; classical `seasonal_decompose`
/// has NaN at the first/last `period/2` positions where the centered
/// moving average can't be computed.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposeMode {
    Additive,
    Multiplicative,
}

/// Policy for handling non-finite (NaN / ±Inf) entries in the input.
///
/// `Error` (default) preserves the existing behaviour: any non-finite
/// value returns `NonFinite`. `Interpolate` fills non-finite entries via
/// linear interpolation between adjacent finite values (leading/trailing
/// runs are filled with the nearest finite value), runs the decomposition
/// on the filled series, and propagates `NaN` back into the residual at
/// originally-missing positions. The trend and seasonal stay finite
/// everywhere — they are the model's estimates at the imputed points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Missing {
    #[default]
    Error,
    Interpolate,
}

/// Linearly interpolate non-finite values in `y`. Leading and trailing
/// runs of non-finite values are filled with the nearest finite value;
/// interior runs are filled by linear interpolation between the
/// surrounding finite endpoints. Returns `None` if `y` has no finite
/// values.
pub(crate) fn interpolate_missing(y: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let first = y.iter().position(|v| v.is_finite())?;
    let last = y.iter().rposition(|v| v.is_finite()).expect("first exists ⇒ last exists");

    let mut out = y.to_vec();
    let lead = out[first];
    for slot in out.iter_mut().take(first) {
        *slot = lead;
    }
    let tail = out[last];
    for slot in out.iter_mut().take(n).skip(last + 1) {
        *slot = tail;
    }

    let mut i = first + 1;
    while i < last {
        if out[i].is_finite() {
            i += 1;
            continue;
        }
        // Find next finite index j > i (must exist before `last` since
        // out[last] is finite).
        let mut j = i + 1;
        while !out[j].is_finite() {
            j += 1;
        }
        let lo = out[i - 1];
        let hi = out[j];
        let gap = (j - (i - 1)) as f64;
        for k in i..j {
            let alpha = (k - (i - 1)) as f64 / gap;
            out[k] = lo + alpha * (hi - lo);
        }
        i = j + 1;
    }
    Some(out)
}

/// Options for `stl`. Construct via `StlOpts::new(period)` for Cleveland
/// defaults and override individual fields with struct-update syntax.
#[derive(Debug, Clone)]
pub struct StlOpts {
    pub period: u32,
    /// LOESS span (in points) for cycle-subseries smoothing.
    /// Must be odd and >= 7.
    pub seasonal_window: u32,
    /// LOESS span for the trend smoother. `None` uses Cleveland's
    /// recommended formula: smallest odd >=
    /// `1.5 * period / (1 - 1.5 / seasonal_window)`.
    pub trend_window: Option<u32>,
    /// Number of inner-loop iterations. Cleveland recommends 2.
    pub inner_iters: u32,
    /// Number of robust outer-loop iterations (Cleveland 1990 §3.5).
    /// `0` (default) skips the outer loop and produces a non-robust
    /// decomposition. `15` matches the default used by R's stl(robust=TRUE)
    /// and statsmodels' STL(robust=True). Each outer iteration recomputes
    /// per-point robustness weights from the previous residuals using the
    /// bisquare (Tukey biweight) function, then re-runs the inner loop with
    /// those weights folded into the cycle-subseries and trend LOESS fits.
    pub outer_iters: u32,
    pub mode: DecomposeMode,
    /// Cleveland 1990 "jump" parameter for the cycle-subseries smoother:
    /// fit LOESS at every `seasonal_jump`-th point in each subseries and
    /// linearly interpolate between them. Must be `>= 1`; `1` is exact.
    /// Cleveland recommends `round(period / 10)` for typical workloads.
    pub seasonal_jump: u32,
    /// Jump parameter for the trend LOESS. Must be `>= 1`; `1` is exact.
    pub trend_jump: u32,
    /// Jump parameter for the low-pass LOESS. Must be `>= 1`; `1` is exact.
    pub low_pass_jump: u32,
    /// How to handle non-finite values in the input. Defaults to `Error`.
    pub missing: Missing,
}

impl StlOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            seasonal_window: 7,
            trend_window: None,
            inner_iters: 2,
            outer_iters: 0,
            mode: DecomposeMode::Additive,
            seasonal_jump: 1,
            trend_jump: 1,
            low_pass_jump: 1,
            missing: Missing::Error,
        }
    }
}

/// Options for `seasonal_decompose`. Construct via
/// `SeasonalDecomposeOpts::new(period)` for additive defaults.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposeOpts {
    pub period: u32,
    pub mode: DecomposeMode,
    /// How to handle non-finite values in the input. Defaults to `Error`.
    pub missing: Missing,
}

impl SeasonalDecomposeOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            mode: DecomposeMode::Additive,
            missing: Missing::Error,
        }
    }
}
