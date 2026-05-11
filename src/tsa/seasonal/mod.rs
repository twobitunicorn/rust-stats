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
        }
    }
}

/// Options for `seasonal_decompose`. Construct via
/// `SeasonalDecomposeOpts::new(period)` for additive defaults.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposeOpts {
    pub period: u32,
    pub mode: DecomposeMode,
}

impl SeasonalDecomposeOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            mode: DecomposeMode::Additive,
        }
    }
}
