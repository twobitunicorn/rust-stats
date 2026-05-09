//! Seasonal-trend decomposition: Cleveland 1990 STL and the classical
//! moving-average `seasonal_decompose`. Both produce identically-shaped
//! `Decomposition` output.

pub mod decompose;
pub mod stl;

use faer::Col;

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
    pub trend: Col<f64>,
    pub seasonal: Col<f64>,
    pub residual: Col<f64>,
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
    pub mode: DecomposeMode,
}

impl StlOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            seasonal_window: 7,
            trend_window: None,
            inner_iters: 2,
            mode: DecomposeMode::Additive,
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
