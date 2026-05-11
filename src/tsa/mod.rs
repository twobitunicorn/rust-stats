//! Time-series analysis. Currently: seasonal decomposition and
//! Holt-Winters exponential smoothing.

pub mod holt_winters;
pub mod seasonal;

pub use holt_winters::{holt_winters, HoltWintersOpts};
pub use seasonal::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
    SeasonalWindow, StlOpts,
};
