//! rust-stats: pure-Rust statistical modeling.
//!
//! Smoothing and seasonal-trend decomposition: LOESS, Cleveland 1990 STL,
//! and classical seasonal_decompose.

pub mod error;
pub mod smoothing;
pub mod transforms;
pub mod tsa;

#[cfg(feature = "arrow")]
pub mod arrow_compat;

#[cfg(feature = "polars")]
pub mod polars_compat;

pub use error::{
    ArimaError, BoxCoxError, HoltWintersError, LoessError, SeasonalDecomposeError, StlError,
};
pub use smoothing::{loess, loess_at};
pub use transforms::{
    box_cox, center, inv_box_cox, min_max_scale, z_score, BoxCoxOutput, Lambda,
};
pub use tsa::{
    arima, arima_with_exog, auto_arima, holt_winters, seasonal_decompose, stl, ArimaFit,
    ArimaMethod, ArimaOpts, AutoArimaOpts, DecomposeMode, Decomposition, ForecastResult,
    HoltWintersOpts, Missing, SeasonalDecomposeOpts, SeasonalWindow, StlOpts,
};
