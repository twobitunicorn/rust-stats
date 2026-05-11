//! Time-series analysis. Currently: ARIMA, seasonal decomposition, and
//! Holt-Winters exponential smoothing.

pub mod arima;
pub mod holt_winters;
pub mod seasonal;

pub use arima::{arima, arima_with_exog, ArimaFit, ArimaMethod, ArimaOpts, ForecastResult};
pub use holt_winters::{holt_winters, HoltWintersOpts};
pub use seasonal::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
    SeasonalWindow, StlOpts,
};
