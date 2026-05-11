//! Time-series analysis. Currently: ARIMA, seasonal decomposition, and
//! Holt-Winters exponential smoothing.

pub mod arima;
pub mod diagnostics;
pub mod holt_winters;
pub mod seasonal;
pub mod stationarity;

pub use arima::{
    arima, arima_with_exog, auto_arima, ArimaFit, ArimaMethod, ArimaOpts, AutoArimaOpts,
    ForecastResult,
};
pub use holt_winters::{holt_winters, HoltWintersOpts};
pub use seasonal::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
    SeasonalWindow, StlOpts,
};
