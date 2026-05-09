//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS) and LOESS-based smoothing /
//! seasonal decomposition. See `regression::Ols`, `smoothing::loess`,
//! `tsa::seasonal::stl`, and `tsa::seasonal::seasonal_decompose`.

pub mod distributions;
pub mod error;
pub mod regression;
pub mod smoothing;
pub mod tsa;

pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue, z_quantile, z_two_sided_pvalue};
pub use error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
pub use regression::{CovType, Inference, Ols, OlsResults};
