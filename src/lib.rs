//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS). See `regression::Ols`.

pub mod error;
pub mod distributions;
pub mod regression;

pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};
pub use error::OlsError;
pub use regression::{CovType, Inference, Ols, OlsResults};
