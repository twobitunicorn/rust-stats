//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS). See `regression::Ols`.

pub mod error;
pub mod distributions;
pub mod regression;

pub use error::OlsError;
pub use regression::{CovType, Inference, Ols, OlsResults};
