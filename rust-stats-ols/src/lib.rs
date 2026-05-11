//! rust-stats-ols: ordinary least squares regression.
//!
//! Column-pivoted QR via faer; classical and HC0–HC3 heteroskedasticity-
//! consistent covariance; predictions with 95% prediction intervals;
//! rank-deficient inputs error rather than silently pseudo-invert.
//!
//! Shares distribution helpers (t_cdf, t_quantile, ...) with the sibling
//! crate `rust-stats`.

pub mod error;
pub mod regression;

#[cfg(feature = "arrow")]
pub mod arrow_compat;

pub use faer::{Mat as Matrix, MatRef as Block, MatRef as SubMatrix};

pub use error::OlsError;
pub use regression::{CovType, Inference, Ols, OlsResults};
