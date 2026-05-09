//! Regression models. v1: OLS only.

pub mod ols;
pub mod results;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
