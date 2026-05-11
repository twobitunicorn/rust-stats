//! Regression models. v1: OLS only.

pub mod design;
pub mod ols;
pub mod predict;
pub mod results;
pub mod robust;
pub mod summary;

pub use ols::Ols;
pub use results::{CovType, Inference, OlsResults};
