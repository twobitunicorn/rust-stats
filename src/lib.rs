//! rust-stats: pure-Rust statistical modeling.
//!
//! This crate covers smoothing and seasonal-trend decomposition: LOESS,
//! Cleveland 1990 STL, and classical seasonal_decompose. OLS regression
//! lives in the sibling crate `rust-stats-ols`.
//!
//! Linear algebra (dense matrix type, column-pivoted QR, triangular solves)
//! comes from the `faer` crate, which is re-exported below so callers don't
//! need a direct `faer` dependency.

pub mod distributions;
pub mod error;
pub mod smoothing;
pub mod tsa;

#[cfg(feature = "arrow")]
pub mod arrow_compat;

pub use faer::{Mat as Matrix, MatRef as Block, MatRef as SubMatrix};

pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue, z_quantile, z_two_sided_pvalue};
pub use error::{LoessError, SeasonalDecomposeError, StlError};
pub use smoothing::{loess, loess_at};
pub use tsa::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, SeasonalDecomposeOpts, StlOpts,
};
