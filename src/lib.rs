//! rust-stats: pure-Rust statistical modeling.
//!
//! Smoothing and seasonal-trend decomposition: LOESS, Cleveland 1990 STL,
//! and classical seasonal_decompose.

pub mod error;
pub mod smoothing;
pub mod tsa;

#[cfg(feature = "arrow")]
pub mod arrow_compat;

pub use error::{LoessError, SeasonalDecomposeError, StlError};
pub use smoothing::{loess, loess_at};
pub use tsa::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
    SeasonalWindow, StlOpts,
};
