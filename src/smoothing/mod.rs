//! Smoothing — currently LOESS.

pub mod loess;

#[cfg(feature = "arrow")]
pub(crate) mod loess_batch;

pub use loess::{loess, loess_at};
