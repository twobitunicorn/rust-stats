//! Smoothing — currently LOESS.

pub mod loess;
pub(crate) mod loess_batch;

pub use loess::{loess, loess_at};
