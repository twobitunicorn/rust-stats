//! Smoothing — currently LOESS.

pub mod loess;

pub use loess::{loess, loess_at};
