//! Error types for rust-stats.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum OlsError {
    #[error("dimension mismatch: y has {y} rows but X has {x}")]
    DimensionMismatch { y: usize, x: usize },

    #[error("not enough observations: n={n} must exceed p={p}")]
    InsufficientObservations { n: usize, p: usize },

    #[error("rank deficient design matrix: rank {rank} < p {p}")]
    RankDeficient { rank: usize, p: usize },

    #[error("input contains non-finite values")]
    NonFinite,

    #[error("predict X has {got} columns, expected {expected}")]
    NewXShapeMismatch { got: usize, expected: usize },

    #[error("invalid alpha {0}: must be in (0, 1)")]
    InvalidAlpha(f64),
}
