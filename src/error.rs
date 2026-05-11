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

#[derive(Debug, Error, PartialEq)]
pub enum LoessError {
    #[error("span must be in (0, 1]; got {0}")]
    InvalidSpan(f64),
    #[error("degree must be 0, 1, or 2; got {0}")]
    InvalidDegree(u8),
    #[error("input is empty")]
    Empty,
    #[error("input contains non-finite values")]
    NonFinite,
}

#[derive(Debug, Error, PartialEq)]
pub enum StlError {
    #[error("period must be >= 2; got {0}")]
    InvalidPeriod(u32),
    #[error("seasonal_window must be odd and >= 7; got {0}")]
    InvalidSeasonalWindow(u32),
    #[error("trend_window must be odd; got {0}")]
    InvalidTrendWindow(u32),
    #[error("inner_iters must be >= 1; got 0")]
    InvalidInnerIters,
    #[error("{which}_jump must be >= 1; got 0")]
    InvalidJump { which: &'static str },
    #[error("series too short: needs >= 2*period samples, got {n} < {min}")]
    SeriesTooShort { n: usize, min: usize },
    #[error("multiplicative mode requires strictly positive values; got {min}")]
    NonPositiveForMultiplicative { min: f64 },
    #[error("input contains non-finite values")]
    NonFinite,
    #[error(transparent)]
    Loess(#[from] LoessError),
}

#[derive(Debug, Error, PartialEq)]
pub enum SeasonalDecomposeError {
    #[error("period must be >= 2; got {0}")]
    InvalidPeriod(u32),
    #[error("series too short: needs >= 2*period samples, got {n} < {min}")]
    SeriesTooShort { n: usize, min: usize },
    #[error("multiplicative mode requires strictly positive values; got {min}")]
    NonPositiveForMultiplicative { min: f64 },
    #[error("input contains non-finite values")]
    NonFinite,
}
