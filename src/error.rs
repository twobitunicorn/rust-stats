//! Error types for rust-stats.

use thiserror::Error;

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

#[derive(Debug, Error, PartialEq)]
pub enum BoxCoxError {
    #[error("Box-Cox requires strictly positive values; got minimum {min}")]
    NonPositive { min: f64 },
    #[error("inverse Box-Cox is undefined: 1 + λ·y must be > 0; got y={value}, λ={lambda}")]
    NonInvertible { value: f64, lambda: f64 },
    #[error("Box-Cox λ estimation needs at least {min} observations; got {n}")]
    TooFewObservations { n: usize, min: usize },
    #[error("Guerrero λ estimation needs period >= 2 and a full cycle; got period={0}")]
    InvalidPeriod(usize),
}

#[derive(Debug, Error, PartialEq)]
pub enum ArimaError {
    #[error("invalid order: p={p}, d={d}, q={q}; need p,q in [0, 10] and d in [0, 2]")]
    InvalidOrder { p: u32, d: u32, q: u32 },
    #[error("series too short: need >= {min} observations for ARIMA({p},{d},{q}), got {n}")]
    SeriesTooShort {
        n: usize,
        min: usize,
        p: u32,
        d: u32,
        q: u32,
    },
    #[error("input contains non-finite values")]
    NonFinite,
    #[error("Nelder-Mead failed to converge within {iters} iterations")]
    OptimizationFailed { iters: usize },
    #[error("starting-value regression failed: singular normal-equations matrix")]
    Singular,
}

#[derive(Debug, Error, PartialEq)]
pub enum HoltWintersError {
    #[error("alpha must be in [0, 1]; got {0}")]
    InvalidAlpha(f64),
    #[error("beta must be in [0, 1]; got {0}")]
    InvalidBeta(f64),
    #[error("gamma must be in [0, 1]; got {0}")]
    InvalidGamma(f64),
    #[error("series too short: needs >= {min} samples, got {n}")]
    SeriesTooShort { n: usize, min: usize },
    #[error("multiplicative mode requires strictly positive values; got {min}")]
    NonPositiveForMultiplicative { min: f64 },
    #[error("input contains non-finite values")]
    NonFinite,
}
