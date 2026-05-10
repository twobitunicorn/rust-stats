//! rust-stats: pure-Rust statistical modeling.
//!
//! v1 ships ordinary least squares (OLS) and LOESS-based smoothing /
//! seasonal decomposition. See `regression::Ols`, `smoothing::loess`,
//! `tsa::seasonal::stl`, and `tsa::seasonal::seasonal_decompose`.
//!
//! Linear algebra (dense matrix type, column-pivoted QR, triangular solves)
//! comes from the `faer` crate, which is re-exported below so callers don't
//! need a direct `faer` dependency.

pub mod distributions;
pub mod error;
pub mod regression;
pub mod smoothing;
pub mod tsa;

// Re-export the matrix types so downstream crates don't need a direct
// faer dep just to call us.
pub use faer::{Mat as Matrix, MatRef as MatrixView};

pub use distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue, z_quantile, z_two_sided_pvalue};
pub use error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
pub use regression::{CovType, Inference, Ols, OlsResults};
pub use smoothing::{loess, loess_at};
pub use tsa::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, SeasonalDecomposeOpts, StlOpts,
};
