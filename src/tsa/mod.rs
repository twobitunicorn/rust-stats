//! Time-series analysis. Currently: seasonal decomposition.

pub mod seasonal;

pub use seasonal::{
    seasonal_decompose, stl, DecomposeMode, Decomposition, Missing, SeasonalDecomposeOpts,
    SeasonalWindow, StlOpts,
};
