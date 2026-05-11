//! Polars adapters for rust-stats. Enabled with the `polars` feature.
//!
//! Thin wrappers that pull a `Series`' contiguous `f64` buffer, run the
//! core algorithm on it, and pack the result back as a `Series` or
//! `DataFrame`. The cost is at most one contiguous-slice extraction per
//! call (free for single-chunk no-null Series, one `rechunk` otherwise).
//!
//! Null policy: any null in an input column returns
//! `PolarsCompatError::HasNulls`. Use `Series::drop_nulls` upstream or
//! the `Missing::Interpolate` option for STL / `seasonal_decompose` if
//! you want imputation.

use polars::prelude::*;

use crate::error::{LoessError, SeasonalDecomposeError, StlError};
use crate::smoothing::loess as core_loess;
use crate::tsa::{
    seasonal_decompose as core_sd, stl as core_stl, SeasonalDecomposeOpts, StlOpts,
};

#[derive(Debug, thiserror::Error)]
pub enum PolarsCompatError {
    #[error("column '{0}' is not Float64")]
    NotFloat64(String),
    #[error("column '{col}' has {nulls} null(s); rust-stats requires non-null inputs")]
    HasNulls { col: String, nulls: usize },
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
    #[error(transparent)]
    Loess(#[from] LoessError),
    #[error(transparent)]
    Stl(#[from] StlError),
    #[error(transparent)]
    SeasonalDecompose(#[from] SeasonalDecomposeError),
}

/// Pull a `Series`' contiguous `f64` values into a `Vec<f64>` — copy
/// path because Polars chunked arrays may not be contiguous in memory.
/// Errors if the series isn't Float64 or contains any nulls.
fn series_to_vec(s: &Series) -> Result<Vec<f64>, PolarsCompatError> {
    let ca = s
        .f64()
        .map_err(|_| PolarsCompatError::NotFloat64(s.name().to_string()))?;
    if ca.null_count() > 0 {
        return Err(PolarsCompatError::HasNulls {
            col: s.name().to_string(),
            nulls: ca.null_count(),
        });
    }
    // ChunkedArray::to_vec walks the (possibly multi-chunk) array once.
    // For typical single-chunk Series this is essentially a memcpy.
    Ok(ca.into_no_null_iter().collect())
}

/// LOESS on a Polars `Series`. Output keeps the input's name.
pub fn loess(s: &Series, span: f64, degree: u8) -> Result<Series, PolarsCompatError> {
    let v = series_to_vec(s)?;
    let out = core_loess(&v, span, degree)?;
    Ok(Series::new(s.name().clone(), out))
}

/// Build a 3-column DataFrame {trend, seasonal, residual} from a
/// `Decomposition`.
fn decomp_to_df(d: crate::tsa::Decomposition) -> Result<DataFrame, PolarsCompatError> {
    df![
        "trend"    => d.trend,
        "seasonal" => d.seasonal,
        "residual" => d.residual,
    ]
    .map_err(PolarsCompatError::Polars)
}

/// Cleveland 1990 STL on a Polars `Series`. Returns a `DataFrame` with
/// `trend`, `seasonal`, `residual` columns of length `s.len()`.
pub fn stl(s: &Series, opts: StlOpts) -> Result<DataFrame, PolarsCompatError> {
    let v = series_to_vec(s)?;
    Ok(decomp_to_df(core_stl(&v, opts)?)?)
}

/// Classical seasonal_decompose on a Polars `Series`. Same output shape
/// as `stl`, but `trend` and `residual` have nulls (NaN) at the
/// first/last `period/2` positions where the centred MA can't be
/// computed.
pub fn seasonal_decompose(
    s: &Series,
    opts: SeasonalDecomposeOpts,
) -> Result<DataFrame, PolarsCompatError> {
    let v = series_to_vec(s)?;
    Ok(decomp_to_df(core_sd(&v, opts)?)?)
}

// ── Batched (multi-column) adapters ─────────────────────────────────────

/// Output of a batched seasonal-trend decomposition. Each field is a
/// `DataFrame` with the same schema as the input (one Float64 column
/// per input series).
#[derive(Debug, Clone)]
pub struct PolarsDecompositionBatch {
    pub trend:    DataFrame,
    pub seasonal: DataFrame,
    pub residual: DataFrame,
}

/// Validate that every column is Float64 + null-free, returning the
/// per-column data as `Vec<f64>` in input order. Each column is
/// materialised once.
fn validated_columns(df: &DataFrame) -> Result<Vec<Vec<f64>>, PolarsCompatError> {
    let mut out = Vec::with_capacity(df.width());
    for col in df.columns() {
        out.push(series_to_vec(col.as_materialized_series())?);
    }
    Ok(out)
}

fn rebuild_df(schema_src: &DataFrame, cols: Vec<Vec<f64>>) -> Result<DataFrame, PolarsCompatError> {
    let height = cols.first().map(|c| c.len()).unwrap_or(0);
    let columns: Vec<Column> = schema_src
        .columns()
        .iter()
        .zip(cols.into_iter())
        .map(|(src, vals)| Series::new(src.name().clone(), vals).into_column())
        .collect();
    DataFrame::new(height, columns).map_err(PolarsCompatError::Polars)
}

/// LOESS over every column of `df`. Returns a `DataFrame` with the same
/// schema where each column is the smoothed input. Routes through the
/// shared batched LOESS path when the `arrow` feature is also enabled
/// (so polars+arrow users get the SIMD kernel); otherwise falls back to
/// rayon-over-columns scalar LOESS.
pub fn loess_batch(df: &DataFrame, span: f64, degree: u8) -> Result<DataFrame, PolarsCompatError> {
    let cols = validated_columns(df)?;
    let outs = run_loess_batch(&cols, span, degree)?;
    rebuild_df(df, outs)
}

#[cfg(feature = "arrow")]
fn run_loess_batch(
    cols: &[Vec<f64>],
    span: f64,
    degree: u8,
) -> Result<Vec<Vec<f64>>, PolarsCompatError> {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut outs: Vec<Vec<f64>> = (0..cols.len()).map(|_| vec![0.0; n]).collect();
    let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    crate::smoothing::loess_batch::loess_batch_simd(&col_refs, span, degree, &mut outs)?;
    Ok(outs)
}

#[cfg(not(feature = "arrow"))]
fn run_loess_batch(
    cols: &[Vec<f64>],
    span: f64,
    degree: u8,
) -> Result<Vec<Vec<f64>>, PolarsCompatError> {
    use rayon::prelude::*;
    let results: Result<Vec<Vec<f64>>, LoessError> =
        cols.par_iter().map(|c| core_loess(c, span, degree)).collect();
    Ok(results?)
}

fn allocate_outs(n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..p).map(|_| vec![0.0; n]).collect()
}

/// STL over every column. Returns three same-shape DataFrames in a
/// `PolarsDecompositionBatch`.
pub fn stl_batch(
    df: &DataFrame,
    opts: StlOpts,
) -> Result<PolarsDecompositionBatch, PolarsCompatError> {
    use rayon::prelude::*;
    let cols = validated_columns(df)?;
    let parts: Result<Vec<crate::tsa::Decomposition>, StlError> =
        cols.par_iter().map(|c| core_stl(c, opts.clone())).collect();
    let parts = parts?;
    let n = if cols.is_empty() { 0 } else { cols[0].len() };
    let p = cols.len();
    let mut trend    = allocate_outs(n, p);
    let mut seasonal = allocate_outs(n, p);
    let mut residual = allocate_outs(n, p);
    for (j, d) in parts.into_iter().enumerate() {
        trend[j]    = d.trend;
        seasonal[j] = d.seasonal;
        residual[j] = d.residual;
    }
    Ok(PolarsDecompositionBatch {
        trend:    rebuild_df(df, trend)?,
        seasonal: rebuild_df(df, seasonal)?,
        residual: rebuild_df(df, residual)?,
    })
}

/// Classical seasonal_decompose over every column.
pub fn seasonal_decompose_batch(
    df: &DataFrame,
    opts: SeasonalDecomposeOpts,
) -> Result<PolarsDecompositionBatch, PolarsCompatError> {
    use rayon::prelude::*;
    let cols = validated_columns(df)?;
    let parts: Result<Vec<crate::tsa::Decomposition>, SeasonalDecomposeError> =
        cols.par_iter().map(|c| core_sd(c, opts.clone())).collect();
    let parts = parts?;
    let n = if cols.is_empty() { 0 } else { cols[0].len() };
    let p = cols.len();
    let mut trend    = allocate_outs(n, p);
    let mut seasonal = allocate_outs(n, p);
    let mut residual = allocate_outs(n, p);
    for (j, d) in parts.into_iter().enumerate() {
        trend[j]    = d.trend;
        seasonal[j] = d.seasonal;
        residual[j] = d.residual;
    }
    Ok(PolarsDecompositionBatch {
        trend:    rebuild_df(df, trend)?,
        seasonal: rebuild_df(df, seasonal)?,
        residual: rebuild_df(df, residual)?,
    })
}
