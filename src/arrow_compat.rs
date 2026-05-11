//! Apache Arrow adapters for rust-stats. Enabled with the `arrow` feature.
//!
//! Thin wrappers that unpack `Float64Array` / `RecordBatch` into the
//! borrowed-slice and `Matrix<f64>` forms the core API uses, then
//! repackage the outputs as Arrow. The point is interop with Polars,
//! DataFusion, DuckDB, PyArrow, and Parquet — not a performance win.
//!
//! Null policy: any null in an input array returns `ArrowError::HasNulls`.
//! Use `arrow::compute::filter` or Polars' `drop_nulls` upstream if you
//! want statsmodels-style `missing='drop'` semantics.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use rayon::prelude::*;

use crate::error::{LoessError, OlsError, SeasonalDecomposeError, StlError};
use crate::smoothing::loess as loess_core;
use crate::tsa::{
    seasonal_decompose as sd_core, stl as stl_core, Decomposition,
    SeasonalDecomposeOpts, StlOpts,
};
use crate::{Matrix, Ols, OlsResults};

#[derive(Debug, thiserror::Error)]
pub enum ArrowError {
    #[error("column '{col}' has {nulls} nulls; rust-stats requires non-null inputs")]
    HasNulls { col: String, nulls: usize },
    #[error("column '{col}' has type {got}; expected Float64")]
    WrongType { col: String, got: DataType },
    #[error("y length {ny} does not match x rows {nx}")]
    LengthMismatch { ny: usize, nx: usize },
    #[error(transparent)]
    Ols(#[from] OlsError),
    #[error(transparent)]
    Loess(#[from] LoessError),
    #[error(transparent)]
    Stl(#[from] StlError),
    #[error(transparent)]
    SeasonalDecompose(#[from] SeasonalDecomposeError),
}

fn as_slice<'a>(arr: &'a Float64Array, name: &str) -> Result<&'a [f64], ArrowError> {
    if arr.null_count() > 0 {
        return Err(ArrowError::HasNulls {
            col: name.into(),
            nulls: arr.null_count(),
        });
    }
    Ok(arr.values())
}

fn float_col<'a>(batch: &'a RecordBatch, j: usize) -> Result<&'a Float64Array, ArrowError> {
    let field = batch.schema().field(j).clone();
    let arr = batch.column(j);
    arr.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(ArrowError::WrongType {
            col: field.name().clone(),
            got: arr.data_type().clone(),
        })
}

/// Pack a RecordBatch of Float64 columns into a column-major `Matrix<f64>`.
fn batch_to_matrix(batch: &RecordBatch) -> Result<Matrix<f64>, ArrowError> {
    let n = batch.num_rows();
    let p = batch.num_columns();
    let mut cols: Vec<&[f64]> = Vec::with_capacity(p);
    for j in 0..p {
        let arr = float_col(batch, j)?;
        cols.push(as_slice(arr, batch.schema().field(j).name())?);
    }
    Ok(Matrix::from_fn(n, p, |i, j| cols[j][i]))
}

/// Fit OLS where `x` is a `RecordBatch` of `Float64` feature columns. An
/// intercept is auto-prepended; coefficient names become `["(Intercept)",
/// <field-names>...]`.
pub fn fit_ols(y: &Float64Array, x: &RecordBatch) -> Result<OlsResults, ArrowError> {
    if y.len() != x.num_rows() {
        return Err(ArrowError::LengthMismatch {
            ny: y.len(),
            nx: x.num_rows(),
        });
    }
    let y_slice = as_slice(y, "y")?;
    let x_mat = batch_to_matrix(x)?;
    let mut names: Vec<String> = Vec::with_capacity(x.num_columns() + 1);
    names.push("(Intercept)".to_string());
    for f in x.schema().fields() {
        names.push(f.name().clone());
    }
    Ok(Ols::new(y_slice, x_mat.as_ref()).fit()?.with_names(names))
}

/// LOESS on an Arrow `Float64Array`. Output array length equals input length.
pub fn loess(y: &Float64Array, span: f64, degree: u8) -> Result<Float64Array, ArrowError> {
    let out = loess_core(as_slice(y, "y")?, span, degree)?;
    Ok(Float64Array::from(out))
}

fn decomposition_to_batch(d: Decomposition) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("trend",    DataType::Float64, true),
        Field::new("seasonal", DataType::Float64, true),
        Field::new("residual", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(d.trend)),
        Arc::new(Float64Array::from(d.seasonal)),
        Arc::new(Float64Array::from(d.residual)),
    ];
    RecordBatch::try_new(schema, cols).expect("schema/columns match by construction")
}

/// Cleveland 1990 STL on an Arrow series. Returns a `RecordBatch` with
/// `trend`, `seasonal`, `residual` columns.
pub fn stl(y: &Float64Array, opts: StlOpts) -> Result<RecordBatch, ArrowError> {
    Ok(decomposition_to_batch(stl_core(as_slice(y, "y")?, opts)?))
}

/// Classical seasonal_decompose on an Arrow series. Returns a `RecordBatch`
/// with `trend`, `seasonal`, `residual` columns; the first/last `period/2`
/// rows of `trend`/`residual` are NaN (encoded as Arrow NaN, not nulls).
pub fn seasonal_decompose(
    y: &Float64Array,
    opts: SeasonalDecomposeOpts,
) -> Result<RecordBatch, ArrowError> {
    Ok(decomposition_to_batch(sd_core(as_slice(y, "y")?, opts)?))
}

// ── Batched (multi-column) adapters ─────────────────────────────────────
//
// All three functions take a `RecordBatch` of `Float64` series and apply
// the same operation to every column. The output schema matches the
// input (same field names, same order). Validation is performed up
// front for the whole batch, so a malformed column fails fast before
// any compute starts.
//
// `loess_batch` for degree 0/1 runs through a `pulp`-dispatched
// cross-column SIMD kernel — see `crate::smoothing::loess_batch`. The
// tricube weights and `Σw·dx^r` moments depend only on the shared
// x-grid, so one scalar pass per output point feeds an L-wide SIMD
// accumulator for the per-column `Σw·y` and `Σw·dx·y`, with the 2×2
// normal-equation solve broadcast across lanes. The scalar fallback is
// the same kernel via `pulp`'s `f64s = f64` impl — one source.
// `stl_batch` and `seasonal_decompose_batch` still parallelise
// per-column with rayon; STL's inner LOESS is the natural next
// candidate but isn't done yet.

/// Output of a batched seasonal-trend decomposition. Each component is a
/// `RecordBatch` with the same schema as the input — column `j` of `trend`
/// is the trend of input column `j`, and similarly for `seasonal` /
/// `residual`.
#[derive(Debug, Clone)]
pub struct DecompositionBatch {
    pub trend:    RecordBatch,
    pub seasonal: RecordBatch,
    pub residual: RecordBatch,
}

/// Validate that every column is `Float64` and null-free, returning the
/// per-column borrowed slices in input order.
fn validated_columns<'a>(batch: &'a RecordBatch) -> Result<Vec<&'a [f64]>, ArrowError> {
    let p = batch.num_columns();
    let mut out = Vec::with_capacity(p);
    for j in 0..p {
        let arr = float_col(batch, j)?;
        out.push(as_slice(arr, batch.schema().field(j).name())?);
    }
    Ok(out)
}

/// Build a `RecordBatch` from `cols` using the supplied schema.
fn batch_from_columns(schema: SchemaRef, cols: Vec<Vec<f64>>) -> RecordBatch {
    let arrays: Vec<ArrayRef> =
        cols.into_iter().map(|c| Arc::new(Float64Array::from(c)) as ArrayRef).collect();
    RecordBatch::try_new(schema, arrays).expect("schema/columns match by construction")
}

/// LOESS over every column of `batch`. Returns a `RecordBatch` with the
/// same schema, where each column is the smoothed input.
///
/// For `degree` 0 and 1 this uses a `pulp`-dispatched cross-column SIMD
/// kernel (`crate::smoothing::loess_batch::loess_batch_simd`); for
/// `degree=2` it falls back to per-column scalar LOESS parallelised
/// across columns with rayon.
pub fn loess_batch(
    batch: &RecordBatch,
    span: f64,
    degree: u8,
) -> Result<RecordBatch, ArrowError> {
    let cols = validated_columns(batch)?;
    let n = batch.num_rows();

    if degree <= 1 {
        let mut out: Vec<Vec<f64>> = (0..cols.len()).map(|_| vec![0.0; n]).collect();
        crate::smoothing::loess_batch::loess_batch_simd(&cols, span, degree, &mut out)?;
        return Ok(batch_from_columns(batch.schema(), out));
    }

    let smoothed: Result<Vec<Vec<f64>>, LoessError> =
        cols.par_iter().map(|c| loess_core(c, span, degree)).collect();
    Ok(batch_from_columns(batch.schema(), smoothed?))
}

fn empty_like(schema: SchemaRef, n: usize) -> Vec<Vec<f64>> {
    (0..schema.fields().len()).map(|_| vec![0.0; n]).collect()
}

/// STL over every column of `batch`. Returns `DecompositionBatch` whose
/// three fields share the input schema; column `j` of each field is the
/// decomposition of input column `j`.
pub fn stl_batch(
    batch: &RecordBatch,
    opts: StlOpts,
) -> Result<DecompositionBatch, ArrowError> {
    let cols = validated_columns(batch)?;
    let parts: Result<Vec<Decomposition>, StlError> =
        cols.par_iter().map(|c| stl_core(c, opts.clone())).collect();
    let parts = parts?;
    let n = batch.num_rows();
    let mut trend    = empty_like(batch.schema(), n);
    let mut seasonal = empty_like(batch.schema(), n);
    let mut residual = empty_like(batch.schema(), n);
    for (j, d) in parts.into_iter().enumerate() {
        trend[j]    = d.trend;
        seasonal[j] = d.seasonal;
        residual[j] = d.residual;
    }
    Ok(DecompositionBatch {
        trend:    batch_from_columns(batch.schema(), trend),
        seasonal: batch_from_columns(batch.schema(), seasonal),
        residual: batch_from_columns(batch.schema(), residual),
    })
}

/// Classical seasonal_decompose over every column of `batch`. Returns
/// `DecompositionBatch` whose three fields share the input schema. NaN
/// edges (first/last `period/2` rows of trend and residual) are preserved
/// per column.
pub fn seasonal_decompose_batch(
    batch: &RecordBatch,
    opts: SeasonalDecomposeOpts,
) -> Result<DecompositionBatch, ArrowError> {
    let cols = validated_columns(batch)?;
    let parts: Result<Vec<Decomposition>, SeasonalDecomposeError> =
        cols.par_iter().map(|c| sd_core(c, opts.clone())).collect();
    let parts = parts?;
    let n = batch.num_rows();
    let mut trend    = empty_like(batch.schema(), n);
    let mut seasonal = empty_like(batch.schema(), n);
    let mut residual = empty_like(batch.schema(), n);
    for (j, d) in parts.into_iter().enumerate() {
        trend[j]    = d.trend;
        seasonal[j] = d.seasonal;
        residual[j] = d.residual;
    }
    Ok(DecompositionBatch {
        trend:    batch_from_columns(batch.schema(), trend),
        seasonal: batch_from_columns(batch.schema(), seasonal),
        residual: batch_from_columns(batch.schema(), residual),
    })
}
