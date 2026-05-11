//! Apache Arrow adapter for rust-stats-ols. Enabled with the `arrow`
//! feature.
//!
//! Thin wrapper that unpacks a `Float64Array` (y) and a `RecordBatch`
//! of `Float64` feature columns (x) into the borrowed-slice / `Matrix`
//! forms `Ols::new` consumes, fits the model, and returns `OlsResults`.

use arrow::array::{Array, Float64Array, RecordBatch};
use arrow::datatypes::DataType;

use crate::error::OlsError;
use crate::{Matrix, Ols, OlsResults};

#[derive(Debug, thiserror::Error)]
pub enum ArrowError {
    #[error("column '{col}' has {nulls} nulls; rust-stats-ols requires non-null inputs")]
    HasNulls { col: String, nulls: usize },
    #[error("column '{col}' has type {got}; expected Float64")]
    WrongType { col: String, got: DataType },
    #[error("y length {ny} does not match x rows {nx}")]
    LengthMismatch { ny: usize, nx: usize },
    #[error(transparent)]
    Ols(#[from] OlsError),
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
/// intercept is auto-prepended; coefficient names become
/// `["(Intercept)", <field-names>...]`.
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
