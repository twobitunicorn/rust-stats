//! Arrow-feature tests for rust-stats-ols.

#![cfg(feature = "arrow")]

use std::sync::Arc;
use std::path::PathBuf;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use serde::Deserialize;

use rust_stats_ols::arrow_compat::{self, fit_ols, ArrowError};
use rust_stats_ols::CovType;

#[derive(Deserialize)]
struct OlsGolden {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    coef: Vec<f64>,
    r_squared: f64,
}

fn load_ols(name: &str) -> OlsGolden {
    let path: PathBuf = ["tests", "golden", &format!("{name}.json")].iter().collect();
    let bytes = std::fs::read(&path).expect("missing golden — run tests/golden/generate.py");
    serde_json::from_slice(&bytes).expect("invalid JSON")
}

fn batch_from_rows(rows: &[Vec<f64>], names: &[&str]) -> RecordBatch {
    let n = rows.len();
    let p = rows[0].len();
    assert_eq!(p, names.len());
    let mut cols: Vec<Arc<dyn Array>> = Vec::with_capacity(p);
    let mut fields = Vec::with_capacity(p);
    for j in 0..p {
        let col: Float64Array = (0..n).map(|i| rows[i][j]).collect();
        cols.push(Arc::new(col));
        fields.push(Field::new(names[j], DataType::Float64, true));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

#[test]
fn ols_matches_slice_api_on_longley() {
    let g = load_ols("longley");
    let y_arr = Float64Array::from(g.y.clone());
    let names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"];
    let x_batch = batch_from_rows(&g.x, &names);

    let res = fit_ols(&y_arr, &x_batch).unwrap();

    for i in 0..res.coef().len() {
        let denom = g.coef[i].abs().max(1.0);
        let rel = (res.coef()[i] - g.coef[i]).abs() / denom;
        assert!(rel < 1e-9, "coef[{i}]: {} vs {} (rel {rel})", res.coef()[i], g.coef[i]);
    }
    assert!((res.r_squared() - g.r_squared).abs() < 1e-10);

    let expected_names = [
        "(Intercept)", "GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR",
    ];
    let got_names = res.names().expect("names should be set");
    for i in 0..expected_names.len() {
        assert_eq!(got_names[i], expected_names[i]);
    }

    let inf = res.inference(CovType::HC3);
    assert_eq!(inf.std_err.len(), 7);
}

#[test]
fn ols_rejects_nulls() {
    let g = load_ols("longley");
    let mut y_vals: Vec<Option<f64>> = g.y.iter().map(|v| Some(*v)).collect();
    y_vals[3] = None;
    let y_arr = Float64Array::from(y_vals);
    let names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"];
    let x_batch = batch_from_rows(&g.x, &names);

    let err = fit_ols(&y_arr, &x_batch).unwrap_err();
    assert!(matches!(err, ArrowError::HasNulls { col, nulls: 1 } if col == "y"));
}

#[test]
fn ols_rejects_wrong_column_type() {
    let g = load_ols("longley");
    let y_arr = Float64Array::from(g.y.clone());

    let n = g.x.len();
    let int_col: Int64Array = (0..n as i64).collect();
    let f64_col: Float64Array = (0..n).map(|i| g.x[i][1]).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("bad", DataType::Int64, true),
        Field::new("ok",  DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(int_col), Arc::new(f64_col)],
    )
    .unwrap();

    let err = fit_ols(&y_arr, &batch).unwrap_err();
    assert!(matches!(err, ArrowError::WrongType { col, .. } if col == "bad"));
}

#[test]
fn ols_rejects_length_mismatch() {
    let g = load_ols("longley");
    let y_arr = Float64Array::from(vec![1.0; 5]);
    let names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"];
    let x_batch = batch_from_rows(&g.x, &names);

    let err = fit_ols(&y_arr, &x_batch).unwrap_err();
    assert!(matches!(err, ArrowError::LengthMismatch { ny: 5, nx: 16 }));
}

#[test]
fn module_surface() {
    let _: fn(&Float64Array, &RecordBatch)
        -> Result<rust_stats_ols::OlsResults, ArrowError> = arrow_compat::fit_ols;
}
