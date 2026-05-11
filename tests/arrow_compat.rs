//! Tests for the optional `arrow` feature. The whole file compiles to
//! nothing when the feature is off so `cargo test` stays clean for users
//! who haven't opted in.

#![cfg(feature = "arrow")]

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use serde::Deserialize;
use std::path::PathBuf;

use rust_stats::arrow_compat::{
    self, fit_ols, loess, seasonal_decompose, stl, ArrowError,
};
use rust_stats::{CovType, DecomposeMode, SeasonalDecomposeOpts, StlOpts};

// ── Shared fixture helpers ──────────────────────────────────────────────

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

// ── OLS ─────────────────────────────────────────────────────────────────

#[test]
fn ols_matches_slice_api_on_longley() {
    let g = load_ols("longley");
    let y_arr = Float64Array::from(g.y.clone());
    let names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"];
    let x_batch = batch_from_rows(&g.x, &names);

    let res = fit_ols(&y_arr, &x_batch).unwrap();

    // Coefficients match the golden to floating-point precision (relative,
    // because the Longley intercept is ~3.5e6 and even ulp-level noise from
    // the Arrow → faer pack vs reading from JSON produces ~1 unit of drift).
    for i in 0..res.coef().len() {
        let denom = g.coef[i].abs().max(1.0);
        let rel = (res.coef()[i] - g.coef[i]).abs() / denom;
        assert!(
            rel < 1e-9,
            "coef[{i}]: {} vs {} (rel {rel})", res.coef()[i], g.coef[i]
        );
    }
    assert!((res.r_squared() - g.r_squared).abs() < 1e-10);

    // Schema field names flow through to the summary table.
    let expected_names = [
        "(Intercept)", "GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR",
    ];
    let got_names = res.names().expect("names should be set");
    for i in 0..expected_names.len() {
        assert_eq!(got_names[i], expected_names[i]);
    }

    // Inference still works through the returned OlsResults.
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

    // Build a batch with an Int64 column where Float64 is expected.
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
    let y_arr = Float64Array::from(vec![1.0; 5]); // wrong length
    let names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"];
    let x_batch = batch_from_rows(&g.x, &names);

    let err = fit_ols(&y_arr, &x_batch).unwrap_err();
    assert!(matches!(err, ArrowError::LengthMismatch { ny: 5, nx: 16 }));
}

// ── LOESS ───────────────────────────────────────────────────────────────

#[test]
fn loess_matches_slice_api() {
    let n = 50;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_arr = Float64Array::from(y.clone());

    let out_arr = loess(&y_arr, 0.5, 1).unwrap();
    let out_slice = rust_stats::smoothing::loess(&y, 0.5, 1).unwrap();

    assert_eq!(out_arr.len(), n);
    for i in 0..n {
        assert!((out_arr.value(i) - out_slice[i]).abs() < 1e-12);
    }
}

#[test]
fn loess_rejects_nulls() {
    let mut vals: Vec<Option<f64>> = (0..20).map(|i| Some(i as f64)).collect();
    vals[7] = None;
    let y_arr = Float64Array::from(vals);
    let err = loess(&y_arr, 0.5, 1).unwrap_err();
    assert!(matches!(err, ArrowError::HasNulls { .. }));
}

// ── STL ─────────────────────────────────────────────────────────────────

#[test]
fn stl_returns_record_batch_with_expected_schema() {
    // Use a simple seasonal series that the core STL test suite also covers.
    let period = 4u32;
    let n = 32usize;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let s = [1.0, 2.0, 3.0, 2.0][i % 4];
            10.0 + 0.1 * t + s
        })
        .collect();
    let y_arr = Float64Array::from(y.clone());

    let batch = stl(&y_arr, StlOpts::new(period)).unwrap();
    assert_eq!(batch.num_rows(), n);
    assert_eq!(batch.num_columns(), 3);
    let schema = batch.schema();
    assert_eq!(schema.field(0).name(), "trend");
    assert_eq!(schema.field(1).name(), "seasonal");
    assert_eq!(schema.field(2).name(), "residual");

    let trend = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    let seas  = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let resid = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..n {
        let recon = trend.value(i) + seas.value(i) + resid.value(i);
        assert!((recon - y[i]).abs() < 1e-9);
    }
}

// ── seasonal_decompose ──────────────────────────────────────────────────

#[test]
fn seasonal_decompose_returns_record_batch_with_nan_edges() {
    let period = 4u32;
    let half = (period as usize) / 2;
    let n = 24usize;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_arr = Float64Array::from(y);

    let mut opts = SeasonalDecomposeOpts::new(period);
    opts.mode = DecomposeMode::Additive;
    let batch = seasonal_decompose(&y_arr, opts).unwrap();
    assert_eq!(batch.num_rows(), n);
    assert_eq!(batch.num_columns(), 3);

    let trend = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    let resid = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..half {
        assert!(trend.value(i).is_nan());
        assert!(resid.value(i).is_nan());
    }
    for i in (n - half)..n {
        assert!(trend.value(i).is_nan());
        assert!(resid.value(i).is_nan());
    }
}

// ── Sanity: the public adapter module re-exports nothing surprising ────

#[test]
fn module_surface() {
    // Forces the symbols we expect to be public to exist.
    let _: fn(&Float64Array, &RecordBatch) -> Result<rust_stats::OlsResults, ArrowError> =
        arrow_compat::fit_ols;
    let _: fn(&Float64Array, f64, u8) -> Result<Float64Array, ArrowError> = arrow_compat::loess;
    let _: fn(&Float64Array, StlOpts) -> Result<RecordBatch, ArrowError> = arrow_compat::stl;
    let _: fn(&Float64Array, SeasonalDecomposeOpts) -> Result<RecordBatch, ArrowError> =
        arrow_compat::seasonal_decompose;
}
