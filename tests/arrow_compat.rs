//! Tests for the optional `arrow` feature. Compiles to nothing when the
//! feature is off so `cargo test` stays clean for users who haven't opted in.

#![cfg(feature = "arrow")]

use std::sync::Arc;

use arrow::array::{Array, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use rust_stats::arrow_compat::{
    self, loess, loess_batch, seasonal_decompose, seasonal_decompose_batch, stl, stl_batch,
    ArrowError,
};
use rust_stats::{DecomposeMode, SeasonalDecomposeOpts, StlOpts};

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

#[test]
fn module_surface() {
    let _: fn(&Float64Array, f64, u8) -> Result<Float64Array, ArrowError> = arrow_compat::loess;
    let _: fn(&Float64Array, StlOpts) -> Result<RecordBatch, ArrowError> = arrow_compat::stl;
    let _: fn(&Float64Array, SeasonalDecomposeOpts) -> Result<RecordBatch, ArrowError> =
        arrow_compat::seasonal_decompose;
}

// ── Batched (multi-column) adapters ────────────────────────────────────

fn series_with_seasonality(n: usize, period: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) - 0.5
    };
    (0..n)
        .map(|i| {
            let t = i as f64;
            let ph = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            10.0 + 0.05 * t + 3.0 * ph.sin() + 0.5 * next()
        })
        .collect()
}

fn floatcol(batch: &RecordBatch, j: usize) -> &Float64Array {
    batch.column(j).as_any().downcast_ref::<Float64Array>().unwrap()
}

fn batch_of_series(names: &[&str], series: Vec<Vec<f64>>) -> RecordBatch {
    assert_eq!(names.len(), series.len());
    let mut fields = Vec::with_capacity(names.len());
    let mut cols: Vec<Arc<dyn Array>> = Vec::with_capacity(names.len());
    for (name, s) in names.iter().zip(series.into_iter()) {
        fields.push(Field::new(*name, DataType::Float64, true));
        cols.push(Arc::new(Float64Array::from(s)));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

#[test]
fn loess_batch_matches_scalar_column_by_column() {
    let n = 200;
    let series: Vec<Vec<f64>> = (0..5)
        .map(|s| series_with_seasonality(n, 12, 1234 + s as u64))
        .collect();
    let names = ["s0", "s1", "s2", "s3", "s4"];
    let batch = batch_of_series(&names, series.clone());

    let out_batch = loess_batch(&batch, 0.3, 1).unwrap();
    assert_eq!(out_batch.schema(), batch.schema());
    assert_eq!(out_batch.num_rows(), n);

    for j in 0..5 {
        let scalar = rust_stats::smoothing::loess(&series[j], 0.3, 1).unwrap();
        let arr = floatcol(&out_batch, j);
        for i in 0..n {
            assert!(
                (arr.value(i) - scalar[i]).abs() < 1e-12,
                "col {j}, row {i}: batch={} scalar={}",
                arr.value(i),
                scalar[i]
            );
        }
    }
}

#[test]
fn stl_batch_matches_scalar_column_by_column() {
    let n = 144;
    let period = 12u32;
    let series: Vec<Vec<f64>> = (0..3)
        .map(|s| series_with_seasonality(n, period as usize, 9000 + s as u64))
        .collect();
    let names = ["a", "b", "c"];
    let batch = batch_of_series(&names, series.clone());

    let res = stl_batch(&batch, StlOpts::new(period)).unwrap();
    assert_eq!(res.trend.schema(),    batch.schema());
    assert_eq!(res.seasonal.schema(), batch.schema());
    assert_eq!(res.residual.schema(), batch.schema());

    for j in 0..3 {
        let d = rust_stats::tsa::stl(&series[j], StlOpts::new(period)).unwrap();
        let t = floatcol(&res.trend,    j);
        let s = floatcol(&res.seasonal, j);
        let r = floatcol(&res.residual, j);
        for i in 0..n {
            assert!((t.value(i) - d.trend[i]).abs()    < 1e-12);
            assert!((s.value(i) - d.seasonal[i]).abs() < 1e-12);
            assert!((r.value(i) - d.residual[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn seasonal_decompose_batch_preserves_nan_edges_per_column() {
    let period = 4u32;
    let half = (period as usize) / 2;
    let n = 24usize;
    let s1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let s2: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let batch = batch_of_series(&["s1", "s2"], vec![s1.clone(), s2.clone()]);

    let mut opts = SeasonalDecomposeOpts::new(period);
    opts.mode = DecomposeMode::Additive;
    let res = seasonal_decompose_batch(&batch, opts).unwrap();

    for j in 0..2 {
        let t = floatcol(&res.trend, j);
        let r = floatcol(&res.residual, j);
        for i in 0..half {
            assert!(t.value(i).is_nan() && r.value(i).is_nan(), "col {j} edge {i}");
        }
        for i in (n - half)..n {
            assert!(t.value(i).is_nan() && r.value(i).is_nan(), "col {j} edge {i}");
        }
    }
}

#[test]
fn batched_fns_reject_nulls_in_any_column() {
    let n = 144;
    let good = series_with_seasonality(n, 12, 7);
    let mut maybe: Vec<Option<f64>> = good.iter().map(|v| Some(*v)).collect();
    maybe[10] = None;

    let schema = Arc::new(Schema::new(vec![
        Field::new("good",   DataType::Float64, true),
        Field::new("hasnan", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(good)),
            Arc::new(Float64Array::from(maybe)),
        ],
    )
    .unwrap();

    assert!(matches!(
        loess_batch(&batch, 0.3, 1),
        Err(ArrowError::HasNulls { col, .. }) if col == "hasnan"
    ));
    assert!(matches!(
        stl_batch(&batch, StlOpts::new(12)),
        Err(ArrowError::HasNulls { col, .. }) if col == "hasnan"
    ));
}

#[test]
fn batched_fns_reject_wrong_column_type() {
    let n = 144;
    let f = series_with_seasonality(n, 12, 11);
    let schema = Arc::new(Schema::new(vec![
        Field::new("good", DataType::Float64, true),
        Field::new("ints", DataType::Int64,   true),
    ]));
    let ints: arrow::array::Int64Array = (0..n as i64).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(f)), Arc::new(ints)],
    )
    .unwrap();

    assert!(matches!(
        loess_batch(&batch, 0.3, 1),
        Err(ArrowError::WrongType { col, .. }) if col == "ints"
    ));
}
