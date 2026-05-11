//! Tests for the optional `polars` feature.

#![cfg(feature = "polars")]

use polars::prelude::*;

use rust_stats::polars_compat::{
    self, loess, loess_batch, seasonal_decompose, seasonal_decompose_batch, stl, stl_batch,
    PolarsCompatError,
};
use rust_stats::{DecomposeMode, SeasonalDecomposeOpts, StlOpts};

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

fn float_col<'a>(df: &'a DataFrame, name: &str) -> &'a Float64Chunked {
    df.column(name).unwrap().f64().unwrap()
}

// ── LOESS ───────────────────────────────────────────────────────────────

#[test]
fn loess_matches_slice_api() {
    let n = 50;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let s = Series::new("y".into(), y.clone());

    let out = loess(&s, 0.5, 1).unwrap();
    let scalar = rust_stats::smoothing::loess(&y, 0.5, 1).unwrap();

    assert_eq!(out.name().as_str(), "y");
    let ca = out.f64().unwrap();
    for (i, v) in ca.into_no_null_iter().enumerate() {
        assert!((v - scalar[i]).abs() < 1e-12);
    }
}

#[test]
fn loess_rejects_nulls() {
    let mut vals: Vec<Option<f64>> = (0..20).map(|i| Some(i as f64)).collect();
    vals[7] = None;
    let s = Series::new("y".into(), vals);
    let err = loess(&s, 0.5, 1).unwrap_err();
    assert!(matches!(err, PolarsCompatError::HasNulls { .. }));
}

#[test]
fn loess_rejects_non_float64() {
    let s = Series::new("y".into(), (0..10i64).collect::<Vec<_>>());
    let err = loess(&s, 0.5, 1).unwrap_err();
    assert!(matches!(err, PolarsCompatError::NotFloat64(name) if name == "y"));
}

// ── STL ─────────────────────────────────────────────────────────────────

#[test]
fn stl_returns_dataframe_with_expected_schema() {
    let period = 4u32;
    let n = 32usize;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let s = [1.0, 2.0, 3.0, 2.0][i % 4];
            10.0 + 0.1 * i as f64 + s
        })
        .collect();
    let series = Series::new("y".into(), y.clone());

    let df = stl(&series, StlOpts::new(period)).unwrap();
    assert_eq!(df.height(), n);
    let names: Vec<&str> = df.columns().iter().map(|c| c.name().as_str()).collect();
    assert_eq!(names, vec!["trend", "seasonal", "residual"]);

    let trend    = float_col(&df, "trend");
    let seasonal = float_col(&df, "seasonal");
    let residual = float_col(&df, "residual");
    for i in 0..n {
        let recon = trend.get(i).unwrap()
            + seasonal.get(i).unwrap()
            + residual.get(i).unwrap();
        assert!((recon - y[i]).abs() < 1e-9);
    }
}

// ── seasonal_decompose ──────────────────────────────────────────────────

#[test]
fn seasonal_decompose_preserves_nan_edges() {
    let period = 4u32;
    let half = (period as usize) / 2;
    let n = 24usize;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let series = Series::new("y".into(), y);

    let mut opts = SeasonalDecomposeOpts::new(period);
    opts.mode = DecomposeMode::Additive;
    let df = seasonal_decompose(&series, opts).unwrap();
    assert_eq!(df.height(), n);

    let trend = float_col(&df, "trend");
    let residual = float_col(&df, "residual");
    for i in 0..half {
        assert!(trend.get(i).unwrap().is_nan());
        assert!(residual.get(i).unwrap().is_nan());
    }
    for i in (n - half)..n {
        assert!(trend.get(i).unwrap().is_nan());
        assert!(residual.get(i).unwrap().is_nan());
    }
}

// ── Batched ─────────────────────────────────────────────────────────────

fn df_of_series(names: &[&str], series: Vec<Vec<f64>>) -> DataFrame {
    let columns: Vec<Column> = names
        .iter()
        .zip(series.into_iter())
        .map(|(name, vals)| Series::new((*name).into(), vals).into_column())
        .collect();
    let h = columns[0].len();
    DataFrame::new(h, columns).unwrap()
}

#[test]
fn loess_batch_matches_scalar_column_by_column() {
    let n = 200;
    let series: Vec<Vec<f64>> = (0..5)
        .map(|s| series_with_seasonality(n, 12, 1234 + s as u64))
        .collect();
    let names = ["s0", "s1", "s2", "s3", "s4"];
    let df = df_of_series(&names, series.clone());

    let out_df = loess_batch(&df, 0.3, 1).unwrap();
    assert_eq!(out_df.height(), n);
    for (j, name) in names.iter().enumerate() {
        let scalar = rust_stats::smoothing::loess(&series[j], 0.3, 1).unwrap();
        let ca = float_col(&out_df, name);
        for (i, v) in ca.into_no_null_iter().enumerate() {
            assert!(
                (v - scalar[i]).abs() < 1e-12,
                "col {name}, row {i}: batch={v} scalar={}", scalar[i]
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
    let df = df_of_series(&names, series.clone());

    let res = stl_batch(&df, StlOpts::new(period)).unwrap();
    for (j, name) in names.iter().enumerate() {
        let d = rust_stats::tsa::stl(&series[j], StlOpts::new(period)).unwrap();
        let t = float_col(&res.trend, name);
        let s = float_col(&res.seasonal, name);
        let r = float_col(&res.residual, name);
        for i in 0..n {
            assert!((t.get(i).unwrap() - d.trend[i]).abs()    < 1e-12);
            assert!((s.get(i).unwrap() - d.seasonal[i]).abs() < 1e-12);
            assert!((r.get(i).unwrap() - d.residual[i]).abs() < 1e-12);
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
    let df = df_of_series(&["s1", "s2"], vec![s1, s2]);

    let mut opts = SeasonalDecomposeOpts::new(period);
    opts.mode = DecomposeMode::Additive;
    let res = seasonal_decompose_batch(&df, opts).unwrap();

    for name in ["s1", "s2"] {
        let t = float_col(&res.trend, name);
        let r = float_col(&res.residual, name);
        for i in 0..half {
            assert!(t.get(i).unwrap().is_nan() && r.get(i).unwrap().is_nan());
        }
        for i in (n - half)..n {
            assert!(t.get(i).unwrap().is_nan() && r.get(i).unwrap().is_nan());
        }
    }
}

#[test]
fn batched_rejects_nulls_in_any_column() {
    let n = 144;
    let good = series_with_seasonality(n, 12, 7);
    let mut maybe: Vec<Option<f64>> = good.iter().map(|v| Some(*v)).collect();
    maybe[10] = None;
    let df = DataFrame::new(
        n,
        vec![
            Series::new("good".into(), good).into_column(),
            Series::new("hasnan".into(), maybe).into_column(),
        ],
    )
    .unwrap();

    assert!(matches!(
        loess_batch(&df, 0.3, 1),
        Err(PolarsCompatError::HasNulls { col, .. }) if col == "hasnan"
    ));
    assert!(matches!(
        stl_batch(&df, StlOpts::new(12)),
        Err(PolarsCompatError::HasNulls { col, .. }) if col == "hasnan"
    ));
}

#[test]
fn module_surface() {
    let _: fn(&Series, f64, u8) -> Result<Series, PolarsCompatError> = polars_compat::loess;
    let _: fn(&Series, StlOpts) -> Result<DataFrame, PolarsCompatError> = polars_compat::stl;
    let _: fn(&Series, SeasonalDecomposeOpts) -> Result<DataFrame, PolarsCompatError> =
        polars_compat::seasonal_decompose;
}
