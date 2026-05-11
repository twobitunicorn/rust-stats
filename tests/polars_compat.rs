//! Tests for the optional `polars` feature.

#![cfg(feature = "polars")]

use polars::prelude::*;

use rust_stats::polars_compat::{
    self, loess, loess_batch, seasonal_decompose, seasonal_decompose_batch, stl, stl_batch,
    PolarsCompatError,
};
use rust_stats::{DecomposeMode, Missing, SeasonalDecomposeOpts, StlOpts};

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

fn f64s(s: &Series) -> &Float64Chunked {
    s.f64().unwrap()
}

// ── LOESS ───────────────────────────────────────────────────────────────

#[test]
fn loess_matches_slice_api() {
    let n = 50;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let s = Series::new("y".into(), y.clone());

    let out = loess(&s, 0.5, 1, Missing::Error).unwrap();
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
    let err = loess(&s, 0.5, 1, Missing::Error).unwrap_err();
    assert!(matches!(err, PolarsCompatError::HasNulls { .. }));
}

#[test]
fn loess_rejects_non_float64() {
    let s = Series::new("y".into(), (0..10i64).collect::<Vec<_>>());
    let err = loess(&s, 0.5, 1, Missing::Error).unwrap_err();
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

    let d = stl(&series, StlOpts::new(period)).unwrap();
    assert_eq!(d.trend.len(), n);
    assert_eq!(d.trend.name().as_str(),    "trend");
    assert_eq!(d.seasonal.name().as_str(), "seasonal");
    assert_eq!(d.residual.name().as_str(), "residual");

    let trend    = f64s(&d.trend);
    let seasonal = f64s(&d.seasonal);
    let residual = f64s(&d.residual);
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
    let d = seasonal_decompose(&series, opts).unwrap();
    assert_eq!(d.trend.len(), n);

    let trend = f64s(&d.trend);
    let residual = f64s(&d.residual);
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

    let out_df = loess_batch(&df, 0.3, 1, Missing::Error).unwrap();
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
        loess_batch(&df, 0.3, 1, Missing::Error),
        Err(PolarsCompatError::HasNulls { col, .. }) if col == "hasnan"
    ));
    assert!(matches!(
        stl_batch(&df, StlOpts::new(12)),
        Err(PolarsCompatError::HasNulls { col, .. }) if col == "hasnan"
    ));
}

#[test]
fn module_surface() {
    use rust_stats::polars_compat::PolarsDecomposition;
    let _: fn(&Series, f64, u8, Missing) -> Result<Series, PolarsCompatError> =
        polars_compat::loess;
    let _: fn(&Series, StlOpts) -> Result<PolarsDecomposition, PolarsCompatError> =
        polars_compat::stl;
    let _: fn(&Series, SeasonalDecomposeOpts) -> Result<PolarsDecomposition, PolarsCompatError> =
        polars_compat::seasonal_decompose;
}

#[test]
fn loess_interpolate_handles_polars_nulls() {
    let n = 200;
    let mut y: Vec<Option<f64>> = (0..n)
        .map(|i| Some((i as f64 * 0.05).sin() + 0.1 * i as f64))
        .collect();
    y[40] = None;
    y[41] = None;
    y[150] = None;
    let s = Series::new("y".into(), y.clone());

    let out = loess(&s, 0.3, 1, Missing::Interpolate).unwrap();
    assert_eq!(out.len(), n);

    // Every output value is finite — the polars adapter linear-fills
    // the nulls before calling LOESS, so the smoother sees a complete
    // series and emits a smoothed value at every position.
    let ca = out.f64().unwrap();
    for i in 0..n {
        let v = ca.get(i).unwrap();
        assert!(v.is_finite(), "loess output at i={i} should be finite, got {v}");
    }
}

#[test]
fn loess_batch_interpolate_handles_polars_nulls() {
    let n = 200;
    let a: Vec<Option<f64>> = (0..n).map(|i| Some((i as f64 * 0.05).sin())).collect();
    let mut b = a.clone();
    b[80] = None;
    let df = DataFrame::new(
        n,
        vec![
            Series::new("clean".into(), a).into_column(),
            Series::new("gappy".into(), b).into_column(),
        ],
    )
    .unwrap();

    let out = loess_batch(&df, 0.3, 1, Missing::Interpolate).unwrap();
    for name in ["clean", "gappy"] {
        let ca = out.column(name).unwrap().f64().unwrap();
        for i in 0..n {
            assert!(ca.get(i).unwrap().is_finite(), "{name}[{i}] should be finite");
        }
    }
}

// ── Missing::Interpolate via the polars layer ──────────────────────────

fn airpassengers_like(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64;
            100.0 + 0.5 * i as f64 + 30.0 * phase.sin()
        })
        .collect()
}

#[test]
fn stl_interpolate_handles_polars_nulls() {
    let period = 12;
    let n = 144;
    let raw = airpassengers_like(n, period);
    let mut with_nulls: Vec<Option<f64>> = raw.iter().map(|v| Some(*v)).collect();
    with_nulls[20] = None;
    with_nulls[21] = None;
    with_nulls[80] = None;

    let s = Series::new("y".into(), with_nulls);

    let opts = StlOpts {
        missing: Missing::Interpolate,
        ..StlOpts::new(period as u32)
    };
    let d = stl(&s, opts).unwrap();
    assert_eq!(d.trend.len(), n);

    let trend    = f64s(&d.trend);
    let seasonal = f64s(&d.seasonal);
    let residual = f64s(&d.residual);

    let null_positions = [20usize, 21, 80];
    for i in 0..n {
        assert!(trend.get(i).unwrap().is_finite(), "trend at i={i} should be finite");
        assert!(seasonal.get(i).unwrap().is_finite(), "seasonal at i={i} should be finite");

        let was_null = null_positions.contains(&i);
        let r = residual.get(i).unwrap();
        if was_null {
            assert!(r.is_nan(), "residual at originally-null i={i} should be NaN; got {r}");
        } else {
            assert!(r.is_finite(), "residual at i={i} should be finite");
        }
    }
}

#[test]
fn stl_error_default_still_rejects_polars_nulls() {
    // With Missing::Error (the default) the polars adapter must still
    // fail fast on any null.
    let mut v: Vec<Option<f64>> = airpassengers_like(48, 4).iter().map(|x| Some(*x)).collect();
    v[3] = None;
    let s = Series::new("y".into(), v);
    let err = stl(&s, StlOpts::new(4)).unwrap_err();
    assert!(matches!(err, PolarsCompatError::HasNulls { col, .. } if col == "y"));
}

#[test]
fn stl_batch_interpolate_handles_polars_nulls() {
    let n = 144;
    let period = 12u32;

    let a = airpassengers_like(n, period as usize);
    let mut b: Vec<Option<f64>> = a.iter().map(|v| Some(*v)).collect();
    b[40] = None;
    b[41] = None;

    let df = DataFrame::new(
        n,
        vec![
            Series::new("clean".into(), a).into_column(),
            Series::new("gappy".into(), b).into_column(),
        ],
    )
    .unwrap();

    let opts = StlOpts {
        missing: Missing::Interpolate,
        ..StlOpts::new(period)
    };
    let res = stl_batch(&df, opts).unwrap();

    // Clean column: finite residual everywhere.
    let r_clean = float_col(&res.residual, "clean");
    for i in 0..n {
        assert!(r_clean.get(i).unwrap().is_finite());
    }

    // Gappy column: NaN residual at originally-null positions only.
    let r_gappy = float_col(&res.residual, "gappy");
    for i in 0..n {
        let was_null = i == 40 || i == 41;
        if was_null {
            assert!(r_gappy.get(i).unwrap().is_nan(), "i={i}");
        } else {
            assert!(r_gappy.get(i).unwrap().is_finite(), "i={i}");
        }
    }
}
