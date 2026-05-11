use approx::assert_relative_eq;
use rust_stats_ols::{Matrix, Ols};

/// Build a deterministic small problem with non-zero residuals so we can
/// assert specific values.
fn small_fit() -> rust_stats_ols::OlsResults {
    let y: Vec<f64> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    let x = Matrix::from_fn(6, 1, |i, _| (i as f64) + 1.0);
    Ols::new(&y, x.as_ref()).fit().unwrap()
}

#[test]
fn fitted_plus_residuals_recovers_y() {
    let res = small_fit();
    let f = res.fitted_values();
    let e = res.residuals();
    let y_true = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    for i in 0..6 {
        assert_relative_eq!(f[i] + e[i], y_true[i], epsilon = 1e-12);
    }
}

#[test]
fn r_squared_in_zero_one() {
    let res = small_fit();
    let r2 = res.r_squared();
    assert!(r2 > 0.0 && r2 < 1.0);
    let adj = res.adj_r_squared();
    assert!(adj <= r2);
}

#[test]
fn f_statistic_is_positive_with_nonzero_signal() {
    let res = small_fit();
    let f = res.f_statistic();
    let p = res.f_pvalue();
    assert!(f > 0.0);
    assert!(p > 0.0 && p < 1.0);
}

#[test]
fn sigma_squared_matches_rss_over_df_resid() {
    let res = small_fit();
    let rss: f64 = res.residuals().iter().map(|r| r * r).sum();
    let expected_sigma = (rss / res.df_resid() as f64).sqrt();
    assert_relative_eq!(res.sigma(), expected_sigma, epsilon = 1e-12);
}
