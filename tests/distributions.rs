use approx::assert_relative_eq;
use rust_stats::distributions::{f_sf, t_cdf, t_quantile, t_two_sided_pvalue};

#[test]
fn t_cdf_at_zero_is_half() {
    assert_relative_eq!(t_cdf(0.0, 10.0), 0.5, epsilon = 1e-12);
}

#[test]
fn t_two_sided_pvalue_known_values() {
    // df=10, |t|=2.22814... is the exact 0.025 quantile; use its inverse for accuracy.
    // scipy.stats.t.ppf(0.025, 10) = -2.2281388366041255
    let t_exact = 2.228_138_836_604_125_5_f64;
    let p = t_two_sided_pvalue(t_exact, 10.0);
    assert_relative_eq!(p, 0.05, epsilon = 1e-5);
}

#[test]
fn t_quantile_symmetry() {
    let df = 12.0;
    let q_upper = t_quantile(0.975, df);
    let q_lower = t_quantile(0.025, df);
    assert_relative_eq!(q_upper, -q_lower, epsilon = 1e-10);
}

#[test]
fn f_survival_at_one_for_df1_df2() {
    // Sanity: F(1, 1) survival at 1.0 is 0.5.
    assert_relative_eq!(f_sf(1.0, 1.0, 1.0), 0.5, epsilon = 1e-10);
}
