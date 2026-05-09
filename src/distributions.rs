//! Thin wrappers over `statrs` distributions used for inference.
//!
//! Centralizing these makes call sites readable and the implementation
//! swappable.

use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

/// CDF of Student's t with `df` degrees of freedom at `x`.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    StudentsT::new(0.0, 1.0, df)
        .expect("df must be > 0")
        .cdf(x)
}

/// Inverse CDF (quantile) of Student's t with `df` degrees of freedom at `p`.
pub fn t_quantile(p: f64, df: f64) -> f64 {
    StudentsT::new(0.0, 1.0, df)
        .expect("df must be > 0")
        .inverse_cdf(p)
}

/// Two-sided p-value for a t-statistic with `df` degrees of freedom.
///
/// Uses the survival function `sf` from `statrs` (`ContinuousCDF` trait)
/// rather than `1 - cdf`, which would underflow to 0.0 for very large `|t|`
/// due to f64 cancellation.
pub fn t_two_sided_pvalue(t: f64, df: f64) -> f64 {
    let dist = StudentsT::new(0.0, 1.0, df).expect("df must be > 0");
    2.0 * dist.sf(t.abs())
}

/// Survival function (1 - CDF) of F-distribution with (df1, df2) at `x`.
///
/// Uses the survival function `sf` from `statrs` (`ContinuousCDF` trait)
/// rather than `1 - cdf`, which would underflow to 0.0 for very large F-statistics
/// due to f64 cancellation.
pub fn f_sf(x: f64, df1: f64, df2: f64) -> f64 {
    let dist = FisherSnedecor::new(df1, df2).expect("df1, df2 must be > 0");
    dist.sf(x)
}
