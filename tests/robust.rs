use approx::assert_relative_eq;
use rust_stats::{CovType, Matrix, Ols};

fn small_heteroskedastic() -> rust_stats::OlsResults {
    // y = 1 + 2x + ε with Var(ε) ∝ x²
    let n = 30;
    let x = Matrix::from_fn(n, 1, |i, _| (i as f64) * 0.1 + 0.5);
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let xi = x[(i, 0)];
            1.0 + 2.0 * xi + 0.05 * xi * ((i as f64).sin())
        })
        .collect();
    Ols::new(&y, x.as_ref()).fit().unwrap()
}

#[test]
fn hc1_equals_hc0_times_n_over_n_minus_p() {
    let res = small_heteroskedastic();
    let hc0 = res.cov_hc0();
    let hc1 = res.cov_hc1();
    let scale = res.n_obs() as f64 / res.df_resid() as f64;
    for i in 0..res.coef().len() {
        for j in 0..res.coef().len() {
            assert_relative_eq!(hc1[(i, j)], hc0[(i, j)] * scale, epsilon = 1e-10);
        }
    }
}

#[test]
fn hc_diagonals_are_positive() {
    let res = small_heteroskedastic();
    for cov in [
        res.cov_hc0(), res.cov_hc1(), res.cov_hc2(), res.cov_hc3(),
    ] {
        for i in 0..res.coef().len() {
            assert!(cov[(i, i)] > 0.0);
        }
    }
}

#[test]
fn cov_dispatches_to_robust_variants() {
    let res = small_heteroskedastic();
    for (variant, direct) in [
        (CovType::HC0, res.cov_hc0()),
        (CovType::HC1, res.cov_hc1()),
        (CovType::HC2, res.cov_hc2()),
        (CovType::HC3, res.cov_hc3()),
    ] {
        let via = res.cov(variant);
        for i in 0..res.coef().len() {
            for j in 0..res.coef().len() {
                assert_relative_eq!(via[(i, j)], direct[(i, j)], epsilon = 1e-12);
            }
        }
    }
}
