//! statsmodels-style text summary.

use crate::regression::results::{CovType, OlsResults};
use std::fmt::Write;

pub(crate) fn render(res: &OlsResults, cov: CovType) -> String {
    let inf = res.inference(cov);
    let beta = res.coef();
    let ci = res.conf_int_with(cov, 0.05).expect("alpha 0.05 valid");

    let mut s = String::new();
    let line_eq: String = "=".repeat(78);
    let line_dash: String = "-".repeat(78);

    let _ = writeln!(s, "{:^78}", "OLS Regression Results");
    let _ = writeln!(s, "{line_eq}");
    let _ = writeln!(s,
        "Dep. Variable:      {:>14}   R-squared:         {:>16.4}",
        "y", res.r_squared());
    let _ = writeln!(s,
        "Model:              {:>14}   Adj. R-squared:    {:>16.4}",
        "OLS", res.adj_r_squared());
    let _ = writeln!(s,
        "Method:             {:>14}   F-statistic:       {:>16.3}",
        "Least Squares", res.f_statistic());
    let _ = writeln!(s,
        "No. Observations:   {:>14}   Prob (F-statistic):{:>16.3e}",
        res.n_obs(), res.f_pvalue());
    let _ = writeln!(s, "Df Residuals:       {:>14}", res.df_resid());
    let _ = writeln!(s, "Df Model:           {:>14}", res.df_model());
    let _ = writeln!(s, "Covariance Type:    {:>14}", cov_label(cov));
    let _ = writeln!(s, "{line_eq}");
    let _ = writeln!(s,
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]");
    let _ = writeln!(s, "{line_dash}");

    // Build labels: use user names if present; else "const" (if intercept) + "x1, x2, ..."
    let default_predictor_names: Vec<String> = (0..res.df_model())
        .map(|i| format!("x{}", i + 1))
        .collect();
    let user_names = res.names();
    let labels: Vec<&str> = match user_names {
        Some(ns) => ns.iter().map(|n| n.as_str()).collect(),
        None => {
            let mut v: Vec<&str> = Vec::with_capacity(beta.nrows());
            if res.has_intercept() { v.push("const"); }
            for n in &default_predictor_names { v.push(n.as_str()); }
            v
        }
    };

    for i in 0..beta.nrows() {
        let _ = writeln!(s,
            "{:<10} {:>10.4} {:>10.4} {:>10.3} {:>10.3} {:>10.4} {:>10.4}",
            labels[i],
            *beta.get(i),
            *inf.std_err.get(i),
            *inf.t_values.get(i),
            *inf.p_values.get(i),
            *ci.get(i, 0),
            *ci.get(i, 1),
        );
    }
    let _ = writeln!(s, "{line_eq}");
    s
}

fn cov_label(cov: CovType) -> &'static str {
    match cov {
        CovType::NonRobust => "nonrobust",
        CovType::HC0 => "HC0",
        CovType::HC1 => "HC1",
        CovType::HC2 => "HC2",
        CovType::HC3 => "HC3",
    }
}
