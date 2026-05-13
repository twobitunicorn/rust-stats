//! Parity tests against the canonical catch22 C reference
//! (https://github.com/DynamicsAndNeuralSystems/catch22, repo's
//! `testData/` fixtures).
//!
//! Each fixture is two committed files in `tests/golden/`:
//!   - `catch22_c_<name>.in`  — one f64 per line (the input series)
//!   - `catch22_c_<name>.out` — one feature per line, format
//!     `value, feature_name, time_ms`. The C reference writes feature
//!     values to 14 decimal places.
//!
//! These fixtures complement the pycatch22 goldens: pycatch22 wraps the
//! same C kernel, but the C lib here is built with our local toolchain,
//! so a parity match at this level rules out compiler/optimisation
//! drift in the upstream wheel.

use rust_stats::catch22::catch22;
use std::collections::HashMap;
use std::path::PathBuf;

fn read_input(name: &str) -> Vec<f64> {
    let path: PathBuf = ["tests", "golden", &format!("catch22_c_{name}.in")]
        .iter()
        .collect();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().expect("non-float line in input"))
        .collect()
}

fn read_expected(name: &str) -> HashMap<String, f64> {
    let path: PathBuf = ["tests", "golden", &format!("catch22_c_{name}.out")]
        .iter()
        .collect();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            // "value, feature_name, time_ms"
            let mut parts = line.splitn(3, ',');
            let v = parts.next().unwrap().trim().parse::<f64>()
                .unwrap_or_else(|_| panic!("bad value in {line:?}"));
            let n = parts.next().unwrap().trim().to_string();
            (n, v)
        })
        .collect()
}

fn assert_matches_c(name: &str, rel_tol: f64, abs_tol: f64, skip: &[&str]) {
    let y = read_input(name);
    let expected = read_expected(name);
    let ours = catch22(&y);
    let our_named: Vec<(&'static str, f64)> = rust_stats::catch22::CATCH22_NAMES
        .iter()
        .copied()
        .zip(ours.iter().copied())
        .collect();

    let mut failures: Vec<String> = Vec::new();
    for (feature_name, ours_v) in &our_named {
        if skip.contains(feature_name) {
            continue;
        }
        let ref_v = match expected.get(*feature_name) {
            Some(v) => *v,
            None => panic!("C fixture {name} is missing reference value for {feature_name}"),
        };
        let diff = (ours_v - ref_v).abs();
        let close = if ref_v.is_nan() && ours_v.is_nan() {
            true
        } else {
            diff <= abs_tol.max(rel_tol * ours_v.abs().max(ref_v.abs()))
        };
        if !close {
            failures.push(format!(
                "{feature_name}: ours={ours_v}, c_ref={ref_v}, |Δ|={diff}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} feature(s) disagree on C fixture {name}:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Canonical fixture from `catch22/testData/test.txt` (n=270, mixed
/// real-valued series). Used as the C reference's main parity test.
#[test]
fn matches_c_reference_test() {
    assert_matches_c("test", 1e-6, 1e-9, &[]);
}

/// Short series (n=12) — exercises features at the lower end of valid
/// input length.
#[test]
fn matches_c_reference_test_short() {
    assert_matches_c("testShort", 1e-6, 1e-9, &[]);
}

/// Long sinusoid (n=5001) — the C reference's stress fixture. Slightly
/// looser tolerance: this input has periodic structure where
/// integer-quantised features (longstretch / transition matrix) sit on
/// thresholds that can flip ±1 between any two FFT implementations.
#[test]
fn matches_c_reference_test_sinusoid() {
    assert_matches_c("testSinusoid", 1e-5, 1e-8, &[]);
}
