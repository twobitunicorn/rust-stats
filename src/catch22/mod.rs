//! catch22 / catch24 feature extraction.
//!
//! Computes the canonical 22-feature catch22 set (Lubba et al. 2019) and
//! the two catch24 extras (`DN_Mean`, `DN_Spread_Std`) on a slice of
//! `f64` values. The 22 features are bit-for-bit comparable to the
//! canonical C implementation
//! ([pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22))
//! to ~`1e-6` relative tolerance on multiple seeded inputs.
//!
//! ```ignore
//! use rust_stats::catch22::catch22_named;
//!
//! let y: Vec<f64> = (0..200).map(|i| (i as f64).sin()).collect();
//! for (name, v) in catch22_named(&y) {
//!     println!("{name} = {v}");
//! }
//! ```
//!
//! ## Input requirements
//!
//! Inputs must be finite. NaN or ±∞ entries are not silently filtered;
//! they propagate through the kernels and corrupt every feature. Strip
//! them upstream. In debug builds this is checked with `debug_assert!`.
//!
//! ## Algorithm shape
//!
//! - Input is z-scored internally (sample std, ddof = 1) before the 22
//!   features are computed. The catch24 extras (`DN_Mean` /
//!   `DN_Spread_Std`) are computed on the **raw** series — that's the
//!   canonical pycatch22 convention.
//! - A single FFT-based autocorrelation pass is shared across five
//!   features (`CO_f1ecac`, `CO_FirstMin_ac`, `SB_TransitionMatrix*`,
//!   `FC_LocalSimple_mean1*`, `CO_Embed2_Dist*`).
//! - The 22 features are computed in parallel via `rayon`; per-thread
//!   FFT planners are cached.
//!
//! Individual feature functions are public under [`features`] for
//! advanced callers that want to compute a single statistic without
//! paying for the full panel.

use rayon::prelude::*;

pub mod features;

/// Names of the 22 catch22 features in the canonical pycatch22 order.
/// Reference: <https://github.com/DynamicsAndNeuralSystems/catch22>
pub const CATCH22_NAMES: [&str; 22] = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
];

/// Short feature names used by `pycatch22.catch22_all(short_names=True)`.
/// Order matches [`CATCH22_NAMES`] index-for-index. Note pycatch22's
/// mapping has `centroid_freq` and `low_freq_power` swapped relative to
/// what the long names suggest — we mirror that verbatim.
pub const CATCH22_SHORT_NAMES: [&str; 22] = [
    "mode_5",
    "mode_10",
    "acf_timescale",
    "acf_first_min",
    "ami2",
    "trev",
    "high_fluctuation",
    "stretch_high",
    "transition_matrix",
    "periodicity",
    "embedding_dist",
    "ami_timescale",
    "whiten_timescale",
    "outlier_timing_pos",
    "outlier_timing_neg",
    "centroid_freq",
    "stretch_decreasing",
    "entropy_pairs",
    "rs_range",
    "dfa",
    "low_freq_power",
    "forecast_error",
];

/// Two extra features added by catch24 (computed on the raw,
/// non-z-scored series): `DN_Mean` and `DN_Spread_Std`.
pub const CATCH24_EXTRA_NAMES: [&str; 2] = ["DN_Mean", "DN_Spread_Std"];

/// Short names for the catch24 extras: `DN_Mean → "mean"`,
/// `DN_Spread_Std → "SD"`.
pub const CATCH24_EXTRA_SHORT_NAMES: [&str; 2] = ["mean", "SD"];

/// Compute the canonical 22 catch22 features.
///
/// Returns the values in the order given by [`CATCH22_NAMES`]. The
/// input is z-scored internally (sample std, ddof = 1) before feature
/// computation; constant inputs are handled via the same fallbacks as
/// the reference C implementation.
///
/// `y` must contain only finite values. NaN or ±∞ entries propagate
/// through the kernels and corrupt every feature; strip them upstream.
/// In debug builds this is checked with `debug_assert!`.
pub fn catch22(y: &[f64]) -> [f64; 22] {
    debug_assert!(
        y.iter().all(|v| v.is_finite()),
        "catch22: input contains non-finite values; strip them upstream"
    );
    let (raw_mean, raw_std) = (features::dn_mean(y), features::dn_spread_std(y));
    compute(y, raw_mean, raw_std)
}

/// Compute the catch22 panel plus the two catch24 extras
/// (`DN_Mean`, `DN_Spread_Std`).
///
/// Returns the values in the order given by [`CATCH22_NAMES`] followed
/// by [`CATCH24_EXTRA_NAMES`]. The extras are computed on the **raw**
/// (non-z-scored) series — this matches `pycatch22.catch22_all(..,
/// catch24=True)`. See [`catch22`] for input requirements.
pub fn catch24(y: &[f64]) -> [f64; 24] {
    debug_assert!(
        y.iter().all(|v| v.is_finite()),
        "catch24: input contains non-finite values; strip them upstream"
    );
    let raw_mean = features::dn_mean(y);
    let raw_std = features::dn_spread_std(y);
    let core = compute(y, raw_mean, raw_std);
    let mut out = [0.0f64; 24];
    out[..22].copy_from_slice(&core);
    out[22] = raw_mean;
    out[23] = raw_std;
    out
}

/// Convenience wrapper around [`catch22`] that pairs each value with its
/// canonical long name. Order matches [`CATCH22_NAMES`].
pub fn catch22_named(y: &[f64]) -> [(&'static str, f64); 22] {
    let values = catch22(y);
    let mut out: [(&'static str, f64); 22] = [("", 0.0); 22];
    for i in 0..22 {
        out[i] = (CATCH22_NAMES[i], values[i]);
    }
    out
}

/// Convenience wrapper around [`catch24`] that pairs each value with its
/// canonical long name. Order matches [`CATCH22_NAMES`] followed by
/// [`CATCH24_EXTRA_NAMES`].
pub fn catch24_named(y: &[f64]) -> [(&'static str, f64); 24] {
    let values = catch24(y);
    let mut out: [(&'static str, f64); 24] = [("", 0.0); 24];
    for i in 0..22 {
        out[i] = (CATCH22_NAMES[i], values[i]);
    }
    out[22] = (CATCH24_EXTRA_NAMES[0], values[22]);
    out[23] = (CATCH24_EXTRA_NAMES[1], values[23]);
    out
}

fn compute(raw: &[f64], raw_mean: f64, raw_std: f64) -> [f64; 22] {
    // Constant input (std == 0) carries no information. pycatch22's
    // contract here is NaN for 19 of the 22 features; the three lag-
    // based ones (CO_f1ecac, CO_FirstMin_ac, PD_PeriodicityWang_th0_01)
    // canonicalise to 0. Match that exactly rather than running the
    // kernels on a degenerate series — the C kernels short-circuit
    // similarly, but doing it here avoids 22 redundant std checks.
    if !raw_std.is_finite() || raw_std == 0.0 {
        let mut out = [f64::NAN; 22];
        out[2] = 0.0; // CO_f1ecac
        out[3] = 0.0; // CO_FirstMin_ac
        out[9] = 0.0; // PD_PeriodicityWang_th0_01
        return out;
    }
    // catch22 z-scores its input internally (sample std, ddof = 1)
    // before running any per-feature kernel.
    let data: Vec<f64> = raw.iter().map(|v| (v - raw_mean) / raw_std).collect();

    // Shared state: one FFT-based ACF pass, reused by 5 features.
    let acf = features::autocorr_fft(&data);
    let first_zero = features::first_zero_in_acf(&acf);

    // Compute the 22 features in parallel; rayon's work-stealing handles
    // the uneven costs (PD_PeriodicityWang and SC_FluctAnal dominate).
    // Order MUST match CATCH22_NAMES.
    let mut values: [f64; 22] = [0.0; 22];
    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, slot)| {
            *slot = compute_feature(idx, &data, &acf, first_zero);
        });
    values
}

fn compute_feature(idx: usize, data: &[f64], acf: &[f64], first_zero: usize) -> f64 {
    match idx {
        0 => features::dn_histogram_mode(data, 5),
        1 => features::dn_histogram_mode(data, 10),
        2 => features::co_f1ecac(acf),
        3 => features::co_first_min_ac(acf),
        4 => features::co_histogram_ami_even_2_5(data),
        5 => features::co_trev_1_num(data),
        6 => features::md_hrv_classic_pnn40(data),
        7 => features::sb_binary_stats_mean_longstretch1(data),
        8 => features::sb_transition_matrix_3ac_sumdiagcov(data, first_zero),
        9 => {
            let residuals = features::pd_compute_residuals(data);
            features::pd_periodicity_wang_th0_01(residuals.as_deref())
        }
        10 => features::co_embed2_dist_tau_d_expfit_meandiff(data, first_zero),
        11 => features::in_automutualinfostats_40_gaussian_fmmi(data),
        12 => features::fc_local_simple_mean1_tauresrat(data, first_zero),
        13 => features::dn_outlier_include(data, 1),
        14 => features::dn_outlier_include(data, -1),
        15 => features::sp_summaries_welch_rect_area_5_1(data),
        16 => features::sb_binary_stats_diff_longstretch0(data),
        17 => features::sb_motifthree_quantile_hh(data),
        18 => features::sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1(data),
        19 => features::sc_fluctanal_2_dfa_50_1_2_logi_prop_r1(data),
        20 => features::sp_summaries_welch_rect_centroid(data),
        21 => features::fc_local_simple_mean3_stderr(data),
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dn_mean_matches_python_mean() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(features::dn_mean(&x), 3.0);
    }

    #[test]
    fn dn_spread_std_matches_sample_std() {
        // Sample std (ddof=1) of 1..5 is sqrt(2.5).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = (2.5_f64).sqrt();
        assert_relative_eq!(features::dn_spread_std(&x), expected, max_relative = 1e-12);
    }

    #[test]
    fn dn_histogram_mode_constant_input_returns_zero() {
        let x = [3.0_f64; 10];
        // catch22 returns 0 (not the constant value) for constant input.
        assert_eq!(features::dn_histogram_mode(&x, 5), 0.0);
        assert_eq!(features::dn_histogram_mode(&x, 10), 0.0);
    }

    #[test]
    fn catch24_extras_match_raw_aggregates() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let out = catch24(&x);
        assert_relative_eq!(out[22], features::dn_mean(&x), max_relative = 1e-12);
        assert_relative_eq!(out[23], features::dn_spread_std(&x), max_relative = 1e-12);
    }

    #[test]
    fn catch22_has_expected_shape() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let out = catch22(&x);
        assert_eq!(out.len(), 22);
        // All implemented features should produce finite values on this
        // monotonic input (no NaN stubs left in the panel).
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "feature {} ({}) returned non-finite: {v}", i, CATCH22_NAMES[i]);
        }
    }

    #[test]
    fn no_panic_on_random_input() {
        // Box-Muller from a small LCG to avoid pulling in a dep.
        let mut state: u64 = 42;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 32) as f64 / u32::MAX as f64
        };
        let x: Vec<f64> = (0..500)
            .map(|_| {
                let u1 = next().max(1e-12);
                let u2 = next();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let out = catch24(&x);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "feature {} returned non-finite: {v}", i);
        }
    }

    #[test]
    fn name_arrays_have_matching_length() {
        assert_eq!(CATCH22_NAMES.len(), 22);
        assert_eq!(CATCH22_SHORT_NAMES.len(), 22);
        assert_eq!(CATCH24_EXTRA_NAMES.len(), 2);
        assert_eq!(CATCH24_EXTRA_SHORT_NAMES.len(), 2);
    }

    #[test]
    fn catch22_named_returns_canonical_long_names() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let named = catch22_named(&x);
        let raw = catch22(&x);
        for (i, (name, value)) in named.iter().enumerate() {
            assert_eq!(*name, CATCH22_NAMES[i]);
            assert_eq!(*value, raw[i]);
        }
    }

    #[test]
    fn catch24_named_includes_dn_extras() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let named = catch24_named(&x);
        assert_eq!(named[22].0, "DN_Mean");
        assert_eq!(named[23].0, "DN_Spread_Std");
        assert_relative_eq!(named[22].1, features::dn_mean(&x), max_relative = 1e-12);
        assert_relative_eq!(named[23].1, features::dn_spread_std(&x), max_relative = 1e-12);
    }

    /// In release builds (debug_assert disabled) NaN inputs propagate
    /// silently — that's the documented behavior. Pin it so the policy
    /// can't change without an explicit code review.
    #[test]
    #[cfg(not(debug_assertions))]
    fn nan_inputs_propagate_in_release() {
        let mut x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        x[10] = f64::NAN;
        let out = catch22(&x);
        assert!(
            out.iter().any(|v| v.is_nan()),
            "expected NaN to leak into the catch22 panel; instead got {:?}",
            out
        );
    }

    /// In debug builds the contract is enforced via debug_assert, so a
    /// NaN-containing input panics rather than silently producing
    /// garbage. Pin that too.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-finite")]
    fn nan_inputs_panic_in_debug() {
        let mut x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        x[10] = f64::NAN;
        let _ = catch22(&x);
    }
}
