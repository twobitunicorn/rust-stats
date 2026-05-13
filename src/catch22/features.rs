//! catch22 feature implementations.
//!
//! Reference: https://github.com/DynamicsAndNeuralSystems/catch22
//!
//! Each function takes a slice of finite f64 values (nulls already stripped).
//! For features that catch22 computes on a normalised series, the caller
//! z-scores the input before calling these (see catch22/mod.rs::compute).

use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::cell::RefCell;

thread_local! {
    static FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::<f64>::new());
}

pub fn dn_mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn dn_spread_std(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = dn_mean(x);
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Mode of a histogram with `n_bins` equal-width bins between min and max.
/// Returns 0.0 for constant input (matches catch22 reference).
pub fn dn_histogram_mode(x: &[f64], n_bins: usize) -> f64 {
    if x.is_empty() || n_bins == 0 {
        return f64::NAN;
    }
    let (min, max) = match minmax(x) {
        Some(m) => m,
        None => return f64::NAN,
    };
    if max - min == 0.0 {
        return 0.0;
    }
    let bin_width = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &v in x {
        counts[bin_index(v, min, bin_width, n_bins)] += 1;
    }
    let max_count = *counts.iter().max().unwrap();
    let mut sum = 0.0;
    let mut tied = 0usize;
    for (i, &c) in counts.iter().enumerate() {
        if c == max_count {
            sum += min + bin_width * (i as f64 + 0.5);
            tied += 1;
        }
    }
    sum / tied as f64
}

/// First crossing of the 1/e level by the autocorrelation function (linearly interpolated).
pub fn co_f1ecac(acf: &[f64]) -> f64 {
    if acf.len() < 3 {
        return f64::NAN;
    }
    let threshold = 1.0 / std::f64::consts::E;
    for lag in 1..acf.len() {
        if acf[lag - 1] >= threshold && acf[lag] < threshold {
            let denom = acf[lag - 1] - acf[lag];
            if denom == 0.0 {
                return lag as f64;
            }
            let frac = (acf[lag - 1] - threshold) / denom;
            return (lag - 1) as f64 + frac;
        }
    }
    acf.len() as f64
}

/// First local minimum of the autocorrelation function.
pub fn co_first_min_ac(acf: &[f64]) -> f64 {
    if acf.len() < 3 {
        return f64::NAN;
    }
    for lag in 1..acf.len() - 1 {
        if acf[lag] < acf[lag - 1] && acf[lag] < acf[lag + 1] {
            return lag as f64;
        }
    }
    acf.len() as f64
}

/// First lag where ACF drops to <= 0. Returns acf.len() if none.
pub fn first_zero_in_acf(acf: &[f64]) -> usize {
    if acf.len() < 2 {
        return acf.len();
    }
    for lag in 1..acf.len() {
        if acf[lag] <= 0.0 {
            return lag;
        }
    }
    acf.len()
}

/// CO_HistogramAMI_even_2_5: histogram-based mutual information of (x[t], x[t+2])
/// using 5 even-width bins over [min - 0.1, max + 0.1].
pub fn co_histogram_ami_even_2_5(x: &[f64]) -> f64 {
    let tau = 2usize;
    let n_bins = 5usize;
    if x.len() <= tau + 1 {
        return f64::NAN;
    }
    let (min, max) = match minmax(x) {
        Some(m) => m,
        None => return f64::NAN,
    };
    if max - min == 0.0 {
        return 0.0;
    }
    let lo = min - 0.1;
    let hi = max + 0.1;
    let bin_width = (hi - lo) / n_bins as f64;

    let pair_count = x.len() - tau;
    let mut joint = vec![0usize; n_bins * n_bins];
    let mut p_x = vec![0usize; n_bins];
    let mut p_y = vec![0usize; n_bins];

    for i in 0..pair_count {
        let bx = bin_index(x[i], lo, bin_width, n_bins);
        let by = bin_index(x[i + tau], lo, bin_width, n_bins);
        joint[bx * n_bins + by] += 1;
        p_x[bx] += 1;
        p_y[by] += 1;
    }

    let n = pair_count as f64;
    let mut ami = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            let p_ij = joint[i * n_bins + j] as f64 / n;
            if p_ij > 0.0 {
                let p_i = p_x[i] as f64 / n;
                let p_j = p_y[j] as f64 / n;
                ami += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }
    ami
}

/// Mean of cubed first differences.
pub fn co_trev_1_num(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let n = (x.len() - 1) as f64;
    x.windows(2).map(|w| (w[1] - w[0]).powi(3)).sum::<f64>() / n
}

/// pNN40 with threshold 0.04 (catch22 expects z-scored input internally).
pub fn md_hrv_classic_pnn40(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let count = x.windows(2).filter(|w| (w[1] - w[0]).abs() > 0.04).count();
    count as f64 / (x.len() - 1) as f64
}

/// Longest stretch of values strictly above the mean. Matches the catch22 C
/// algorithm: only `size-1` binarized entries are considered (last sample
/// dropped) and stretches are computed as differences between consecutive
/// terminator positions, not run lengths.
pub fn sb_binary_stats_mean_longstretch1(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 2 {
        return f64::NAN;
    }
    let m = dn_mean(x);
    let y_bin: Vec<u8> = x[..size - 1]
        .iter()
        .map(|&v| if v - m > 0.0 { 1 } else { 0 })
        .collect();
    catch22_longstretch(&y_bin, 0)
}

/// Longest stretch of decreases (sign of first differences). Matches the
/// catch22 C algorithm (yBin = 0 only when diff < 0, terminator on yBin == 1).
pub fn sb_binary_stats_diff_longstretch0(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 2 {
        return f64::NAN;
    }
    let y_bin: Vec<u8> = x
        .windows(2)
        .map(|w| if w[1] - w[0] < 0.0 { 0 } else { 1 })
        .collect();
    catch22_longstretch(&y_bin, 1)
}

/// Reproduces the catch22 longstretch loop: scan up to len-1, terminator on
/// `seq[i] == terminator || i == len-1`, stretch = i - last, last = i.
fn catch22_longstretch(seq: &[u8], terminator: u8) -> f64 {
    let n = seq.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut max_stretch = 0i64;
    let mut last = 0i64;
    for i in 0..n {
        if seq[i] == terminator || i == n - 1 {
            let stretch = i as i64 - last;
            if stretch > max_stretch {
                max_stretch = stretch;
            }
            last = i as i64;
        }
    }
    max_stretch as f64
}

/// FC_LocalSimple_mean1_tauresrat: ratio of first ACF zero crossing of residuals to that of x.
/// Train length = 1, so prediction = previous value, residuals are first differences.
/// Caller passes the input series' first-zero lag (precomputed once and shared).
pub fn fc_local_simple_mean1_tauresrat(x: &[f64], y_first_zero: usize) -> f64 {
    if x.len() < 4 {
        return f64::NAN;
    }
    if y_first_zero == 0 {
        return f64::NAN;
    }
    let residuals: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let res_acf = autocorr_fft(&residuals);
    let res_zero = first_zero_in_acf(&res_acf);
    res_zero as f64 / y_first_zero as f64
}

/// FC_LocalSimple_mean3_stderr: sample std of residuals from rolling mean-of-3 prediction.
pub fn fc_local_simple_mean3_stderr(x: &[f64]) -> f64 {
    let train_len = 3usize;
    if x.len() <= train_len + 1 {
        return f64::NAN;
    }
    let mut residuals = Vec::with_capacity(x.len() - train_len);
    for i in 0..(x.len() - train_len) {
        let yest = (x[i] + x[i + 1] + x[i + 2]) / 3.0;
        residuals.push(x[i + train_len] - yest);
    }
    dn_spread_std(&residuals)
}

/// DN_OutlierInclude with `sign = 1` (positive outliers) or `sign = -1` (negative).
/// Direct port of the catch22 C reference: sweeps thresholds `j * 0.01` for
/// `j = 0 .. floor(max / 0.01)`, and at each threshold records (a) the mean
/// inter-arrival of high indices, (b) the percent of `tot = count(y_work >= 0)`
/// remaining, (c) the centred median index. Trim limit is the smaller of the
/// last `j` with `>2%` density and the first `j` where the inter-arrival mean
/// becomes NaN. Result is the median of the centred-median series up to the
/// trim limit (inclusive).
pub fn dn_outlier_include(x: &[f64], sign: i32) -> f64 {
    let size = x.len();
    if size < 4 {
        return f64::NAN;
    }
    if x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let inc = 0.01;

    // Single pass: constant check, sign-flip, max, tot.
    let mut constant = true;
    let first = x[0];
    let sign_f = sign as f64;
    let mut y_work: Vec<f64> = Vec::with_capacity(size);
    let mut tot = 0usize;
    let mut max_val = f64::NEG_INFINITY;
    for &v in x {
        if v != first {
            constant = false;
        }
        let yw = sign_f * v;
        if yw >= 0.0 {
            tot += 1;
        }
        if yw > max_val {
            max_val = yw;
        }
        y_work.push(yw);
    }
    if constant {
        return 0.0;
    }
    if !max_val.is_finite() || max_val < inc {
        return 0.0;
    }

    let n_thresh = (max_val / inc) as usize + 1;
    let half_size = size as f64 / 2.0;
    let inv_tot = if tot > 0 { 100.0 / tot as f64 } else { 0.0 };

    // Precomputed catch22 threshold values (compared with `>=`).
    let threshes: Vec<f64> = (0..=n_thresh).map(|j| j as f64 * inc).collect();

    // For each position with y_work[i] >= 0, record the threshold-index at
    // which it first fails the test (i.e., the smallest j where threshes[j] > v).
    let mut events: Vec<(usize, usize)> = Vec::with_capacity(tot);
    for (i, &v) in y_work.iter().enumerate() {
        if v >= 0.0 {
            let drop_at = threshes.partition_point(|&t| t <= v);
            events.push((drop_at, i));
        }
    }
    events.sort_unstable_by_key(|&(d, _)| d);

    // Fenwick tree of alive positions: supports add/remove and select-by-rank
    // in O(log N). Initialised with every position whose y_work[i] >= 0.
    let n = size;
    let mut tree = vec![0i32; n + 1];
    fn fenwick_add(tree: &mut [i32], n: usize, pos: usize, delta: i32) {
        let mut idx = pos + 1;
        while idx <= n {
            tree[idx] += delta;
            idx += idx & idx.wrapping_neg();
        }
    }
    fn fenwick_kth(tree: &[i32], n: usize, k: i32) -> usize {
        let mut idx = 0usize;
        let mut acc = 0i32;
        let mut step = 1usize;
        while step * 2 <= n {
            step *= 2;
        }
        while step > 0 {
            let next = idx + step;
            if next <= n && acc + tree[next] < k {
                idx = next;
                acc += tree[next];
            }
            step >>= 1;
        }
        idx
    }
    let mut alive = 0usize;
    for (i, &v) in y_work.iter().enumerate() {
        if v >= 0.0 {
            fenwick_add(&mut tree, n, i, 1);
            alive += 1;
        }
    }

    let mut ms_dti1 = vec![0.0; n_thresh];
    let mut ms_dti3 = vec![0.0; n_thresh];
    let mut ms_dti4 = vec![0.0; n_thresh];

    let mut event_idx = 0usize;
    for j in 0..n_thresh {
        while event_idx < events.len() && events[event_idx].0 <= j {
            let pos = events[event_idx].1;
            fenwick_add(&mut tree, n, pos, -1);
            alive -= 1;
            event_idx += 1;
        }
        let high_size = alive;

        ms_dti1[j] = if high_size >= 2 {
            let first_pos = fenwick_kth(&tree, n, 1) + 1;
            let last_pos = fenwick_kth(&tree, n, high_size as i32) + 1;
            (last_pos as f64 - first_pos as f64) / (high_size - 1) as f64
        } else if high_size == 1 {
            f64::NAN
        } else {
            0.0
        };

        ms_dti3[j] = (high_size as f64 - 1.0) * inv_tot;

        ms_dti4[j] = if high_size > 0 {
            let med = if high_size % 2 == 1 {
                (fenwick_kth(&tree, n, high_size as i32 / 2 + 1) + 1) as f64
            } else {
                let p1 = fenwick_kth(&tree, n, high_size as i32 / 2) + 1;
                let p2 = fenwick_kth(&tree, n, high_size as i32 / 2 + 1) + 1;
                (p1 + p2) as f64 / 2.0
            };
            med / half_size - 1.0
        } else {
            0.0
        };
    }

    let trim_thr = 2.0;
    let mut mj = 0usize;
    let mut fbi = n_thresh - 1;
    for i in 0..n_thresh {
        if ms_dti3[i] > trim_thr {
            mj = i;
        }
        if ms_dti1[n_thresh - 1 - i].is_nan() {
            fbi = n_thresh - 1 - i;
        }
    }

    let trim_limit = mj.min(fbi);
    median_f64(&ms_dti4[..=trim_limit])
}

/// IN_AutoMutualInfoStats_40_gaussian_fmmi: first local minimum of the
/// Gaussian-AMI series, where AMI(tau) = -0.5 * ln(1 - rho(tau)^2) and rho is
/// the per-lag Pearson autocorrelation. Returns the index (not lag) of the
/// first interior local minimum, or `tau_max` if none.
///
/// Optimized with prefix sums: each lag's mean and variance are O(1) given
/// `psum` and `psumsq`, so only the cross-product sum still costs O(n-lag).
pub fn in_automutualinfostats_40_gaussian_fmmi(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 4 {
        return f64::NAN;
    }
    let mut tau = 40usize;
    let half = (size as f64 / 2.0).ceil() as usize;
    if tau > half {
        tau = half;
    }
    if tau < 3 {
        return tau as f64;
    }

    let mut psum = vec![0.0f64; size + 1];
    let mut psumsq = vec![0.0f64; size + 1];
    for i in 0..size {
        psum[i + 1] = psum[i] + x[i];
        psumsq[i + 1] = psumsq[i] + x[i] * x[i];
    }

    let mut ami = vec![0.0; tau];
    for lag_idx in 0..tau {
        let lag = lag_idx + 1;
        let nl = (size - lag) as f64;
        let mu_x = psum[size - lag] / nl;
        let mu_y = (psum[size] - psum[lag]) / nl;
        let s_xx = psumsq[size - lag];
        let s_yy = psumsq[size] - psumsq[lag];
        let mut s_xy = 0.0;
        for i in 0..(size - lag) {
            s_xy += x[i] * x[i + lag];
        }
        let denom_x = s_xx - nl * mu_x * mu_x;
        let denom_y = s_yy - nl * mu_y * mu_y;
        let nom = s_xy - nl * mu_x * mu_y;
        let ac = if denom_x > 0.0 && denom_y > 0.0 {
            nom / (denom_x * denom_y).sqrt()
        } else {
            0.0
        };
        ami[lag_idx] = -0.5 * (1.0 - ac * ac).ln();
    }

    for i in 1..tau - 1 {
        if ami[i] < ami[i - 1] && ami[i] < ami[i + 1] {
            return i as f64;
        }
    }
    tau as f64
}

/// SB_MotifThree_quantile_hh: sum of row-wise entropies of the length-2 motif
/// distribution under 3-quantile coarse-graining.
pub fn sb_motifthree_quantile_hh(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 3 {
        return f64::NAN;
    }
    let yt = sb_coarsegrain_quantile(x, 3);

    let mut r1: Vec<Vec<usize>> = vec![Vec::new(); 3];
    for j in 0..size {
        let label = yt[j];
        if (1..=3).contains(&label) {
            r1[(label - 1) as usize].push(j);
        }
    }
    for r in r1.iter_mut() {
        if let Some(&last) = r.last() {
            if last == size - 1 {
                r.pop();
            }
        }
    }

    let denom = (size - 1) as f64;
    let mut hh = 0.0;
    for i in 0..3 {
        let mut counts = [0usize; 3];
        for &k in &r1[i] {
            let next = yt[k + 1];
            if (1..=3).contains(&next) {
                counts[(next - 1) as usize] += 1;
            }
        }
        for &c in counts.iter() {
            let p = c as f64 / denom;
            if p > 0.0 {
                hh += -p * p.ln();
            }
        }
    }
    hh
}

/// SB_TransitionMatrix_3ac_sumdiagcov: subsample at tau = first ACF zero,
/// quantile-bin to 3 groups, build transition matrix, return sum of diagonal
/// of column covariance matrix.
pub fn sb_transition_matrix_3ac_sumdiagcov(x: &[f64], tau: usize) -> f64 {
    let size = x.len();
    if size < 3 {
        return f64::NAN;
    }
    if x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if x.iter().all(|&v| v == x[0]) {
        return f64::NAN;
    }

    if tau == 0 || tau >= size {
        return f64::NAN;
    }

    let n_down = (size - 1) / tau + 1;
    if n_down < 2 {
        return f64::NAN;
    }
    let y_down: Vec<f64> = (0..n_down).map(|i| x[i * tau]).collect();
    let y_cg = sb_coarsegrain_quantile(&y_down, 3);

    let mut t = [[0.0f64; 3]; 3];
    for j in 0..(n_down - 1) {
        let from = y_cg[j];
        let to = y_cg[j + 1];
        if (1..=3).contains(&from) && (1..=3).contains(&to) {
            t[(from - 1) as usize][(to - 1) as usize] += 1.0;
        }
    }
    let denom = (n_down - 1) as f64;
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] /= denom;
        }
    }

    // Sum of variances of each column (ddof=1, length-3 columns).
    let mut sum_diag = 0.0;
    for col in 0..3 {
        let column: [f64; 3] = [t[0][col], t[1][col], t[2][col]];
        let mean: f64 = column.iter().sum::<f64>() / 3.0;
        let var: f64 = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 2.0;
        sum_diag += var;
    }
    sum_diag
}

/// Detrend `x` with a cubic spline (one interior knot) and return the residuals.
/// Returns `None` if the input is too short or contains NaN. This is split out
/// from `pd_periodicity_wang_th0_01` so the spline fit can run in parallel with
/// the shared autocorr_fft during sequential setup, shrinking PD's
/// parallel-section critical path to just the autocov FFT + peak detection.
pub fn pd_compute_residuals(x: &[f64]) -> Option<Vec<f64>> {
    let n = x.len();
    if n < 5 || x.iter().any(|v| v.is_nan()) {
        return None;
    }
    let spline = splinefit_3knot_cubic(x)?;
    Some(x.iter().zip(spline.iter()).map(|(a, b)| a - b).collect())
}

/// PD_PeriodicityWang_th0_01: given precomputed spline residuals (`y_sub`),
/// return the first peak in the residuals' autocovariance that (a) follows a
/// trough, (b) exceeds it by at least 0.01, (c) is positive. Returns 0 if no
/// qualifying peak is found, or if the caller passes `None`.
pub fn pd_periodicity_wang_th0_01(y_sub: Option<&[f64]>) -> f64 {
    let y_sub = match y_sub {
        Some(y) => y,
        None => return 0.0,
    };
    let size = y_sub.len();
    if size < 5 {
        return 0.0;
    }
    let th = 0.01;

    let acmax = ((size as f64) / 3.0).ceil() as usize;
    if acmax < 3 {
        return 0.0;
    }
    // FFT-based autocovariance: O(N log N) instead of O(N²/3) naive.
    let acov_raw = autocov_unnormalized_fft(y_sub);
    let acf: Vec<f64> = (1..=acmax)
        .map(|tau| {
            if tau < acov_raw.len() {
                acov_raw[tau] / (size - tau) as f64
            } else {
                0.0
            }
        })
        .collect();

    // Detect troughs and peaks (strict interior, slope sign change).
    let mut troughs: Vec<usize> = Vec::new();
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..acmax - 1 {
        let slope_in = acf[i] - acf[i - 1];
        let slope_out = acf[i + 1] - acf[i];
        if slope_in < 0.0 && slope_out > 0.0 {
            troughs.push(i);
        } else if slope_in > 0.0 && slope_out < 0.0 {
            peaks.push(i);
        }
    }

    for &i_peak in &peaks {
        let the_peak = acf[i_peak];
        // Find latest trough strictly before this peak.
        let trough_before = troughs.iter().copied().take_while(|&t| t < i_peak).last();
        let i_trough = match trough_before {
            Some(t) => t,
            None => continue,
        };
        let the_trough = acf[i_trough];
        if the_peak - the_trough < th {
            continue;
        }
        if the_peak < 0.0 {
            continue;
        }
        return i_peak as f64;
    }
    0.0
}

/// Unnormalized linear autocovariance via FFT: r[lag] = sum y[i]*y[i+lag] for
/// lag in 0..n. Assumes y already has zero mean (or that the caller doesn't care).
///
/// `realfft` doesn't scale on inverse (forward + inverse = m * identity), so we
/// divide by the zero-padded FFT length here to land on the same numerator the
/// canonical catch22 C kernel computes by direct summation. Without this, the
/// absolute threshold check inside `pd_periodicity_wang_th0_01` (peak − trough
/// ≥ 0.01) is bypassed by FFT-scaled noise peaks at small lags.
fn autocov_unnormalized_fft(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n < 2 {
        return vec![0.0; n.max(1)];
    }
    let m = (2 * n).next_power_of_two().max(2);
    match fft_squared_magnitude_inverse(y, 0.0) {
        Some(out) => {
            let scale = 1.0 / m as f64;
            out[..n].iter().map(|&v| v * scale).collect()
        }
        None => vec![0.0; n],
    }
}

/// Shared FFT helper: pads `x` (after subtracting `offset`) to `2 * n` rounded
/// up to a power of two, takes |FFT|², and inverse-transforms back to the time
/// domain. Returns `None` on FFT failure. Uses a thread-local planner so plans
/// are cached across all callers within a thread.
fn fft_squared_magnitude_inverse(x: &[f64], offset: f64) -> Option<Vec<f64>> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let m = (2 * n).next_power_of_two().max(2);
    let mut buf: Vec<f64> = vec![0.0; m];
    if offset == 0.0 {
        buf[..n].copy_from_slice(x);
    } else {
        for i in 0..n {
            buf[i] = x[i] - offset;
        }
    }
    FFT_PLANNER.with(|cell| {
        let mut planner = cell.borrow_mut();
        let r2c = planner.plan_fft_forward(m);
        let mut spec = r2c.make_output_vec();
        r2c.process(&mut buf, &mut spec).ok()?;
        for s in spec.iter_mut() {
            let mag2 = s.re * s.re + s.im * s.im;
            s.re = mag2;
            s.im = 0.0;
        }
        let c2r = planner.plan_fft_inverse(m);
        let mut out = c2r.make_output_vec();
        c2r.process(&mut spec, &mut out).ok()?;
        Some(out)
    })
}

/// Least-squares cubic spline with one interior knot at `floor(n/2) - 1`,
/// using the truncated-power basis `[1, t, t^2, t^3, max(0, t-knot)^3]`. Spans
/// the same 5-dimensional cubic spline space as catch22's B-spline construction
/// (same fitted curve under any non-degenerate basis). Returns the fit
/// evaluated at integer positions 0..n-1.
fn splinefit_3knot_cubic(y: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    if n < 5 {
        return None;
    }
    let knot_int = (n / 2) as f64 - 1.0;
    let scale = (n - 1) as f64;
    let knot = knot_int / scale;

    // Fuse row construction with ATA/ATb accumulation — avoids the n*5 design
    // matrix allocation (4 MB for n=100k) and improves cache locality.
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    for k in 0..n {
        let t = k as f64 / scale;
        let t2 = t * t;
        let t3 = t2 * t;
        let truncated = if t > knot { (t - knot).powi(3) } else { 0.0 };
        let row = [1.0, t, t2, t3, truncated];
        let yk = y[k];
        for i in 0..5 {
            atb[i] += row[i] * yk;
            for j in 0..5 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }

    let coefs = gauss_solve_5(ata, atb)?;

    // Evaluate fit by recomputing each row's basis (cheap, no storage).
    let mut fit = vec![0.0; n];
    for k in 0..n {
        let t = k as f64 / scale;
        let t2 = t * t;
        let t3 = t2 * t;
        let truncated = if t > knot { (t - knot).powi(3) } else { 0.0 };
        fit[k] = coefs[0]
            + coefs[1] * t
            + coefs[2] * t2
            + coefs[3] * t3
            + coefs[4] * truncated;
    }
    Some(fit)
}

fn gauss_solve_5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> Option<[f64; 5]> {
    const N: usize = 5;
    // Partial pivoting + elimination.
    for i in 0..N {
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..N {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                pivot = r;
            }
        }
        if best == 0.0 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        for j in (i + 1)..N {
            let factor = a[j][i] / a[i][i];
            b[j] -= factor * b[i];
            for k in i..N {
                a[j][k] -= factor * a[i][k];
            }
        }
    }
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut s = b[i];
        for j in (i + 1)..N {
            s -= x[j] * a[i][j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[derive(Copy, Clone)]
enum FluctMode {
    Dfa,
    RsRange,
}

/// SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1: detrended fluctuation analysis with
/// stride-2 cumulative sum, log-spaced taus from 5 to size/2.
pub fn sc_fluctanal_2_dfa_50_1_2_logi_prop_r1(x: &[f64]) -> f64 {
    sc_fluctanal_logi_prop(x, 2, FluctMode::Dfa)
}

/// SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1: rs-range fluctuation analysis
/// with stride-1 cumulative sum.
pub fn sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1(x: &[f64]) -> f64 {
    sc_fluctanal_logi_prop(x, 1, FluctMode::RsRange)
}

fn sc_fluctanal_logi_prop(x: &[f64], lag: usize, mode: FluctMode) -> f64 {
    let size = x.len();
    if size < 4 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    // Use C integer division (size / 2) — not real division — to match the
    // canonical kernel exactly. With real division, the upper end of the
    // tau schedule rounds *up* one unit (e.g. n=5001 → tau[49] = 2501),
    // which gives n_buffer = 0 → F = NaN → NaN poisons every sserr slot
    // → first_min_ind sticks at 0 → wildly wrong result.
    let half_log = ((size / 2) as f64).ln();
    let low_log = 5.0_f64.ln();
    if !half_log.is_finite() || half_log <= low_log {
        return 0.0;
    }
    let n_tau_steps = 50usize;
    let tau_step = (half_log - low_log) / (n_tau_steps - 1) as f64;
    let mut tau: Vec<usize> = (0..n_tau_steps)
        .map(|i| (low_log + i as f64 * tau_step).exp().round() as usize)
        .collect();
    tau.dedup();
    let n_tau = tau.len();
    if n_tau < 12 {
        return 0.0;
    }

    let size_cs = size / lag;
    if size_cs < 2 {
        return 0.0;
    }
    let mut y_cs = vec![0.0; size_cs];
    y_cs[0] = x[0];
    for i in 0..(size_cs - 1) {
        y_cs[i + 1] = y_cs[i] + x[(i + 1) * lag];
    }

    // Each tau is independent; parallelise so idle workers from the outer
    // catch22 par_iter can be picked up here for the heaviest feature.
    let f_arr: Vec<f64> = tau
        .par_iter()
        .map(|&t| {
            let n_buffer = size_cs / t;
            if n_buffer == 0 {
                return 0.0;
            }
            let tf = t as f64;
            let sum_x = tf * (tf + 1.0) / 2.0;
            let sum_x2 = tf * (tf + 1.0) * (2.0 * tf + 1.0) / 6.0;
            let denom = tf * sum_x2 - sum_x * sum_x;
            let denom_ok = denom != 0.0;

            let mut acc = 0.0;
            for j in 0..n_buffer {
                let chunk = &y_cs[j * t..j * t + t];
                let mut sum_xy = 0.0;
                let mut sum_y = 0.0;
                for k in 0..t {
                    let kv = chunk[k];
                    sum_xy += (k + 1) as f64 * kv;
                    sum_y += kv;
                }
                let (m, b) = if denom_ok {
                    let m = (tf * sum_xy - sum_x * sum_y) / denom;
                    let b = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
                    (m, b)
                } else {
                    (0.0, 0.0)
                };

                match mode {
                    FluctMode::Dfa => {
                        let mut s = 0.0;
                        for k in 0..t {
                            let r = chunk[k] - (m * (k + 1) as f64 + b);
                            s += r * r;
                        }
                        acc += s;
                    }
                    FluctMode::RsRange => {
                        let mut mn = f64::INFINITY;
                        let mut mx = f64::NEG_INFINITY;
                        for k in 0..t {
                            let r = chunk[k] - (m * (k + 1) as f64 + b);
                            if r < mn {
                                mn = r;
                            }
                            if r > mx {
                                mx = r;
                            }
                        }
                        let range = mx - mn;
                        acc += range * range;
                    }
                }
            }
            match mode {
                FluctMode::Dfa => (acc / (n_buffer as f64 * tf)).sqrt(),
                FluctMode::RsRange => (acc / n_buffer as f64).sqrt(),
            }
        })
        .collect();

    let log_tt: Vec<f64> = tau.iter().map(|&t| (t as f64).ln()).collect();
    let log_ff: Vec<f64> = f_arr.iter().map(|&v| v.ln()).collect();
    let ntt = n_tau;
    let min_points = 6usize;
    if ntt < 2 * min_points {
        return 0.0;
    }
    let nsserr = ntt - 2 * min_points + 1;
    let mut sserr = vec![0.0f64; nsserr];

    for i in min_points..=ntt - min_points {
        let (m1, b1) = linreg(&log_tt[..i], &log_ff[..i]);
        let (m2, b2) = linreg(&log_tt[i - 1..], &log_ff[i - 1..]);
        let mut sum_sq1 = 0.0;
        for j in 0..i {
            let r = log_tt[j] * m1 + b1 - log_ff[j];
            sum_sq1 += r * r;
        }
        let mut sum_sq2 = 0.0;
        for j in 0..(ntt - i + 1) {
            let r = log_tt[j + i - 1] * m2 + b2 - log_ff[j + i - 1];
            sum_sq2 += r * r;
        }
        sserr[i - min_points] = sum_sq1.sqrt() + sum_sq2.sqrt();
    }

    let min_val = sserr
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, b| if a < b { a } else { b });
    let mut first_min_ind = 0usize;
    for (i, &v) in sserr.iter().enumerate() {
        if v == min_val {
            first_min_ind = i + min_points - 1;
            break;
        }
    }
    (first_min_ind + 1) as f64 / ntt as f64
}

fn linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mut sumx = 0.0;
    let mut sumx2 = 0.0;
    let mut sumxy = 0.0;
    let mut sumy = 0.0;
    for i in 0..x.len() {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
    }
    let denom = n * sumx2 - sumx * sumx;
    if denom == 0.0 {
        return (0.0, 0.0);
    }
    let m = (n * sumxy - sumx * sumy) / denom;
    let b = (sumy * sumx2 - sumx * sumxy) / denom;
    (m, b)
}

/// SP_Summaries_welch_rect_area_5_1: integrated power in the lowest fifth of
/// the Welch (rectangular-window, single-segment) spectrum, in angular units.
pub fn sp_summaries_welch_rect_area_5_1(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 4 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let (sw, _w) = match welch_rect_angular(x) {
        Some(v) => v,
        None => return f64::NAN,
    };
    if sw.iter().any(|v| v.is_infinite()) {
        return 0.0;
    }
    let nfft = size.next_power_of_two();
    let dw = 2.0 * std::f64::consts::PI / nfft as f64;
    let limit = sw.len() / 5;
    let area: f64 = sw[..limit].iter().sum::<f64>() * dw;
    area
}

/// SP_Summaries_welch_rect_centroid: angular frequency at which the cumulative
/// Welch (rectangular) power crosses 50%.
pub fn sp_summaries_welch_rect_centroid(x: &[f64]) -> f64 {
    let size = x.len();
    if size < 4 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let (sw, w) = match welch_rect_angular(x) {
        Some(v) => v,
        None => return f64::NAN,
    };
    if sw.iter().any(|v| v.is_infinite()) {
        return 0.0;
    }
    let n = sw.len();
    let mut cs = vec![0.0; n];
    cs[0] = sw[0];
    for i in 1..n {
        cs[i] = cs[i - 1] + sw[i];
    }
    let thresh = cs[n - 1] * 0.5;
    for i in 0..n {
        if cs[i] > thresh {
            return w[i];
        }
    }
    0.0
}

/// CO_Embed2_Dist_tau_d_expfit_meandiff: 2D phase-space embedding distances,
/// fit exponential, return mean absolute residual to histogram-normalised pdf.
pub fn co_embed2_dist_tau_d_expfit_meandiff(x: &[f64], first_zero: usize) -> f64 {
    let size = x.len();
    if size < 4 || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut tau = first_zero;
    let cap = (size as f64 / 10.0).floor() as usize;
    if tau as f64 > size as f64 / 10.0 {
        tau = cap;
    }
    if tau == 0 || size <= tau + 1 {
        return f64::NAN;
    }
    let m = size - tau - 1;
    if m < 2 {
        return f64::NAN;
    }
    let mut d = Vec::with_capacity(m);
    for i in 0..m {
        let dx = x[i + 1] - x[i];
        let dy = x[i + tau] - x[i + tau + 1];
        let v = (dx * dx + dy * dy).sqrt();
        if v.is_nan() {
            return f64::NAN;
        }
        d.push(v);
    }

    // Exponential rate (mean of d).
    let l = dn_mean(&d);

    // Auto-binned histogram (Scott's rule).
    let n_bins = num_bins_auto(&d);
    if n_bins == 0 {
        return 0.0;
    }
    let (counts, edges) = histcounts_uniform(&d, n_bins);
    let m_f = m as f64;

    let mut sum_abs = 0.0;
    for i in 0..n_bins {
        let center = (edges[i] + edges[i + 1]) * 0.5;
        let mut expf = (-center / l).exp() / l;
        if expf < 0.0 {
            expf = 0.0;
        }
        let p = counts[i] as f64 / m_f;
        sum_abs += (p - expf).abs();
    }
    sum_abs / n_bins as f64
}

// ---------- helpers ----------

/// catch22-style quantile: linear interpolation with edge-clipping at 0.5/n
/// from each end (returns min/max if quant is in those bands).
fn catch22_quantile(sorted: &[f64], quant: f64) -> f64 {
    let size = sorted.len();
    if size == 0 {
        return f64::NAN;
    }
    let q = 0.5 / size as f64;
    if quant < q {
        return sorted[0];
    }
    if quant > 1.0 - q {
        return sorted[size - 1];
    }
    let quant_idx = size as f64 * quant - 0.5;
    let idx_left = quant_idx.floor() as usize;
    let idx_right = quant_idx.ceil() as usize;
    if idx_left == idx_right {
        return sorted[idx_left];
    }
    sorted[idx_left]
        + (quant_idx - idx_left as f64) * (sorted[idx_right] - sorted[idx_left])
            / (idx_right as f64 - idx_left as f64)
}

/// catch22 coarse-grain (quantile mode): num_groups equal-frequency bins,
/// labels in 1..num_groups.
fn sb_coarsegrain_quantile(x: &[f64], num_groups: usize) -> Vec<i32> {
    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut th: Vec<f64> = (0..=num_groups)
        .map(|i| catch22_quantile(&sorted, i as f64 / num_groups as f64))
        .collect();
    th[0] -= 1.0;

    let mut labels = vec![0i32; x.len()];
    for (j, &v) in x.iter().enumerate() {
        for i in 0..num_groups {
            if v > th[i] && v <= th[i + 1] {
                labels[j] = (i + 1) as i32;
                break;
            }
        }
    }
    labels
}

fn num_bins_auto(y: &[f64]) -> usize {
    let n = y.len();
    if n < 2 {
        return 0;
    }
    let s = dn_spread_std(y);
    if !s.is_finite() || s < 0.001 {
        return 0;
    }
    let (min, max) = match minmax(y) {
        Some(m) => m,
        None => return 0,
    };
    let bin_width = 3.5 * s / (n as f64).powf(1.0 / 3.0);
    if bin_width <= 0.0 {
        return 0;
    }
    ((max - min) / bin_width).ceil() as usize
}

/// One-sided angular-frequency Welch PSD (rectangular window, single segment,
/// zero-padded to next power of two). Returns (Sw, w) where Sw = Pxx / (2π) and
/// w = 2π * f. Matches catch22's SP_Summaries_welch_rect.
fn welch_rect_angular(x: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let size = x.len();
    if size < 2 {
        return None;
    }
    let nfft = size.next_power_of_two().max(2);
    let mean = dn_mean(x);

    let mut buf: Vec<f64> = vec![0.0; nfft];
    for i in 0..size {
        buf[i] = x[i] - mean;
    }
    let nout = nfft / 2 + 1;
    let kmu = size as f64; // single rect window: k * |window|^2 = 1 * size
    let pxx = FFT_PLANNER.with(|cell| -> Option<Vec<f64>> {
        let mut planner = cell.borrow_mut();
        let r2c = planner.plan_fft_forward(nfft);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut buf, &mut spectrum).ok()?;
        let mut pxx = vec![0.0; nout];
        for i in 0..nout {
            pxx[i] = spectrum[i].norm_sqr() / kmu;
            if i > 0 && i < nout - 1 {
                pxx[i] *= 2.0;
            }
        }
        Some(pxx)
    })?;

    let pi = std::f64::consts::PI;
    let df = 1.0 / nfft as f64;
    let sw: Vec<f64> = pxx.iter().map(|p| p / (2.0 * pi)).collect();
    let w: Vec<f64> = (0..nout).map(|i| 2.0 * pi * (i as f64 * df)).collect();
    Some((sw, w))
}

fn histcounts_uniform(y: &[f64], n_bins: usize) -> (Vec<usize>, Vec<f64>) {
    let (min, max) = minmax(y).unwrap_or((0.0, 0.0));
    let bin_step = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &v in y {
        if bin_step <= 0.0 {
            counts[0] += 1;
            continue;
        }
        let mut idx = ((v - min) / bin_step).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let idx = (idx as usize).min(n_bins - 1);
        counts[idx] += 1;
    }
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| i as f64 * bin_step + min)
        .collect();
    (counts, edges)
}

fn minmax(x: &[f64]) -> Option<(f64, f64)> {
    let mut iter = x.iter().copied().filter(|v| v.is_finite());
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for v in iter {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    Some((min, max))
}

fn bin_index(v: f64, lo: f64, bin_width: f64, n_bins: usize) -> usize {
    let mut idx = ((v - lo) / bin_width).floor() as isize;
    if idx < 0 {
        idx = 0;
    }
    let idx = idx as usize;
    if idx >= n_bins {
        n_bins - 1
    } else {
        idx
    }
}

/// Biased autocorrelation via FFT: O(N log N), normalized so acf[0] = 1.
/// Returns acf[0..n] (length n).
pub fn autocorr_fft(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![1.0; n.max(1)];
    }
    let mean = dn_mean(x);
    if !mean.is_finite() {
        return vec![0.0; n];
    }
    let out = fft_squared_magnitude_inverse(x, mean);
    let out = match out {
        Some(v) => v,
        None => return vec![0.0; n],
    };
    let denom = out[0];
    if denom == 0.0 || !denom.is_finite() {
        return vec![0.0; n];
    }
    out[..n].iter().map(|&v| v / denom).collect()
}

fn median_f64(v: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}
