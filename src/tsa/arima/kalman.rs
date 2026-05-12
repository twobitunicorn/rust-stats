//! Kalman filter for ARMA exact MLE.
//!
//! State-space representation (Harvey 1989, companion form) for an
//! ARMA(p, q) on a *centered, fully-differenced* series `w`:
//!
//! - `r = max(p, q + 1)` — state dimension.
//! - Transition matrix `T` (r × r): companion form with the AR
//!   coefficients in the first column and a shifted identity in the
//!   upper right.
//! - Selection vector `R` (r): `[1, θ₁, θ₂, …, θ_{r−1}]`, with θ
//!   padded by zero.
//! - Observation: `y_t = α_t[0]`.
//! - State equation: `α_{t+1} = T α_t + R ε_{t+1}`, with `ε ∼ N(0, σ²)`.
//!
//! For a stationary AR (guaranteed by our PACF reparameterisation) the
//! initial state has zero mean and the covariance `P_0` solves the
//! discrete Lyapunov equation `P_0 = T P_0 Tᵀ + R Rᵀ` (using `σ² = 1`;
//! `σ²` is profiled out of the concentrated likelihood). We solve it
//! iteratively — the iteration converges geometrically with rate equal
//! to the spectral radius of `T`, which is bounded below 1 inside the
//! stationary region.
//!
//! Given the filter output `(Σ v_t²/F_t, Σ log F_t)` (where `F_t` is
//! the innovation variance and `v_t` the innovation at step `t`), the
//! concentrated negative Gaussian log-likelihood is
//!
//! ```text
//! n · log(σ̂²) + Σ log F_t   (up to constants),
//!   where σ̂² = (1/n) Σ v_t²/F_t.
//! ```

/// State-space ARMA(p, q) representation.
pub(super) struct ArmaSs {
    pub r: usize,
    /// Transition matrix `T`, row-major (length r·r).
    pub t_matrix: Vec<f64>,
    /// Selection vector `R` (length r).
    pub r_vec: Vec<f64>,
}

impl ArmaSs {
    pub fn build(phi: &[f64], theta: &[f64]) -> Self {
        let p = phi.len();
        let q = theta.len();
        let r = p.max(q + 1).max(1);
        let mut t_matrix = vec![0.0f64; r * r];
        // First column = phi (padded with zeros if p < r).
        for i in 0..r {
            t_matrix[i * r] = if i < p { phi[i] } else { 0.0 };
        }
        // Super-diagonal = 1.
        for i in 0..r.saturating_sub(1) {
            t_matrix[i * r + (i + 1)] = 1.0;
        }
        // R = [1, θ_1, …, θ_{r-1}].
        let mut r_vec = vec![0.0f64; r];
        r_vec[0] = 1.0;
        for i in 1..r {
            r_vec[i] = if i - 1 < q { theta[i - 1] } else { 0.0 };
        }
        Self { r, t_matrix, r_vec }
    }

    /// Solve the discrete Lyapunov equation `P = T P Tᵀ + R Rᵀ`
    /// iteratively. Bails after `max_iter` if convergence is too slow
    /// (returns whatever we have, which is still a positive-semidefinite
    /// upper bound on `P_0`).
    pub fn lyapunov_p0(&self) -> Vec<f64> {
        let r = self.r;
        let rrt: Vec<f64> = (0..r * r)
            .map(|k| self.r_vec[k / r] * self.r_vec[k % r])
            .collect();
        let mut p = rrt.clone();
        let mut work_a = vec![0.0f64; r * r];
        let mut work_b = vec![0.0f64; r * r];
        for _ in 0..500 {
            mat_mul(&self.t_matrix, &p, &mut work_a, r, r, r);
            mat_mul_b_transpose(&work_a, &self.t_matrix, &mut work_b, r, r, r);
            let mut max_diff = 0.0f64;
            for k in 0..r * r {
                let new_v = work_b[k] + rrt[k];
                let diff = (new_v - p[k]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                p[k] = new_v;
            }
            if max_diff < 1e-12 {
                break;
            }
        }
        p
    }

    /// Run the Kalman filter over `y`. Returns
    /// `(Σ v_t²/F_t, Σ log F_t)`, both used by the concentrated
    /// likelihood.
    pub fn filter(&self, y: &[f64]) -> (f64, f64) {
        let (sum_v2_f, sum_log_f, _, _) = self.filter_inner::<false>(y);
        (sum_v2_f, sum_log_f)
    }

    /// Run the Kalman filter and also collect the one-step-ahead
    /// predictions `ŷ_t = E[y_t | y_1…y_{t-1}]` and innovations
    /// `v_t = y_t − ŷ_t`. These are what R's `fitted(arima)` returns —
    /// the filter handles the diffuse start-up naturally so every step
    /// has a meaningful prediction (no zero warm-up).
    pub fn filter_with_predictions(&self, y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (_, _, predicted, innovations) = self.filter_inner::<true>(y);
        (predicted, innovations)
    }

    /// Inner loop. `COLLECT = true` allocates and fills per-step
    /// prediction/innovation buffers; `false` skips that work so the
    /// likelihood path stays as cheap as before.
    fn filter_inner<const COLLECT: bool>(
        &self,
        y: &[f64],
    ) -> (f64, f64, Vec<f64>, Vec<f64>) {
        let r = self.r;
        let mut a = vec![0.0f64; r];
        let mut p_mat = self.lyapunov_p0();
        let rrt: Vec<f64> = (0..r * r)
            .map(|k| self.r_vec[k / r] * self.r_vec[k % r])
            .collect();
        let mut sum_v2_f = 0.0f64;
        let mut sum_log_f = 0.0f64;

        let mut k_gain = vec![0.0f64; r];
        let mut a_upd = vec![0.0f64; r];
        let mut p_upd = vec![0.0f64; r * r];
        let mut work_a = vec![0.0f64; r * r];
        let mut work_b = vec![0.0f64; r * r];

        let mut predicted = if COLLECT { Vec::with_capacity(y.len()) } else { Vec::new() };
        let mut innovations = if COLLECT { Vec::with_capacity(y.len()) } else { Vec::new() };

        for &y_t in y {
            // Innovation v = y_t − a[0], innovation variance F = P[0,0].
            let v = y_t - a[0];
            let f = p_mat[0];
            if COLLECT {
                predicted.push(a[0]);
                innovations.push(v);
            }
            if !f.is_finite() || f <= 0.0 {
                if COLLECT {
                    // Pad remaining slots with NaN so the buffers stay length-y.
                    while predicted.len() < y.len() {
                        predicted.push(f64::NAN);
                        innovations.push(f64::NAN);
                    }
                }
                return (f64::INFINITY, 0.0, predicted, innovations);
            }
            sum_v2_f += v * v / f;
            sum_log_f += f.ln();
            // K = P · Zᵀ / F  = column 0 of P, scaled.
            for i in 0..r {
                k_gain[i] = p_mat[i * r] / f;
            }
            // a_upd = a + K · v.
            for i in 0..r {
                a_upd[i] = a[i] + k_gain[i] * v;
            }
            // P_upd = P − K F Kᵀ.
            for i in 0..r {
                for j in 0..r {
                    p_upd[i * r + j] = p_mat[i * r + j] - k_gain[i] * f * k_gain[j];
                }
            }
            // Predict next: a ← T · a_upd; P ← T · P_upd · Tᵀ + R Rᵀ.
            mat_vec(&self.t_matrix, &a_upd, &mut a, r, r);
            mat_mul(&self.t_matrix, &p_upd, &mut work_a, r, r, r);
            mat_mul_b_transpose(&work_a, &self.t_matrix, &mut work_b, r, r, r);
            for k in 0..r * r {
                p_mat[k] = work_b[k] + rrt[k];
            }
        }
        (sum_v2_f, sum_log_f, predicted, innovations)
    }
}

/// Concentrated negative log-likelihood: minimise this. σ² is profiled
/// out via `σ̂² = (1/n) Σ v_t²/F_t`. Drops the `n · log(2π) + n`
/// constants that don't affect the optimisation.
pub(super) fn concentrated_neg_loglik(y: &[f64], phi: &[f64], theta: &[f64]) -> f64 {
    let ss = ArmaSs::build(phi, theta);
    let (sum_v2_f, sum_log_f) = ss.filter(y);
    if !sum_v2_f.is_finite() || sum_v2_f <= 0.0 {
        return f64::INFINITY;
    }
    let n = y.len() as f64;
    let sigma2_hat = sum_v2_f / n;
    if !sigma2_hat.is_finite() || sigma2_hat <= 0.0 {
        return f64::INFINITY;
    }
    n * sigma2_hat.ln() + sum_log_f
}

/// Concentrated σ̂² at the given (φ, θ). Computed alongside the
/// objective but exposed separately so the fitting routine can populate
/// `ArimaFit::sigma2` once the optimiser is done.
pub(super) fn concentrated_sigma2(y: &[f64], phi: &[f64], theta: &[f64]) -> f64 {
    let ss = ArmaSs::build(phi, theta);
    let (sum_v2_f, _) = ss.filter(y);
    sum_v2_f / y.len() as f64
}

/// One-step-ahead predictions and innovations from the Kalman filter
/// run at (φ, θ) over `y`. Both vectors have the same length as `y`
/// and are well-defined at every step (the filter's diffuse start-up
/// means even `t = 0` has a meaningful prediction — the unconditional
/// mean of the stationary process).
pub(super) fn fitted_residuals(y: &[f64], phi: &[f64], theta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let ss = ArmaSs::build(phi, theta);
    ss.filter_with_predictions(y)
}

// ----------------------------------------------------------------------
// Tiny row-major linear algebra primitives.
// ----------------------------------------------------------------------

fn mat_mul(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

/// `out (m×n) = a (m×k) * bᵀ` where `b` is stored as `(n×k)` row-major.
fn mat_mul_b_transpose(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i * k + l] * b[j * k + l];
            }
            out[i * n + j] = s;
        }
    }
}

fn mat_vec(a: &[f64], x: &[f64], out: &mut [f64], m: usize, n: usize) {
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        out[i] = s;
    }
}

#[cfg(test)]
mod kalman_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ar1_lyapunov_matches_closed_form() {
        // For AR(1): P_0 = σ² / (1 − φ²). With σ² = 1 here.
        let phi = 0.7_f64;
        let ss = ArmaSs::build(&[phi], &[]);
        let p = ss.lyapunov_p0();
        assert_eq!(ss.r, 1);
        assert_relative_eq!(p[0], 1.0 / (1.0 - phi * phi), max_relative = 1e-8);
    }

    #[test]
    fn ma1_lyapunov_matches_closed_form() {
        // For MA(1): P_0 element [0,0] = 1 + θ². Two-state with θ_1.
        let theta = 0.4_f64;
        let ss = ArmaSs::build(&[], &[theta]);
        let p = ss.lyapunov_p0();
        assert_eq!(ss.r, 2);
        // y_t = ε_t + θ ε_{t-1} → Var(y_t) = 1 + θ².
        assert_relative_eq!(p[0], 1.0 + theta * theta, max_relative = 1e-8);
    }

    #[test]
    fn filter_runs_on_short_series() {
        let phi = vec![0.5];
        let theta = vec![];
        let y = vec![0.0, 0.5, 0.25, 0.125, 0.0625];
        let ss = ArmaSs::build(&phi, &theta);
        let (sum_v2_f, sum_log_f) = ss.filter(&y);
        assert!(sum_v2_f.is_finite());
        assert!(sum_log_f.is_finite());
    }
}
