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
        let t_col0: Vec<f64> = (0..r).map(|i| self.t_matrix[i * r]).collect();
        let rrt: Vec<f64> = (0..r * r)
            .map(|k| self.r_vec[k / r] * self.r_vec[k % r])
            .collect();
        let mut p = rrt.clone();
        let mut work_b = vec![0.0f64; r * r];
        let mut scratch = vec![0.0f64; r];
        for _ in 0..500 {
            t_x_tt(&t_col0, &p, &mut work_b, &mut scratch, r);
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

    /// Kalman filter forward pass that *also* propagates sensitivities
    /// of `(α_t, P_t)` with respect to each parameter. `dt_col0_stack[i]`
    /// is the **column 0** of `∂T/∂param_i` — for ARMA companion-form
    /// state-spaces all `∂T` entries outside column 0 are zero, so we
    /// only store the column and exploit the sparsity in the inner
    /// products (`∂T · M` and `M · ∂Tᵀ` both become O(r²) instead of
    /// O(r³)). `dr_stack[i]` is the full `∂R/∂param_i`. Returns the
    /// usual `(Σ v²/F, Σ log F)` plus their gradients.
    fn filter_grad_inner(
        &self,
        y: &[f64],
        dt_col0_stack: &[Vec<f64>],
        dr_stack: &[Vec<f64>],
    ) -> (f64, f64, Vec<f64>, Vec<f64>) {
        let r = self.r;
        let n_params = dt_col0_stack.len();
        debug_assert_eq!(dr_stack.len(), n_params);

        let t_col0: Vec<f64> = (0..r).map(|i| self.t_matrix[i * r]).collect();

        let p0 = self.lyapunov_p0();
        let mut a = vec![0.0f64; r];
        let mut p_mat = p0.clone();

        let mut da_stack: Vec<Vec<f64>> = vec![vec![0.0; r]; n_params];
        let mut dp_stack: Vec<Vec<f64>> = self.lyapunov_p0_grad(&p0, dt_col0_stack, dr_stack);

        let rrt: Vec<f64> = (0..r * r)
            .map(|k| self.r_vec[k / r] * self.r_vec[k % r])
            .collect();

        let mut sum_v2_f = 0.0f64;
        let mut sum_log_f = 0.0f64;
        let mut d_sum_v2_f = vec![0.0f64; n_params];
        let mut d_sum_log_f = vec![0.0f64; n_params];

        let mut k_gain = vec![0.0f64; r];
        let mut a_upd = vec![0.0f64; r];
        let mut p_upd = vec![0.0f64; r * r];

        let mut work_b = vec![0.0f64; r * r];
        let mut scratch_v = vec![0.0f64; r];

        let mut dk = vec![0.0f64; r];
        let mut da_upd = vec![0.0f64; r];
        let mut dp_upd = vec![0.0f64; r * r];

        // Per-step scratch vectors for the rank-1 ∂T contributions.
        let mut t_p_upd_col0 = vec![0.0f64; r]; // column 0 of T·P_upd
        let mut p_upd_row0_t = vec![0.0f64; r]; // (T·(row 0 of P_upd)ᵀ)

        for &y_t in y {
            let v = y_t - a[0];
            let f = p_mat[0];
            if !f.is_finite() || f <= 0.0 {
                return (f64::INFINITY, 0.0, d_sum_v2_f, d_sum_log_f);
            }
            sum_v2_f += v * v / f;
            sum_log_f += f.ln();

            for i in 0..r {
                k_gain[i] = p_mat[i * r] / f;
            }
            for i in 0..r {
                a_upd[i] = a[i] + k_gain[i] * v;
            }
            for i in 0..r {
                for j in 0..r {
                    p_upd[i * r + j] = p_mat[i * r + j] - k_gain[i] * f * k_gain[j];
                }
            }

            // Column 0 of T·P_upd: (T·P_upd)[i, 0] = phi[i]·P_upd[0,0]
            //   + (P_upd[i+1, 0] if i+1<r else 0).
            let p00 = p_upd[0];
            for i in 0..r {
                let shifted = if i + 1 < r { p_upd[(i + 1) * r] } else { 0.0 };
                t_p_upd_col0[i] = t_col0[i] * p00 + shifted;
            }
            // T applied to row 0 of P_upd (as a column vec):
            //   w[j] = phi[j]·P_upd[0,0] + (P_upd[0, j+1] if j+1<r else 0).
            for j in 0..r {
                let shifted = if j + 1 < r { p_upd[j + 1] } else { 0.0 };
                p_upd_row0_t[j] = t_col0[j] * p00 + shifted;
            }

            // Per-parameter sensitivity update — every term is O(r²)
            // with the companion-form helpers below.
            let f2 = f * f;
            for ip in 0..n_params {
                let da = &mut da_stack[ip];
                let dp = &mut dp_stack[ip];
                let dt_col0 = &dt_col0_stack[ip];
                let dr = &dr_stack[ip];

                let dv = -da[0];
                let df = dp[0];

                for i in 0..r {
                    dk[i] = (dp[i * r] * f - p_mat[i * r] * df) / f2;
                }

                d_sum_v2_f[ip] += (2.0 * v * dv * f - v * v * df) / f2;
                d_sum_log_f[ip] += df / f;

                for i in 0..r {
                    da_upd[i] = da[i] + dk[i] * v + k_gain[i] * dv;
                }

                for i in 0..r {
                    for j in 0..r {
                        dp_upd[i * r + j] = dp[i * r + j]
                            - dk[i] * f * k_gain[j]
                            - k_gain[i] * df * k_gain[j]
                            - k_gain[i] * f * dk[j];
                    }
                }

                // ∂α_{t+1} = ∂T · α_upd + T · ∂α_upd
                //         = dt_col0 · α_upd[0] + T · ∂α_upd.
                for i in 0..r {
                    da[i] = dt_col0[i] * a_upd[0];
                }
                t_vec_add(&t_col0, &da_upd, da, r);

                // ∂P_{t+1} = T · ∂P_upd · Tᵀ            (O(r²) via t_x_tt)
                //          + ∂T · P_upd · Tᵀ            (rank-1: dt_col0 ⊗ p_upd_row0_t)
                //          + T · P_upd · ∂Tᵀ            (rank-1: t_p_upd_col0 ⊗ dt_col0)
                //          + ∂R · Rᵀ + R · ∂Rᵀ          (rank-1 outer)
                t_x_tt(&t_col0, &dp_upd, &mut work_b, &mut scratch_v, r);
                for i in 0..r {
                    let dt_i = dt_col0[i];
                    let t_p_col0_i = t_p_upd_col0[i];
                    let r_i = self.r_vec[i];
                    let dr_i = dr[i];
                    for j in 0..r {
                        dp[i * r + j] = work_b[i * r + j]
                            + dt_i * p_upd_row0_t[j]
                            + t_p_col0_i * dt_col0[j]
                            + dr_i * self.r_vec[j]
                            + r_i * dr[j];
                    }
                }
            }

            // Predict next: a ← T · a_upd; P ← T · P_upd · Tᵀ + R Rᵀ.
            t_vec(&t_col0, &a_upd, &mut a, r);
            t_x_tt(&t_col0, &p_upd, &mut work_b, &mut scratch_v, r);
            for k in 0..r * r {
                p_mat[k] = work_b[k] + rrt[k];
            }
        }
        (sum_v2_f, sum_log_f, d_sum_v2_f, d_sum_log_f)
    }

    /// Solve `∂P = T · ∂P · Tᵀ + G` for each parameter, where the
    /// forcing `G = ∂T·P·Tᵀ + T·P·∂Tᵀ + ∂R·Rᵀ + R·∂Rᵀ` comes from
    /// differentiating the steady-state Lyapunov equation. `dt_col0`
    /// stores only column 0 of `∂T` (the rest is zero for ARMA
    /// companion form).
    fn lyapunov_p0_grad(
        &self,
        p0: &[f64],
        dt_col0_stack: &[Vec<f64>],
        dr_stack: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let r = self.r;
        let n_params = dt_col0_stack.len();
        let mut out = Vec::with_capacity(n_params);
        let mut work_a = vec![0.0f64; r * r];
        let mut work_b = vec![0.0f64; r * r];
        for i in 0..n_params {
            let dt_col0 = &dt_col0_stack[i];
            let dr = &dr_stack[i];
            // G = ∂T·P·Tᵀ + T·P·∂Tᵀ + ∂R·Rᵀ + R·∂Rᵀ.
            // Note: ∂T·P has rows equal to dt_col0[i] · (row 0 of P);
            // (∂T·P)·Tᵀ likewise factors. Each is rank-1 in the row-0
            // pattern, so we compute it in O(r²) instead of O(r³).
            let mut g = vec![0.0f64; r * r];
            // ∂T · P: row i = dt_col0[i] · (row 0 of P).
            // Then ·Tᵀ: column j of result = (∂T·P) · (row j of T)ᵀ.
            //          (row i of (∂T·P)) · (column j of Tᵀ)
            //          = (dt_col0[i] · row0(P)) · (column j of Tᵀ)
            //          = dt_col0[i] · (row0(P) · column j of Tᵀ)
            //          = dt_col0[i] · w_b[j]
            // where w_b[j] = row0(P) · column j of Tᵀ.
            // First build w_b (length r): w_b[j] = sum_k P[0,k] * T[j,k].
            let mut wb = vec![0.0f64; r];
            for j in 0..r {
                let mut s = 0.0f64;
                for k in 0..r {
                    s += p0[k] * self.t_matrix[j * r + k];
                }
                wb[j] = s;
            }
            for ii in 0..r {
                for jj in 0..r {
                    g[ii * r + jj] += dt_col0[ii] * wb[jj];
                }
            }
            // T·P·∂Tᵀ — the transpose of the above (G is symmetric).
            for ii in 0..r {
                for jj in 0..r {
                    g[ii * r + jj] += dt_col0[jj] * wb[ii];
                }
            }
            for ii in 0..r {
                for jj in 0..r {
                    g[ii * r + jj] += dr[ii] * self.r_vec[jj] + self.r_vec[ii] * dr[jj];
                }
            }
            // Iterate ∂P = T·∂P·Tᵀ + G — uses the O(r²) companion-aware
            // `t_x_tt` since T has the same companion structure here.
            let t_col0: Vec<f64> = (0..r).map(|i| self.t_matrix[i * r]).collect();
            let mut dp = g.clone();
            let mut scratch_v = vec![0.0f64; r];
            for _ in 0..500 {
                t_x_tt(&t_col0, &dp, &mut work_b, &mut scratch_v, r);
                let mut max_diff = 0.0f64;
                for k in 0..r * r {
                    let new_v = work_b[k] + g[k];
                    let diff = (new_v - dp[k]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    dp[k] = new_v;
                }
                if max_diff < 1e-12 {
                    break;
                }
            }
            out.push(dp);
        }
        // work_a is unused now; silence the warning by binding.
        let _ = &work_a;
        out
    }

    /// Inner loop. `COLLECT = true` allocates and fills per-step
    /// prediction/innovation buffers; `false` skips that work so the
    /// likelihood path stays as cheap as before.
    fn filter_inner<const COLLECT: bool>(
        &self,
        y: &[f64],
    ) -> (f64, f64, Vec<f64>, Vec<f64>) {
        let r = self.r;
        let t_col0: Vec<f64> = (0..r).map(|i| self.t_matrix[i * r]).collect();
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
        let mut work_b = vec![0.0f64; r * r];
        let mut scratch_v = vec![0.0f64; r];

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
            t_vec(&t_col0, &a_upd, &mut a, r);
            t_x_tt(&t_col0, &p_upd, &mut work_b, &mut scratch_v, r);
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

/// Concentrated NLL together with its gradient with respect to a
/// caller-supplied stack of `(∂T, ∂R)` directions. One entry of
/// `dt_stack` / `dr_stack` corresponds to one free parameter; the
/// returned `grad` is in the same order.
///
/// The gradient is hand-derived from the Kalman recursion — forward
/// sensitivity propagation of `(α_t, P_t)` per parameter, no finite
/// differences in the Kalman pass itself. Cost is O(n_params · r³ ·
/// n_t), roughly `n_params` Kalman-pass-equivalents per call, so it
/// beats `2·n_params + 1` central-difference passes once `n_params ≥ 2`.
///
/// The chain rule for SARIMA's convolved AR/MA polynomial is handled
/// by the caller: build a `∂T_i, ∂R_i` pair for each free parameter
/// (`φ`, `Φ`, `θ`, `Θ`) — these come from differentiating the
/// convolution `convolve_ar(φ, Φ, m)` / `convolve_ma(θ, Θ, m)`, which
/// is a cheap polynomial bookkeeping step.
pub(super) fn concentrated_nll_and_grad_with_stacks(
    y: &[f64],
    total_ar: &[f64],
    total_ma: &[f64],
    dt_col0_stack: &[Vec<f64>],
    dr_stack: &[Vec<f64>],
) -> (f64, Vec<f64>) {
    let n_params = dt_col0_stack.len();
    debug_assert_eq!(dr_stack.len(), n_params);
    if n_params == 0 {
        return (concentrated_neg_loglik(y, total_ar, total_ma), Vec::new());
    }
    let ss = ArmaSs::build(total_ar, total_ma);

    let (sum_v2_f, sum_log_f, d_sum_v2_f, d_sum_log_f) =
        ss.filter_grad_inner(y, dt_col0_stack, dr_stack);

    let n = y.len() as f64;
    if !sum_v2_f.is_finite() || sum_v2_f <= 0.0 {
        return (f64::INFINITY, vec![0.0; n_params]);
    }
    let sigma2_hat = sum_v2_f / n;
    if !sigma2_hat.is_finite() || sigma2_hat <= 0.0 {
        return (f64::INFINITY, vec![0.0; n_params]);
    }
    let nll = n * sigma2_hat.ln() + sum_log_f;

    // ∂NLL/∂param = (1/σ̂²) · ∂sum_v²/F + ∂sum_log_f.
    let grad: Vec<f64> = (0..n_params)
        .map(|i| d_sum_v2_f[i] / sigma2_hat + d_sum_log_f[i])
        .collect();
    (nll, grad)
}

/// Convenience wrapper for the non-seasonal ARMA(p, q) case: builds
/// the trivial `(∂T, ∂R)` stacks where each free parameter perturbs a
/// single entry of `T` (for `φ`) or `R` (for `θ`). The gradient is
/// returned in the order `[∂/∂φ_1, …, ∂/∂φ_p, ∂/∂θ_1, …, ∂/∂θ_q]`.
#[cfg(test)]
pub(super) fn concentrated_nll_and_grad(
    y: &[f64],
    phi: &[f64],
    theta: &[f64],
) -> (f64, Vec<f64>) {
    let p = phi.len();
    let q = theta.len();
    let n_params = p + q;
    if n_params == 0 {
        return (concentrated_neg_loglik(y, phi, theta), Vec::new());
    }
    let ss = ArmaSs::build(phi, theta);
    let r = ss.r;
    let mut dt_col0_stack: Vec<Vec<f64>> = vec![vec![0.0; r]; n_params];
    let mut dr_stack: Vec<Vec<f64>> = vec![vec![0.0; r]; n_params];
    for j in 0..p {
        // ∂T/∂φ_{j+1} = unit at row j, column 0 → dt_col0[j] = 1.
        dt_col0_stack[j][j] = 1.0;
    }
    for j in 0..q {
        dr_stack[p + j][j + 1] = 1.0;
    }
    concentrated_nll_and_grad_with_stacks(y, phi, theta, &dt_col0_stack, &dr_stack)
}

// ----------------------------------------------------------------------
// ARMA companion-form linear algebra primitives. The generic `mat_mul`
// helpers are gone — every appearance of `T · X · Tᵀ`, `T · X`, and
// `T · v` in the Kalman filter is now routed through the structure-
// aware kernels below (O(r²) / O(r) instead of O(r³) / O(r²)).
// ----------------------------------------------------------------------

/// Companion-form `T · X · Tᵀ` in O(r²). Exploits `T = phi·e_0ᵀ + S`
/// (where `phi = t_col0` and `S` is the super-diagonal shift):
///
/// ```text
/// T·X·Tᵀ = X[0,0]·(phi⊗phi)
///        + phi ⊗ (row 0 of X shifted left, last entry 0)
///        + (col 0 of X shifted up, last entry 0) ⊗ phi
///        + S·X·Sᵀ   (= X with first row & column dropped,
///                     placed at (0,0) and zero-padded)
/// ```
///
/// Inner loops are branch-free over `j ∈ 0..r` — `scratch` is a
/// caller-supplied length-r work buffer (avoids the per-call Vec
/// allocation that would otherwise dominate at small `r`).
fn t_x_tt(t_col0: &[f64], x: &[f64], out: &mut [f64], scratch: &mut [f64], r: usize) {
    let x00 = x[0];
    // row0_shift[j] = X[0, j+1] for j<r-1, else 0.
    scratch[..r - 1].copy_from_slice(&x[1..r]);
    scratch[r - 1] = 0.0;

    for i in 0..r - 1 {
        let phi_i = t_col0[i];
        let col0_shift_i = x[(i + 1) * r];
        let alpha_phi = phi_i;
        let beta_phi = col0_shift_i + x00 * phi_i;
        let src_row = &x[(i + 1) * r + 1..(i + 1) * r + r];
        let dst_row = &mut out[i * r..i * r + r];
        for j in 0..r - 1 {
            dst_row[j] = src_row[j]
                + alpha_phi * scratch[j]
                + beta_phi * t_col0[j];
        }
        dst_row[r - 1] = beta_phi * t_col0[r - 1];
    }
    let phi_last = t_col0[r - 1];
    let beta_last = x00 * phi_last;
    let dst_last = &mut out[(r - 1) * r..(r - 1) * r + r];
    for j in 0..r - 1 {
        dst_last[j] = phi_last * scratch[j] + beta_last * t_col0[j];
    }
    dst_last[r - 1] = beta_last * t_col0[r - 1];
}

/// Companion-form `T · v` in O(r). `(T·v)[i] = phi[i]·v[0] +
/// v[i+1]` (with the second term zero when `i+1 ≥ r`).
fn t_vec(t_col0: &[f64], v: &[f64], out: &mut [f64], r: usize) {
    for i in 0..r {
        let shifted = if i + 1 < r { v[i + 1] } else { 0.0 };
        out[i] = t_col0[i] * v[0] + shifted;
    }
}

/// Like `t_vec` but accumulates into `out` instead of overwriting.
fn t_vec_add(t_col0: &[f64], v: &[f64], out: &mut [f64], r: usize) {
    for i in 0..r {
        let shifted = if i + 1 < r { v[i + 1] } else { 0.0 };
        out[i] += t_col0[i] * v[0] + shifted;
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
    fn analytic_gradient_matches_central_difference() {
        // ARMA(2, 1): pick a stationary / invertible point and verify
        // the analytic gradient agrees with a tight central-difference
        // gradient on the same series.
        let phi = vec![0.6, -0.2];
        let theta = vec![0.3];
        let y: Vec<f64> = (0..150)
            .map(|i| (i as f64 * 0.17).sin() + 0.5 * (i as f64 * 0.03).cos())
            .collect();
        let (nll, grad) = concentrated_nll_and_grad(&y, &phi, &theta);
        assert!(nll.is_finite());

        let h = 1e-6;
        // ∂/∂φ_1
        let nll_pp = {
            let mut p = phi.clone();
            p[0] += h;
            concentrated_neg_loglik(&y, &p, &theta)
        };
        let nll_pm = {
            let mut p = phi.clone();
            p[0] -= h;
            concentrated_neg_loglik(&y, &p, &theta)
        };
        let g_phi1 = (nll_pp - nll_pm) / (2.0 * h);
        assert_relative_eq!(grad[0], g_phi1, max_relative = 1e-4, epsilon = 1e-6);

        // ∂/∂φ_2
        let nll_pp = {
            let mut p = phi.clone();
            p[1] += h;
            concentrated_neg_loglik(&y, &p, &theta)
        };
        let nll_pm = {
            let mut p = phi.clone();
            p[1] -= h;
            concentrated_neg_loglik(&y, &p, &theta)
        };
        let g_phi2 = (nll_pp - nll_pm) / (2.0 * h);
        assert_relative_eq!(grad[1], g_phi2, max_relative = 1e-4, epsilon = 1e-6);

        // ∂/∂θ_1
        let nll_pp = {
            let mut t = theta.clone();
            t[0] += h;
            concentrated_neg_loglik(&y, &phi, &t)
        };
        let nll_pm = {
            let mut t = theta.clone();
            t[0] -= h;
            concentrated_neg_loglik(&y, &phi, &t)
        };
        let g_theta1 = (nll_pp - nll_pm) / (2.0 * h);
        assert_relative_eq!(grad[2], g_theta1, max_relative = 1e-4, epsilon = 1e-6);
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
