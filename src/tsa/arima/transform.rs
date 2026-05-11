//! Parameter transformations for ARMA stationarity / invertibility
//! (Jones 1980).
//!
//! The optimiser sees an unconstrained ℝ^(p+q) vector. We map each
//! component → a partial autocorrelation ∈ (-1, 1) via the smooth
//! `r / sqrt(1 + r²)` squashing (cheaper and just as well-behaved as
//! `tanh`), then transform the PACF vector into AR / MA polynomial
//! coefficients via the Durbin-Levinson recursion.
//!
//! Key identities:
//!
//! - AR(p) is stationary iff every step of `ar_poly_to_pacf` produces a
//!   PACF in (-1, 1) — equivalently, iff every root of
//!   `1 − φ₁z − ⋯ − φ_p z^p` lies *outside* the unit circle.
//! - MA(q) is invertible iff `1 + θ₁z + ⋯ + θ_q z^q` has all roots
//!   outside the unit circle. With `ψ_i = −θ_i`, this is the AR
//!   stationarity condition on `ψ`, so we re-use the AR machinery and
//!   negate at the boundary.

#[inline]
pub fn real_to_pacf(r: f64) -> f64 {
    r / (1.0 + r * r).sqrt()
}

#[inline]
pub fn pacf_to_real(p: f64) -> f64 {
    // Inverse of `r / sqrt(1 + r²)`: r = p / sqrt(1 - p²).
    let clamped = p.clamp(-0.999_999, 0.999_999);
    clamped / (1.0 - clamped * clamped).sqrt()
}

/// Durbin-Levinson: PACF → AR polynomial coefficients `[φ₁, …, φ_p]`.
pub fn pacf_to_ar_poly(pacf: &[f64]) -> Vec<f64> {
    let p = pacf.len();
    if p == 0 {
        return Vec::new();
    }
    let mut phi = vec![0.0f64; p];
    phi[0] = pacf[0];
    for k in 1..p {
        let r_k = pacf[k];
        // Build new coefficients in `tmp`, then commit.
        let mut tmp = vec![0.0f64; k + 1];
        for j in 0..k {
            tmp[j] = phi[j] - r_k * phi[k - 1 - j];
        }
        tmp[k] = r_k;
        phi[..=k].copy_from_slice(&tmp);
    }
    phi
}

/// Inverse Durbin-Levinson: AR polynomial → PACF. The caller is
/// responsible for any out-of-region handling; we clamp internally to
/// avoid divide-by-zero on degenerate inputs.
pub fn ar_poly_to_pacf(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    if p == 0 {
        return Vec::new();
    }
    let mut cur: Vec<f64> = phi.to_vec();
    let mut pacf = vec![0.0f64; p];
    for k in (0..p).rev() {
        pacf[k] = cur[k];
        if k == 0 {
            break;
        }
        let r = cur[k];
        let denom = 1.0 - r * r;
        if denom.abs() < 1e-12 {
            // Degenerate — leave the remaining PACFs at their current
            // value and bail. shrink_to_stationary() in the parent
            // module will then back off the seed.
            return pacf;
        }
        let prev: Vec<f64> = (0..k)
            .map(|j| (cur[j] + r * cur[k - 1 - j]) / denom)
            .collect();
        cur[..k].copy_from_slice(&prev);
    }
    pacf
}

/// PACF → MA polynomial coefficients `[θ₁, …, θ_q]`. Uses the AR
/// machinery on the sign-flipped polynomial (MA invertibility ↔ AR
/// stationarity of `−θ`).
pub fn pacf_to_ma_poly(pacf: &[f64]) -> Vec<f64> {
    pacf_to_ar_poly(pacf).into_iter().map(|c| -c).collect()
}

pub fn ma_poly_to_pacf(theta: &[f64]) -> Vec<f64> {
    let neg: Vec<f64> = theta.iter().map(|t| -t).collect();
    ar_poly_to_pacf(&neg)
}

#[cfg(test)]
mod transform_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn real_pacf_roundtrip() {
        for r in [-3.0_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0] {
            let p = real_to_pacf(r);
            assert!(p.abs() < 1.0);
            let r2 = pacf_to_real(p);
            assert_relative_eq!(r, r2, max_relative = 1e-12);
        }
    }

    #[test]
    fn ar_poly_pacf_roundtrip() {
        // Hand-chosen stationary AR(3): pacf = [0.5, 0.3, -0.2].
        let pacf = vec![0.5, 0.3, -0.2];
        let phi = pacf_to_ar_poly(&pacf);
        let back = ar_poly_to_pacf(&phi);
        for (a, b) in pacf.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn ma_poly_pacf_roundtrip() {
        let pacf = vec![0.4, -0.2];
        let theta = pacf_to_ma_poly(&pacf);
        let back = ma_poly_to_pacf(&theta);
        for (a, b) in pacf.iter().zip(back.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    #[test]
    fn ar2_known() {
        // For AR(2) with pacf = [r_1, r_2]:
        //   φ_1 = r_1 (1 - r_2), φ_2 = r_2.
        let phi = pacf_to_ar_poly(&[0.4, 0.3]);
        assert_relative_eq!(phi[0], 0.4 * (1.0 - 0.3), max_relative = 1e-12);
        assert_relative_eq!(phi[1], 0.3, max_relative = 1e-12);
    }
}
