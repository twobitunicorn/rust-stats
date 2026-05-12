//! L-BFGS quasi-Newton optimiser with **strong-Wolfe line search**.
//!
//! Built around Nocedal & Wright (2nd ed., 2006) Algorithm 7.5 plus
//! Algorithms 3.5 / 3.6 for the line search. The strong-Wolfe
//! conditions on the step length `α`:
//!
//! ```text
//!     f(x + α d) ≤ f(x) + c₁ α ∇f(x)ᵀd                  (Armijo)
//!     |∇f(x + α d)ᵀd| ≤ c₂ |∇f(x)ᵀd|                    (curvature)
//! ```
//!
//! with `c₁ = 1e-4`, `c₂ = 0.9`. Together they guarantee both
//! sufficient decrease and meaningful progress along the search
//! direction — the latter is what backtracking-Armijo alone misses.
//! On the Kalman likelihood surface with finite-difference gradients,
//! Armijo alone accepts tiny-step iterates that stall L-BFGS; strong
//! Wolfe rejects those and forces actual movement.
//!
//! Gradients are computed by central differences (`2·n+1` objective
//! evaluations per gradient call, scaled step `h = 1e-5·(1+|xᵢ|)`).
//! The line search reuses the gradient it computes at the accepted
//! step, so the outer L-BFGS loop pays no extra gradient cost per
//! iteration.

const HISTORY_SIZE: usize = 10;
const FD_STEP: f64 = 1e-5;
const C1: f64 = 1e-4;
const C2: f64 = 0.9;
const MAX_LINE_SEARCH_OUTER: usize = 25;
const MAX_LINE_SEARCH_ZOOM: usize = 25;
const ALPHA_MAX: f64 = 1e6;
const ALPHA_MIN_RESOLUTION: f64 = 1e-15;

/// Minimise `f` from `x0` by L-BFGS with strong-Wolfe line search.
/// Returns `(x_star, f_star, converged)`; `converged` is true iff
/// `‖∇f‖₂ < grad_tol` within `max_iter` outer steps.
pub fn minimize<F>(
    x0: &[f64],
    f: &F,
    max_iter: usize,
    grad_tol: f64,
) -> (Vec<f64>, f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return (Vec::new(), f(x0), true);
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut grad = numerical_gradient(f, &mut x);
    if !fx.is_finite() || grad.iter().any(|g| !g.is_finite()) {
        return (x, fx, false);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY_SIZE);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(HISTORY_SIZE);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(HISTORY_SIZE);

    for _ in 0..max_iter {
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < grad_tol {
            return (x, fx, true);
        }

        let mut direction = two_loop(&grad, &s_hist, &y_hist, &rho_hist);

        // Ensure descent. If the L-BFGS direction has bad curvature info
        // (early iterations or a stale history), fall back to scaled
        // steepest descent.
        let g_dot_d: f64 = grad.iter().zip(&direction).map(|(g, d)| g * d).sum();
        let descent = g_dot_d.is_finite() && g_dot_d < 0.0;
        if !descent {
            direction = grad.iter().map(|g| -g).collect();
        }

        // First iteration: scale the steepest-descent step so its norm
        // is ~1 (Nocedal & Wright §7.2 initial guess). For later
        // iterations the L-BFGS curvature info already produces a good
        // initial α = 1.
        let alpha_init = if s_hist.is_empty() {
            let dir_norm: f64 = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
            if dir_norm > 1.0 { 1.0 / dir_norm } else { 1.0 }
        } else {
            1.0
        };

        let g_dot_d: f64 = grad.iter().zip(&direction).map(|(g, d)| g * d).sum();
        let ls = strong_wolfe(f, &x, fx, &grad, &direction, g_dot_d, alpha_init);
        let Some((_alpha, x_new, fx_new, grad_new)) = ls else {
            return (x, fx, false);
        };

        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = grad_new.iter().zip(&grad).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        if sy > 1e-10 {
            if s_hist.len() == HISTORY_SIZE {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    (x, fx, false)
}

/// Central-difference gradient. `x` is borrowed mutably and restored
/// in-place so each per-coordinate perturbation is allocation-free.
fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &mut [f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    for i in 0..n {
        let orig = x[i];
        let h = FD_STEP * (1.0 + orig.abs());
        x[i] = orig + h;
        let f_plus = f(x);
        x[i] = orig - h;
        let f_minus = f(x);
        x[i] = orig;
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

/// L-BFGS two-loop recursion. Returns the search direction `-Hₖ·∇f`.
fn two_loop(grad: &[f64], s_hist: &[Vec<f64>], y_hist: &[Vec<f64>], rho_hist: &[f64]) -> Vec<f64> {
    let n = grad.len();
    let m = s_hist.len();
    let mut q: Vec<f64> = grad.to_vec();
    let mut alpha = vec![0.0f64; m];

    for i in (0..m).rev() {
        let sq: f64 = s_hist[i].iter().zip(&q).map(|(a, b)| a * b).sum();
        alpha[i] = rho_hist[i] * sq;
        for j in 0..n {
            q[j] -= alpha[i] * y_hist[i][j];
        }
    }

    let gamma = if m > 0 {
        let last = m - 1;
        let sy: f64 = s_hist[last]
            .iter()
            .zip(&y_hist[last])
            .map(|(a, b)| a * b)
            .sum();
        let yy: f64 = y_hist[last].iter().map(|v| v * v).sum();
        if yy > 1e-10 { sy / yy } else { 1.0 }
    } else {
        1.0
    };
    let mut r: Vec<f64> = q.iter().map(|v| gamma * v).collect();

    for i in 0..m {
        let yr: f64 = y_hist[i].iter().zip(&r).map(|(a, b)| a * b).sum();
        let beta = rho_hist[i] * yr;
        for j in 0..n {
            r[j] += (alpha[i] - beta) * s_hist[i][j];
        }
    }

    for v in r.iter_mut() {
        *v = -*v;
    }
    r
}

/// Strong-Wolfe line search (Nocedal & Wright Algorithm 3.5).
/// Returns `(α, x_new, f_new, grad_new)` or `None` on failure.
fn strong_wolfe<F: Fn(&[f64]) -> f64>(
    f: &F,
    x: &[f64],
    fx: f64,
    _grad: &[f64],
    direction: &[f64],
    g_dot_d: f64,
    alpha_init: f64,
) -> Option<(f64, Vec<f64>, f64, Vec<f64>)> {
    if g_dot_d >= 0.0 {
        return None;
    }
    let n = x.len();
    let mut alpha_prev = 0.0_f64;
    let mut fx_prev = fx;
    let mut alpha = alpha_init.min(ALPHA_MAX);

    for i in 0..MAX_LINE_SEARCH_OUTER {
        let mut x_try = vec![0.0; n];
        for j in 0..n {
            x_try[j] = x[j] + alpha * direction[j];
        }
        let fx_try = f(&x_try);

        // Armijo failure or non-monotone: zoom in [α_prev, α].
        if !fx_try.is_finite()
            || fx_try > fx + C1 * alpha * g_dot_d
            || (i > 0 && fx_try >= fx_prev)
        {
            return zoom(
                f, x, fx, direction, g_dot_d, alpha_prev, alpha, fx_prev, fx_try,
            );
        }

        let mut x_for_grad = x_try.clone();
        let grad_try = numerical_gradient(f, &mut x_for_grad);
        let g_dot_d_try: f64 = grad_try.iter().zip(direction).map(|(g, d)| g * d).sum();

        // Strong curvature condition satisfied → accept.
        if g_dot_d_try.abs() <= -C2 * g_dot_d {
            return Some((alpha, x_try, fx_try, grad_try));
        }

        // Past the minimum (positive directional derivative): zoom
        // with reversed bracket.
        if g_dot_d_try >= 0.0 {
            return zoom(
                f, x, fx, direction, g_dot_d, alpha, alpha_prev, fx_try, fx_prev,
            );
        }

        // Step still too short — expand.
        alpha_prev = alpha;
        fx_prev = fx_try;
        let new_alpha = (alpha * 2.0).min(ALPHA_MAX);
        if (new_alpha - alpha).abs() < ALPHA_MIN_RESOLUTION {
            return None;
        }
        alpha = new_alpha;
    }
    None
}

/// Zoom phase of the strong-Wolfe search (Nocedal & Wright Alg. 3.6).
/// `alpha_lo`/`alpha_hi` bracket a step satisfying both Wolfe
/// conditions; we narrow by bisection until one is found.
#[allow(clippy::too_many_arguments)]
fn zoom<F: Fn(&[f64]) -> f64>(
    f: &F,
    x: &[f64],
    fx: f64,
    direction: &[f64],
    g_dot_d: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut fx_lo: f64,
    mut _fx_hi: f64,
) -> Option<(f64, Vec<f64>, f64, Vec<f64>)> {
    let n = x.len();
    for _ in 0..MAX_LINE_SEARCH_ZOOM {
        // Bisection. Cubic / quadratic interpolation would converge
        // a step or two faster, but bisection is shorter to write and
        // is what More-Thuente safeguard back-stops to anyway.
        let alpha = 0.5 * (alpha_lo + alpha_hi);
        if (alpha_hi - alpha_lo).abs() < ALPHA_MIN_RESOLUTION {
            // Bracket collapsed — accept what we have.
            let mut x_try = vec![0.0; n];
            for j in 0..n {
                x_try[j] = x[j] + alpha_lo * direction[j];
            }
            let fx_try = f(&x_try);
            let mut x_for_grad = x_try.clone();
            let grad_try = numerical_gradient(f, &mut x_for_grad);
            return Some((alpha_lo, x_try, fx_try, grad_try));
        }

        let mut x_try = vec![0.0; n];
        for j in 0..n {
            x_try[j] = x[j] + alpha * direction[j];
        }
        let fx_try = f(&x_try);

        if !fx_try.is_finite() || fx_try > fx + C1 * alpha * g_dot_d || fx_try >= fx_lo {
            alpha_hi = alpha;
            _fx_hi = fx_try;
        } else {
            let mut x_for_grad = x_try.clone();
            let grad_try = numerical_gradient(f, &mut x_for_grad);
            let g_dot_d_try: f64 = grad_try.iter().zip(direction).map(|(g, d)| g * d).sum();
            if g_dot_d_try.abs() <= -C2 * g_dot_d {
                return Some((alpha, x_try, fx_try, grad_try));
            }
            if g_dot_d_try * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                _fx_hi = fx_lo;
            }
            alpha_lo = alpha;
            fx_lo = fx_try;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn finds_minimum_of_quadratic() {
        let f = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 2.0).powi(2);
        let (x, fmin, ok) = minimize(&[0.0, 0.0], &f, 100, 1e-8);
        assert!(ok);
        assert_relative_eq!(x[0], 3.0, max_relative = 1e-4);
        assert_relative_eq!(x[1], -2.0, max_relative = 1e-4);
        assert!(fmin < 1e-8);
    }

    #[test]
    fn finds_minimum_of_rosenbrock_2d() {
        let f = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        // Strong Wolfe makes this much easier than backtracking Armijo.
        let (x, fmin, ok) = minimize(&[-1.2, 1.0], &f, 500, 1e-6);
        assert!(ok, "Rosenbrock failed to converge");
        assert_relative_eq!(x[0], 1.0, max_relative = 1e-4);
        assert_relative_eq!(x[1], 1.0, max_relative = 1e-4);
        assert!(fmin < 1e-8);
    }

    #[test]
    fn handles_higher_dimension() {
        let f = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, x)| (x - i as f64).powi(2))
                .sum::<f64>()
        };
        let (x, fmin, ok) = minimize(&vec![0.0; 10], &f, 200, 1e-8);
        assert!(ok);
        for (i, xi) in x.iter().enumerate() {
            assert_relative_eq!(*xi, i as f64, max_relative = 1e-4);
        }
        assert!(fmin < 1e-6);
    }

    #[test]
    fn handles_flat_plateau() {
        // f(x, y) = (x² + y² - 1)² has a circular minimum at radius 1.
        // The plateau is exactly the kind of surface where Armijo
        // line search stalls — strong Wolfe should handle it cleanly.
        let f = |v: &[f64]| (v[0] * v[0] + v[1] * v[1] - 1.0).powi(2);
        let (_x, fmin, _ok) = minimize(&[0.5, 0.5], &f, 200, 1e-6);
        assert!(fmin < 1e-6, "fmin = {fmin}");
    }
}
