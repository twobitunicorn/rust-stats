//! Nelder-Mead simplex optimiser for the CSS objective.
//!
//! No gradient required, which suits the PACF-reparameterised CSS
//! surface (smooth but non-convex). The textbook coefficient settings
//! (α=1, γ=2, ρ=0.5, σ=0.5) are robust for the small parameter spaces
//! typical of ARIMA (`p + q ≤ 20`). Convergence is declared when the
//! simplex shrinks below `tol` in both function-value spread and vertex
//! spread.

/// Minimise `f` starting from `x0`. Returns `(x_min, f_min, converged)`.
pub fn minimize<F>(x0: &[f64], f: &F, max_iter: usize, tol: f64) -> (Vec<f64>, f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return (Vec::new(), f(x0), true);
    }

    // Initialise simplex: x0 plus n perturbed vertices.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        let step = if v[i].abs() > 1e-8 { 0.05 * v[i] } else { 0.05 };
        v[i] += step;
        simplex.push(v);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5;   // contraction
    let sigma = 0.5; // shrink

    let mut iter = 0;
    while iter < max_iter {
        // 1. Order by function value (best first).
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        // 2. Convergence checks.
        let f_spread = fvals[n] - fvals[0];
        let x_spread = vertex_spread(&simplex);
        if f_spread < tol && x_spread < tol {
            return (simplex[0].clone(), fvals[0], true);
        }

        // 3. Centroid of the n best vertices (exclude worst).
        let mut centroid = vec![0.0f64; n];
        for v in &simplex[..n] {
            for i in 0..n {
                centroid[i] += v[i];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // 4. Reflection.
        let worst = &simplex[n];
        let xr: Vec<f64> = (0..n).map(|i| centroid[i] + alpha * (centroid[i] - worst[i])).collect();
        let fr = f(&xr);
        if fvals[0] <= fr && fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
            iter += 1;
            continue;
        }

        // 5. Expansion.
        if fr < fvals[0] {
            let xe: Vec<f64> = (0..n).map(|i| centroid[i] + gamma * (xr[i] - centroid[i])).collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
            iter += 1;
            continue;
        }

        // 6. Contraction.
        let xc: Vec<f64> = (0..n).map(|i| centroid[i] + rho * (worst[i] - centroid[i])).collect();
        let fc = f(&xc);
        if fc < fvals[n] {
            simplex[n] = xc;
            fvals[n] = fc;
            iter += 1;
            continue;
        }

        // 7. Shrink toward the best vertex.
        let best = simplex[0].clone();
        for i in 1..=n {
            for j in 0..n {
                simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
            }
            fvals[i] = f(&simplex[i]);
        }
        iter += 1;
    }

    let mut best_i = 0;
    for i in 1..=n {
        if fvals[i] < fvals[best_i] {
            best_i = i;
        }
    }
    (simplex[best_i].clone(), fvals[best_i], false)
}

fn vertex_spread(simplex: &[Vec<f64>]) -> f64 {
    let n = simplex[0].len();
    let mut max = 0.0f64;
    for j in 0..n {
        let lo = simplex.iter().map(|v| v[j]).fold(f64::INFINITY, f64::min);
        let hi = simplex.iter().map(|v| v[j]).fold(f64::NEG_INFINITY, f64::max);
        let spread = hi - lo;
        if spread > max {
            max = spread;
        }
    }
    max
}

#[cfg(test)]
mod nm_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn finds_minimum_of_quadratic() {
        // f(x, y) = (x - 3)² + (y + 2)² has minimum at (3, -2), f = 0.
        let f = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 2.0).powi(2);
        let (x, fmin, ok) = minimize(&[0.0, 0.0], &f, 1_000, 1e-10);
        assert!(ok);
        assert_relative_eq!(x[0], 3.0, max_relative = 1e-4);
        assert_relative_eq!(x[1], -2.0, max_relative = 1e-4);
        assert!(fmin < 1e-8, "fmin should be near 0, got {fmin}");
    }

    #[test]
    fn finds_minimum_of_rosenbrock_2d() {
        // Standard Rosenbrock test, minimum at (1, 1).
        let f = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        let (x, _f, ok) = minimize(&[-1.2, 1.0], &f, 5_000, 1e-10);
        assert!(ok);
        assert_relative_eq!(x[0], 1.0, max_relative = 1e-3);
        assert_relative_eq!(x[1], 1.0, max_relative = 1e-3);
    }
}
