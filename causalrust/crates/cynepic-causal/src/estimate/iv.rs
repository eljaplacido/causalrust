//! Instrumental variable estimation via Two-Stage Least Squares (2SLS).
//!
//! Uses instruments to identify causal effects when there is unmeasured
//! confounding between treatment and outcome.

use ndarray::{Array1, Array2};

use super::linear::ATEResult;

/// Instrumental variable estimator.
pub struct IVEstimator;

impl IVEstimator {
    /// Two-stage least squares (2SLS) estimation.
    ///
    /// Stage 1: Regress treatment on instrument(s) to get predicted treatment.
    /// Stage 2: Regress outcome on predicted treatment to get causal effect.
    ///
    /// Assumes instruments are valid: correlated with treatment but not with
    /// the error term in the outcome equation.
    pub fn two_stage_ls(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        instruments: &Array2<f64>,
    ) -> ATEResult {
        let n = treatment.len();
        assert_eq!(n, outcome.len());
        assert_eq!(n, instruments.nrows());

        let k = instruments.ncols();

        // Stage 1: Regress treatment on instruments (with intercept).
        // Design matrix Z = [1 | instruments]
        // Solve: treatment = Z * gamma via normal equations.
        let cols = k + 1;
        let mut zt_z = vec![0.0; cols * cols]; // Z'Z
        let mut zt_t = vec![0.0; cols]; // Z'T

        for i in 0..n {
            // Build row of Z
            let mut z_row = vec![1.0]; // intercept
            for j in 0..k {
                z_row.push(instruments[[i, j]]);
            }

            // Accumulate Z'Z
            for r in 0..cols {
                for c in 0..cols {
                    zt_z[r * cols + c] += z_row[r] * z_row[c];
                }
                zt_t[r] += z_row[r] * treatment[i];
            }
        }

        // Solve Z'Z * gamma = Z'T
        let gamma = solve_linear_system(&zt_z, &zt_t, cols);

        // Compute predicted treatment: T_hat = Z * gamma
        let mut t_hat = Array1::zeros(n);
        for i in 0..n {
            t_hat[i] = gamma[0]; // intercept
            for j in 0..k {
                t_hat[i] += gamma[j + 1] * instruments[[i, j]];
            }
        }

        // Stage 2: Regress outcome on predicted treatment (with intercept).
        // Design matrix X2 = [1 | T_hat]
        let cols2 = 2;
        let mut xt_x = vec![0.0; cols2 * cols2];
        let mut xt_y = vec![0.0; cols2];

        for i in 0..n {
            let x_row = [1.0, t_hat[i]];
            for r in 0..cols2 {
                for c in 0..cols2 {
                    xt_x[r * cols2 + c] += x_row[r] * x_row[c];
                }
                xt_y[r] += x_row[r] * outcome[i];
            }
        }

        let beta = solve_linear_system(&xt_x, &xt_y, cols2);
        let ate = beta[1]; // coefficient on predicted treatment

        // Compute residuals and standard error
        let mut rss = 0.0;
        for i in 0..n {
            let y_hat = beta[0] + beta[1] * t_hat[i];
            rss += (outcome[i] - y_hat).powi(2);
        }
        let sigma2 = rss / (n as f64 - 2.0);

        // SE of beta[1]: sqrt(sigma^2 * (X'X)^{-1}[1,1])
        let xt_x_inv = invert_matrix(&xt_x, cols2);
        let std_error = (sigma2 * xt_x_inv[1 * cols2 + 1]).sqrt();

        ATEResult {
            ate,
            std_error,
            n_obs: n,
        }
    }
}

/// Solve a linear system A*x = b using Gaussian elimination with partial pivoting.
/// `a` is stored row-major with dimension `dim x dim`.
fn solve_linear_system(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
    // Augmented matrix [A | b]
    let mut aug = vec![0.0; dim * (dim + 1)];
    for r in 0..dim {
        for c in 0..dim {
            aug[r * (dim + 1) + c] = a[r * dim + c];
        }
        aug[r * (dim + 1) + dim] = b[r];
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_val = aug[col * (dim + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..dim {
            let val = aug[row * (dim + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for c in 0..=dim {
                let tmp = aug[col * (dim + 1) + c];
                aug[col * (dim + 1) + c] = aug[max_row * (dim + 1) + c];
                aug[max_row * (dim + 1) + c] = tmp;
            }
        }

        let pivot = aug[col * (dim + 1) + col];
        if pivot.abs() < 1e-12 {
            // Singular or near-singular: return zeros
            return vec![0.0; dim];
        }

        // Eliminate below
        for row in (col + 1)..dim {
            let factor = aug[row * (dim + 1) + col] / pivot;
            for c in col..=dim {
                aug[row * (dim + 1) + c] -= factor * aug[col * (dim + 1) + c];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; dim];
    for row in (0..dim).rev() {
        let mut sum = aug[row * (dim + 1) + dim];
        for c in (row + 1)..dim {
            sum -= aug[row * (dim + 1) + c] * x[c];
        }
        x[row] = sum / aug[row * (dim + 1) + row];
    }

    x
}

/// Invert a matrix stored row-major with dimension `dim x dim`.
fn invert_matrix(a: &[f64], dim: usize) -> Vec<f64> {
    // Augmented matrix [A | I]
    let mut aug = vec![0.0; dim * (2 * dim)];
    for r in 0..dim {
        for c in 0..dim {
            aug[r * (2 * dim) + c] = a[r * dim + c];
        }
        aug[r * (2 * dim) + dim + r] = 1.0;
    }

    // Forward elimination
    for col in 0..dim {
        let mut max_val = aug[col * (2 * dim) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..dim {
            let val = aug[row * (2 * dim) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            for c in 0..(2 * dim) {
                let tmp = aug[col * (2 * dim) + c];
                aug[col * (2 * dim) + c] = aug[max_row * (2 * dim) + c];
                aug[max_row * (2 * dim) + c] = tmp;
            }
        }

        let pivot = aug[col * (2 * dim) + col];
        if pivot.abs() < 1e-12 {
            return vec![0.0; dim * dim];
        }

        for c in 0..(2 * dim) {
            aug[col * (2 * dim) + c] /= pivot;
        }

        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row * (2 * dim) + col];
            for c in 0..(2 * dim) {
                aug[row * (2 * dim) + c] -= factor * aug[col * (2 * dim) + c];
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            inv[r * dim + c] = aug[r * (2 * dim) + dim + c];
        }
    }

    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn iv_known_effect() {
        // Scenario: Z -> T -> Y, with unmeasured confounder U -> T, U -> Y.
        // True causal effect of T on Y is 3.0.
        // Instrument Z affects T but not Y directly.
        let n = 500;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut instruments = Array2::zeros((n, 1));

        let mut rng_state: u64 = 99999;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (2.0_f64.powi(31)) - 0.5
        };

        for i in 0..n {
            let z = next_rng(&mut rng_state) * 2.0; // instrument
            let u = next_rng(&mut rng_state); // unmeasured confounder
            let noise_t = next_rng(&mut rng_state) * 0.3;
            let noise_y = next_rng(&mut rng_state) * 0.3;

            instruments[[i, 0]] = z;
            treatment[i] = 2.0 * z + 1.5 * u + noise_t; // T depends on Z and U
            outcome[i] = 3.0 * treatment[i] - 1.5 * u + noise_y; // Y depends on T and U
            // OLS would be biased because U confounds T->Y.
            // 2SLS using Z should recover ~3.0.
        }

        let result = IVEstimator::two_stage_ls(&treatment, &outcome, &instruments);

        assert!(
            (result.ate - 3.0).abs() < 1.0,
            "2SLS ATE = {}, expected ~3.0",
            result.ate
        );
        assert_eq!(result.n_obs, n);
    }

    #[test]
    fn iv_simple_wald() {
        // Simple binary-like instrument scenario.
        // Z = 0 or 1, T = Z + noise, Y = 2*T + noise.
        // Wald estimator: ATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[T|Z=1] - E[T|Z=0])
        let n = 400;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut instruments = Array2::zeros((n, 1));

        let mut rng_state: u64 = 77777;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (2.0_f64.powi(31))
        };

        for i in 0..n {
            let z = if i < n / 2 { 0.0 } else { 1.0 };
            instruments[[i, 0]] = z;
            treatment[i] = z + next_rng(&mut rng_state) * 0.2;
            outcome[i] = 2.0 * treatment[i] + next_rng(&mut rng_state) * 0.3;
        }

        let result = IVEstimator::two_stage_ls(&treatment, &outcome, &instruments);

        assert!(
            (result.ate - 2.0).abs() < 0.5,
            "2SLS ATE = {}, expected ~2.0",
            result.ate
        );
    }
}
