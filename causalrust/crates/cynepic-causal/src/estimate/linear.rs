use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Average Treatment Effect (ATE) estimator via ordinary least squares.
///
/// Given treatment assignment T, outcome Y, and covariates X,
/// fits Y = β₀ + β₁·T + β₂·X + ε and returns β₁ as the ATE estimate.
pub struct LinearATEEstimator;

/// Result of an ATE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATEResult {
    /// The estimated average treatment effect.
    pub ate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Number of observations.
    pub n_obs: usize,
}

impl LinearATEEstimator {
    /// Estimate the ATE using simple difference-in-means.
    ///
    /// This is the simplest estimator: ATE = E[Y|T=1] - E[Y|T=0].
    /// Suitable when the backdoor criterion is satisfied and adjustment
    /// is handled by sample selection.
    pub fn difference_in_means(treatment: &Array1<f64>, outcome: &Array1<f64>) -> ATEResult {
        assert_eq!(treatment.len(), outcome.len(), "treatment and outcome must have same length");

        let n = treatment.len();
        let mut sum_treated = 0.0;
        let mut sum_control = 0.0;
        let mut n_treated = 0usize;
        let mut n_control = 0usize;

        for i in 0..n {
            if treatment[i] > 0.5 {
                sum_treated += outcome[i];
                n_treated += 1;
            } else {
                sum_control += outcome[i];
                n_control += 1;
            }
        }

        let mean_treated = if n_treated > 0 {
            sum_treated / n_treated as f64
        } else {
            0.0
        };
        let mean_control = if n_control > 0 {
            sum_control / n_control as f64
        } else {
            0.0
        };

        let ate = mean_treated - mean_control;

        // Standard error via pooled variance
        let mut var_treated = 0.0;
        let mut var_control = 0.0;
        for i in 0..n {
            if treatment[i] > 0.5 {
                var_treated += (outcome[i] - mean_treated).powi(2);
            } else {
                var_control += (outcome[i] - mean_control).powi(2);
            }
        }
        let se = if n_treated > 1 && n_control > 1 {
            let var_t = var_treated / (n_treated - 1) as f64;
            let var_c = var_control / (n_control - 1) as f64;
            (var_t / n_treated as f64 + var_c / n_control as f64).sqrt()
        } else {
            f64::NAN
        };

        ATEResult {
            ate,
            std_error: se,
            n_obs: n,
        }
    }

    /// Estimate ATE via OLS with covariate adjustment.
    ///
    /// Fits Y = β₀ + β₁·T + X·β₂ via the normal equation: β = (X'X)⁻¹X'Y
    /// Returns β₁ as the ATE.
    pub fn ols_adjusted(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> ATEResult {
        let n = treatment.len();
        assert_eq!(n, outcome.len());
        assert_eq!(n, covariates.nrows());

        let p = covariates.ncols();
        let cols = p + 2; // intercept + treatment + covariates

        // Build X'X and X'Y using the design matrix X = [1 | T | covariates]
        let mut xt_x = vec![0.0; cols * cols];
        let mut xt_y = vec![0.0; cols];

        for i in 0..n {
            // Build row: [1, T_i, cov_i1, ..., cov_ip]
            let mut x_row = Vec::with_capacity(cols);
            x_row.push(1.0);
            x_row.push(treatment[i]);
            for j in 0..p {
                x_row.push(covariates[[i, j]]);
            }

            // Accumulate X'X
            for r in 0..cols {
                for c in 0..cols {
                    xt_x[r * cols + c] += x_row[r] * x_row[c];
                }
                xt_y[r] += x_row[r] * outcome[i];
            }
        }

        // Solve (X'X) * beta = X'Y via Gaussian elimination with partial pivoting
        let beta = solve_normal_equation(&xt_x, &xt_y, cols);
        let ate = beta[1]; // coefficient on treatment

        // Compute residual standard error
        let mut rss = 0.0;
        for i in 0..n {
            let mut y_hat = beta[0] + beta[1] * treatment[i];
            for j in 0..p {
                y_hat += beta[j + 2] * covariates[[i, j]];
            }
            rss += (outcome[i] - y_hat).powi(2);
        }
        let dof = n as f64 - cols as f64;
        let sigma2 = if dof > 0.0 { rss / dof } else { f64::NAN };

        // SE of beta[1] = sqrt(sigma^2 * (X'X)^{-1}[1,1])
        let xt_x_inv = invert_matrix(&xt_x, cols);
        let std_error = (sigma2 * xt_x_inv[1 * cols + 1]).sqrt();

        ATEResult {
            ate,
            std_error,
            n_obs: n,
        }
    }
}

/// Solve a linear system A*x = b using Gaussian elimination with partial pivoting.
/// `a` is stored row-major with dimension `dim x dim`.
fn solve_normal_equation(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
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
        let mut max_val = aug[col * (dim + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..dim {
            let val = aug[row * (dim + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            for c in 0..=dim {
                let tmp = aug[col * (dim + 1) + c];
                aug[col * (dim + 1) + c] = aug[max_row * (dim + 1) + c];
                aug[max_row * (dim + 1) + c] = tmp;
            }
        }

        let pivot = aug[col * (dim + 1) + col];
        if pivot.abs() < 1e-12 {
            return vec![0.0; dim];
        }

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

/// Invert a matrix stored row-major with dimension `dim x dim` using Gauss-Jordan.
fn invert_matrix(a: &[f64], dim: usize) -> Vec<f64> {
    let mut aug = vec![0.0; dim * (2 * dim)];
    for r in 0..dim {
        for c in 0..dim {
            aug[r * (2 * dim) + c] = a[r * dim + c];
        }
        aug[r * (2 * dim) + dim + r] = 1.0;
    }

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
    use ndarray::{array, Array2};

    #[test]
    fn difference_in_means_basic() {
        // Treatment group: outcomes [10, 12, 11]
        // Control group: outcomes [5, 6, 7]
        // True ATE = 11 - 6 = 5
        let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let outcome = array![10.0, 12.0, 11.0, 5.0, 6.0, 7.0];

        let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);
        assert!((result.ate - 5.0).abs() < 1e-10);
        assert_eq!(result.n_obs, 6);
    }

    #[test]
    fn difference_in_means_no_effect() {
        let treatment = array![1.0, 1.0, 0.0, 0.0];
        let outcome = array![5.0, 5.0, 5.0, 5.0];

        let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);
        assert!((result.ate).abs() < 1e-10);
    }

    #[test]
    fn ols_adjusted_known_effect_with_confounder() {
        // Y = 2 + 5*T + 3*Z + noise
        // Z confounds: affects both T and Y.
        // True causal effect of T on Y is 5.0.
        let n = 200;
        let mut treatment = ndarray::Array1::zeros(n);
        let mut outcome = ndarray::Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));

        let mut rng_state: u64 = 42;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (2.0_f64.powi(31)) - 0.5
        };

        for i in 0..n {
            let z = next_rng(&mut rng_state) * 4.0; // confounder
            covariates[[i, 0]] = z;
            treatment[i] = if z + next_rng(&mut rng_state) > 0.0 {
                1.0
            } else {
                0.0
            };
            outcome[i] = 2.0 + 5.0 * treatment[i] + 3.0 * z + next_rng(&mut rng_state) * 0.5;
        }

        let result = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates);

        assert!(
            (result.ate - 5.0).abs() < 0.5,
            "OLS adjusted ATE = {}, expected ~5.0",
            result.ate
        );
        assert_eq!(result.n_obs, n);
    }

    #[test]
    fn ols_adjustment_changes_estimate() {
        // Without adjustment, confounding biases the estimate.
        // With adjustment, we recover the true effect.
        // Y = 3*T + 4*Z, where Z -> T and Z -> Y
        let n = 300;
        let mut treatment = ndarray::Array1::zeros(n);
        let mut outcome = ndarray::Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));

        let mut rng_state: u64 = 12345;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (2.0_f64.powi(31)) - 0.5
        };

        for i in 0..n {
            let z = next_rng(&mut rng_state) * 2.0;
            covariates[[i, 0]] = z;
            // Treatment strongly influenced by confounder
            treatment[i] = if z + next_rng(&mut rng_state) * 0.3 > 0.0 {
                1.0
            } else {
                0.0
            };
            outcome[i] = 3.0 * treatment[i] + 4.0 * z + next_rng(&mut rng_state) * 0.2;
        }

        let naive = LinearATEEstimator::difference_in_means(&treatment, &outcome);
        let adjusted = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates);

        // The naive estimate should be biased (far from 3.0)
        // The adjusted estimate should be close to 3.0
        assert!(
            (adjusted.ate - 3.0).abs() < (naive.ate - 3.0).abs(),
            "Adjustment should reduce bias: naive={}, adjusted={}",
            naive.ate,
            adjusted.ate
        );
        assert!(
            (adjusted.ate - 3.0).abs() < 0.5,
            "OLS adjusted ATE = {}, expected ~3.0",
            adjusted.ate
        );
    }
}
