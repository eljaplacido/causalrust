//! Propensity score estimator using inverse probability weighting (IPW).
//!
//! Estimates the Average Treatment Effect by:
//! 1. Fitting a logistic regression model for treatment assignment
//! 2. Computing inverse probability weights
//! 3. Calculating the weighted mean difference in outcomes

use ndarray::{Array1, Array2};

use super::linear::ATEResult;

/// Propensity score estimator for causal effect estimation.
pub struct PropensityScoreEstimator;

impl PropensityScoreEstimator {
    /// Estimate ATE using inverse probability weighting (IPW).
    ///
    /// 1. Estimate propensity scores via logistic regression (gradient descent)
    /// 2. Compute IPW weights: w_i = T_i/e_i + (1-T_i)/(1-e_i)
    /// 3. ATE = weighted mean difference of outcomes
    pub fn ipw(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> ATEResult {
        let n = treatment.len();
        assert_eq!(n, outcome.len());
        assert_eq!(n, covariates.nrows());

        let p = covariates.ncols();

        // Fit logistic regression: P(T=1 | X) via gradient descent.
        // Parameters: intercept + one per covariate.
        let mut beta = vec![0.0; p + 1]; // [intercept, beta_1, ..., beta_p]

        let learning_rate = 0.1;
        let n_iterations = 100;

        for _ in 0..n_iterations {
            let mut grad = vec![0.0; p + 1];

            for i in 0..n {
                let mut linear_pred = beta[0]; // intercept
                for j in 0..p {
                    linear_pred += beta[j + 1] * covariates[[i, j]];
                }
                let prob = sigmoid(linear_pred);
                let error = treatment[i] - prob;

                // Gradient of log-likelihood
                grad[0] += error; // intercept
                for j in 0..p {
                    grad[j + 1] += error * covariates[[i, j]];
                }
            }

            // Update parameters
            for k in 0..beta.len() {
                beta[k] += learning_rate * grad[k] / n as f64;
            }
        }

        // Compute propensity scores
        let propensity: Vec<f64> = (0..n)
            .map(|i| {
                let mut lp = beta[0];
                for j in 0..p {
                    lp += beta[j + 1] * covariates[[i, j]];
                }
                // Clip to avoid division by zero
                sigmoid(lp).clamp(0.01, 0.99)
            })
            .collect();

        // Compute IPW estimate of ATE
        let mut weighted_treated_sum = 0.0;
        let mut weighted_treated_count = 0.0;
        let mut weighted_control_sum = 0.0;
        let mut weighted_control_count = 0.0;

        for i in 0..n {
            let e = propensity[i];
            if treatment[i] > 0.5 {
                let w = 1.0 / e;
                weighted_treated_sum += w * outcome[i];
                weighted_treated_count += w;
            } else {
                let w = 1.0 / (1.0 - e);
                weighted_control_sum += w * outcome[i];
                weighted_control_count += w;
            }
        }

        let mean_treated = if weighted_treated_count > 0.0 {
            weighted_treated_sum / weighted_treated_count
        } else {
            0.0
        };
        let mean_control = if weighted_control_count > 0.0 {
            weighted_control_sum / weighted_control_count
        } else {
            0.0
        };

        let ate = mean_treated - mean_control;

        // Approximate standard error via weighted variance
        let mut var_sum = 0.0;
        for i in 0..n {
            let e = propensity[i];
            let ipw_contribution = if treatment[i] > 0.5 {
                outcome[i] / e
            } else {
                -outcome[i] / (1.0 - e)
            };
            var_sum += (ipw_contribution - ate).powi(2);
        }
        let std_error = (var_sum / (n as f64 * (n as f64 - 1.0))).sqrt();

        ATEResult {
            ate,
            std_error,
            n_obs: n,
        }
    }
}

/// Sigmoid (logistic) function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn ipw_known_effect() {
        // Deterministic dataset with known treatment effect of 5.0.
        // Confounder X determines treatment assignment and also affects outcome.
        //
        // Group 1 (X=1, treated):   outcome = 5*1 + 3*1 = 8
        // Group 2 (X=1, control):   outcome = 5*0 + 3*1 = 3
        // Group 3 (X=-1, treated):  outcome = 5*1 + 3*(-1) = 2
        // Group 4 (X=-1, control):  outcome = 5*0 + 3*(-1) = -3
        //
        // Naive diff-in-means would be confounded because X=1 units are
        // more likely to be treated. IPW should recover ATE ≈ 5.0.

        let n = 80;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));

        for i in 0..n {
            let x = if i < n / 2 { 1.0 } else { -1.0 };
            covariates[[i, 0]] = x;
            // In X=1 group: 75% treated. In X=-1 group: 25% treated.
            let is_treated = if x > 0.0 {
                (i % 4) != 0 // 3 out of 4 treated
            } else {
                (i % 4) == 0 // 1 out of 4 treated
            };
            treatment[i] = if is_treated { 1.0 } else { 0.0 };
            // Outcome = 5*T + 3*X (no noise for deterministic test)
            outcome[i] = 5.0 * treatment[i] + 3.0 * x;
        }

        let result = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates);

        assert!(
            (result.ate - 5.0).abs() < 2.0,
            "IPW ATE = {}, expected ~5.0",
            result.ate
        );
        assert_eq!(result.n_obs, n);
    }

    #[test]
    fn ipw_no_effect() {
        // No treatment effect, confounder only affects outcome.
        let n = 200;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));

        let mut rng_state: u64 = 54321;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f64) / (2.0_f64.powi(31))
        };

        for i in 0..n {
            let x = next_rng(&mut rng_state) * 2.0 - 1.0;
            covariates[[i, 0]] = x;
            treatment[i] = if next_rng(&mut rng_state) > 0.5 { 1.0 } else { 0.0 };
            // No treatment effect, outcome depends only on X
            outcome[i] = 3.0 * x + next_rng(&mut rng_state) * 0.5;
        }

        let result = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates);

        assert!(
            result.ate.abs() < 2.0,
            "IPW ATE = {}, expected ~0.0",
            result.ate
        );
    }
}
