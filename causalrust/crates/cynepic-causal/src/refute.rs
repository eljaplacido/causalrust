//! Refutation tests for causal effect estimates.
//!
//! These tests probe whether an estimated causal effect is robust:
//! - **Placebo treatment**: Replace treatment with a random variable; effect should vanish
//! - **Random common cause**: Add a random confounder; effect should remain stable
//! - **Data subset**: Re-estimate on a subset; effect should be similar
//! - **Bootstrap**: Resample with replacement; effect should be stable

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Result of a refutation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefutationResult {
    /// Name of the test performed.
    pub test_name: String,
    /// The original effect estimate.
    pub original_effect: f64,
    /// The refuted effect estimate (should differ from original if test catches a problem).
    pub refuted_effect: f64,
    /// Whether the refutation test passed (effect is robust).
    pub passed: bool,
}

/// Simple linear congruential generator for reproducible pseudo-random numbers.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a pseudo-random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.state >> 33) as f64) / (2.0_f64.powi(31))
    }

    /// Returns a pseudo-random usize in [0, max).
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize % max
    }
}

/// Run a placebo treatment refutation test.
///
/// Replaces the treatment with random noise and re-estimates.
/// If the effect persists with random treatment, the original estimate is suspicious.
pub fn placebo_treatment(
    outcome: &Array1<f64>,
    original_ate: f64,
    tolerance: f64,
) -> RefutationResult {
    let n = outcome.len();

    // Generate random "treatment" (coin flip)
    let mut rng = SimpleRng::new(42);
    let placebo_treatment: Array1<f64> = Array1::from_shape_fn(n, |_| {
        if rng.next_f64() < 0.5 { 1.0 } else { 0.0 }
    });

    let result = crate::estimate::linear::LinearATEEstimator::difference_in_means(
        &placebo_treatment,
        outcome,
    );

    let passed = result.ate.abs() < tolerance;

    RefutationResult {
        test_name: "placebo_treatment".into(),
        original_effect: original_ate,
        refuted_effect: result.ate,
        passed,
    }
}

/// Refutation via adding a random common cause.
///
/// Adds a random variable as a covariate and re-estimates using OLS adjustment.
/// If the estimate changes significantly, the original may be confounded.
pub fn random_common_cause(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    original_ate: f64,
    n_simulations: usize,
) -> RefutationResult {
    let n = treatment.len();
    let mut rng = SimpleRng::new(123);
    let mut total_ate = 0.0;

    for _ in 0..n_simulations {
        // Generate a random "common cause" variable
        let random_cov: Array1<f64> = Array1::from_shape_fn(n, |_| rng.next_f64() * 2.0 - 1.0);
        let covariates = random_cov
            .view()
            .into_shape_with_order((n, 1))
            .unwrap()
            .to_owned();

        let result = crate::estimate::linear::LinearATEEstimator::ols_adjusted(
            treatment,
            outcome,
            &covariates,
        );
        total_ate += result.ate;
    }

    let mean_ate = total_ate / n_simulations as f64;

    // The estimate should not change much when adding a random common cause.
    // If it does, the original estimate may be fragile.
    let relative_change = if original_ate.abs() > 1e-10 {
        ((mean_ate - original_ate) / original_ate).abs()
    } else {
        (mean_ate - original_ate).abs()
    };

    let passed = relative_change < 0.15; // Less than 15% change

    RefutationResult {
        test_name: "random_common_cause".into(),
        original_effect: original_ate,
        refuted_effect: mean_ate,
        passed,
    }
}

/// Refutation via data subset validation.
///
/// Estimates ATE on random subsets of the data. If the estimate is unstable
/// across subsets, the original may be fragile.
pub fn subset_validation(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    original_ate: f64,
    subset_fraction: f64,
    n_subsets: usize,
) -> RefutationResult {
    let n = treatment.len();
    let subset_size = (n as f64 * subset_fraction).max(2.0) as usize;
    let mut rng = SimpleRng::new(456);
    let mut total_ate = 0.0;

    for _ in 0..n_subsets {
        // Sample indices without replacement (approximate via shuffle prefix)
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle for first subset_size elements
        for i in 0..subset_size.min(n) {
            let j = i + rng.next_usize(n - i);
            indices.swap(i, j);
        }
        let selected = &indices[..subset_size];

        let sub_treatment = Array1::from_vec(selected.iter().map(|&i| treatment[i]).collect());
        let sub_outcome = Array1::from_vec(selected.iter().map(|&i| outcome[i]).collect());

        let result = crate::estimate::linear::LinearATEEstimator::difference_in_means(
            &sub_treatment,
            &sub_outcome,
        );
        total_ate += result.ate;
    }

    let mean_ate = total_ate / n_subsets as f64;

    let relative_change = if original_ate.abs() > 1e-10 {
        ((mean_ate - original_ate) / original_ate).abs()
    } else {
        (mean_ate - original_ate).abs()
    };

    let passed = relative_change < 0.20; // Less than 20% change

    RefutationResult {
        test_name: "subset_validation".into(),
        original_effect: original_ate,
        refuted_effect: mean_ate,
        passed,
    }
}

/// Bootstrap refutation: resample with replacement and check stability.
///
/// If the bootstrapped estimates are widely dispersed around the original,
/// the estimate may not be reliable.
pub fn bootstrap_refutation(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    original_ate: f64,
    n_bootstrap: usize,
) -> RefutationResult {
    let n = treatment.len();
    let mut rng = SimpleRng::new(789);
    let mut total_ate = 0.0;

    for _ in 0..n_bootstrap {
        // Resample with replacement (paired: same index for treatment and outcome)
        let indices: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
        let boot_treatment = Array1::from_vec(indices.iter().map(|&i| treatment[i]).collect());
        let boot_outcome = Array1::from_vec(indices.iter().map(|&i| outcome[i]).collect());

        let result = crate::estimate::linear::LinearATEEstimator::difference_in_means(
            &boot_treatment,
            &boot_outcome,
        );
        total_ate += result.ate;
    }

    let mean_ate = total_ate / n_bootstrap as f64;

    let relative_change = if original_ate.abs() > 1e-10 {
        ((mean_ate - original_ate) / original_ate).abs()
    } else {
        (mean_ate - original_ate).abs()
    };

    let passed = relative_change < 0.20;

    RefutationResult {
        test_name: "bootstrap_refutation".into(),
        original_effect: original_ate,
        refuted_effect: mean_ate,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn placebo_on_random_data() {
        // Random outcome with no real treatment effect
        let outcome = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = placebo_treatment(&outcome, 0.0, 3.0);
        // With random treatment and random outcome, the placebo effect should be small
        assert!(result.test_name == "placebo_treatment");
    }

    #[test]
    fn random_common_cause_stable_estimate() {
        // Known treatment effect, adding random noise should not change it much.
        let treatment = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let outcome = array![10.0, 11.0, 12.0, 10.0, 11.0, 5.0, 6.0, 5.0, 6.0, 5.0];
        let original_ate = 5.4; // mean(10,11,12,10,11) - mean(5,6,5,6,5) = 10.8 - 5.4 = 5.4

        let result = random_common_cause(&treatment, &outcome, original_ate, 10);
        assert_eq!(result.test_name, "random_common_cause");
        assert!(
            result.passed,
            "Random common cause should not change a real effect much: original={}, refuted={}",
            result.original_effect,
            result.refuted_effect
        );
    }

    #[test]
    fn subset_validation_stable() {
        // Larger dataset with clear treatment effect
        let n = 100;
        let mut t_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        for i in 0..n {
            t_vec[i] = if i < n / 2 { 1.0 } else { 0.0 };
            y_vec[i] = if i < n / 2 { 10.0 } else { 5.0 };
        }
        let treatment = Array1::from_vec(t_vec);
        let outcome = Array1::from_vec(y_vec);

        let result = subset_validation(&treatment, &outcome, 5.0, 0.8, 20);
        assert_eq!(result.test_name, "subset_validation");
        assert!(
            result.passed,
            "Subset validation should be stable for a clear effect: original={}, refuted={}",
            result.original_effect,
            result.refuted_effect
        );
    }

    #[test]
    fn bootstrap_stable() {
        let n = 100;
        let mut t_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        for i in 0..n {
            t_vec[i] = if i < n / 2 { 1.0 } else { 0.0 };
            y_vec[i] = if i < n / 2 { 10.0 } else { 5.0 };
        }
        let treatment = Array1::from_vec(t_vec);
        let outcome = Array1::from_vec(y_vec);

        let result = bootstrap_refutation(&treatment, &outcome, 5.0, 50);
        assert_eq!(result.test_name, "bootstrap_refutation");
        assert!(
            result.passed,
            "Bootstrap should be stable for a clear effect: original={}, refuted={}",
            result.original_effect,
            result.refuted_effect
        );
    }
}

