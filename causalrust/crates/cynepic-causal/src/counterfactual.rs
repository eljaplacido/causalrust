//! Counterfactual reasoning — Level 3 on Pearl's causal ladder.
//!
//! Answers questions of the form: "Given that we observed Y=y under treatment T=t,
//! what *would* Y have been if T had instead been t'?"
//!
//! This module provides counterfactual estimation using linear structural equations.
//! For a treatment variable T with estimated average treatment effect (ATE) β:
//!
//! ```text
//! Y_counterfactual = Y_observed + β × (T_counterfactual - T_factual)
//! ```
//!
//! This is exact for linear additive models and a reasonable first-order
//! approximation for nonlinear models when the treatment shift is small.

use crate::estimate::linear::{ATEResult, LinearATEEstimator};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A counterfactual query: "What would outcome be if treatment had been different?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualQuery {
    /// Name of the treatment variable.
    pub treatment: String,
    /// Name of the outcome variable.
    pub outcome: String,
    /// The treatment value that was actually observed.
    pub factual_treatment: f64,
    /// The hypothetical treatment value we want to reason about.
    pub counterfactual_treatment: f64,
    /// The outcome that was actually observed.
    pub observed_outcome: f64,
}

/// Result of a counterfactual analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// The query that produced this result.
    pub query: CounterfactualQuery,
    /// The estimated counterfactual outcome.
    pub counterfactual_outcome: f64,
    /// The estimated treatment effect (ATE) used for projection.
    pub treatment_effect: f64,
    /// Standard error of the treatment effect estimate.
    pub std_error: f64,
    /// 95% confidence interval for the counterfactual outcome.
    pub confidence_interval: (f64, f64),
}

/// Engine for counterfactual reasoning using linear structural equations.
pub struct CounterfactualEngine;

impl CounterfactualEngine {
    /// Compute a counterfactual outcome using a pre-estimated ATE.
    ///
    /// Given an observed outcome Y under treatment T, projects what Y would
    /// have been under a different treatment value T' using the linear model:
    ///
    /// `Y_cf = Y_obs + ATE × (T_cf - T_factual)`
    pub fn query_with_ate(
        query: &CounterfactualQuery,
        ate_result: &ATEResult,
    ) -> CounterfactualResult {
        let treatment_shift = query.counterfactual_treatment - query.factual_treatment;
        let cf_outcome = query.observed_outcome + ate_result.ate * treatment_shift;

        // Propagate uncertainty: SE of counterfactual = |shift| × SE(ATE)
        let cf_se = treatment_shift.abs() * ate_result.std_error;
        let ci_lower = cf_outcome - 1.96 * cf_se;
        let ci_upper = cf_outcome + 1.96 * cf_se;

        CounterfactualResult {
            query: query.clone(),
            counterfactual_outcome: cf_outcome,
            treatment_effect: ate_result.ate,
            std_error: cf_se,
            confidence_interval: (ci_lower, ci_upper),
        }
    }

    /// Estimate a counterfactual outcome directly from data.
    ///
    /// First estimates the ATE via difference-in-means, then projects the
    /// counterfactual outcome for the given query.
    pub fn estimate_from_data(
        query: &CounterfactualQuery,
        treatment_data: &Array1<f64>,
        outcome_data: &Array1<f64>,
    ) -> CounterfactualResult {
        let ate_result = LinearATEEstimator::difference_in_means(treatment_data, outcome_data);
        Self::query_with_ate(query, &ate_result)
    }

    /// Batch counterfactual: compute counterfactuals for multiple units.
    ///
    /// Given vectors of observed treatments, observed outcomes, and a single
    /// counterfactual treatment value, returns per-unit counterfactual outcomes.
    pub fn batch(
        treatment_data: &Array1<f64>,
        outcome_data: &Array1<f64>,
        counterfactual_treatment: f64,
    ) -> Vec<CounterfactualResult> {
        let ate_result = LinearATEEstimator::difference_in_means(treatment_data, outcome_data);

        treatment_data
            .iter()
            .zip(outcome_data.iter())
            .enumerate()
            .map(|(i, (&t, &y))| {
                let query = CounterfactualQuery {
                    treatment: format!("unit_{i}"),
                    outcome: "outcome".into(),
                    factual_treatment: t,
                    counterfactual_treatment,
                    observed_outcome: y,
                };
                Self::query_with_ate(&query, &ate_result)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn counterfactual_with_known_ate() {
        let ate = ATEResult {
            ate: 5.0,
            std_error: 1.0,
            n_obs: 100,
        };

        let query = CounterfactualQuery {
            treatment: "drug".into(),
            outcome: "recovery_days".into(),
            factual_treatment: 1.0,
            counterfactual_treatment: 0.0,
            observed_outcome: 10.0,
        };

        let result = CounterfactualEngine::query_with_ate(&query, &ate);

        // Y_cf = 10.0 + 5.0 * (0.0 - 1.0) = 5.0
        assert!((result.counterfactual_outcome - 5.0).abs() < 1e-10);
        assert!((result.treatment_effect - 5.0).abs() < 1e-10);
        // CI should bracket the counterfactual
        assert!(result.confidence_interval.0 < 5.0);
        assert!(result.confidence_interval.1 > 5.0);
    }

    #[test]
    fn counterfactual_no_shift() {
        let ate = ATEResult {
            ate: 5.0,
            std_error: 1.0,
            n_obs: 100,
        };

        let query = CounterfactualQuery {
            treatment: "drug".into(),
            outcome: "recovery".into(),
            factual_treatment: 1.0,
            counterfactual_treatment: 1.0, // same as factual
            observed_outcome: 10.0,
        };

        let result = CounterfactualEngine::query_with_ate(&query, &ate);
        // No shift -> counterfactual = observed
        assert!((result.counterfactual_outcome - 10.0).abs() < 1e-10);
        assert!((result.std_error).abs() < 1e-10);
    }

    #[test]
    fn counterfactual_from_data() {
        let treatment = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let outcome = array![10.0, 12.0, 11.0, 9.0, 13.0, 5.0, 6.0, 7.0, 4.0, 8.0];

        // Ask: "Unit 0 got treatment (T=1, Y=10). What if they hadn't (T=0)?"
        let query = CounterfactualQuery {
            treatment: "treatment".into(),
            outcome: "outcome".into(),
            factual_treatment: 1.0,
            counterfactual_treatment: 0.0,
            observed_outcome: 10.0,
        };

        let result = CounterfactualEngine::estimate_from_data(&query, &treatment, &outcome);

        // ATE ≈ 5.0, so counterfactual ≈ 10.0 - 5.0 = 5.0
        assert!((result.counterfactual_outcome - 5.0).abs() < 1.0);
        assert!(result.treatment_effect > 0.0);
    }

    #[test]
    fn batch_counterfactuals() {
        let treatment = array![1.0, 1.0, 0.0, 0.0];
        let outcome = array![10.0, 12.0, 5.0, 6.0];

        let results = CounterfactualEngine::batch(&treatment, &outcome, 0.0);
        assert_eq!(results.len(), 4);

        // Treated units should have lower counterfactual outcomes
        assert!(results[0].counterfactual_outcome < results[0].query.observed_outcome);
        assert!(results[1].counterfactual_outcome < results[1].query.observed_outcome);

        // Untreated units: counterfactual = observed (no shift)
        assert!((results[2].counterfactual_outcome - results[2].query.observed_outcome).abs() < 1e-10);
    }
}
