//! Incremental belief updating API.
//!
//! Wraps conjugate prior models with a unified interface for
//! real-time belief updating as evidence arrives.

use crate::priors::{BetaBinomial, DirichletMultinomial, GammaPoisson, NormalNormal};
use serde::{Deserialize, Serialize};

/// A belief state that can be incrementally updated with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeliefState {
    /// Binary outcome belief (success/failure).
    Binary(BetaBinomial),
    /// Continuous outcome belief (mean estimation).
    Continuous(NormalNormal),
    /// Count outcome belief (rate estimation).
    Count(GammaPoisson),
    /// Categorical outcome belief (probability vector estimation).
    Categorical(DirichletMultinomial),
}

impl BeliefState {
    /// Get the current posterior mean.
    ///
    /// For `Categorical`, returns the mean of the first category.
    /// Use the inner `DirichletMultinomial` directly for full vector means.
    pub fn mean(&self) -> f64 {
        match self {
            BeliefState::Binary(m) => m.mean(),
            BeliefState::Continuous(m) => m.mean(),
            BeliefState::Count(m) => m.mean(),
            BeliefState::Categorical(m) => m.marginal_mean(0),
        }
    }

    /// Get the current posterior variance.
    ///
    /// For `Categorical`, returns the marginal variance of the first category.
    pub fn variance(&self) -> f64 {
        match self {
            BeliefState::Binary(m) => m.variance(),
            BeliefState::Continuous(m) => m.variance(),
            BeliefState::Count(m) => m.variance(),
            BeliefState::Categorical(m) => m.marginal_variance(0),
        }
    }

    /// Get the 95% credible interval (where applicable).
    ///
    /// For `Categorical`, returns the interval for the first category.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        match self {
            BeliefState::Binary(m) => m.credible_interval_95(),
            BeliefState::Continuous(m) => m.credible_interval_95(),
            BeliefState::Count(m) => {
                let std = m.variance().sqrt();
                (m.mean() - 1.96 * std, m.mean() + 1.96 * std)
            }
            BeliefState::Categorical(m) => {
                let mean = m.marginal_mean(0);
                let std = m.marginal_variance(0).sqrt();
                ((mean - 1.96 * std).max(0.0), (mean + 1.96 * std).min(1.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_state_binary() {
        let state = BeliefState::Binary(BetaBinomial::new(10.0, 10.0));
        assert!((state.mean() - 0.5).abs() < 1e-10);
    }
}
