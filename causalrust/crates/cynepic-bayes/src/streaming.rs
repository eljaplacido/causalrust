//! Streaming belief tracker for real-time observation processing.
//!
//! Designed for systems that receive observations one at a time
//! (e.g., Kafka consumers, event streams, WebSocket feeds).

use crate::belief::BeliefState;
use serde::{Deserialize, Serialize};

/// An observation for streaming updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Observation {
    /// Binary outcome (success/failure).
    Binary {
        /// Number of successes observed.
        successes: usize,
        /// Number of failures observed.
        failures: usize,
    },
    /// Continuous measurement.
    Continuous {
        /// The observed value.
        value: f64,
    },
    /// Count observation.
    Count {
        /// Observed count.
        count: usize,
        /// Number of observation periods.
        periods: usize,
    },
    /// Categorical counts.
    Categorical {
        /// Counts per category.
        counts: Vec<usize>,
    },
}

/// Error type for belief tracking operations.
#[derive(Debug, thiserror::Error)]
pub enum BeliefError {
    /// The observation type does not match the belief state type.
    #[error("Observation type mismatch: expected {expected}, got {got}")]
    TypeMismatch {
        /// The expected observation type name.
        expected: String,
        /// The received observation type name.
        got: String,
    },
}

/// A streaming belief tracker that processes observations one at a time.
///
/// Wraps a `BeliefState` and provides a unified API for incrementally
/// updating it with typed observations. Tracks observation count and
/// last update timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefTracker {
    state: BeliefState,
    observation_count: u64,
    last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

impl BeliefTracker {
    /// Create a new belief tracker with the given initial state.
    pub fn new(initial: BeliefState) -> Self {
        Self {
            state: initial,
            observation_count: 0,
            last_updated: None,
        }
    }

    /// Process a single observation, updating the belief state.
    ///
    /// Returns an error if the observation type does not match the belief state.
    pub fn observe(&mut self, obs: Observation) -> Result<(), BeliefError> {
        match (&mut self.state, &obs) {
            (BeliefState::Binary(model), Observation::Binary { successes, failures }) => {
                model.update(*successes as u64, *failures as u64);
            }
            (BeliefState::Continuous(model), Observation::Continuous { value }) => {
                model.update(&[*value]);
            }
            (BeliefState::Count(model), Observation::Count { count, periods }) => {
                let counts: Vec<u64> = std::iter::repeat(*count as u64)
                    .take(*periods)
                    .collect();
                model.update(&counts);
            }
            (BeliefState::Categorical(model), Observation::Categorical { counts }) => {
                model.update(counts);
            }
            (state, obs) => {
                let expected = match state {
                    BeliefState::Binary(_) => "Binary",
                    BeliefState::Continuous(_) => "Continuous",
                    BeliefState::Count(_) => "Count",
                    BeliefState::Categorical(_) => "Categorical",
                };
                let got = match obs {
                    Observation::Binary { .. } => "Binary",
                    Observation::Continuous { .. } => "Continuous",
                    Observation::Count { .. } => "Count",
                    Observation::Categorical { .. } => "Categorical",
                };
                return Err(BeliefError::TypeMismatch {
                    expected: expected.to_string(),
                    got: got.to_string(),
                });
            }
        }
        self.observation_count += 1;
        self.last_updated = Some(chrono::Utc::now());
        Ok(())
    }

    /// Process a batch of observations.
    ///
    /// Stops at the first error and returns it.
    pub fn observe_batch(&mut self, observations: &[Observation]) -> Result<(), BeliefError> {
        for obs in observations {
            self.observe(obs.clone())?;
        }
        Ok(())
    }

    /// Get a reference to the current belief state.
    pub fn state(&self) -> &BeliefState {
        &self.state
    }

    /// Get the total number of observations processed.
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Get the posterior mean of the belief state.
    ///
    /// For `Categorical`, returns the mean of the first category.
    pub fn mean(&self) -> f64 {
        self.state.mean()
    }

    /// Check if the belief is confident (posterior variance below threshold).
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.state.variance() < threshold
    }

    /// Reset the tracker with a new belief state, clearing observation count.
    pub fn reset(&mut self, new_state: BeliefState) {
        self.state = new_state;
        self.observation_count = 0;
        self.last_updated = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priors::{BetaBinomial, NormalNormal};

    #[test]
    fn streaming_binary_updates() {
        let mut tracker = BeliefTracker::new(BeliefState::Binary(BetaBinomial::uniform()));

        // Process several binary observations
        tracker
            .observe(Observation::Binary {
                successes: 5,
                failures: 2,
            })
            .unwrap();
        tracker
            .observe(Observation::Binary {
                successes: 3,
                failures: 1,
            })
            .unwrap();

        assert_eq!(tracker.observation_count(), 2);
        // Posterior: Beta(1+5+3, 1+2+1) = Beta(9, 4), mean = 9/13
        assert!((tracker.mean() - 9.0 / 13.0).abs() < 1e-10);
        assert!(tracker.last_updated.is_some());
    }

    #[test]
    fn batch_continuous_updates() {
        let mut tracker =
            BeliefTracker::new(BeliefState::Continuous(NormalNormal::new(0.0, 100.0, 1.0)));

        let observations = vec![
            Observation::Continuous { value: 5.0 },
            Observation::Continuous { value: 5.0 },
            Observation::Continuous { value: 5.0 },
        ];
        tracker.observe_batch(&observations).unwrap();

        assert_eq!(tracker.observation_count(), 3);
        // Mean should be pulled toward 5.0
        assert!((tracker.mean() - 5.0).abs() < 1.0);
    }

    #[test]
    fn type_mismatch_error() {
        let mut tracker = BeliefTracker::new(BeliefState::Binary(BetaBinomial::uniform()));

        let result = tracker.observe(Observation::Continuous { value: 1.0 });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Binary"));
        assert!(msg.contains("Continuous"));

        // Observation count should not increment on error
        assert_eq!(tracker.observation_count(), 0);
    }
}
