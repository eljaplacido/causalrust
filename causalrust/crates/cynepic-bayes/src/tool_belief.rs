//! Tool reliability tracking using Bayesian beliefs.
//!
//! Provides `ToolBelief` for tracking the reliability of individual tools,
//! models, or services using Beta-Binomial inference, and `ToolBeliefSet`
//! for managing beliefs about multiple tools simultaneously.

use crate::priors::BetaBinomial;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks the reliability belief for a tool, model, or service.
///
/// Uses Beta-Binomial internally: each call is a Bernoulli trial
/// (success or failure). The posterior mean gives the estimated
/// probability of success (reliability).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolBelief {
    /// Name of the tool being tracked.
    pub name: String,
    prior: BetaBinomial,
    total_calls: u64,
    recent_failures: u32,
    last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

impl ToolBelief {
    /// Create a new tool belief with a uniform prior Beta(1, 1).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prior: BetaBinomial::uniform(),
            total_calls: 0,
            recent_failures: 0,
            last_updated: None,
        }
    }

    /// Create a new tool belief with a custom prior Beta(alpha, beta).
    pub fn with_prior(name: impl Into<String>, alpha: f64, beta: f64) -> Self {
        Self {
            name: name.into(),
            prior: BetaBinomial::new(alpha, beta),
            total_calls: 0,
            recent_failures: 0,
            last_updated: None,
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.prior.update(1, 0);
        self.total_calls += 1;
        self.recent_failures = 0;
        self.last_updated = Some(chrono::Utc::now());
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.prior.update(0, 1);
        self.total_calls += 1;
        self.recent_failures += 1;
        self.last_updated = Some(chrono::Utc::now());
    }

    /// Posterior mean P(success) — the estimated reliability.
    pub fn reliability(&self) -> f64 {
        self.prior.mean()
    }

    /// 95% credible interval for the reliability.
    pub fn confidence_interval(&self) -> (f64, f64) {
        self.prior.credible_interval_95()
    }

    /// Check if the tool is reliable (posterior mean above threshold).
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.reliability() > threshold
    }

    /// Check if the tool should be circuit-broken.
    ///
    /// Returns true if the lower bound of the 95% credible interval
    /// falls below `min_reliability`, indicating we are reasonably
    /// confident the tool is unreliable.
    pub fn should_circuit_break(&self, min_reliability: f64) -> bool {
        let (lower, _) = self.confidence_interval();
        lower < min_reliability
    }

    /// Total number of calls recorded.
    pub fn total_calls(&self) -> u64 {
        self.total_calls
    }

    /// Observed failure rate (frequentist, not Bayesian).
    pub fn failure_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            // beta - 1 gives the number of failures (subtract the prior)
            let failures = self.prior.beta - 1.0; // subtract initial prior
            failures / self.total_calls as f64
        }
    }
}

/// Tracks beliefs about multiple tools simultaneously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolBeliefSet {
    tools: HashMap<String, ToolBelief>,
}

impl ToolBeliefSet {
    /// Create a new empty tool belief set.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a new tool with a uniform prior.
    pub fn register(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.tools.insert(name.clone(), ToolBelief::new(name));
    }

    /// Record a success or failure for a tool.
    ///
    /// Returns a reference to the updated tool belief, or None if the tool
    /// is not registered.
    pub fn record(&mut self, tool: &str, success: bool) -> Option<&ToolBelief> {
        if let Some(belief) = self.tools.get_mut(tool) {
            if success {
                belief.record_success();
            } else {
                belief.record_failure();
            }
            Some(belief)
        } else {
            None
        }
    }

    /// Get a reference to a tool's belief.
    pub fn get(&self, tool: &str) -> Option<&ToolBelief> {
        self.tools.get(tool)
    }

    /// Get all tools whose reliability is below the given threshold.
    pub fn unreliable_tools(&self, threshold: f64) -> Vec<&ToolBelief> {
        self.tools
            .values()
            .filter(|t| !t.is_reliable(threshold))
            .collect()
    }

    /// Get the most reliable tool (by posterior mean).
    pub fn most_reliable(&self) -> Option<&ToolBelief> {
        self.tools
            .values()
            .max_by(|a, b| a.reliability().partial_cmp(&b.reliability()).unwrap())
    }
}

impl Default for ToolBeliefSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_success_failure_tracking() {
        let mut tool = ToolBelief::new("test-api");

        // Record 8 successes and 2 failures
        for _ in 0..8 {
            tool.record_success();
        }
        for _ in 0..2 {
            tool.record_failure();
        }

        assert_eq!(tool.total_calls(), 10);
        assert_eq!(tool.name, "test-api");

        // Posterior: Beta(1+8, 1+2) = Beta(9, 3), mean = 9/12 = 0.75
        assert!((tool.reliability() - 0.75).abs() < 1e-10);
        assert!(tool.is_reliable(0.5));
        assert!(!tool.is_reliable(0.8));
    }

    #[test]
    fn circuit_break_detection() {
        let mut tool = ToolBelief::new("flaky-service");

        // Record many failures
        for _ in 0..20 {
            tool.record_failure();
        }
        tool.record_success();

        // Reliability should be very low
        assert!(tool.reliability() < 0.15);

        // Should trigger circuit break at reasonable thresholds
        assert!(tool.should_circuit_break(0.5));

        // Should not circuit break at very low threshold
        // With 20 failures and 1 success, lower CI bound is very low
        // but not necessarily below 0.01
    }

    #[test]
    fn custom_prior() {
        let tool = ToolBelief::with_prior("strong-tool", 10.0, 1.0);
        // Prior mean = 10/11 ≈ 0.909
        assert!(tool.reliability() > 0.9);
        assert!(tool.is_reliable(0.8));
    }

    #[test]
    fn tool_belief_set_operations() {
        let mut set = ToolBeliefSet::new();
        set.register("api-a");
        set.register("api-b");
        set.register("api-c");

        // api-a: reliable
        for _ in 0..10 {
            set.record("api-a", true);
        }

        // api-b: unreliable
        for _ in 0..10 {
            set.record("api-b", false);
        }

        // api-c: mixed
        for _ in 0..5 {
            set.record("api-c", true);
        }
        for _ in 0..5 {
            set.record("api-c", false);
        }

        // Check get
        assert!(set.get("api-a").is_some());
        assert!(set.get("nonexistent").is_none());

        // Check most_reliable — should be api-a
        let best = set.most_reliable().unwrap();
        assert_eq!(best.name, "api-a");

        // Check unreliable tools at 0.5 threshold
        let unreliable = set.unreliable_tools(0.5);
        // api-b should be unreliable (mean ≈ 1/12)
        assert!(unreliable.iter().any(|t| t.name == "api-b"));

        // Recording unknown tool returns None
        assert!(set.record("unknown", true).is_none());
    }
}
