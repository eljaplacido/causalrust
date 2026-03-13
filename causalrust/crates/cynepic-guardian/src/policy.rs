use async_trait::async_trait;
use cynepic_core::PolicyDecision;
use serde_json::Value;
use std::sync::Arc;

/// Trait for any policy engine that can evaluate an action.
#[async_trait]
pub trait PolicyEvaluator: Send + Sync {
    /// Evaluate whether the given action (as JSON context) is permitted.
    async fn evaluate(&self, action: &str, context: &Value) -> Result<PolicyDecision, GuardianError>;

    /// Human-readable name of this policy engine.
    fn name(&self) -> &str;
}

/// A chain of policy evaluators applied in sequence.
///
/// The chain short-circuits on the first `Reject` or `Escalate`.
/// All evaluators must approve for the overall result to be `Approve`.
pub struct PolicyChain {
    evaluators: Vec<Arc<dyn PolicyEvaluator>>,
}

impl PolicyChain {
    /// Create an empty policy chain.
    pub fn new() -> Self {
        Self {
            evaluators: Vec::new(),
        }
    }

    /// Add a policy evaluator to the chain.
    pub fn add(mut self, evaluator: Arc<dyn PolicyEvaluator>) -> Self {
        self.evaluators.push(evaluator);
        self
    }

    /// Evaluate all policies in sequence. Short-circuits on first non-Approve.
    pub async fn evaluate(&self, action: &str, context: &Value) -> Result<PolicyDecision, GuardianError> {
        for evaluator in &self.evaluators {
            let decision = evaluator.evaluate(action, context).await?;
            match &decision {
                PolicyDecision::Approve => continue,
                PolicyDecision::Reject { .. } | PolicyDecision::Escalate { .. } => {
                    tracing::info!(
                        engine = evaluator.name(),
                        ?decision,
                        "Policy chain short-circuited"
                    );
                    return Ok(decision);
                }
            }
        }
        Ok(PolicyDecision::Approve)
    }
}

impl Default for PolicyChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Rego-based policy evaluator using the `regorus` engine.
#[cfg(feature = "rego")]
pub struct RegoPolicyEvaluator {
    engine: regorus::Engine,
}

#[cfg(feature = "rego")]
impl RegoPolicyEvaluator {
    /// Create a new Rego evaluator from a policy string.
    pub fn from_policy(policy: &str) -> Result<Self, GuardianError> {
        let mut engine = regorus::Engine::new();
        engine
            .add_policy("policy.rego".into(), policy.into())
            .map_err(|e| GuardianError::PolicyLoad(e.to_string()))?;
        Ok(Self { engine })
    }
}

#[cfg(feature = "rego")]
#[async_trait]
impl PolicyEvaluator for RegoPolicyEvaluator {
    async fn evaluate(&self, _action: &str, context: &Value) -> Result<PolicyDecision, GuardianError> {
        let mut engine = self.engine.clone();

        // Set input data
        let input_value = regorus::Value::from_json_str(&context.to_string())
            .map_err(|e| GuardianError::Evaluation(e.to_string()))?;
        engine.set_input(input_value);

        // Evaluate the policy
        let result = engine
            .eval_rule("data.policy.allow".into())
            .map_err(|e| GuardianError::Evaluation(e.to_string()))?;

        // Interpret result
        match result.as_bool() {
            Ok(true) => Ok(PolicyDecision::Approve),
            _ => Ok(PolicyDecision::Reject {
                reason: "Rego policy denied the action".into(),
            }),
        }
    }

    fn name(&self) -> &str {
        "rego"
    }
}

/// Errors from the guardian policy layer.
#[derive(Debug, thiserror::Error)]
pub enum GuardianError {
    #[error("Failed to load policy: {0}")]
    PolicyLoad(String),

    #[error("Policy evaluation failed: {0}")]
    Evaluation(String),

    #[error("Circuit breaker tripped: {0}")]
    CircuitBreakerTripped(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_allow_policy() {
        let policy = r#"
            package policy
            default allow = false
            allow {
                input.role == "admin"
            }
        "#;

        let evaluator = RegoPolicyEvaluator::from_policy(policy).unwrap();
        let context = serde_json::json!({ "role": "admin" });
        let decision = evaluator.evaluate("test_action", &context).await.unwrap();
        assert!(decision.is_approved());
    }

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_deny_policy() {
        let policy = r#"
            package policy
            default allow = false
            allow {
                input.role == "admin"
            }
        "#;

        let evaluator = RegoPolicyEvaluator::from_policy(policy).unwrap();
        let context = serde_json::json!({ "role": "viewer" });
        let decision = evaluator.evaluate("test_action", &context).await.unwrap();
        assert!(!decision.is_approved());
    }

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn policy_chain_short_circuits() {
        let deny_policy = r#"
            package policy
            default allow = false
        "#;

        let evaluator = Arc::new(RegoPolicyEvaluator::from_policy(deny_policy).unwrap());
        let chain = PolicyChain::new().add(evaluator);

        let context = serde_json::json!({});
        let decision = chain.evaluate("test_action", &context).await.unwrap();
        assert!(!decision.is_approved());
    }

    #[tokio::test]
    async fn empty_chain_approves() {
        let chain = PolicyChain::new();
        let context = serde_json::json!({});
        let decision = chain.evaluate("any_action", &context).await.unwrap();
        assert!(decision.is_approved());
    }
}
