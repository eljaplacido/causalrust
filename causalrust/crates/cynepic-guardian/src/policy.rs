use async_trait::async_trait;
use cynepic_core::PolicyDecision;
use serde_json::Value;
use std::sync::Arc;

/// Trait for any policy engine that can evaluate an action.
#[async_trait]
pub trait PolicyEvaluator: Send + Sync {
    /// Evaluate whether the given action (as JSON context) is permitted.
    async fn evaluate(
        &self,
        action: &str,
        context: &Value,
    ) -> Result<PolicyDecision, GuardianError>;

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

// Trait objects are not `Debug`, but every evaluator can name itself — so print
// the chain in evaluation order, which is the thing you actually want to see
// when a decision came out unexpectedly.
impl std::fmt::Debug for PolicyChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyChain")
            .field(
                "evaluators",
                &self.evaluators.iter().map(|e| e.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PolicyChain {
    /// Create an empty policy chain.
    pub fn new() -> Self {
        Self {
            evaluators: Vec::new(),
        }
    }

    /// Add a policy evaluator to the chain.
    // Not `std::ops::Add`: this is a consuming builder step, and `chain.add(e)`
    // reads better at the call site than any name chosen to appease the lint.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, evaluator: Arc<dyn PolicyEvaluator>) -> Self {
        self.evaluators.push(evaluator);
        self
    }

    /// Evaluate all policies in sequence. Short-circuits on first non-Approve.
    pub async fn evaluate(
        &self,
        action: &str,
        context: &Value,
    ) -> Result<PolicyDecision, GuardianError> {
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

// `regorus::Engine` is not `Debug`, so derive by hand rather than leaking the
// engine's internals. Deliberately opaque: policy source can carry sensitive
// authorization logic and should not land in logs via `{:?}`.
#[cfg(feature = "rego")]
impl std::fmt::Debug for RegoPolicyEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegoPolicyEvaluator")
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "rego")]
impl RegoPolicyEvaluator {
    /// Create a new Rego evaluator from a policy string written in **Rego v1**.
    ///
    /// Rego v1 requires the `if` keyword before a rule body and `:=` for
    /// assignment, e.g.:
    ///
    /// ```rego
    /// package policy
    /// default allow := false
    /// allow if {
    ///     input.role == "admin"
    /// }
    /// ```
    ///
    /// Policies written in the legacy v0 dialect (a bare `allow { ... }` body)
    /// are rejected with [`GuardianError::PolicyLoad`]. Use
    /// [`RegoPolicyEvaluator::from_policy_v0`] to opt in to the legacy dialect
    /// explicitly.
    pub fn from_policy(policy: &str) -> Result<Self, GuardianError> {
        Self::build(policy, false)
    }

    /// Create a new Rego evaluator from a policy string written in the legacy
    /// **Rego v0** dialect.
    ///
    /// Prefer [`RegoPolicyEvaluator::from_policy`]. This constructor exists so
    /// that callers migrating an existing v0 policy corpus can do so
    /// deliberately, rather than by accident.
    pub fn from_policy_v0(policy: &str) -> Result<Self, GuardianError> {
        Self::build(policy, true)
    }

    fn build(policy: &str, rego_v0: bool) -> Result<Self, GuardianError> {
        let mut engine = regorus::Engine::new();
        engine.set_rego_v0(rego_v0);
        engine
            .add_policy("policy.rego".into(), policy.into())
            .map_err(|e| GuardianError::PolicyLoad(e.to_string()))?;
        Ok(Self { engine })
    }
}

#[cfg(feature = "rego")]
#[async_trait]
impl PolicyEvaluator for RegoPolicyEvaluator {
    async fn evaluate(
        &self,
        _action: &str,
        context: &Value,
    ) -> Result<PolicyDecision, GuardianError> {
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

    /// An admin-only `allow` rule in Rego v1 syntax.
    #[cfg(feature = "rego")]
    const ADMIN_ONLY_V1: &str = r#"
        package policy
        default allow := false
        allow if {
            input.role == "admin"
        }
    "#;

    /// The same rule in the legacy Rego v0 dialect (no `if` before the body).
    #[cfg(feature = "rego")]
    const ADMIN_ONLY_V0: &str = r#"
        package policy
        default allow = false
        allow {
            input.role == "admin"
        }
    "#;

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_allow_policy() {
        let evaluator = RegoPolicyEvaluator::from_policy(ADMIN_ONLY_V1).unwrap();
        let context = serde_json::json!({ "role": "admin" });
        let decision = evaluator.evaluate("test_action", &context).await.unwrap();
        assert!(decision.is_approved());
    }

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_deny_policy() {
        let evaluator = RegoPolicyEvaluator::from_policy(ADMIN_ONLY_V1).unwrap();
        let context = serde_json::json!({ "role": "viewer" });
        let decision = evaluator.evaluate("test_action", &context).await.unwrap();
        assert!(!decision.is_approved());
    }

    /// The default constructor targets Rego v1, so a v0-dialect policy must be
    /// rejected at load time rather than silently accepted. This pins the
    /// dialect the crate compiles against, so a `regorus` upgrade that flips the
    /// default fails here instead of in a user's policy.
    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_v0_policy_is_rejected_by_default() {
        let err = RegoPolicyEvaluator::from_policy(ADMIN_ONLY_V0)
            .expect_err("v0 dialect must not load through the v1 constructor");
        assert!(matches!(err, GuardianError::PolicyLoad(_)), "got {err:?}");
    }

    /// ...but the same policy loads through the explicit v0 constructor and
    /// behaves identically to its v1 counterpart.
    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn rego_v0_policy_loads_via_explicit_opt_in() {
        let evaluator = RegoPolicyEvaluator::from_policy_v0(ADMIN_ONLY_V0).unwrap();

        let admin = serde_json::json!({ "role": "admin" });
        assert!(
            evaluator
                .evaluate("test_action", &admin)
                .await
                .unwrap()
                .is_approved()
        );

        let viewer = serde_json::json!({ "role": "viewer" });
        assert!(
            !evaluator
                .evaluate("test_action", &viewer)
                .await
                .unwrap()
                .is_approved()
        );
    }

    #[cfg(feature = "rego")]
    #[tokio::test]
    async fn policy_chain_short_circuits() {
        let deny_policy = r#"
            package policy
            default allow := false
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
