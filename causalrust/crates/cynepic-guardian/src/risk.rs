use async_trait::async_trait;
use cynepic_core::{EscalationTarget, PolicyDecision};
use serde_json::Value;

use crate::policy::{GuardianError, PolicyEvaluator};

/// A policy evaluator that makes decisions based on risk scores.
///
/// Risk scores are expected in the context JSON. This bridges Bayesian uncertainty
/// (from cynepic-bayes) with policy decisions: agents compute a risk score
/// (e.g., posterior probability of failure), and this evaluator enforces thresholds.
pub struct RiskAwareEvaluator {
    /// JSON path to the risk score in the context (e.g., "risk_score" or "reliability").
    risk_field: String,
    /// Actions with risk above this threshold are rejected.
    reject_threshold: f64,
    /// Actions with risk between escalate and reject are escalated to humans.
    escalate_threshold: f64,
    /// Who to escalate to.
    escalation_target: EscalationTarget,
}

impl RiskAwareEvaluator {
    /// Create a new risk-aware evaluator.
    ///
    /// - `risk_field`: the JSON key containing the risk score (0.0 to 1.0)
    /// - `escalate_threshold`: risk at or above this triggers human review
    /// - `reject_threshold`: risk at or above this is automatically rejected
    /// - `escalation_target`: where to escalate when risk is in the escalation range
    pub fn new(
        risk_field: impl Into<String>,
        escalate_threshold: f64,
        reject_threshold: f64,
        escalation_target: EscalationTarget,
    ) -> Self {
        Self {
            risk_field: risk_field.into(),
            reject_threshold,
            escalate_threshold,
            escalation_target,
        }
    }

    /// Extract risk score from context JSON.
    fn extract_risk(&self, context: &Value) -> Result<f64, GuardianError> {
        context
            .get(&self.risk_field)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                GuardianError::Evaluation(format!(
                    "Missing or invalid risk field '{}' in context",
                    self.risk_field
                ))
            })
    }
}

#[async_trait]
impl PolicyEvaluator for RiskAwareEvaluator {
    /// Evaluate an action based on its risk score.
    ///
    /// - risk >= reject_threshold: Reject
    /// - risk >= escalate_threshold: Escalate
    /// - otherwise: Approve
    async fn evaluate(
        &self,
        _action: &str,
        context: &Value,
    ) -> Result<PolicyDecision, GuardianError> {
        let risk = self.extract_risk(context)?;

        if risk >= self.reject_threshold {
            Ok(PolicyDecision::Reject {
                reason: format!(
                    "Risk score {:.3} exceeds reject threshold {:.3}",
                    risk, self.reject_threshold
                ),
            })
        } else if risk >= self.escalate_threshold {
            Ok(PolicyDecision::Escalate {
                target: self.escalation_target.clone(),
            })
        } else {
            Ok(PolicyDecision::Approve)
        }
    }

    /// Human-readable name of this policy engine.
    fn name(&self) -> &str {
        "risk-aware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_evaluator() -> RiskAwareEvaluator {
        RiskAwareEvaluator::new(
            "risk_score",
            0.5,  // escalate threshold
            0.8,  // reject threshold
            EscalationTarget::Slack {
                channel: "#risk-review".into(),
            },
        )
    }

    #[tokio::test]
    async fn approve_low_risk() {
        let evaluator = make_evaluator();
        let context = serde_json::json!({ "risk_score": 0.2 });
        let decision = evaluator.evaluate("deploy", &context).await.unwrap();
        assert!(decision.is_approved());
    }

    #[tokio::test]
    async fn escalate_medium_risk() {
        let evaluator = make_evaluator();
        let context = serde_json::json!({ "risk_score": 0.6 });
        let decision = evaluator.evaluate("deploy", &context).await.unwrap();
        assert!(decision.requires_human());
    }

    #[tokio::test]
    async fn reject_high_risk() {
        let evaluator = make_evaluator();
        let context = serde_json::json!({ "risk_score": 0.9 });
        let decision = evaluator.evaluate("deploy", &context).await.unwrap();
        assert!(!decision.is_approved());
        assert!(!decision.requires_human());
        match decision {
            PolicyDecision::Reject { reason } => {
                assert!(reason.contains("0.900"));
            }
            _ => panic!("Expected Reject"),
        }
    }
}
