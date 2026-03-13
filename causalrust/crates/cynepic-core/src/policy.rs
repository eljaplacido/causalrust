use serde::{Deserialize, Serialize};

/// Result of evaluating an action against a policy engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// The action is permitted.
    Approve,
    /// The action is denied with a reason.
    Reject { reason: String },
    /// The action requires human approval.
    Escalate { target: EscalationTarget },
}

impl PolicyDecision {
    /// Returns `true` if the action is approved.
    pub fn is_approved(&self) -> bool {
        matches!(self, PolicyDecision::Approve)
    }

    /// Returns `true` if the action requires human involvement.
    pub fn requires_human(&self) -> bool {
        matches!(self, PolicyDecision::Escalate { .. })
    }
}

/// Where to escalate a policy decision that requires human input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationTarget {
    /// Escalate via Slack notification.
    Slack { channel: String },
    /// Escalate via email.
    Email { address: String },
    /// Escalate to a named role or team.
    Role { name: String },
    /// Generic webhook escalation.
    Webhook { url: String },
}

/// A single entry in the policy audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier for this audit event.
    pub id: uuid::Uuid,
    /// When this event occurred.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// The action that was evaluated.
    pub action: String,
    /// Which policy engine produced this decision.
    pub engine: String,
    /// The decision that was made.
    pub decision: PolicyDecision,
    /// Optional metadata (e.g., matched policy rule, input context).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl AuditEntry {
    /// Create a new audit entry with the current timestamp.
    pub fn new(action: impl Into<String>, engine: impl Into<String>, decision: PolicyDecision) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            action: action.into(),
            engine: engine.into(),
            decision,
            metadata: None,
        }
    }

    /// Attach metadata to this audit entry.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approve_is_approved() {
        assert!(PolicyDecision::Approve.is_approved());
    }

    #[test]
    fn reject_is_not_approved() {
        let d = PolicyDecision::Reject {
            reason: "budget exceeded".into(),
        };
        assert!(!d.is_approved());
    }

    #[test]
    fn escalate_requires_human() {
        let d = PolicyDecision::Escalate {
            target: EscalationTarget::Slack {
                channel: "#approvals".into(),
            },
        };
        assert!(d.requires_human());
    }

    #[test]
    fn audit_entry_roundtrip() {
        let entry = AuditEntry::new("deploy_model", "cynepic-guardian", PolicyDecision::Approve);
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.action, "deploy_model");
        assert!(parsed.decision.is_approved());
    }
}
