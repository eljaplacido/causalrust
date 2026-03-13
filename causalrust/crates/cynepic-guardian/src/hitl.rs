use cynepic_core::EscalationTarget;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::policy::GuardianError;

/// Events generated when human-in-the-loop intervention is needed.
///
/// These events can be sent to external systems (Slack, webhooks, etc.)
/// for approval, then used to resume paused workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationEvent {
    /// Unique ID for this escalation request.
    pub id: uuid::Uuid,
    /// When the escalation was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// The action that triggered escalation.
    pub action: String,
    /// Why escalation was triggered.
    pub reason: String,
    /// Who/where to escalate to.
    pub target: EscalationTarget,
    /// Context for the human reviewer.
    pub context: serde_json::Value,
    /// Current status of the escalation.
    pub status: EscalationStatus,
}

/// Status of an escalation event through its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EscalationStatus {
    /// Awaiting human review.
    Pending,
    /// Approved by a human reviewer.
    Approved {
        by: String,
        at: chrono::DateTime<chrono::Utc>,
    },
    /// Rejected by a human reviewer.
    Rejected {
        by: String,
        reason: String,
        at: chrono::DateTime<chrono::Utc>,
    },
    /// The escalation timed out without a response.
    TimedOut {
        at: chrono::DateTime<chrono::Utc>,
    },
}

/// Manages pending escalation events.
///
/// Tracks escalation lifecycle from creation through approval, rejection,
/// or timeout. Used in conjunction with policy evaluators that produce
/// `Escalate` decisions.
#[derive(Debug)]
pub struct EscalationManager {
    pending: HashMap<uuid::Uuid, EscalationEvent>,
    timeout: Duration,
}

impl EscalationManager {
    /// Create a new escalation manager with the given timeout duration.
    ///
    /// Escalations that are not resolved within `timeout` will be marked
    /// as timed out when `check_timeouts` is called.
    pub fn new(timeout: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            timeout,
        }
    }

    /// Create a new escalation event.
    ///
    /// Returns a reference to the newly created event.
    pub fn create_escalation(
        &mut self,
        action: String,
        reason: String,
        target: EscalationTarget,
        context: serde_json::Value,
    ) -> &EscalationEvent {
        let id = uuid::Uuid::new_v4();
        let event = EscalationEvent {
            id,
            created_at: chrono::Utc::now(),
            action,
            reason,
            target,
            context,
            status: EscalationStatus::Pending,
        };
        self.pending.insert(id, event);
        self.pending.get(&id).expect("just inserted")
    }

    /// Record approval for an escalation.
    pub fn approve(
        &mut self,
        id: &uuid::Uuid,
        by: String,
    ) -> Result<&EscalationEvent, GuardianError> {
        let event = self
            .pending
            .get_mut(id)
            .ok_or_else(|| GuardianError::NotFound(format!("Escalation {} not found", id)))?;
        event.status = EscalationStatus::Approved {
            by,
            at: chrono::Utc::now(),
        };
        Ok(event)
    }

    /// Record rejection for an escalation.
    pub fn reject(
        &mut self,
        id: &uuid::Uuid,
        by: String,
        reason: String,
    ) -> Result<&EscalationEvent, GuardianError> {
        let event = self
            .pending
            .get_mut(id)
            .ok_or_else(|| GuardianError::NotFound(format!("Escalation {} not found", id)))?;
        event.status = EscalationStatus::Rejected {
            by,
            reason,
            at: chrono::Utc::now(),
        };
        Ok(event)
    }

    /// Check and mark timed-out escalations.
    ///
    /// Returns the IDs of escalations that were marked as timed out.
    pub fn check_timeouts(&mut self) -> Vec<uuid::Uuid> {
        let timeout = self.timeout;
        let now = chrono::Utc::now();
        let mut timed_out = Vec::new();

        for (id, event) in self.pending.iter_mut() {
            if event.status == EscalationStatus::Pending {
                let elapsed = now
                    .signed_duration_since(event.created_at)
                    .to_std()
                    .unwrap_or(Duration::ZERO);
                if elapsed >= timeout {
                    event.status = EscalationStatus::TimedOut { at: now };
                    timed_out.push(*id);
                }
            }
        }

        timed_out
    }

    /// Get all pending escalations.
    pub fn pending(&self) -> Vec<&EscalationEvent> {
        self.pending
            .values()
            .filter(|e| e.status == EscalationStatus::Pending)
            .collect()
    }

    /// Get a specific escalation by ID.
    pub fn get(&self, id: &uuid::Uuid) -> Option<&EscalationEvent> {
        self.pending.get(id)
    }

    /// Check if a specific escalation was approved.
    pub fn is_approved(&self, id: &uuid::Uuid) -> bool {
        self.pending
            .get(id)
            .map(|e| matches!(e.status, EscalationStatus::Approved { .. }))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> EscalationManager {
        EscalationManager::new(Duration::from_millis(50))
    }

    #[test]
    fn create_escalation() {
        let mut mgr = make_manager();
        let event = mgr.create_escalation(
            "deploy_model".into(),
            "High risk score".into(),
            EscalationTarget::Slack {
                channel: "#approvals".into(),
            },
            serde_json::json!({"model": "gpt-4"}),
        );
        assert_eq!(event.action, "deploy_model");
        assert_eq!(event.status, EscalationStatus::Pending);
        assert_eq!(mgr.pending().len(), 1);
    }

    #[test]
    fn approve_escalation() {
        let mut mgr = make_manager();
        let event = mgr.create_escalation(
            "deploy".into(),
            "needs review".into(),
            EscalationTarget::Role {
                name: "admin".into(),
            },
            serde_json::json!({}),
        );
        let id = event.id;

        let approved = mgr.approve(&id, "alice".into()).unwrap();
        match &approved.status {
            EscalationStatus::Approved { by, .. } => assert_eq!(by, "alice"),
            _ => panic!("Expected Approved"),
        }
        assert!(mgr.is_approved(&id));
        // No longer in pending list
        assert_eq!(mgr.pending().len(), 0);
    }

    #[test]
    fn reject_escalation() {
        let mut mgr = make_manager();
        let event = mgr.create_escalation(
            "delete_data".into(),
            "destructive action".into(),
            EscalationTarget::Email {
                address: "admin@example.com".into(),
            },
            serde_json::json!({}),
        );
        let id = event.id;

        let rejected = mgr.reject(&id, "bob".into(), "too risky".into()).unwrap();
        match &rejected.status {
            EscalationStatus::Rejected { by, reason, .. } => {
                assert_eq!(by, "bob");
                assert_eq!(reason, "too risky");
            }
            _ => panic!("Expected Rejected"),
        }
        assert!(!mgr.is_approved(&id));
    }

    #[test]
    fn timeout_check() {
        let mut mgr = EscalationManager::new(Duration::from_millis(10));
        let event = mgr.create_escalation(
            "action".into(),
            "reason".into(),
            EscalationTarget::Webhook {
                url: "https://example.com".into(),
            },
            serde_json::json!({}),
        );
        let id = event.id;

        // Initially no timeouts
        assert!(mgr.check_timeouts().is_empty());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));

        let timed_out = mgr.check_timeouts();
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0], id);

        match &mgr.get(&id).unwrap().status {
            EscalationStatus::TimedOut { .. } => {}
            other => panic!("Expected TimedOut, got {:?}", other),
        }
    }
}
