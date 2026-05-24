use cynepic_core::AuditEntry;
use std::sync::{Arc, Mutex};

/// Append-only audit trail for policy decisions.
///
/// Thread-safe via `Arc<Mutex<_>>` for use across async tasks.
#[derive(Debug, Clone)]
pub struct AuditTrail {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
}

impl AuditTrail {
    /// Create a new empty audit trail.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record a new audit entry.
    pub fn record(&self, entry: AuditEntry) {
        let mut entries = self.entries.lock().expect("audit trail lock poisoned");
        tracing::debug!(
            action = %entry.action,
            engine = %entry.engine,
            decision = ?entry.decision,
            "Audit entry recorded"
        );
        entries.push(entry);
    }

    /// Get all entries (clones the entire Vec — O(n) memory).
    ///
    /// For large audit trails, prefer [`recent_entries`] or [`iter`].
    pub fn entries(&self) -> Vec<AuditEntry> {
        self.entries.lock().expect("audit trail lock poisoned").clone()
    }

    /// Get the most recent `limit` entries (O(limit) memory).
    pub fn recent_entries(&self, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.lock().expect("audit trail lock poisoned");
        entries.iter().rev().take(limit).rev().cloned().collect()
    }

    /// Apply a closure to audit entries while holding the lock.
    ///
    /// Prefer this over [`entries`] for large audit trails to avoid full clone.
    pub fn with_entries<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[AuditEntry]) -> R,
    {
        let entries = self.entries.lock().expect("audit trail lock poisoned");
        f(&entries)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("audit trail lock poisoned").len()
    }

    /// Whether the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Export all entries as a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let entries = self.entries();
        serde_json::to_string_pretty(&entries)
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cynepic_core::PolicyDecision;

    #[test]
    fn record_and_retrieve() {
        let trail = AuditTrail::new();
        assert!(trail.is_empty());

        trail.record(AuditEntry::new("action_1", "test_engine", PolicyDecision::Approve));
        trail.record(AuditEntry::new(
            "action_2",
            "test_engine",
            PolicyDecision::Reject { reason: "denied".into() },
        ));

        assert_eq!(trail.len(), 2);
        let entries = trail.entries();
        assert!(entries[0].decision.is_approved());
        assert!(!entries[1].decision.is_approved());
    }

    #[test]
    fn json_export() {
        let trail = AuditTrail::new();
        trail.record(AuditEntry::new("deploy", "guardian", PolicyDecision::Approve));
        let json = trail.to_json().unwrap();
        assert!(json.contains("deploy"));
    }
}
