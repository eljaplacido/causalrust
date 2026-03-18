use serde::{Deserialize, Serialize};

use crate::{AuditEntry, CynefinDomain, PolicyDecision};

/// Discretized confidence level for epistemic assessments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceLevel {
    /// Strong evidence, narrow uncertainty bounds.
    High,
    /// Reasonable evidence, moderate uncertainty.
    Medium,
    /// Weak evidence, wide uncertainty bounds.
    Low,
    /// Insufficient data for any assessment.
    Unknown,
}

impl ConfidenceLevel {
    /// Map a numeric confidence score (0.0–1.0) to a discretized level.
    pub fn from_score(score: f64) -> Self {
        if score >= 0.85 {
            ConfidenceLevel::High
        } else if score >= 0.5 {
            ConfidenceLevel::Medium
        } else if score > 0.0 {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::Unknown
        }
    }
}

/// A single reasoning step in the epistemic chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Which engine or component produced this step.
    pub engine: String,
    /// What was concluded.
    pub conclusion: String,
    /// Confidence in this step's conclusion (0.0–1.0).
    pub confidence: f64,
    /// Optional supporting evidence or context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<serde_json::Value>,
}

impl ReasoningStep {
    /// Create a new reasoning step.
    pub fn new(
        engine: impl Into<String>,
        conclusion: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            engine: engine.into(),
            conclusion: conclusion.into(),
            confidence,
            evidence: None,
        }
    }

    /// Attach evidence to this reasoning step.
    pub fn with_evidence(mut self, evidence: serde_json::Value) -> Self {
        self.evidence = Some(evidence);
        self
    }
}

/// Unified epistemic session state carrying provenance through the decision pipeline.
///
/// Tracks domain classification, confidence, reasoning chain, and audit entries
/// as a query flows through router → engine → guardian → graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicState {
    /// Unique session identifier.
    pub session_id: uuid::Uuid,
    /// The classified complexity domain.
    pub domain: CynefinDomain,
    /// Overall confidence score (0.0–1.0).
    pub confidence: f64,
    /// Discretized confidence level.
    pub confidence_level: ConfidenceLevel,
    /// Ordered chain of reasoning steps.
    pub reasoning_chain: Vec<ReasoningStep>,
    /// Policy decisions made during this session.
    pub decisions: Vec<PolicyDecision>,
    /// Audit entries accumulated during this session.
    pub audit_trail: Vec<AuditEntry>,
    /// Session-level metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// When this state was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// When this state was last updated.
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl EpistemicState {
    /// Create a new epistemic state for a given domain and confidence.
    pub fn new(domain: CynefinDomain, confidence: f64) -> Self {
        let now = chrono::Utc::now();
        Self {
            session_id: uuid::Uuid::new_v4(),
            domain,
            confidence,
            confidence_level: ConfidenceLevel::from_score(confidence),
            reasoning_chain: Vec::new(),
            decisions: Vec::new(),
            audit_trail: Vec::new(),
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a reasoning step to the chain.
    pub fn add_reasoning(&mut self, step: ReasoningStep) {
        self.reasoning_chain.push(step);
        self.updated_at = chrono::Utc::now();
    }

    /// Record a policy decision.
    pub fn record_decision(&mut self, decision: PolicyDecision) {
        self.decisions.push(decision);
        self.updated_at = chrono::Utc::now();
    }

    /// Append an audit entry.
    pub fn record_audit(&mut self, entry: AuditEntry) {
        self.audit_trail.push(entry);
        self.updated_at = chrono::Utc::now();
    }

    /// Update the domain classification (e.g., after reclassification mid-pipeline).
    pub fn reclassify(&mut self, domain: CynefinDomain, confidence: f64) {
        self.domain = domain;
        self.confidence = confidence;
        self.confidence_level = ConfidenceLevel::from_score(confidence);
        self.updated_at = chrono::Utc::now();
    }

    /// Whether any decision in this session was rejected.
    pub fn has_rejection(&self) -> bool {
        self.decisions
            .iter()
            .any(|d| matches!(d, PolicyDecision::Reject { .. }))
    }

    /// Whether the session requires human escalation.
    pub fn requires_human(&self) -> bool {
        self.domain.requires_human() || self.decisions.iter().any(|d| d.requires_human())
    }

    /// Number of reasoning steps recorded.
    pub fn reasoning_depth(&self) -> usize {
        self.reasoning_chain.len()
    }

    /// Attach session-level metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_level_mapping() {
        assert_eq!(ConfidenceLevel::from_score(0.95), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_score(0.85), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_score(0.6), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_score(0.3), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::from_score(0.0), ConfidenceLevel::Unknown);
    }

    #[test]
    fn epistemic_state_lifecycle() {
        let mut state = EpistemicState::new(CynefinDomain::Complicated, 0.87);

        assert_eq!(state.domain, CynefinDomain::Complicated);
        assert_eq!(state.confidence_level, ConfidenceLevel::High);
        assert!(!state.requires_human());
        assert!(!state.has_rejection());
        assert_eq!(state.reasoning_depth(), 0);

        state.add_reasoning(ReasoningStep::new(
            "cynepic-causal",
            "Identified 2 confounders via backdoor criterion",
            0.92,
        ));
        assert_eq!(state.reasoning_depth(), 1);

        state.record_decision(PolicyDecision::Approve);
        assert!(!state.has_rejection());

        state.record_decision(PolicyDecision::Reject {
            reason: "budget exceeded".into(),
        });
        assert!(state.has_rejection());
    }

    #[test]
    fn reclassification_updates_confidence() {
        let mut state = EpistemicState::new(CynefinDomain::Clear, 0.95);
        assert_eq!(state.confidence_level, ConfidenceLevel::High);

        state.reclassify(CynefinDomain::Complex, 0.45);
        assert_eq!(state.domain, CynefinDomain::Complex);
        assert_eq!(state.confidence_level, ConfidenceLevel::Low);
    }

    #[test]
    fn requires_human_for_disorder() {
        let state = EpistemicState::new(CynefinDomain::Disorder, 0.3);
        assert!(state.requires_human());
    }

    #[test]
    fn serde_roundtrip() {
        let mut state = EpistemicState::new(CynefinDomain::Complex, 0.72);
        state.add_reasoning(ReasoningStep::new("bayes", "Updated posterior", 0.8));
        state.record_decision(PolicyDecision::Approve);

        let json = serde_json::to_string(&state).unwrap();
        let parsed: EpistemicState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, CynefinDomain::Complex);
        assert_eq!(parsed.reasoning_chain.len(), 1);
        assert_eq!(parsed.decisions.len(), 1);
    }
}
