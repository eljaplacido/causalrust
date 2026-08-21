use serde::{Deserialize, Serialize};
use std::fmt;

/// The five Cynefin complexity domains used for semantic query classification.
///
/// Each domain maps to a different analytical strategy:
/// - **Clear**: Deterministic lookup — the answer is known and retrievable.
/// - **Complicated**: Causal inference — the answer requires expert analysis via DAGs.
/// - **Complex**: Bayesian exploration — irreducible uncertainty, probe and update beliefs.
/// - **Chaotic**: Circuit breaker — emergency stop, stabilize before analyzing.
/// - **Disorder**: Human escalation — the domain itself is unclear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CynefinDomain {
    Clear,
    Complicated,
    Complex,
    Chaotic,
    #[default]
    Disorder,
}

impl CynefinDomain {
    /// Returns all domain variants in canonical order.
    pub fn all() -> &'static [CynefinDomain] {
        &[
            CynefinDomain::Clear,
            CynefinDomain::Complicated,
            CynefinDomain::Complex,
            CynefinDomain::Chaotic,
            CynefinDomain::Disorder,
        ]
    }

    /// Whether this domain requires human involvement.
    pub fn requires_human(&self) -> bool {
        matches!(self, CynefinDomain::Disorder)
    }

    /// Whether this domain should trigger safety mechanisms.
    pub fn is_emergency(&self) -> bool {
        matches!(self, CynefinDomain::Chaotic)
    }
}

impl fmt::Display for CynefinDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CynefinDomain::Clear => write!(f, "Clear"),
            CynefinDomain::Complicated => write!(f, "Complicated"),
            CynefinDomain::Complex => write!(f, "Complex"),
            CynefinDomain::Chaotic => write!(f, "Chaotic"),
            CynefinDomain::Disorder => write!(f, "Disorder"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_domains_are_exhaustive() {
        assert_eq!(CynefinDomain::all().len(), 5);
    }

    #[test]
    fn disorder_requires_human() {
        assert!(CynefinDomain::Disorder.requires_human());
        assert!(!CynefinDomain::Clear.requires_human());
    }

    #[test]
    fn chaotic_is_emergency() {
        assert!(CynefinDomain::Chaotic.is_emergency());
        assert!(!CynefinDomain::Complicated.is_emergency());
    }

    #[test]
    fn serde_roundtrip() {
        let domain = CynefinDomain::Complicated;
        let json = serde_json::to_string(&domain).unwrap();
        assert_eq!(json, "\"complicated\"");
        let parsed: CynefinDomain = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, domain);
    }
}
