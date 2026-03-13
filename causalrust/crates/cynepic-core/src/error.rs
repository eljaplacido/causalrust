use thiserror::Error;

/// Top-level error type shared across the cynepic workspace.
#[derive(Debug, Error)]
pub enum CynepicError {
    /// A policy evaluation rejected the action.
    #[error("Policy rejected: {reason}")]
    PolicyRejection { reason: String },

    /// The requested analytical engine is not available.
    #[error("Engine not available for domain: {domain}")]
    EngineUnavailable { domain: crate::CynefinDomain },

    /// A classification produced no confident result.
    #[error("Classification failed: confidence {confidence:.2} below threshold {threshold:.2}")]
    LowConfidence { confidence: f64, threshold: f64 },

    /// The circuit breaker tripped (Chaotic domain).
    #[error("Circuit breaker tripped: {reason}")]
    CircuitBreakerTripped { reason: String },

    /// Human escalation is required.
    #[error("Human escalation required: {reason}")]
    HumanEscalationRequired { reason: String },

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}
