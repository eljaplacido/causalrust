//! # cynepic-guardian
//!
//! AI policy guardrails for the cynepic-rs workspace.
//!
//! Provides a composable policy evaluation framework that chains multiple policy
//! engines (Rego via `regorus`, Cedar, custom rules) with a unified audit trail
//! and circuit-breaker pattern for LLM output interception.
//!
//! # Architecture
//!
//! ```text
//! Action → PolicyEvaluator chain → [Approve | Reject | Escalate]
//!                                       ↓
//!                                  AuditTrail (append-only log)
//! ```

pub mod audit;
pub mod circuit_breaker;
pub mod hitl;
pub mod loop_detector;
pub mod policy;
pub mod rate_limiter;
pub mod risk;

pub use audit::AuditTrail;
pub use circuit_breaker::CircuitBreaker;
pub use hitl::{EscalationEvent, EscalationManager, EscalationStatus};
pub use loop_detector::{LoopDetector, LoopViolation};
pub use policy::{PolicyChain, PolicyEvaluator};
pub use rate_limiter::{RateLimitDecision, RateLimiter};
pub use risk::RiskAwareEvaluator;
#[cfg(feature = "rego")]
pub use policy::RegoPolicyEvaluator;
