//! # cynepic-core
//!
//! Shared types, traits, and contracts for the `cynepic-rs` decision intelligence workspace.
//!
//! This crate defines the foundational abstractions that all other `cynepic-*` crates
//! depend on, including the Cynefin domain model, analytical engine traits, policy
//! decision types, and audit entry structures.

mod domain;
mod engine;
pub mod epistemic;
mod error;
mod policy;
pub mod special;

pub use domain::CynefinDomain;
pub use engine::AnalyticalEngine;
pub use epistemic::{ConfidenceLevel, EpistemicState, ReasoningStep};
pub use error::CynepicError;
pub use policy::{AuditEntry, EscalationTarget, PolicyDecision};
pub use special::{beta_cdf, beta_quantile, gamma_cdf, gamma_quantile, ln_gamma, t_quantile};
