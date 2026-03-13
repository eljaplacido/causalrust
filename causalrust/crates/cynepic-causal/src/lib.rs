//! # cynepic-causal
//!
//! A causal inference engine inspired by Microsoft's DoWhy, built in Rust.
//!
//! Provides the full causal inference pipeline:
//! 1. **Model** — Define a Structural Causal Model (SCM) as a DAG
//! 2. **Identify** — Find valid adjustment sets (backdoor, frontdoor, IV)
//! 3. **Estimate** — Compute causal effects (ATE via linear regression, propensity scoring, IV)
//! 4. **Refute** — Validate estimates via placebo treatments, random common causes, bootstrap
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use cynepic_causal::dag::CausalDag;
//! use cynepic_causal::identify::BackdoorCriterion;
//!
//! let mut dag = CausalDag::new();
//! dag.add_edge("smoking", "tar");
//! dag.add_edge("tar", "cancer");
//! dag.add_edge("smoking", "cancer");
//!
//! let adjustment = BackdoorCriterion::find(&dag, "smoking", "cancer");
//! ```

pub mod dag;
pub mod dsep;
pub mod estimate;
pub mod identify;
pub mod refute;

// Re-export key public types for convenience.
pub use dag::CausalDag;
pub use dsep::d_separated;
pub use estimate::iv::IVEstimator;
pub use estimate::linear::{ATEResult, LinearATEEstimator};
pub use estimate::propensity::PropensityScoreEstimator;
pub use identify::{BackdoorCriterion, FrontDoorCriterion};
pub use refute::RefutationResult;
