//! # cynepic-causal
//!
//! Causal inference: model, identify, estimate, refute.
//!
//! # Status
//!
//! `LinearATEEstimator` is validated by simulation — interval coverage 94.7% to
//! 97.3% across the standard grid in `cynepic-testkit`, with bias below 0.02.
//! The other estimators are checked by unit tests and by the same coverage
//! harness; see `docs/FINDINGS.md` for what is measured and what is not.
//!
//! # Design
//!
//! Three things distinguish this crate from a collection of formulas:
//!
//! **Estimates carry their provenance.** [`ATEResult`] has no public
//! constructor. It can only come from an estimator, and it always names its
//! [`Estimand`] — ATE, ATT, ATC or LATE are different quantities — the kind of
//! standard error used, and diagnostics taken during the fit. A number cannot
//! be separated from what it means.
//!
//! **Failure is representable.** Rank-deficient designs, empty treatment arms,
//! non-convergent propensity models, insufficient overlap and unidentifiable
//! effects are all errors rather than plausible-looking numbers. Each one used
//! to be a value a caller could not distinguish from a real result.
//!
//! **Invariants are enforced where they are stated.** [`CausalDag`] rejects
//! cycles at insertion, so every value of the type is acyclic. [`d_separated`]
//! rejects variables the graph does not contain rather than reporting them
//! independent.
//!
//! # Quick start
//!
//! ```rust
//! use cynepic_causal::dag::CausalDag;
//! use cynepic_causal::identify::BackdoorCriterion;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut dag = CausalDag::new();
//! dag.add_edge("smoking", "tar")?;
//! dag.add_edge("tar", "cancer")?;
//! dag.add_edge("smoking", "cancer")?;
//!
//! // Returns a verified adjustment set, or explains why none exists.
//! let adjustment = BackdoorCriterion::find(&dag, "smoking", "cancer")?;
//! assert!(adjustment.is_empty()); // no confounding in this graph
//! # Ok(())
//! # }
//! ```
//!
//! Estimating, with the estimand attached to the answer:
//!
//! ```rust
//! use cynepic_causal::estimate::linear::LinearATEEstimator;
//! use ndarray::array;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
//! let outcome = array![10.0, 12.0, 14.0, 4.0, 6.0, 8.0];
//!
//! let result = LinearATEEstimator::difference_in_means(&treatment, &outcome)?;
//! assert_eq!(result.estimand().label(), "ATE");
//! assert!((result.ate() - 6.0).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```

pub mod counterfactual;
pub mod dag;
pub mod dsep;
pub mod error;
pub mod estimand;
pub mod estimate;
pub mod identify;
pub mod refute;

pub use counterfactual::{CounterfactualEngine, CounterfactualQuery, CounterfactualResult};
pub use dag::{CausalDag, VarKind};
pub use dsep::d_separated;
pub use error::{DagError, DsepError, EstimationError, IdentificationError};
pub use estimand::{ATEResult, Convergence, Diagnostics, Estimand, StdErrorKind};
pub use estimate::iv::IVEstimator;
pub use estimate::linear::LinearATEEstimator;
pub use estimate::propensity::{PropensityModel, PropensityScoreEstimator};
pub use identify::{AdjustmentSet, BackdoorCriterion, FrontDoorCriterion};
pub use refute::{RefutationResult, Refuter, Study};
