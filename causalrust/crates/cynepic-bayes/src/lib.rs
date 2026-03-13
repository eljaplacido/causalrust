//! # cynepic-bayes
//!
//! Probabilistic programming and Bayesian inference for the cynepic-rs workspace.
//!
//! Provides conjugate prior families with closed-form posterior updates,
//! MCMC sampling for non-conjugate models, and an incremental belief-updating
//! API for real-time decision systems.
//!
//! # Quick Start
//!
//! ```rust
//! use cynepic_bayes::priors::BetaBinomial;
//!
//! // Start with a weak prior
//! let mut model = BetaBinomial::new(1.0, 1.0);
//!
//! // Update with observations: 7 successes, 3 failures
//! model.update(7, 3);
//!
//! // Posterior mean
//! assert!((model.mean() - 0.6667).abs() < 0.01);
//! ```

pub mod belief;
pub mod priors;
pub mod sampler;
pub mod streaming;
pub mod tool_belief;
