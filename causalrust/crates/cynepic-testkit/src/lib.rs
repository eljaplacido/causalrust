//! # cynepic-testkit
//!
//! Ground-truth simulation and statistical validation for the cynepic-rs
//! workspace.
//!
//! ## Why this crate exists
//!
//! A causal estimate cannot be cross-validated. There is no held-out set that
//! tells you the true effect, which is the single most-cited barrier to trusting
//! causal methods in production.
//!
//! That objection is about *one estimate*. An **estimator** can be validated:
//! generate data from a process whose true effect you chose yourself, run the
//! estimator many times, and measure how often it is right and how often its
//! confidence intervals actually contain the truth.
//!
//! That is what this crate provides:
//!
//! - [`dgp`] — seeded data-generating processes with a known effect, and a
//!   factorial grid over the conditions that break estimators.
//! - [`validate`] — bias, RMSE and **confidence-interval coverage** over many
//!   replications.
//! - [`metamorphic`] — relations that must hold for *any* correct estimator,
//!   regardless of the data.
//! - [`calibration`] — the Bayesian counterpart: credible-interval coverage
//!   under draws from the prior, and simulation-based calibration for
//!   samplers, whose rank histogram diagnoses *how* a posterior is wrong
//!   rather than only that it is.
//!
//! ## The metric that matters
//!
//! Coverage. Simulate 1,000 datasets with a true ATE of 2.0, run the estimator,
//! and count how many nominal 95% intervals contain 2.0. A correct estimator
//! gives ≈950. An estimator that gives 780 is broken — either the point estimate
//! is biased or the standard error is understated — and no amount of speed
//! compensates.
//!
//! Nobody in the Rust causal ecosystem currently publishes coverage. It is the
//! most credible artifact this project could produce, precisely because it is a
//! measurement that can come back bad.
//!
//! ## Determinism
//!
//! Everything here is seeded with `ChaCha8Rng`. The same seed produces the same
//! dataset on every platform, so a failing validation run is always reproducible
//! from the seed printed in the failure message.

#![forbid(unsafe_code)]

pub mod calibration;
pub mod dgp;
pub mod metamorphic;
pub mod validate;

pub use calibration::{
    CalibrationReport, SbcReport, credible_coverage, simulation_based_calibration,
};
pub use dgp::{Dataset, Dgp, DgpGrid, GroundTruth};
pub use validate::{CoverageReport, Replication, ValidationHarness};
