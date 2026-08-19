//! Causal effect estimators.
//!
//! - [`linear`] — difference in means and OLS with covariate adjustment,
//!   solved by rank-revealing QR, with classical, HC1 and HC3 standard errors.
//! - [`propensity`] — inverse probability weighting over an IRLS-fitted
//!   propensity model, with influence-function variance and overlap checks.
//! - [`iv`] — two-stage least squares, which estimates a LATE and says so.
//!
//! Every estimator returns [`crate::ATEResult`], which names the estimand it
//! computed rather than leaving the caller to assume.

pub mod iv;
pub mod linear;
pub mod propensity;

pub use crate::estimand::ATEResult;
