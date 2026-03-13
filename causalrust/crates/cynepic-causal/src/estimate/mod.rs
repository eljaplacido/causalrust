//! Causal effect estimators.
//!
//! Provides:
//! - **Linear regression ATE**: Average Treatment Effect via OLS
//! - **Propensity score IPW**: Inverse Probability Weighting
//! - **Instrumental variables**: Two-Stage Least Squares (2SLS)

pub mod iv;
pub mod linear;
pub mod propensity;
