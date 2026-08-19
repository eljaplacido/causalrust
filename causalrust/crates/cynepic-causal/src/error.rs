//! Errors, and the reason this crate has them at all.
//!
//! Every variant here replaces something that used to be a plausible-looking
//! number. `RankDeficient` replaces an ATE reported with a standard error of
//! exactly 0.0; `EmptyArm` replaces an effect computed against a mean of 0.0
//! for a group with no members; `NotConverged` replaces a propensity model that
//! stopped halfway and said nothing.
//!
//! The through-line is that a caller cannot detect any of those from the value
//! alone. A `Result` moves the failure from the number into the type, where it
//! cannot be ignored by accident.

use thiserror::Error;

/// Failures that can occur while estimating a causal effect.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EstimationError {
    /// Input arrays disagree about how many units there are.
    #[error("length mismatch: {name_a} has {len_a} rows, {name_b} has {len_b}")]
    LengthMismatch {
        /// Name of the first input.
        name_a: &'static str,
        /// Length of the first input.
        len_a: usize,
        /// Name of the second input.
        name_b: &'static str,
        /// Length of the second input.
        len_b: usize,
    },

    /// No units at all.
    #[error("no observations")]
    NoObservations,

    /// One of the treatment arms is empty, so no contrast exists.
    ///
    /// Previously the missing arm's mean was taken to be 0.0 and the difference
    /// returned, which yields a confident effect equal to the other arm's mean.
    #[error(
        "treatment arm '{arm}' is empty ({n_treated} treated, {n_control} control); \
         no contrast is defined"
    )]
    EmptyArm {
        /// Which arm is empty.
        arm: &'static str,
        /// Units with T=1.
        n_treated: usize,
        /// Units with T=0.
        n_control: usize,
    },

    /// Fewer units than parameters, so the model is not estimable.
    #[error("{n} observations cannot identify {p} parameters")]
    InsufficientData {
        /// Units available.
        n: usize,
        /// Parameters required.
        p: usize,
    },

    /// The design matrix is rank deficient: some column is an exact (or
    /// numerically exact) linear combination of others.
    ///
    /// The offending columns are named because "your matrix is singular" is not
    /// actionable and "columns 3 and 7 are aliased with column 1" is.
    #[error(
        "design matrix is rank deficient: rank {rank} of {expected} expected. \
         Aliased columns: {aliased:?}. Drop one of each aliased group, or use a \
         regularised estimator."
    )]
    RankDeficient {
        /// Numerical rank actually found.
        rank: usize,
        /// Rank a full-rank design would have.
        expected: usize,
        /// Human-readable names of the columns that are linearly dependent.
        aliased: Vec<String>,
    },

    /// An iterative fit did not reach its convergence tolerance.
    ///
    /// The previous implementation ran a fixed iteration count and returned
    /// whatever it had, which is indistinguishable from a converged fit at the
    /// call site and was worth roughly 86% of IPW's bias.
    #[error(
        "{model} did not converge after {iterations} iterations \
         (gradient norm {gradient_norm:e}, tolerance {tolerance:e})"
    )]
    NotConverged {
        /// Which model failed to converge.
        model: &'static str,
        /// Iterations performed.
        iterations: usize,
        /// Norm of the score at the last iterate.
        gradient_norm: f64,
        /// Tolerance that was not met.
        tolerance: f64,
    },

    /// The covariates perfectly (or near-perfectly) predict treatment, so no
    /// finite maximum-likelihood estimate exists.
    ///
    /// Detected because the gradient test cannot see it: under separation the
    /// coefficients diverge, but the score goes to zero as the fitted
    /// probabilities saturate, so an iterative fit reports convergence at
    /// whatever finite value it happened to reach. The fitted probabilities
    /// pinned at 0 and 1 are the observable symptom.
    #[error(
        "separation: {n_separated} of {n} units have a fitted probability within \
         {tolerance:e} of 0 or 1. The covariates predict treatment perfectly, so no \
         finite coefficient vector maximises the likelihood. Drop the offending \
         covariate or use a penalised fit."
    )]
    Separation {
        /// Units whose fitted probability is saturated.
        n_separated: usize,
        /// Total units.
        n: usize,
        /// How close to 0 or 1 counts as saturated.
        tolerance: f64,
    },

    /// Estimated propensities leave no region where treated and control units
    /// are comparable.
    ///
    /// Weighting cannot repair this. It is a property of the data, and the
    /// correct response is to say so rather than to return a number computed
    /// from a handful of units carrying enormous weight.
    #[error(
        "insufficient overlap: {n_extreme} of {n} units have propensity outside \
         [{lower}, {upper}] (min {min_propensity:.4}, max {max_propensity:.4}). \
         The effect is not estimable by weighting on this sample."
    )]
    InsufficientOverlap {
        /// Units outside the trimming bounds.
        n_extreme: usize,
        /// Total units.
        n: usize,
        /// Lower propensity bound.
        lower: f64,
        /// Upper propensity bound.
        upper: f64,
        /// Smallest fitted propensity.
        min_propensity: f64,
        /// Largest fitted propensity.
        max_propensity: f64,
    },

    /// An instrument is too weakly correlated with treatment for 2SLS to be
    /// meaningful.
    ///
    /// Below the conventional first-stage F of 10 (Stock–Yogo), 2SLS is more
    /// biased than OLS, so returning an estimate would be actively misleading.
    #[error(
        "weak instrument: first-stage F = {first_stage_f:.2}, below the \
         Stock-Yogo threshold of {threshold:.1}. 2SLS is more biased than OLS here."
    )]
    WeakInstrument {
        /// First-stage F statistic.
        first_stage_f: f64,
        /// Threshold applied.
        threshold: f64,
    },

    /// Treatment is constant, so there is nothing to contrast.
    #[error("treatment is constant at {value}; no contrast exists")]
    ConstantTreatment {
        /// The single value treatment takes.
        value: f64,
    },

    /// A computation produced a non-finite value.
    ///
    /// The last line of defence. Reaching this means an input was pathological
    /// in a way not caught above, and returning NaN to a caller who will format
    /// it into a report is worse than failing.
    #[error("{what} was not finite ({value})")]
    NotFinite {
        /// What was being computed.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
}

/// Failures when constructing or mutating a [`crate::dag::CausalDag`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DagError {
    /// The edge would introduce a directed cycle.
    ///
    /// A type named `CausalDag` holding a cyclic graph invalidates every
    /// algorithm built on it — d-separation, backdoor search and the
    /// counterfactual engine all assume acyclicity.
    #[error("edge {cause} -> {effect} would create a cycle: {}", path.join(" -> "))]
    WouldCreateCycle {
        /// Source of the rejected edge.
        cause: String,
        /// Target of the rejected edge.
        effect: String,
        /// The cycle the edge would close.
        path: Vec<String>,
    },

    /// An edge from a variable to itself.
    #[error("self-loop on '{variable}': a variable cannot cause itself")]
    SelfLoop {
        /// The variable.
        variable: String,
    },

    /// A variable name that is not in the graph.
    #[error("unknown variable '{name}'; the graph contains {known:?}")]
    UnknownVariable {
        /// The name that was not found.
        name: String,
        /// Variables the graph does contain, for typo-spotting.
        known: Vec<String>,
    },
}

/// Failures when testing d-separation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DsepError {
    /// A variable named in the query is not in the graph.
    ///
    /// This used to return `true`, meaning "d-separated", meaning
    /// "conditionally independent". A typo therefore produced a *positive*
    /// finding of independence — the answer most likely to be acted on.
    #[error("unknown variable '{name}' in d-separation query; the graph contains {known:?}")]
    UnknownVariable {
        /// The name that was not found.
        name: String,
        /// Variables the graph does contain.
        known: Vec<String>,
    },
}

/// Why a causal effect could not be identified from the graph.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IdentificationError {
    /// No observed set blocks every backdoor path.
    ///
    /// Distinct from "no adjustment needed", which is a successful
    /// identification returning the empty set. Collapsing the two into
    /// `Option::None` made a criterion that could never say no.
    #[error(
        "effect of {treatment} on {outcome} is not identifiable by adjustment: \
         {} backdoor path(s) cannot be blocked by observed variables. \
         Unblockable: {unblockable:?}",
        unblockable.len()
    )]
    NotIdentifiable {
        /// Treatment variable.
        treatment: String,
        /// Outcome variable.
        outcome: String,
        /// Paths no observed set can block, each as a variable sequence.
        unblockable: Vec<Vec<String>>,
    },

    /// Every candidate adjustment set requires a variable marked latent.
    #[error(
        "adjustment for {treatment} -> {outcome} requires unobserved variable(s) \
         {latent:?}; mark them observed or find another identification strategy"
    )]
    RequiresLatent {
        /// Treatment variable.
        treatment: String,
        /// Outcome variable.
        outcome: String,
        /// The latent variables that would be needed.
        latent: Vec<String>,
    },

    /// A variable named in the query is not in the graph.
    #[error("unknown variable '{name}'; the graph contains {known:?}")]
    UnknownVariable {
        /// The name that was not found.
        name: String,
        /// Variables the graph does contain.
        known: Vec<String>,
    },
}
