//! What was estimated, how its uncertainty was computed, and what that rests on.
//!
//! # Why a number is not a result
//!
//! `ATEResult { ate: 2.03, std_error: 0.11 }` is not enough to act on. Three
//! things are missing and each of them changes the decision:
//!
//! - **Which estimand?** ATE, ATT and LATE are different quantities that
//!   coincide only under effect homogeneity. A 2SLS fit recovers LATE — the
//!   effect among compliers — and reporting it as "the" treatment effect is a
//!   category error, not a rounding error. Under the heterogeneity levels in
//!   the standard test grid they differ by more than their standard errors.
//! - **Which standard error?** Classical, HC1 and HC3 answer different
//!   questions about the same fit. An interval built from the wrong one makes a
//!   confidence claim the data does not support.
//! - **What did it assume?** An estimate is conditional on an identification
//!   strategy. Detached from that strategy it is a number with no warrant.
//!
//! # Enforced, not documented
//!
//! [`ATEResult`] has private fields and no public constructor. It can only be
//! produced by an estimator in this crate, which means it cannot exist without
//! its estimand, its standard-error kind, and its diagnostics. That is
//! `research.md` recommendation 1 — "a causal estimate cannot be exported
//! without an attached robustness report" — expressed in the type system rather
//! than in a style guide.

use serde::{Deserialize, Serialize};

use crate::error::EstimationError;

/// Which causal quantity an estimate refers to.
///
/// These coincide under effect homogeneity and diverge otherwise. The type
/// exists so that divergence is visible at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Estimand {
    /// Average treatment effect over the whole population: `E[Y(1) - Y(0)]`.
    ///
    /// The effect of treating everyone. What a policy-wide rollout asks about.
    Ate,
    /// Average treatment effect on the treated: `E[Y(1) - Y(0) | T=1]`.
    ///
    /// The effect on those who actually received treatment. What "was this
    /// programme worth running?" asks about.
    Att,
    /// Average treatment effect on the controls: `E[Y(1) - Y(0) | T=0]`.
    ///
    /// The effect treatment would have had on those who did not get it. What
    /// "should we expand?" asks about.
    Atc,
    /// Local average treatment effect: the effect among compliers.
    ///
    /// What instrumental variables recover — and *only* that. Compliers are
    /// units whose treatment status the instrument actually moves; they are
    /// unobservable individually, so this population cannot be described except
    /// by the instrument that defines it.
    Late,
}

impl Estimand {
    /// Short label for tables and logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ate => "ATE",
            Self::Att => "ATT",
            Self::Atc => "ATC",
            Self::Late => "LATE",
        }
    }

    /// The population the estimate describes, in words.
    pub fn population(self) -> &'static str {
        match self {
            Self::Ate => "the whole population",
            Self::Att => "units that received treatment",
            Self::Atc => "units that did not receive treatment",
            Self::Late => "compliers — units the instrument moves",
        }
    }
}

/// How the reported standard error was computed.
///
/// Recorded because an interval is only as good as its variance estimate, and
/// because the wrong choice is silent: a classical standard error under
/// heteroskedasticity produces intervals that look identical to correct ones
/// and cover less often than they claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StdErrorKind {
    /// Assumes homoskedastic, independent errors. Cheapest and the most
    /// commonly wrong.
    Classical,
    /// White's heteroskedasticity-consistent estimator with the `n/(n-k)`
    /// finite-sample correction. A safe default.
    Hc1,
    /// Leverage-corrected robust estimator. Preferred at small `n`, where HC1
    /// still under-covers because high-leverage points shrink their own
    /// residuals.
    Hc3,
    /// Influence-function (sandwich) variance for a weighted estimator.
    ///
    /// The correct construction for Hájek-style IPW: it is derived from the
    /// same estimating equation that defines the point estimate, so the two
    /// cannot disagree about which quantity they describe.
    InfluenceFunction,
}

impl StdErrorKind {
    /// Whether this estimator remains valid under heteroskedasticity.
    pub fn is_robust(self) -> bool {
        !matches!(self, Self::Classical)
    }
}

/// Measurements taken during estimation that bear on whether to trust the
/// result.
///
/// Every field is something that was previously computed, used, and discarded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Numerical rank of the design matrix, and the rank a full-rank design
    /// would have. Equal values mean full rank.
    pub rank: Option<(usize, usize)>,
    /// Iterations used by an iterative fit, and whether it converged.
    pub convergence: Option<Convergence>,
    /// Smallest and largest fitted propensity. Values near 0 or 1 mean some
    /// units carry extreme weight and the effective sample is far below `n`.
    pub propensity_range: Option<(f64, f64)>,
    /// Kish effective sample size, `(sum w)^2 / sum w^2`.
    ///
    /// The honest `n` for a weighted estimator. When this is far below the
    /// nominal `n`, the estimate rests on many fewer units than it appears to.
    pub effective_n: Option<f64>,
    /// First-stage F statistic for an IV fit. Below 10 the instrument is weak.
    pub first_stage_f: Option<f64>,
    /// Units in each arm.
    pub arm_sizes: Option<(usize, usize)>,
}

/// Outcome of an iterative fit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Convergence {
    /// Iterations performed.
    pub iterations: usize,
    /// Norm of the score at the final iterate.
    pub gradient_norm: f64,
    /// Whether the tolerance was met.
    pub converged: bool,
}

/// An estimated causal effect, with everything needed to interpret it.
///
/// Constructible only inside this crate: an estimator produces it, and it
/// therefore always carries its estimand, its standard-error kind, and its
/// diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ATEResult {
    ate: f64,
    std_error: f64,
    n_obs: usize,
    estimand: Estimand,
    se_kind: StdErrorKind,
    diagnostics: Diagnostics,
}

impl ATEResult {
    /// Build a result. Crate-internal: only an estimator may claim to have
    /// estimated something.
    pub(crate) fn new(
        ate: f64,
        std_error: f64,
        n_obs: usize,
        estimand: Estimand,
        se_kind: StdErrorKind,
        diagnostics: Diagnostics,
    ) -> Result<Self, EstimationError> {
        if !ate.is_finite() {
            return Err(EstimationError::NotFinite {
                what: "point estimate",
                value: ate,
            });
        }
        // A standard error may legitimately be NaN — too few units in an arm to
        // form a variance is a real state and better represented than faked.
        // It may never be negative or infinite.
        if std_error < 0.0 || std_error.is_infinite() {
            return Err(EstimationError::NotFinite {
                what: "standard error",
                value: std_error,
            });
        }
        Ok(Self {
            ate,
            std_error,
            n_obs,
            estimand,
            se_kind,
            diagnostics,
        })
    }

    /// The point estimate.
    pub fn ate(&self) -> f64 {
        self.ate
    }

    /// The standard error. `NaN` when the sample cannot support a variance.
    pub fn std_error(&self) -> f64 {
        self.std_error
    }

    /// Units the estimate was computed from.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Which causal quantity this is. Not assumed — recorded by the estimator.
    pub fn estimand(&self) -> Estimand {
        self.estimand
    }

    /// How the standard error was computed.
    pub fn std_error_kind(&self) -> StdErrorKind {
        self.se_kind
    }

    /// Measurements taken during estimation.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// A normal-approximation confidence interval.
    ///
    /// `level` is the coverage, e.g. `0.95`. Returns `None` when the standard
    /// error is not available, because an interval of `NaN` invites a caller to
    /// format it into a report.
    pub fn confidence_interval(&self, level: f64) -> Option<(f64, f64)> {
        if !self.std_error.is_finite() || !(0.0..1.0).contains(&level) {
            return None;
        }
        let z = normal_quantile(0.5 + level / 2.0);
        Some((self.ate - z * self.std_error, self.ate + z * self.std_error))
    }

    /// Whether the interval at `level` excludes zero.
    ///
    /// Returns `None` rather than `false` when no interval exists, so "we could
    /// not tell" is distinguishable from "no effect".
    pub fn excludes_zero(&self, level: f64) -> Option<bool> {
        let (lo, hi) = self.confidence_interval(level)?;
        Some(lo > 0.0 || hi < 0.0)
    }

    /// One-line summary carrying the estimand and SE kind, so a number pasted
    /// out of a log still says what it is.
    pub fn summary(&self) -> String {
        let ci = self.confidence_interval(0.95).map_or_else(
            || "CI unavailable".to_string(),
            |(l, h)| format!("95% CI [{l:.4}, {h:.4}]"),
        );
        format!(
            "{} = {:.4} (se {:.4}, {:?}) {ci}, n = {}",
            self.estimand.label(),
            self.ate,
            self.std_error,
            self.se_kind,
            self.n_obs
        )
    }
}

/// Inverse standard normal CDF, Acklam's rational approximation.
///
/// Accurate to about 1.15e-9 over the whole range, which is far beyond what any
/// confidence interval needs. Used rather than pulling `statrs` into this
/// module because the crate targets `wasm32-wasip1` and every avoided
/// dependency is one less thing to port.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];

    // Breakpoints between the tail and central rational approximations.
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(ate: f64, se: f64) -> ATEResult {
        ATEResult::new(
            ate,
            se,
            100,
            Estimand::Ate,
            StdErrorKind::Hc1,
            Diagnostics::default(),
        )
        .expect("valid")
    }

    #[test]
    fn normal_quantile_matches_known_values() {
        // The three every table lists.
        assert!((normal_quantile(0.975) - 1.959_963_985).abs() < 1e-6);
        assert!((normal_quantile(0.995) - 2.575_829_304).abs() < 1e-6);
        assert!((normal_quantile(0.5)).abs() < 1e-12);
    }

    #[test]
    fn normal_quantile_is_symmetric() {
        for p in [0.001, 0.01, 0.1, 0.3] {
            let lo = normal_quantile(p);
            let hi = normal_quantile(1.0 - p);
            assert!((lo + hi).abs() < 1e-6, "asymmetric at p={p}: {lo} vs {hi}");
        }
    }

    #[test]
    fn confidence_interval_is_centred_on_the_estimate() {
        let r = result(2.0, 0.5);
        let (lo, hi) = r.confidence_interval(0.95).expect("finite se");
        assert!(((lo + hi) / 2.0 - 2.0).abs() < 1e-9);
        assert!((hi - lo - 2.0 * 1.959_963_985 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn wider_level_gives_wider_interval() {
        let r = result(2.0, 0.5);
        let (l90, h90) = r.confidence_interval(0.90).expect("finite se");
        let (l99, h99) = r.confidence_interval(0.99).expect("finite se");
        assert!(h99 - l99 > h90 - l90);
    }

    #[test]
    fn nan_standard_error_yields_no_interval() {
        let r = result(2.0, f64::NAN);
        assert!(r.confidence_interval(0.95).is_none());
        // "We could not tell" must not read as "no effect".
        assert!(r.excludes_zero(0.95).is_none());
    }

    #[test]
    fn excludes_zero_tracks_the_interval() {
        assert_eq!(result(2.0, 0.1).excludes_zero(0.95), Some(true));
        assert_eq!(result(0.05, 1.0).excludes_zero(0.95), Some(false));
        assert_eq!(result(-2.0, 0.1).excludes_zero(0.95), Some(true));
    }

    #[test]
    fn non_finite_point_estimate_is_rejected() {
        let err = ATEResult::new(
            f64::NAN,
            0.1,
            10,
            Estimand::Ate,
            StdErrorKind::Classical,
            Diagnostics::default(),
        );
        assert!(matches!(err, Err(EstimationError::NotFinite { .. })));
    }

    #[test]
    fn summary_names_the_estimand() {
        // A number pasted out of a log must still say which quantity it is.
        let r = ATEResult::new(
            1.5,
            0.2,
            50,
            Estimand::Late,
            StdErrorKind::InfluenceFunction,
            Diagnostics::default(),
        )
        .expect("valid");
        let s = r.summary();
        assert!(s.contains("LATE"), "{s}");
        assert!(s.contains("95% CI"), "{s}");
    }

    #[test]
    fn estimands_describe_distinct_populations() {
        let all = [Estimand::Ate, Estimand::Att, Estimand::Atc, Estimand::Late];
        let mut seen = std::collections::HashSet::new();
        for e in all {
            assert!(
                seen.insert(e.population()),
                "duplicate population for {e:?}"
            );
        }
    }

    #[test]
    fn only_classical_is_non_robust() {
        assert!(!StdErrorKind::Classical.is_robust());
        assert!(StdErrorKind::Hc1.is_robust());
        assert!(StdErrorKind::Hc3.is_robust());
        assert!(StdErrorKind::InfluenceFunction.is_robust());
    }
}
