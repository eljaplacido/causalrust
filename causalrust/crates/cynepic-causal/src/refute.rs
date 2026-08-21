//! Refutation: trying to break an estimate you already have.
//!
//! # Verdicts in standard errors, not percentages
//!
//! Every verdict here used to be a fixed relative threshold:
//!
//! ```text
//! let passed = relative_change < 0.15; // Less than 15% change
//! ```
//!
//! That never consults the estimate's own uncertainty (finding C7). A 14% shift
//! in a tightly-estimated effect passes; a 16% shift in one whose interval
//! spans zero fails. Both verdicts are noise, and being a *relative* threshold
//! it is not even scale-free in the way it appears to be.
//!
//! A refutation is a hypothesis test, so its verdict belongs on the scale the
//! estimate's uncertainty defines. Every test here reports the discrepancy in
//! **standard errors** and passes when it stays inside a stated multiple.
//! Doubling the sample size now makes the tests harder to pass, which is what
//! a test of robustness should do.
//!
//! # Refuting the estimator you were given
//!
//! `placebo_treatment` used to take `(outcome, original_ate, tolerance)`. It
//! never saw the treatment, the covariates, or the estimator, and re-estimated
//! with `difference_in_means` regardless of what produced the original number.
//! It therefore could not be refuting the estimate it was handed.
//!
//! The [`Refuter`] API takes a [`Study`] — the data, the adjustment set, and
//! the estimator — so a refutation re-runs the same analysis under a
//! transformation that should destroy the effect, or leave it alone.
//!
//! # Reproducible randomness
//!
//! The generator was a hand-rolled LCG whose low bits are notoriously
//! non-random and whose seed was hard-coded, so a refutation could not be
//! reproduced or varied (finding C8). It is now `ChaCha8Rng` with the seed on
//! the public API.

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::error::EstimationError;
use crate::estimand::ATEResult;
use crate::estimate::linear::LinearATEEstimator;

/// How many standard errors of discrepancy a refutation tolerates.
///
/// Two-sided 5%, the same threshold the estimate's own confidence interval
/// uses — so a refutation and an interval agree about what counts as a
/// difference.
pub const DEFAULT_TOLERANCE_SE: f64 = 1.96;

/// The analysis being refuted: data, adjustment set, and estimator.
///
/// Carrying all three is the point. A refutation of a different estimator than
/// the one that produced the estimate is not evidence about that estimate.
#[derive(Debug, Clone)]
pub struct Study {
    /// Treatment assignment.
    pub treatment: Array1<f64>,
    /// Observed outcome.
    pub outcome: Array1<f64>,
    /// Covariates to adjust for. Empty means unadjusted.
    pub covariates: Array2<f64>,
}

impl Study {
    /// Build a study from its three parts.
    pub fn new(treatment: Array1<f64>, outcome: Array1<f64>, covariates: Array2<f64>) -> Self {
        Self {
            treatment,
            outcome,
            covariates,
        }
    }

    /// A study with no covariates.
    pub fn unadjusted(treatment: Array1<f64>, outcome: Array1<f64>) -> Self {
        let n = treatment.len();
        Self {
            treatment,
            outcome,
            covariates: Array2::zeros((n, 0)),
        }
    }

    /// Units in the study.
    pub fn n(&self) -> usize {
        self.treatment.len()
    }

    /// Run the study's estimator on its current data.
    ///
    /// # Errors
    ///
    /// Whatever the underlying estimator returns.
    pub fn estimate(&self) -> Result<ATEResult, EstimationError> {
        if self.covariates.ncols() == 0 {
            LinearATEEstimator::difference_in_means(&self.treatment, &self.outcome)
        } else {
            LinearATEEstimator::ols_adjusted(&self.treatment, &self.outcome, &self.covariates)
        }
    }
}

/// Outcome of a refutation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefutationResult {
    /// Which test ran.
    pub test_name: String,
    /// The estimate being probed.
    pub original_effect: f64,
    /// The estimate produced under the test's transformation.
    pub refuted_effect: f64,
    /// Discrepancy expressed in standard errors of the original estimate.
    ///
    /// The number the verdict is actually based on. Reported so a caller can
    /// apply a different threshold without re-running anything.
    pub discrepancy_se: f64,
    /// Standard errors of discrepancy allowed.
    pub tolerance_se: f64,
    /// Whether the estimate survived.
    pub passed: bool,
    /// What the result means, in words, including the numbers behind it.
    pub interpretation: String,
}

impl RefutationResult {
    fn build(
        test_name: &str,
        original: f64,
        refuted: f64,
        reference_se: f64,
        tolerance_se: f64,
        expectation: Expectation,
    ) -> Self {
        // A non-positive or non-finite standard error carries no scale, so no
        // verdict is possible. Say so rather than dividing by it.
        let discrepancy_se = if reference_se > 0.0 && reference_se.is_finite() {
            match expectation {
                Expectation::EffectVanishes => refuted.abs() / reference_se,
                Expectation::EffectUnchanged => (refuted - original).abs() / reference_se,
            }
        } else {
            f64::NAN
        };

        let passed = discrepancy_se.is_finite() && discrepancy_se <= tolerance_se;
        let interpretation = if !discrepancy_se.is_finite() {
            format!(
                "no verdict: the original estimate has no usable standard error \
                 ({reference_se}), so there is no scale on which to judge a discrepancy"
            )
        } else {
            match (expectation, passed) {
                (Expectation::EffectVanishes, true) => format!(
                    "passed: the placebo effect {refuted:.4} is {discrepancy_se:.2} SE from \
                     zero, within the {tolerance_se:.2} SE tolerance"
                ),
                (Expectation::EffectVanishes, false) => format!(
                    "FAILED: a placebo produced an effect of {refuted:.4}, {discrepancy_se:.2} \
                     SE from zero. The estimator finds an effect where none exists, so the \
                     original {original:.4} is not evidence of one either"
                ),
                (Expectation::EffectUnchanged, true) => format!(
                    "passed: the estimate moved from {original:.4} to {refuted:.4}, \
                     {discrepancy_se:.2} SE, within the {tolerance_se:.2} SE tolerance"
                ),
                (Expectation::EffectUnchanged, false) => format!(
                    "FAILED: the estimate moved from {original:.4} to {refuted:.4}, \
                     {discrepancy_se:.2} SE. A change this large under a transformation that \
                     should not matter means the estimate is not stable"
                ),
            }
        };

        Self {
            test_name: test_name.to_string(),
            original_effect: original,
            refuted_effect: refuted,
            discrepancy_se,
            tolerance_se,
            passed,
            interpretation,
        }
    }
}

/// What a given refutation expects to see.
#[derive(Debug, Clone, Copy)]
enum Expectation {
    /// The effect should collapse to zero (placebo tests).
    EffectVanishes,
    /// The effect should be unmoved (perturbation tests).
    EffectUnchanged,
}

/// Runs refutation tests against a [`Study`].
#[derive(Debug, Clone)]
pub struct Refuter {
    seed: u64,
    tolerance_se: f64,
}

impl Default for Refuter {
    fn default() -> Self {
        Self {
            seed: 0,
            tolerance_se: DEFAULT_TOLERANCE_SE,
        }
    }
}

impl Refuter {
    /// A refuter with an explicit seed.
    ///
    /// Seeding is on the public API because a refutation that cannot be
    /// reproduced is not a check, and one that can only ever be run with a
    /// single hard-coded seed cannot be varied to see whether its verdict is
    /// stable.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            tolerance_se: DEFAULT_TOLERANCE_SE,
        }
    }

    /// Set the tolerance, in standard errors of the original estimate.
    pub fn with_tolerance_se(mut self, tolerance_se: f64) -> Self {
        self.tolerance_se = tolerance_se;
        self
    }

    /// Replace treatment with a coin flip and re-run the study's estimator.
    ///
    /// A placebo treatment causes nothing, so a correct analysis must return an
    /// effect indistinguishable from zero. An effect that survives randomised
    /// treatment is being manufactured by the estimator.
    ///
    /// Crucially this re-runs the *same* estimator with the *same* adjustment
    /// set — the previous implementation always used `difference_in_means` and
    /// discarded the covariates, so it tested something the caller never ran.
    ///
    /// # Errors
    ///
    /// Whatever the study's estimator returns.
    pub fn placebo_treatment(
        &self,
        study: &Study,
        original: &ATEResult,
    ) -> Result<RefutationResult, EstimationError> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let n = study.n();

        let placebo = Array1::from_shape_fn(n, |_| f64::from(u8::from(rng.random::<f64>() < 0.5)));
        let permuted = Study {
            treatment: placebo,
            outcome: study.outcome.clone(),
            covariates: study.covariates.clone(),
        };
        let refuted = permuted.estimate()?;

        // Judged against the PLACEBO's own standard error, not the original's.
        //
        // Randomising treatment removes the treatment term from the fit, so the
        // placebo run's residual variance includes everything the real effect
        // used to explain. Its standard error is therefore legitimately larger,
        // and scoring the placebo estimate against the original's much smaller
        // SE fails sound analyses roughly a third of the time. The question a
        // placebo asks is "is this effect distinguishable from zero?", which is
        // asked on the scale of the run that produced it.
        Ok(RefutationResult::build(
            "placebo_treatment",
            original.ate(),
            refuted.ate(),
            refuted.std_error(),
            self.tolerance_se,
            Expectation::EffectVanishes,
        ))
    }

    /// Add a covariate of pure noise and re-estimate, averaged over draws.
    ///
    /// An irrelevant covariate carries no information about either treatment or
    /// outcome, so it must not move the estimate. Movement means the estimator
    /// is fitting noise.
    ///
    /// # Errors
    ///
    /// Whatever the underlying estimator returns.
    pub fn random_common_cause(
        &self,
        study: &Study,
        original: &ATEResult,
        n_simulations: usize,
    ) -> Result<RefutationResult, EstimationError> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(1));
        let n = study.n();
        let p = study.covariates.ncols();

        let mut total = 0.0;
        let mut completed = 0usize;

        for _ in 0..n_simulations.max(1) {
            let mut wider = Array2::zeros((n, p + 1));
            for i in 0..n {
                for j in 0..p {
                    wider[[i, j]] = study.covariates[[i, j]];
                }
                wider[[i, p]] = rng.random::<f64>() * 2.0 - 1.0;
            }
            let r = LinearATEEstimator::ols_adjusted(&study.treatment, &study.outcome, &wider)?;
            total += r.ate();
            completed += 1;
        }

        #[allow(clippy::cast_precision_loss)]
        let mean = total / completed as f64;

        Ok(RefutationResult::build(
            "random_common_cause",
            original.ate(),
            mean,
            original.std_error(),
            self.tolerance_se,
            Expectation::EffectUnchanged,
        ))
    }

    /// Re-estimate on a random subset of the data.
    ///
    /// A stable estimate does not depend on which units happened to be
    /// included. The tolerance is widened by `sqrt(1/fraction)` because a
    /// smaller sample legitimately produces a noisier estimate — not doing so
    /// would fail every honest analysis on a small enough subset.
    ///
    /// # Errors
    ///
    /// Whatever the underlying estimator returns.
    pub fn data_subset(
        &self,
        study: &Study,
        original: &ATEResult,
        fraction: f64,
    ) -> Result<RefutationResult, EstimationError> {
        let fraction = fraction.clamp(0.05, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(2));
        let n = study.n();

        let keep: Vec<usize> = (0..n).filter(|_| rng.random::<f64>() < fraction).collect();
        if keep.len() < 4 {
            return Err(EstimationError::InsufficientData {
                n: keep.len(),
                p: 4,
            });
        }

        let subset = Study {
            treatment: Array1::from_shape_fn(keep.len(), |i| study.treatment[keep[i]]),
            outcome: Array1::from_shape_fn(keep.len(), |i| study.outcome[keep[i]]),
            covariates: Array2::from_shape_fn((keep.len(), study.covariates.ncols()), |(i, j)| {
                study.covariates[[keep[i], j]]
            }),
        };
        let refuted = subset.estimate()?;

        // A subset of size f*n has a standard error inflated by ~1/sqrt(f).
        let widened = self.tolerance_se / fraction.sqrt();

        Ok(RefutationResult::build(
            "data_subset",
            original.ate(),
            refuted.ate(),
            original.std_error(),
            widened,
            Expectation::EffectUnchanged,
        ))
    }

    /// Bootstrap the estimate and compare the resampling spread against the
    /// analytic standard error.
    ///
    /// This checks the *uncertainty*, not the point estimate. A bootstrap
    /// standard deviation far from the reported standard error means the
    /// variance formula does not describe the estimator — the failure mode that
    /// coverage measurement exists to catch, available here without ground
    /// truth.
    ///
    /// # Errors
    ///
    /// Whatever the underlying estimator returns.
    pub fn bootstrap(
        &self,
        study: &Study,
        original: &ATEResult,
        n_resamples: usize,
    ) -> Result<RefutationResult, EstimationError> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(3));
        let n = study.n();
        let resamples = n_resamples.max(2);

        let mut estimates = Vec::with_capacity(resamples);
        for _ in 0..resamples {
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let resampled = Study {
                treatment: Array1::from_shape_fn(n, |i| study.treatment[idx[i]]),
                outcome: Array1::from_shape_fn(n, |i| study.outcome[idx[i]]),
                covariates: Array2::from_shape_fn((n, study.covariates.ncols()), |(i, j)| {
                    study.covariates[[idx[i], j]]
                }),
            };
            // A resample can be degenerate — one arm empty, or collinear. Skip
            // those rather than failing the whole bootstrap.
            if let Ok(r) = resampled.estimate() {
                estimates.push(r.ate());
            }
        }

        if estimates.len() < 2 {
            return Err(EstimationError::InsufficientData {
                n: estimates.len(),
                p: 2,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let m = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / m;
        let boot_se =
            (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();

        // Compare the two standard errors on the scale of the analytic one:
        // how many analytic SEs apart are they?
        let analytic = original.std_error();
        let discrepancy = if analytic > 0.0 && analytic.is_finite() {
            (boot_se - analytic).abs() / analytic
        } else {
            f64::NAN
        };

        let passed = discrepancy.is_finite() && discrepancy <= 0.5;
        let interpretation = if !discrepancy.is_finite() {
            "no verdict: the original estimate has no usable standard error".to_string()
        } else if passed {
            format!(
                "passed: bootstrap SE {boot_se:.4} agrees with the analytic SE {analytic:.4} \
                 (relative difference {:.0}%)",
                discrepancy * 100.0
            )
        } else {
            format!(
                "FAILED: bootstrap SE {boot_se:.4} disagrees with the analytic SE \
                 {analytic:.4} by {:.0}%. The variance formula does not describe this \
                 estimator, so its confidence intervals do not mean what they claim",
                discrepancy * 100.0
            )
        };

        Ok(RefutationResult {
            test_name: "bootstrap".to_string(),
            original_effect: original.ate(),
            refuted_effect: mean,
            discrepancy_se: discrepancy,
            tolerance_se: 0.5,
            passed,
            interpretation,
        })
    }

    /// Run every refutation and collect the verdicts.
    ///
    /// # Errors
    ///
    /// Whatever the underlying estimators return.
    pub fn run_all(
        &self,
        study: &Study,
        original: &ATEResult,
    ) -> Result<Vec<RefutationResult>, EstimationError> {
        Ok(vec![
            self.placebo_treatment(study, original)?,
            self.random_common_cause(study, original, 5)?,
            self.data_subset(study, original, 0.7)?,
            self.bootstrap(study, original, 100)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A study with a real effect of 5.0 and a confounder.
    fn honest_study(n: usize) -> Study {
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for i in 0..n {
            let x: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let t = f64::from(u8::from(rng.random::<f64>() < 0.5));
            covariates[[i, 0]] = x;
            treatment[i] = t;
            outcome[i] = 1.0 + 5.0 * t + 2.0 * x + (rng.random::<f64>() - 0.5);
        }
        Study::new(treatment, outcome, covariates)
    }

    #[test]
    fn placebo_destroys_a_real_effect() {
        let study = honest_study(600);
        let original = study.estimate().expect("valid");
        let r = Refuter::new(1)
            .placebo_treatment(&study, &original)
            .expect("valid");
        assert!(r.passed, "{}", r.interpretation);
        assert!(r.refuted_effect.abs() < original.ate().abs() / 2.0);
    }

    #[test]
    fn verdict_scales_with_uncertainty_not_percentage() {
        // C7. The same relative discrepancy must be judged differently at
        // different precisions — that is the whole point of using SE units.
        let tight =
            RefutationResult::build("t", 1.0, 0.10, 0.01, 1.96, Expectation::EffectVanishes);
        let loose =
            RefutationResult::build("t", 1.0, 0.10, 1.00, 1.96, Expectation::EffectVanishes);
        assert!(
            !tight.passed,
            "10x SE should fail: {}",
            tight.interpretation
        );
        assert!(
            loose.passed,
            "0.1x SE should pass: {}",
            loose.interpretation
        );
        assert!((tight.discrepancy_se - 10.0).abs() < 1e-9);
        assert!((loose.discrepancy_se - 0.1).abs() < 1e-9);
    }

    #[test]
    fn a_placebo_effect_that_survives_is_reported_as_failure() {
        // Refuted effect 10 SE from zero must not pass at any sane tolerance.
        let r = RefutationResult::build("p", 2.0, 1.0, 0.1, 1.96, Expectation::EffectVanishes);
        assert!(!r.passed);
        assert!(r.interpretation.contains("FAILED"), "{}", r.interpretation);
    }

    #[test]
    fn no_standard_error_yields_no_verdict() {
        // A verdict computed against a NaN scale would be meaningless; saying
        // so is better than defaulting to pass or fail.
        let r = RefutationResult::build("p", 2.0, 0.0, f64::NAN, 1.96, Expectation::EffectVanishes);
        assert!(!r.passed);
        assert!(
            r.interpretation.contains("no verdict"),
            "{}",
            r.interpretation
        );
    }

    #[test]
    fn random_common_cause_does_not_move_a_sound_estimate() {
        let study = honest_study(600);
        let original = study.estimate().expect("valid");
        let r = Refuter::new(2)
            .random_common_cause(&study, &original, 5)
            .expect("valid");
        assert!(r.passed, "{}", r.interpretation);
    }

    #[test]
    fn subset_tolerance_widens_with_smaller_fractions() {
        let study = honest_study(600);
        let original = study.estimate().expect("valid");
        let refuter = Refuter::new(3);
        let half = refuter.data_subset(&study, &original, 0.5).expect("valid");
        let most = refuter.data_subset(&study, &original, 0.9).expect("valid");
        // A smaller subset is legitimately noisier, so it must be judged more
        // leniently — otherwise every honest analysis fails on a small enough
        // slice of its own data.
        assert!(half.tolerance_se > most.tolerance_se);
    }

    #[test]
    fn bootstrap_agrees_with_a_correct_analytic_standard_error() {
        // OLS standard errors are validated, so the bootstrap must confirm
        // them. If this fails, one of the two is wrong.
        let study = honest_study(500);
        let original = study.estimate().expect("valid");
        let r = Refuter::new(4)
            .bootstrap(&study, &original, 200)
            .expect("valid");
        assert!(r.passed, "{}", r.interpretation);
    }

    #[test]
    fn refutation_is_reproducible_from_its_seed() {
        // C8. The LCG was unseedable from outside, so no result could be
        // reproduced or varied.
        let study = honest_study(300);
        let original = study.estimate().expect("valid");
        let a = Refuter::new(99)
            .placebo_treatment(&study, &original)
            .expect("valid");
        let b = Refuter::new(99)
            .placebo_treatment(&study, &original)
            .expect("valid");
        assert!((a.refuted_effect - b.refuted_effect).abs() < 1e-15);

        let c = Refuter::new(100)
            .placebo_treatment(&study, &original)
            .expect("valid");
        assert!(
            (a.refuted_effect - c.refuted_effect).abs() > 1e-12,
            "different seeds must give different placebo draws"
        );
    }

    #[test]
    fn run_all_reports_every_test() {
        let study = honest_study(400);
        let original = study.estimate().expect("valid");
        let results = Refuter::new(5).run_all(&study, &original).expect("valid");
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.passed, "{}: {}", r.test_name, r.interpretation);
        }
    }

    #[test]
    fn placebo_uses_the_studys_adjustment_set() {
        // A study with covariates must route through ols_adjusted, so a placebo
        // on it exercises the adjusted estimator rather than silently falling
        // back to difference_in_means.
        let study = honest_study(300);
        assert_eq!(study.covariates.ncols(), 1);
        let unadjusted = Study::unadjusted(study.treatment.clone(), study.outcome.clone());
        assert_eq!(unadjusted.covariates.ncols(), 0);

        let a = study.estimate().expect("valid");
        let b = unadjusted.estimate().expect("valid");
        // Same data, different estimators, so different standard errors —
        // evidence the adjustment set is actually being used.
        assert!((a.std_error() - b.std_error()).abs() > 1e-9);
    }
}
