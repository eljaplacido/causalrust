//! Bias, RMSE and confidence-interval coverage over repeated simulation.
//!
//! # Coverage
//!
//! A nominal 95% confidence interval makes an operational promise: across
//! repeated samples from the same process, 95% of the intervals it produces
//! contain the true value. That promise is directly checkable when you generated
//! the data.
//!
//! ```text
//!   1. Draw 1,000 datasets from a DGP with true ATE = 2.0
//!   2. Run the estimator on each      -> 1,000 estimates + 1,000 intervals
//!   3. Count intervals containing 2.0 -> should be about 950
//! ```
//!
//! A result of 780 means the intervals are lying: either the point estimate is
//! biased, or the standard error is too small. Both are silent failures that no
//! unit test catches, because each individual run looks perfectly reasonable.
//!
//! # How to read the three numbers together
//!
//! | Bias | Coverage | Diagnosis |
//! |---|---|---|
//! | ≈ 0 | ≈ 95% | Estimator is sound on this DGP |
//! | ≈ 0 | too low | Point estimate fine, **standard error understated** |
//! | ≈ 0 | too high | Standard error overstated — intervals wastefully wide |
//! | large | too low | Estimator is biased; the interval is centred wrongly |
//! | large | ≈ 95% | Bias masked by intervals so wide they are uninformative |

use crate::dgp::{Dgp, GroundTruth};

/// One estimator run against one simulated dataset.
#[derive(Debug, Clone, Copy)]
pub struct Replication {
    /// The point estimate produced.
    pub estimate: f64,
    /// Lower bound of the reported interval.
    pub ci_lower: f64,
    /// Upper bound of the reported interval.
    pub ci_upper: f64,
    /// The true value this run was trying to recover.
    pub truth: f64,
    /// The seed that generated the dataset, for reproducing a failure.
    pub seed: u64,
}

impl Replication {
    /// Whether the reported interval contains the truth.
    pub fn covers(&self) -> bool {
        self.truth >= self.ci_lower && self.truth <= self.ci_upper
    }

    /// Signed error of the point estimate.
    pub fn error(&self) -> f64 {
        self.estimate - self.truth
    }

    /// Width of the reported interval.
    pub fn width(&self) -> f64 {
        self.ci_upper - self.ci_lower
    }
}

/// Aggregate performance of an estimator on one DGP.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Label of the DGP cell this report describes.
    pub cell: String,
    /// Replications that produced a usable estimate.
    pub n_replications: usize,
    /// Replications where the estimator returned an error or a non-finite value.
    pub n_failed: usize,
    /// Mean signed error. Near zero means unbiased.
    pub bias: f64,
    /// Root mean squared error — bias and variance together.
    pub rmse: f64,
    /// Fraction of intervals containing the truth. Compare against `nominal`.
    pub coverage: f64,
    /// The coverage the intervals claimed, e.g. 0.95.
    pub nominal: f64,
    /// Mean interval width. Useful for spotting coverage bought with vagueness.
    pub mean_ci_width: f64,
    /// Seeds of replications whose interval missed, for investigation.
    pub misses: Vec<u64>,
}

impl CoverageReport {
    /// Whether coverage sits within `tolerance` of nominal.
    ///
    /// With 1,000 replications the Monte Carlo standard error on a 95% coverage
    /// estimate is about 0.7pp, so a ±2pp band is roughly ±3 MC standard errors —
    /// tight enough to catch real problems, loose enough not to flake.
    pub fn coverage_ok(&self, tolerance: f64) -> bool {
        (self.coverage - self.nominal).abs() <= tolerance
    }

    /// A one-line summary suitable for a CI log or a published table.
    pub fn summary(&self) -> String {
        format!(
            "{:<24} n={:<5} bias={:+.4} rmse={:.4} coverage={:.1}% (nominal {:.0}%) width={:.3}{}",
            self.cell,
            self.n_replications,
            self.bias,
            self.rmse,
            self.coverage * 100.0,
            self.nominal * 100.0,
            self.mean_ci_width,
            if self.n_failed > 0 {
                // "refused", not "failed": with Result-returning estimators
                // this counts replications the estimator declined — no overlap,
                // rank deficiency, a weak instrument. Declining is correct
                // behaviour on a hostile cell, and conflating it with a wrong
                // answer would penalise exactly the estimators that are honest
                // about what they cannot do.
                format!(" refused={}", self.n_failed)
            } else {
                String::new()
            }
        )
    }
}

/// Runs an estimator repeatedly against a DGP and scores it.
///
/// The estimator is supplied as a closure so this harness stays independent of
/// the estimator API — which matters because that API is being rewritten in
/// Tier 1 and this crate should not need to change with it.
#[derive(Debug, Clone)]
pub struct ValidationHarness {
    /// Number of datasets to draw.
    pub replications: usize,
    /// First seed; subsequent replications use consecutive seeds.
    pub base_seed: u64,
    /// The confidence level the estimator claims to produce.
    pub nominal: f64,
}

impl Default for ValidationHarness {
    fn default() -> Self {
        Self {
            replications: 500,
            base_seed: 0xC0FFEE,
            nominal: 0.95,
        }
    }
}

impl ValidationHarness {
    /// A harness with the given number of replications.
    pub fn with_replications(mut self, r: usize) -> Self {
        self.replications = r;
        self
    }

    /// Set the base seed.
    pub fn with_base_seed(mut self, s: u64) -> Self {
        self.base_seed = s;
        self
    }

    /// Run `estimator` against `dgp` and score it.
    ///
    /// `estimator` receives a dataset and the ground truth, and returns the
    /// point estimate and interval, or `None` if it declined to produce one.
    /// Returning `None` is counted as a failure rather than silently skipped —
    /// an estimator that refuses on half the datasets has not "passed".
    ///
    /// `select_truth` picks which estimand is being targeted, so the same
    /// harness validates ATE, ATT and LATE estimators without special-casing.
    pub fn run<E, S>(
        &self,
        cell: &str,
        dgp: &Dgp,
        select_truth: S,
        mut estimator: E,
    ) -> CoverageReport
    where
        E: FnMut(&crate::dgp::Dataset) -> Option<(f64, f64, f64)>,
        S: Fn(&GroundTruth) -> Option<f64>,
    {
        let mut reps: Vec<Replication> = Vec::with_capacity(self.replications);
        let mut n_failed = 0usize;

        for k in 0..self.replications {
            let seed = self.base_seed.wrapping_add(k as u64);
            let data = dgp.sample(seed);

            let Some(truth) = select_truth(&data.truth) else {
                n_failed += 1;
                continue;
            };

            match estimator(&data) {
                Some((estimate, lo, hi))
                    if estimate.is_finite() && lo.is_finite() && hi.is_finite() =>
                {
                    reps.push(Replication {
                        estimate,
                        ci_lower: lo,
                        ci_upper: hi,
                        truth,
                        seed,
                    });
                }
                _ => n_failed += 1,
            }
        }

        Self::score(cell, &reps, n_failed, self.nominal)
    }

    fn score(cell: &str, reps: &[Replication], n_failed: usize, nominal: f64) -> CoverageReport {
        if reps.is_empty() {
            return CoverageReport {
                cell: cell.to_string(),
                n_replications: 0,
                n_failed,
                bias: f64::NAN,
                rmse: f64::NAN,
                coverage: f64::NAN,
                nominal,
                mean_ci_width: f64::NAN,
                misses: Vec::new(),
            };
        }

        let n = reps.len() as f64;
        let bias = reps.iter().map(|r| r.error()).sum::<f64>() / n;
        let rmse = (reps.iter().map(|r| r.error().powi(2)).sum::<f64>() / n).sqrt();
        let covered = reps.iter().filter(|r| r.covers()).count();
        let mean_ci_width = reps.iter().map(|r| r.width()).sum::<f64>() / n;
        let misses = reps
            .iter()
            .filter(|r| !r.covers())
            .map(|r| r.seed)
            .take(10)
            .collect();

        CoverageReport {
            cell: cell.to_string(),
            n_replications: reps.len(),
            n_failed,
            bias,
            rmse,
            coverage: covered as f64 / n,
            nominal,
            mean_ci_width,
            misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgp::Dgp;

    /// A deliberately perfect estimator: returns the truth with an interval that
    /// covers 95% of the time by construction. Validates the harness itself —
    /// if this does not score ~95%, the harness is wrong, not the estimator.
    #[test]
    fn harness_scores_a_calibrated_estimator_at_nominal() {
        let dgp = Dgp::new().with_n(50);
        let mut k = 0usize;
        let report = ValidationHarness::default().with_replications(400).run(
            "synthetic",
            &dgp,
            |t| Some(t.ate),
            |d| {
                // Miss deliberately on 5% of runs, in a fixed pattern.
                k += 1;
                let miss = k % 20 == 0;
                let truth = d.truth.ate;
                if miss {
                    Some((truth + 10.0, truth + 9.0, truth + 11.0))
                } else {
                    Some((truth, truth - 1.0, truth + 1.0))
                }
            },
        );

        assert!(
            report.coverage_ok(0.03),
            "harness mis-scored a calibrated estimator: {}",
            report.summary()
        );
    }

    /// An estimator with a badly understated standard error must be caught.
    /// This is the shape of finding C5.
    #[test]
    fn harness_catches_understated_standard_errors() {
        let dgp = Dgp::new().with_n(200);
        let report = ValidationHarness::default().with_replications(200).run(
            "too-narrow",
            &dgp,
            |t| Some(t.ate),
            |d| {
                // Correct point estimate, absurdly narrow interval.
                let truth = d.truth.ate;
                Some((truth + 0.5, truth + 0.49, truth + 0.51))
            },
        );

        assert!(
            report.coverage < 0.5,
            "an interval this narrow must fail coverage, got {}",
            report.summary()
        );
    }

    /// A biased estimator must show up in `bias`, not be hidden by the average.
    #[test]
    fn harness_detects_bias() {
        let dgp = Dgp::new().with_n(200);
        let report = ValidationHarness::default().with_replications(100).run(
            "biased",
            &dgp,
            |t| Some(t.ate),
            |d| {
                let truth = d.truth.ate;
                Some((truth + 1.0, truth + 0.9, truth + 1.1))
            },
        );

        assert!((report.bias - 1.0).abs() < 1e-6, "{}", report.summary());
        assert_eq!(report.coverage, 0.0);
    }

    /// Declining to estimate is counted as a failure, never silently dropped.
    #[test]
    fn declining_to_estimate_counts_as_failure() {
        let dgp = Dgp::new().with_n(50);
        let mut k = 0;
        let report = ValidationHarness::default().with_replications(100).run(
            "half-declines",
            &dgp,
            |t| Some(t.ate),
            |d| {
                k += 1;
                if k % 2 == 0 {
                    None
                } else {
                    let t = d.truth.ate;
                    Some((t, t - 1.0, t + 1.0))
                }
            },
        );

        assert_eq!(report.n_failed, 50);
        assert_eq!(report.n_replications, 50);
    }

    /// Non-finite output is a failure, not a data point. Guards against a NaN
    /// estimate quietly poisoning the aggregate.
    #[test]
    fn non_finite_output_counts_as_failure() {
        let dgp = Dgp::new().with_n(50);
        let report = ValidationHarness::default().with_replications(20).run(
            "nan",
            &dgp,
            |t| Some(t.ate),
            |_| Some((f64::NAN, f64::NEG_INFINITY, f64::INFINITY)),
        );

        assert_eq!(report.n_failed, 20);
        assert_eq!(report.n_replications, 0);
    }
}
