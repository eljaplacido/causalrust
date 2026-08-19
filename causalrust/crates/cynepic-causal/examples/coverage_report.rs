//! Print the estimator coverage table across the standard DGP grid.
//!
//! This is the credibility artifact: it measures something that can come back
//! bad. Speed benchmarks only ever flatter — you would not publish one that made
//! you look slow. Coverage can, and does, expose broken standard errors.
//!
//! ```bash
//! cargo run -p cynepic-causal --example coverage_report --release
//! ```
//!
//! # Reading the output
//!
//! `coverage` is the fraction of nominal 95% intervals that contained the truth.
//! It should be close to 95%. Read it together with `bias`:
//!
//! - bias ≈ 0, coverage ≈ 95%  → sound on this DGP
//! - bias ≈ 0, coverage low    → standard error understated
//! - bias large, coverage low  → estimator is biased
//! - bias large, coverage ≈95% → intervals so wide they hide the bias
//!
//! Cells where an estimator degrades are expected and should be published, not
//! hidden. A map of where a method works is more useful than a claim that it
//! always does.

use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_testkit::{DgpGrid, ValidationHarness};

fn main() {
    let grid = DgpGrid::standard();
    let harness = ValidationHarness::default().with_replications(300);

    println!("cynepic-causal — confidence-interval coverage");
    println!("{} replications per cell, nominal 95%\n", harness.replications);

    for (label, estimator_name) in [("ols_adjusted", "ols"), ("ipw", "ipw")] {
        println!("── {label} {}", "─".repeat(64 - label.len()));

        for (cell, dgp) in &grid.cells {
            // Instrument-only cells are not meaningful for these two estimators.
            if cell.contains("instrument") {
                continue;
            }

            let report = harness.run(cell, dgp, |t| Some(t.ate), |data| {
                let r = match estimator_name {
                    "ols" => LinearATEEstimator::ols_adjusted(
                        &data.treatment,
                        &data.outcome,
                        &data.covariates,
                    ),
                    _ => PropensityScoreEstimator::ipw(
                        &data.treatment,
                        &data.outcome,
                        &data.covariates,
                    ),
                };
                Some((
                    r.ate,
                    r.ate - 1.96 * r.std_error,
                    r.ate + 1.96 * r.std_error,
                ))
            });

            let flag = if report.coverage.is_nan() {
                " ??"
            } else if report.coverage_ok(0.03) {
                "  ok"
            } else if report.coverage < report.nominal {
                " LOW"
            } else {
                "HIGH"
            };
            println!("{flag}  {}", report.summary());
        }
        println!();
    }

    println!(
        "Note: `LOW` means the intervals do not deliver the confidence they claim.\n\
         That is a correctness defect, not a tuning issue."
    );
}
