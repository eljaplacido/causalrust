//! Print the estimator coverage table across the standard DGP grid.
//!
//! This is the credibility artifact: it measures something that can come back
//! bad. Speed benchmarks only ever flatter — nobody publishes one that made
//! them look slow. Coverage can, and on this crate's first run it did, exposing
//! an IPW implementation with 0.0% coverage that thirty passing unit tests had
//! not noticed.
//!
//! ```bash
//! cargo run -p cynepic-causal --example coverage_report --release
//! ```
//!
//! # Reading the output
//!
//! `coverage` is the fraction of nominal 95% intervals that contained the
//! truth. It should be near 95%. Read it with `bias`:
//!
//! | bias | coverage | diagnosis |
//! |---|---|---|
//! | ~0 | ~95% | sound on this DGP |
//! | ~0 | low | point estimate fine, **standard error understated** |
//! | ~0 | high | standard error overstated; intervals wastefully wide |
//! | large | low | estimator is biased |
//! | large | ~95% | intervals so wide they hide the bias |
//!
//! # Refusals are a result, not a gap
//!
//! `refused` counts replications where the estimator returned an error instead
//! of a number — insufficient overlap, a rank-deficient design, a weak
//! instrument. A high refusal rate on a hostile cell is the estimator working:
//! the alternative is a confident number computed from four units carrying
//! enormous weight. A cell that refuses everything is reported as such rather
//! than left blank, because "cannot be estimated here" is the useful finding.

use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_testkit::{DgpGrid, ValidationHarness};

fn main() {
    let grid = DgpGrid::standard();
    let harness = ValidationHarness::default().with_replications(300);

    println!("cynepic-causal — confidence-interval coverage");
    println!(
        "{} replications per cell, nominal 95%\n",
        harness.replications
    );

    for (label, which) in [
        ("ols_adjusted", "ols"),
        ("ipw", "ipw"),
        ("att (IPW)", "att"),
    ] {
        println!(
            "── {label} {}",
            "─".repeat(60_usize.saturating_sub(label.len()))
        );

        for (cell, dgp) in &grid.cells {
            // Instrument cells exist for 2SLS; these estimators do not use them.
            if cell.contains("instrument") {
                continue;
            }

            let report = harness.run(
                cell,
                dgp,
                // ATT is a different estimand and must be scored against the
                // matching truth, or a correct estimator looks biased.
                |t| {
                    if which == "att" {
                        Some(t.att)
                    } else {
                        Some(t.ate)
                    }
                },
                |data| {
                    let r = match which {
                        "ols" => LinearATEEstimator::ols_adjusted(
                            &data.treatment,
                            &data.outcome,
                            &data.covariates,
                        ),
                        "ipw" => PropensityScoreEstimator::ipw(
                            &data.treatment,
                            &data.outcome,
                            &data.covariates,
                        ),
                        _ => PropensityScoreEstimator::att(
                            &data.treatment,
                            &data.outcome,
                            &data.covariates,
                        ),
                    };
                    // `None` means the estimator declined. The harness counts
                    // it as a refusal rather than scoring it as a bad estimate,
                    // which keeps "refused to guess" separate from "guessed
                    // wrong".
                    let r = r.ok()?;
                    let (lo, hi) = r.confidence_interval(0.95)?;
                    Some((r.ate(), lo, hi))
                },
            );

            let flag = if report.coverage.is_nan() {
                " ?? "
            } else if report.coverage_ok(0.05) {
                " ok "
            } else if report.coverage < 0.90 {
                "LOW "
            } else {
                "high"
            };

            println!("  {flag} {}", report.summary());
        }
        println!();
    }

    println!("`LOW` means the intervals do not deliver the confidence they claim.");
    println!("That is a correctness defect, not a tuning issue.");
    println!();
    println!("`refused` counts replications the estimator declined rather than");
    println!("guessing. On a hostile cell that is the estimator working correctly.");
}
