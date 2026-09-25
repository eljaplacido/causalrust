//! CI3, CI4 and CI5 — refutation, counterfactuals, and refusing to answer.
//!
//! Three Tier-1 hypotheses about `cynepic-causal`, each run against a fixture
//! with a right answer rather than against another estimator.
//!
//! CI5  On unidentifiable data the crate reports non-identification rather than
//!      a number.
//!      FALSIFIED IF it returns a confident estimate where identification
//!      fails — Critical, by the ledger's own severity rule.
//!
//! CI4  Counterfactual reasoning is validated against known potential outcomes.
//!      FALSIFIED IF it is no better than chance — withdraw the counterfactual
//!      claim from the README.
//!
//! CI3  Refutation tests detect injected confounding rather than passing
//!      everything.
//!      FALSIFIED IF the refuters pass on an estimate built to fail — the
//!      refutation suite is decorative.
//!
//! WHAT CI3 CAN AND CANNOT SHOW, STATED BEFORE THE RUN
//! ===================================================
//!
//! The three refuters probe *stability*, not *bias*. Placebo replaces the
//! treatment with noise and expects the effect to collapse; random-common-cause
//! adds an irrelevant covariate and expects the estimate to hold; data-subset
//! expects it to hold on subsamples. A confounded estimate that is stable —
//! and a confounded estimate usually is, because the confounding is a property
//! of the population rather than of the sample — passes all three.
//!
//! So a null here is not automatically "the suite is broken". It may be the
//! suite doing what refutation does and the hypothesis asking for something
//! refutation was never able to give. Both readings are reported: whether the
//! refuters DISCRIMINATE between a known-biased and a known-good estimate on
//! the same data is the discriminating question, and it is measured directly by
//! running the identical battery against both.
//!
//!     cargo run -p cynepic-causal --example ci345_report -- <c2_corpus.jsonl>

use std::collections::BTreeMap;
use std::env;
use std::fs;

use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::{
    BackdoorCriterion, CausalDag, CounterfactualEngine, CounterfactualQuery, Refuter, Study,
};
use ndarray::{Array1, Array2};
use serde_json::{Value, json};

struct Row {
    covariates: Vec<f64>,
    treated: f64,
    outcome: f64,
    y0: f64,
    y1: f64,
}

fn parse(path: &str) -> Result<Vec<Row>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let obj = v["covariates"].as_object().ok_or("no covariates")?;
        let ordered: BTreeMap<&String, &Value> = obj.iter().collect();
        rows.push(Row {
            covariates: ordered.values().filter_map(|x| x.as_f64()).collect(),
            treated: v["treated"].as_f64().ok_or("no treated")?,
            outcome: v["y_observed"].as_f64().ok_or("no y_observed")?,
            y0: v["y0_value"].as_f64().ok_or("no y0_value")?,
            y1: v["y1_value"].as_f64().ok_or("no y1_value")?,
        });
    }
    Ok(rows)
}

/// CI5 — the crate must refuse where adjustment cannot identify the effect.
///
/// Three structures, each unidentifiable for a different reason. A criterion
/// that answers any of them with an adjustment set is claiming to have blocked
/// a path it cannot see.
fn ci5_refusal() -> Value {
    let mut cases = Vec::new();

    // 1. Classic unmeasured confounder: T <- U -> Y with U latent.
    let mut dag = CausalDag::new();
    for v in ["U", "T", "Y"] {
        dag.add_variable(v);
    }
    dag.add_edges([("U", "T"), ("U", "Y"), ("T", "Y")]).unwrap();
    dag.mark_latent("U").unwrap();
    cases.push(("unmeasured_confounder", dag));

    // 2. The confounder is observed, so this one IS identifiable. Included as
    //    the positive control: a criterion that refuses everything would pass
    //    the two negative cases and be useless, and only this row catches that.
    let mut dag = CausalDag::new();
    for v in ["X", "T", "Y"] {
        dag.add_variable(v);
    }
    dag.add_edges([("X", "T"), ("X", "Y"), ("T", "Y")]).unwrap();
    cases.push(("observed_confounder_identifiable", dag));

    // 3. Two latent confounders on distinct paths — no single observed set
    //    blocks both.
    let mut dag = CausalDag::new();
    for v in ["U1", "U2", "T", "M", "Y"] {
        dag.add_variable(v);
    }
    dag.add_edges([
        ("U1", "T"),
        ("U1", "Y"),
        ("U2", "T"),
        ("U2", "M"),
        ("M", "Y"),
        ("T", "Y"),
    ])
    .unwrap();
    dag.mark_latent("U1").unwrap();
    dag.mark_latent("U2").unwrap();
    cases.push(("two_latent_paths", dag));

    let mut out = Vec::new();
    for (name, dag) in cases {
        let expected_identifiable = name == "observed_confounder_identifiable";
        let result = BackdoorCriterion::find(&dag, "T", "Y");
        let (identified, detail) = match &result {
            Ok(set) => (true, format!("adjustment set {:?}", set.sorted())),
            Err(e) => (false, e.to_string()),
        };
        out.push(json!({
            "case": name,
            "expected_identifiable": expected_identifiable,
            "identified": identified,
            "correct": identified == expected_identifiable,
            "detail": detail,
        }));
    }
    let all_correct = out.iter().all(|c| c["correct"].as_bool().unwrap_or(false));
    json!({
        "hypothesis": "CI5",
        "cases": out,
        "passed": all_correct,
        "falsifier": "a confident adjustment set where identification fails is Critical",
    })
}

/// CI4 — project a counterfactual and score it against the arm that was not run.
///
/// C2 generated both arms, so for every unit the outcome under the other
/// treatment is known exactly. The baseline to beat is the factual outcome
/// itself: predicting "the counterfactual equals what we saw" is the
/// do-nothing answer, and a rung-3 claim has to beat it.
fn ci4_counterfactual(rows: &[Row], ate: &cynepic_causal::ATEResult) -> Value {
    let mut sq_err_model = 0.0;
    let mut sq_err_naive = 0.0;
    let mut n = 0usize;
    let mut covered = 0usize;

    for r in rows {
        // The counterfactual arm: if treated, ask what would have happened
        // untreated, and vice versa.
        let (factual_t, cf_t, truth) = if r.treated > 0.5 {
            (1.0, 0.0, r.y0)
        } else {
            (0.0, 1.0, r.y1)
        };
        let query = CounterfactualQuery {
            treatment: "treated".into(),
            outcome: "y_observed".into(),
            factual_treatment: factual_t,
            counterfactual_treatment: cf_t,
            observed_outcome: r.outcome,
        };
        let res = CounterfactualEngine::query_with_ate(&query, ate);
        sq_err_model += (res.counterfactual_outcome - truth).powi(2);
        // The do-nothing baseline: assume the intervention changes nothing.
        sq_err_naive += (r.outcome - truth).powi(2);
        let (lo, hi) = res.confidence_interval;
        if truth >= lo && truth <= hi {
            covered += 1;
        }
        n += 1;
    }

    let rmse_model = (sq_err_model / n as f64).sqrt();
    let rmse_naive = (sq_err_naive / n as f64).sqrt();
    json!({
        "hypothesis": "CI4",
        "n": n,
        "rmse_counterfactual": rmse_model,
        "rmse_assume_no_change": rmse_naive,
        "improvement_over_naive": rmse_naive - rmse_model,
        "beats_naive": rmse_model < rmse_naive,
        "interval_coverage": covered as f64 / n as f64,
        "falsifier": "no better than chance -> withdraw the counterfactual claim",
        "note": "truth is the arm C2 generated but did not observe; the naive \
                 baseline predicts the counterfactual equals the factual",
    })
}

/// CI3 — do the refuters tell a biased estimate from a good one?
///
/// The battery is run twice on the identical data: once against the naive
/// contrast, which C2's confounding makes wrong by a known amount, and once
/// against the adjusted estimate, which is close to right. If both survive
/// everything, the suite does not discriminate on this axis.
fn ci3_refutation(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    covariates: &Array2<f64>,
    truth: f64,
) -> Value {
    let refuter = Refuter::new(20260925);
    let mut arms = Vec::new();

    for (name, study, estimate) in [
        (
            "naive_unadjusted",
            Study::unadjusted(treatment.clone(), outcome.clone()),
            LinearATEEstimator::difference_in_means(treatment, outcome),
        ),
        (
            "ols_adjusted",
            Study::new(treatment.clone(), outcome.clone(), covariates.clone()),
            LinearATEEstimator::ols_adjusted(treatment, outcome, covariates),
        ),
    ] {
        let Ok(est) = estimate else {
            arms.push(json!({"arm": name, "error": "estimation failed"}));
            continue;
        };
        let bias = est.ate() - truth;
        match refuter.run_all(&study, &est) {
            Ok(results) => {
                let tests: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "test": r.test_name,
                            "passed": r.passed,
                            "discrepancy_se": r.discrepancy_se,
                        })
                    })
                    .collect();
                let n_failed = results.iter().filter(|r| !r.passed).count();
                arms.push(json!({
                    "arm": name,
                    "estimate": est.ate(),
                    "bias_vs_truth": bias,
                    "n_tests": results.len(),
                    "n_failed": n_failed,
                    "all_passed": n_failed == 0,
                    "tests": tests,
                }));
            }
            Err(e) => arms.push(json!({"arm": name, "error": e.to_string()})),
        }
    }

    // The discriminating question: does the battery treat the two differently?
    let naive_failed = arms
        .iter()
        .find(|a| a["arm"] == "naive_unadjusted")
        .and_then(|a| a["n_failed"].as_u64());
    let adjusted_failed = arms
        .iter()
        .find(|a| a["arm"] == "ols_adjusted")
        .and_then(|a| a["n_failed"].as_u64());
    let discriminates = match (naive_failed, adjusted_failed) {
        (Some(n), Some(a)) => n > a,
        _ => false,
    };

    json!({
        "hypothesis": "CI3",
        "arms": arms,
        "discriminates_biased_from_good": discriminates,
        "falsifier": "refuters pass on an estimate built to fail -> the suite is decorative",
        "stated_before_the_run": "the three refuters probe STABILITY, not BIAS. A \
             confounded estimate that is stable passes all three, so a null here may \
             be refutation doing what refutation does rather than a broken suite.",
    })
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: ci345_report <c2_corpus.jsonl>");
        std::process::exit(2);
    });
    let rows = match parse(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let n = rows.len();
    let k = rows[0].covariates.len();
    let treatment = Array1::from(rows.iter().map(|r| r.treated).collect::<Vec<_>>());
    let outcome = Array1::from(rows.iter().map(|r| r.outcome).collect::<Vec<_>>());
    let mut covariates = Array2::<f64>::zeros((n, k));
    for (i, r) in rows.iter().enumerate() {
        for (j, v) in r.covariates.iter().enumerate() {
            covariates[[i, j]] = *v;
        }
    }
    let truth: f64 = rows.iter().map(|r| r.y1 - r.y0).sum::<f64>() / n as f64;

    let ci5 = ci5_refusal();
    let ci3 = ci3_refutation(&treatment, &outcome, &covariates, truth);
    let ate = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates)
        .or_else(|_| LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates));
    let ci4 = match &ate {
        Ok(a) => ci4_counterfactual(&rows, a),
        Err(e) => json!({"hypothesis": "CI4", "error": e.to_string()}),
    };

    let report = json!({
        "benchmark": "ci345_on_c2",
        "corpus": path,
        "n_obs": n,
        "true_ate": truth,
        "CI5": ci5,
        "CI4": ci4,
        "CI3": ci3,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
