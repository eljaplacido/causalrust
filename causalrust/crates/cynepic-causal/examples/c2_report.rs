//! `cynepic-causal` against the C2 decision corpus — the real crate, one corpus.
//!
//! Experiment #8 of the benchmark programme replaces the sklearn stand-ins used
//! to establish SE6/SE7/CI7 with the implementations actually shipped, so that
//! those findings become statements about *this stack* rather than about the
//! logic of a two-tier architecture in general.
//!
//! C2 is generated, so the ATE is known exactly rather than estimated. That
//! makes this a calibration check with a right answer: every estimator here is
//! scored against `mean(ite)`, and the interesting output is not which number is
//! biggest but which estimators cover the truth and which do not.
//!
//! Treatment assignment in C2 is confounded on purpose — `x0` and `x1` drive
//! both the propensity and the outcome, in the same direction — so
//! `difference_in_means` is expected to be visibly wrong. It is included for
//! exactly that reason: an adjustment that cannot beat the naive contrast on a
//! corpus built to punish the naive contrast is not adjusting.
//!
//!     cargo run -p cynepic-causal --example c2_report -- <path to c2_corpus.jsonl>

use std::collections::BTreeMap;
use std::env;
use std::fs;

use cynepic_causal::ATEResult;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};

/// One C2 row, reduced to what an estimator needs.
struct Row {
    covariates: Vec<f64>,
    treated: f64,
    outcome: f64,
    ite: f64,
}

fn parse(path: &str) -> Result<Vec<Row>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        // BTreeMap so the covariate order is x0..x5 on every platform rather
        // than whatever the JSON object happened to iterate as.
        let obj = v["covariates"]
            .as_object()
            .ok_or_else(|| format!("line {}: no covariates", i + 1))?;
        let ordered: BTreeMap<&String, &Value> = obj.iter().collect();
        rows.push(Row {
            covariates: ordered.values().filter_map(|x| x.as_f64()).collect(),
            treated: v["treated"].as_f64().ok_or("no treated")?,
            outcome: v["y_observed"].as_f64().ok_or("no y_observed")?,
            ite: v["ite"].as_f64().ok_or("no ite")?,
        });
    }
    if rows.is_empty() {
        return Err("corpus is empty".into());
    }
    Ok(rows)
}

/// Estimate, interval, and whether it covers the known truth.
fn describe(name: &str, r: &ATEResult, truth: f64) -> Value {
    let ci = r.confidence_interval(0.95);
    let covers = ci.map(|(lo, hi)| truth >= lo && truth <= hi);
    json!({
        "estimator": name,
        "ate": r.ate(),
        "std_error": r.std_error(),
        "n_obs": r.n_obs(),
        "estimand": r.estimand().label(),
        "std_error_kind": format!("{:?}", r.std_error_kind()),
        "ci95_low": ci.map(|(lo, _)| lo),
        "ci95_high": ci.map(|(_, hi)| hi),
        "bias": r.ate() - truth,
        "abs_bias": (r.ate() - truth).abs(),
        "covers_truth": covers,
    })
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: c2_report <path to c2_corpus.jsonl>");
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

    // The truth, known because C2 generated both arms.
    let truth: f64 = rows.iter().map(|r| r.ite).sum::<f64>() / n as f64;
    // ATT is a different estimand and has its own truth: the mean effect among
    // the treated. Scoring ATT against the ATE would manufacture a bias that is
    // a category error rather than a finding.
    let n_treated = rows.iter().filter(|r| r.treated > 0.5).count();
    let truth_att: f64 = if n_treated > 0 {
        rows.iter()
            .filter(|r| r.treated > 0.5)
            .map(|r| r.ite)
            .sum::<f64>()
            / n_treated as f64
    } else {
        f64::NAN
    };

    let mut results = Vec::new();
    let mut failures = Vec::new();

    match LinearATEEstimator::difference_in_means(&treatment, &outcome) {
        Ok(r) => results.push(describe("difference_in_means", &r, truth)),
        Err(e) => {
            failures.push(json!({"estimator": "difference_in_means", "error": e.to_string()}))
        }
    }
    match LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates) {
        Ok(r) => results.push(describe("ols_adjusted", &r, truth)),
        Err(e) => failures.push(json!({"estimator": "ols_adjusted", "error": e.to_string()})),
    }
    match PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates) {
        Ok(r) => results.push(describe("ipw", &r, truth)),
        Err(e) => failures.push(json!({"estimator": "ipw", "error": e.to_string()})),
    }
    // NOT a comparison against `ipw`: `ipw` calls `default_propensity`, which
    // already cross-fits at `CROSS_FIT_FOLDS = 5`, so `ipw_cross_fitted(.., 5)`
    // is the identical computation and returns identical bytes. Recorded here
    // because two rows agreeing to seventeen significant figures reads as a bug
    // to anyone who does not know that, and it is not one.
    match PropensityScoreEstimator::ipw_cross_fitted(&treatment, &outcome, &covariates, 10) {
        Ok(r) => results.push(describe("ipw_cross_fitted_10", &r, truth)),
        Err(e) => {
            failures.push(json!({"estimator": "ipw_cross_fitted_10", "error": e.to_string()}))
        }
    }

    // The contrast that actually shows what cross-fitting buys: one in-sample
    // propensity fit, no folds. C14 closed the ATE interval partly on this
    // change, so on a confounded corpus the two should differ.
    match PropensityScoreEstimator::fit_propensity(&treatment, &covariates)
        .and_then(|m| PropensityScoreEstimator::ipw_with_model(&treatment, &outcome, &m))
    {
        Ok(r) => results.push(describe("ipw_in_sample_propensity", &r, truth)),
        Err(e) => {
            failures.push(json!({"estimator": "ipw_in_sample_propensity", "error": e.to_string()}))
        }
    }
    match PropensityScoreEstimator::att(&treatment, &outcome, &covariates) {
        Ok(r) => results.push(describe("att", &r, truth_att)),
        Err(e) => failures.push(json!({"estimator": "att", "error": e.to_string()})),
    }

    let report = json!({
        "benchmark": "cynepic_causal_on_c2",
        "corpus": "C2",
        "corpus_path": path,
        "n_obs": n,
        "n_covariates": k,
        "n_treated": n_treated,
        "true_ate": truth,
        "true_att": truth_att,
        "estimates": results,
        "failures": failures,
        "note": "C2 is confounded by construction (x0 and x1 drive both propensity \
                 and outcome, same direction), so difference_in_means is expected to \
                 miss. ATT is scored against the mean effect among the treated, not \
                 against the ATE. `ipw` already cross-fits at 5 folds via \n                 `default_propensity`, so it is compared against 10 folds and \n                 against a single in-sample fit rather than against itself.",
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
