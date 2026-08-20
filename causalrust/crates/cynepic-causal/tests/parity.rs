//! Cross-implementation parity against numpy, scipy and networkx.
//!
//! # Why this suite exists
//!
//! The README carries assumed speedups over NetworkX, PyMC and OPA. A speedup
//! claim is only worth anything if both sides compute the **same thing**, and
//! nothing checked that. "1000x faster than NetworkX" is not a claim until
//! "identical to NetworkX" is true.
//!
//! So this is the first item in the benchmarking plan and the throughput
//! comparison is the last. Agreement is the load-bearing property; speed is a
//! detail about how fast the agreement arrives.
//!
//! # The references are independent, not restated
//!
//! Each pair uses genuinely different algorithms, which is what makes agreement
//! evidence rather than tautology:
//!
//! | Quantity | Here | Reference |
//! |---|---|---|
//! | Least squares | pivoted Householder QR | numpy SVD (`lstsq`) |
//! | Welch standard error | closed form | recovered from `scipy.stats.ttest_ind` |
//! | Beta / Gamma quantiles | Lentz continued fraction + bisection | scipy (Boost) |
//! | Student-t quantiles | incomplete beta + bisection | scipy (Boost) |
//! | D-separation | Bayes-Ball | `networkx.is_d_separator` |
//!
//! # No Python at test time
//!
//! `scripts/generate_parity_fixtures.py` computes the references once and
//! writes `tests/fixtures/parity.json`, which is committed. CI runs this suite
//! with no Python installed.
//!
//! **If a regenerated fixture differs from the committed one, that is a finding
//! to investigate — not a fixture to overwrite.**

use std::collections::HashSet;

use cynepic_causal::dag::CausalDag;
use cynepic_causal::estimand::StdErrorKind;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::{EstimationError, d_separated};
use cynepic_core::special;
use ndarray::{Array1, Array2};
use serde_json::Value;

/// Load the committed fixtures.
fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/parity.json");
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("cannot read {path}: {e}. Run scripts/generate_parity_fixtures.py")
    });
    serde_json::from_str(&raw).expect("fixtures are valid JSON")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Relative difference, falling back to absolute near zero.
///
/// A pure relative test is meaningless when the reference is ~0; a pure
/// absolute one is meaningless when it is large. This is the standard blend.
fn rel_err(actual: f64, expected: f64) -> f64 {
    let denom = expected.abs().max(1.0);
    (actual - expected).abs() / denom
}

// ===========================================================================
// Least squares
// ===========================================================================

/// The treatment coefficient must match `numpy.linalg.lstsq`.
///
/// numpy solves by SVD; this crate uses pivoted Householder QR. Agreement to
/// 1e-9 says the two arrive at the same answer by different routes.
#[test]
fn ols_ate_matches_numpy_lstsq() {
    let f = fixtures();
    let cases = f["ols"].as_array().expect("ols cases");
    assert!(!cases.is_empty(), "fixtures contain no OLS cases");

    for case in cases {
        let label = case["label"].as_str().expect("label");
        let treatment = Array1::from_vec(floats(&case["treatment"]));
        let outcome = Array1::from_vec(floats(&case["outcome"]));

        let rows: Vec<Vec<f64>> = case["covariates"]
            .as_array()
            .expect("covariates")
            .iter()
            .map(floats)
            .collect();
        let (n, p) = (rows.len(), rows[0].len());
        let covariates = Array2::from_shape_fn((n, p), |(i, j)| rows[i][j]);

        let expected = case["expected_ate"].as_f64().expect("expected_ate");

        match LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates) {
            Ok(got) => assert!(
                rel_err(got.ate(), expected) < 1e-9,
                "{label}: ATE {} vs numpy {expected} (rel err {:.3e})",
                got.ate(),
                rel_err(got.ate(), expected)
            ),
            // A refusal is a legitimate disagreement only when the design is
            // genuinely degenerate. numpy's `lstsq` answers regardless — it
            // returns a minimum-norm solution on a singular system — so this is
            // a case where declining is the better behaviour, not a parity
            // failure. It must still be the *nearly* collinear case and not a
            // well-posed one.
            Err(EstimationError::RankDeficient { .. }) => assert!(
                label.contains("collinear"),
                "{label}: refused a well-posed design as rank deficient"
            ),
            Err(e) => panic!("{label}: unexpected error {e}"),
        }
    }
}

/// The classical standard error must match the one computed from
/// `sigma^2 (X'X)^-1`.
#[test]
fn ols_classical_se_matches_numpy() {
    let f = fixtures();
    for case in f["ols"].as_array().expect("ols cases") {
        let label = case["label"].as_str().expect("label");
        if label.contains("collinear") {
            continue; // refused above, by design
        }

        let treatment = Array1::from_vec(floats(&case["treatment"]));
        let outcome = Array1::from_vec(floats(&case["outcome"]));
        let rows: Vec<Vec<f64>> = case["covariates"]
            .as_array()
            .expect("covariates")
            .iter()
            .map(floats)
            .collect();
        let (n, p) = (rows.len(), rows[0].len());
        let covariates = Array2::from_shape_fn((n, p), |(i, j)| rows[i][j]);

        let expected = case["expected_se_classical"]
            .as_f64()
            .expect("expected_se_classical");

        let got = LinearATEEstimator::ols_adjusted_with(
            &treatment,
            &outcome,
            &covariates,
            StdErrorKind::Classical,
        )
        .expect("well-posed design");

        assert!(
            rel_err(got.std_error(), expected) < 1e-8,
            "{label}: classical SE {} vs numpy {expected}",
            got.std_error()
        );
    }
}

/// Difference in means and its Welch standard error must match scipy.
///
/// The reference SE is recovered from `scipy.stats.ttest_ind`'s statistic
/// rather than recomputed, so this checks against an independent derivation
/// instead of the same formula written twice.
#[test]
fn difference_in_means_matches_scipy_welch() {
    let f = fixtures();
    let cases = f["difference_in_means"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let label = case["label"].as_str().expect("label");
        let treatment = Array1::from_vec(floats(&case["treatment"]));
        let outcome = Array1::from_vec(floats(&case["outcome"]));

        let got = LinearATEEstimator::difference_in_means(&treatment, &outcome)
            .expect("both arms populated");

        let expected_ate = case["expected_ate"].as_f64().expect("ate");
        let expected_se = case["expected_se_welch"].as_f64().expect("se");

        assert!(
            rel_err(got.ate(), expected_ate) < 1e-12,
            "{label}: ATE {} vs scipy {expected_ate}",
            got.ate()
        );
        assert!(
            rel_err(got.std_error(), expected_se) < 1e-9,
            "{label}: Welch SE {} vs scipy {expected_se}",
            got.std_error()
        );
    }
}

// ===========================================================================
// Quantiles
// ===========================================================================

/// Beta, Gamma and Student-t quantiles must match scipy.
///
/// `cynepic-core::special` uses a Lentz continued fraction with bisection;
/// scipy wraps Boost.
///
/// The bar is 1e-9. It was set at 1e-7 defensively, then tightened once the
/// measurement came in: the worst disagreement across all 116 cases is
/// **5.5e-12**, at `t(1000)`. A tolerance two orders of magnitude looser than
/// the observed worst case is not a test, it is a formality.
#[test]
fn quantiles_match_scipy() {
    let f = fixtures();
    let cases = f["quantiles"].as_array().expect("quantile cases");
    assert!(
        cases.len() > 50,
        "expected a broad sweep, got {}",
        cases.len()
    );

    let mut worst = (0.0_f64, String::new());

    for case in cases {
        let p = case["p"].as_f64().expect("p");
        let expected = case["expected"].as_f64().expect("expected");

        let (got, what) = match case["kind"].as_str().expect("kind") {
            "beta" => {
                let a = case["a"].as_f64().expect("a");
                let b = case["b"].as_f64().expect("b");
                (
                    special::beta_quantile(p, a, b),
                    format!("Beta({a},{b}) p={p}"),
                )
            }
            "gamma" => {
                let shape = case["shape"].as_f64().expect("shape");
                let rate = case["rate"].as_f64().expect("rate");
                (
                    special::gamma_quantile(p, shape, rate),
                    format!("Gamma({shape},{rate}) p={p}"),
                )
            }
            "t" => {
                let dof = case["dof"].as_f64().expect("dof");
                (special::t_quantile(p, dof), format!("t({dof}) p={p}"))
            }
            other => panic!("unknown quantile kind {other}"),
        };

        let err = rel_err(got, expected);
        if err > worst.0 {
            worst = (err, format!("{what}: {got} vs scipy {expected}"));
        }
        assert!(
            err < 1e-9,
            "{what}: {got} vs scipy {expected} (rel err {err:.3e})"
        );
    }

    // Printed so a tightening tolerance has a number to be judged against.
    println!(
        "worst quantile disagreement: {:.3e}  ({})",
        worst.0, worst.1
    );
}

// ===========================================================================
// D-separation
// ===========================================================================

/// Every d-separation verdict must match `networkx.is_d_separator`.
///
/// Bayes-Ball here, a different formulation there. The collider cases carry the
/// weight: conditioning on a collider *opens* a path, which is the direction an
/// implementation is most likely to get backwards and the one a unit test
/// written by the same author is least likely to catch.
#[test]
fn d_separation_matches_networkx() {
    let f = fixtures();
    let cases = f["d_separation"].as_array().expect("dsep cases");
    assert!(
        cases.len() > 50,
        "expected a broad sweep, got {}",
        cases.len()
    );

    let mut disagreements = Vec::new();

    for case in cases {
        let graph_name = case["graph"].as_str().expect("graph");
        let mut dag = CausalDag::new();
        for edge in case["edges"].as_array().expect("edges") {
            let pair = edge.as_array().expect("edge pair");
            let from = pair[0].as_str().expect("from");
            let to = pair[1].as_str().expect("to");
            dag.add_edge(from, to).expect("fixture graphs are acyclic");
        }

        let x = case["x"].as_str().expect("x");
        let y = case["y"].as_str().expect("y");
        let z: HashSet<String> = case["z"]
            .as_array()
            .expect("z")
            .iter()
            .map(|v| v.as_str().expect("z name").to_string())
            .collect();
        let expected = case["expected_d_separated"].as_bool().expect("verdict");

        let got = d_separated(&dag, x, y, &z).expect("all names are in the fixture graph");

        if got != expected {
            disagreements.push(format!(
                "{graph_name}: d_sep({x}, {y} | {:?}) = {got}, networkx says {expected}",
                {
                    let mut v: Vec<&String> = z.iter().collect();
                    v.sort();
                    v
                }
            ));
        }
    }

    assert!(
        disagreements.is_empty(),
        "{} of {} d-separation verdicts disagree with networkx:\n  {}",
        disagreements.len(),
        cases.len(),
        disagreements.join("\n  ")
    );
}

/// The fixtures must record what produced them.
///
/// A parity suite whose references cannot be attributed to a version is not
/// reproducible, and a future disagreement could not be pinned on either side.
#[test]
fn fixtures_record_their_provenance() {
    let f = fixtures();
    let meta = &f["_generated_with"];
    for key in ["python", "numpy", "scipy", "networkx", "seed"] {
        assert!(
            !meta[key].is_null(),
            "fixtures do not record '{key}'; regenerate with scripts/generate_parity_fixtures.py"
        );
    }
}
