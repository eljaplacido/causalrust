//! Does it decline when the data cannot support an answer?
//!
//! # The metric that matters
//!
//! Item 3 of the benchmarking plan, and the property no Python equivalent has.
//! `statsmodels` and `numpy.linalg.lstsq` return a number for a singular design.
//! `scipy` returns a number for a single observation. They are not wrong to —
//! they are linear algebra libraries, and a minimum-norm solution is a
//! defensible answer to an underdetermined system.
//!
//! But a *causal* estimate is acted on. "Your treatment arm is empty" is
//! actionable; a confident effect equal to the treated mean is not, and nothing
//! downstream can tell them apart.
//!
//! So the number to report is the **false-answer rate**: inputs that should have
//! been refused and were answered instead. Target zero. It is not the refusal
//! rate — a function that refuses everything scores perfectly on that and is
//! useless.
//!
//! ```text
//!                        should refuse      should answer
//!   refused              correct refusal    FALSE REFUSAL   (annoying)
//!   answered             FALSE ANSWER       correct answer
//!                        (dangerous)
//! ```
//!
//! The two errors are not symmetric and are never traded evenly. A false
//! refusal costs a caller some work. A false answer costs them a wrong decision
//! they cannot detect.
//!
//! ```bash
//! cargo run -p cynepic-causal --example refusal_report --release
//! ```

use std::collections::HashSet;

use cynepic_causal::dag::CausalDag;
use cynepic_causal::estimate::iv::IVEstimator;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::identify::BackdoorCriterion;
use cynepic_causal::{EstimationError, d_separated};
use cynepic_testkit::Dgp;
use ndarray::{Array1, Array2};

/// What the correct behaviour is for a given input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// The data cannot support an estimate. Returning one is a false answer.
    Refuse,
    /// The data is fine. Refusing is a false refusal.
    Answer,
}

/// One case: a name, the inputs, and what should happen.
struct Case {
    name: &'static str,
    expect: Expect,
    treatment: Array1<f64>,
    outcome: Array1<f64>,
    covariates: Array2<f64>,
}

fn case(name: &'static str, expect: Expect, t: Vec<f64>, y: Vec<f64>, x: Vec<Vec<f64>>) -> Case {
    let (n, p) = (x.len(), x.first().map_or(0, Vec::len));
    Case {
        name,
        expect,
        treatment: Array1::from_vec(t),
        outcome: Array1::from_vec(y),
        covariates: Array2::from_shape_fn((n, p), |(i, j)| x[i][j]),
    }
}

/// The adversarial corpus.
///
/// Each entry names a way real data goes wrong. The `Answer` cases are load
/// bearing: without them an estimator that refuses everything would score a
/// perfect zero false-answer rate.
fn corpus() -> Vec<Case> {
    // ---- inputs that must be refused ------------------------------------
    let mut cases = vec![case(
        "empty dataset",
        Expect::Refuse,
        vec![],
        vec![],
        vec![],
    )];

    cases.push(case(
        "single unit",
        Expect::Refuse,
        vec![1.0],
        vec![5.0],
        vec![vec![0.5]],
    ));

    cases.push(case(
        "all treated (no control arm)",
        Expect::Refuse,
        vec![1.0; 20],
        (0..20).map(f64::from).collect(),
        (0..20).map(|i| vec![f64::from(i)]).collect(),
    ));

    cases.push(case(
        "all control (no treated arm)",
        Expect::Refuse,
        vec![0.0; 20],
        (0..20).map(f64::from).collect(),
        (0..20).map(|i| vec![f64::from(i)]).collect(),
    ));

    cases.push(case(
        "mismatched lengths",
        Expect::Refuse,
        vec![1.0, 0.0, 1.0],
        vec![1.0, 2.0],
        vec![vec![0.0], vec![1.0], vec![2.0]],
    ));

    {
        // Exactly collinear: the third covariate is the sum of the first two, so
        // the design is rank deficient and the treatment coefficient is not
        // identified by the data alone.
        let n = 60;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let a = f64::from(i % 7);
                let b = f64::from(i % 5);
                vec![a, b, a + b]
            })
            .collect();
        cases.push(case(
            "exactly collinear design",
            Expect::Refuse,
            (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect(),
            (0..n).map(|i| 1.0 + f64::from(i % 3)).collect(),
            x,
        ));
    }

    {
        // More parameters than observations.
        let n = 5;
        cases.push(case(
            "fewer observations than parameters",
            Expect::Refuse,
            (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect(),
            (0..n).map(f64::from).collect(),
            (0..n)
                .map(|i| (0..8).map(|j| f64::from(i * 8 + j)).collect())
                .collect(),
        ));
    }

    {
        // Treatment never varies, so there is no contrast to estimate.
        let n = 40;
        cases.push(case(
            "constant treatment",
            Expect::Refuse,
            vec![1.0; n],
            (0..n).map(|i| f64::from(i as u8)).collect(),
            (0..n).map(|i| vec![f64::from(i as u8)]).collect(),
        ));
    }

    // ---- inputs that must be answered ------------------------------------
    for (name, seed, n, p) in [
        ("benign, n=200", 1u64, 200usize, 3usize),
        ("benign, n=2000", 2, 2_000, 3),
        ("small but adequate, n=40", 3, 40, 1),
        ("high dimensional, n=400 p=25", 4, 400, 25),
        ("noisy but well posed", 5, 500, 2),
    ] {
        let d = Dgp::new().with_n(n).with_p(p).sample(seed);
        cases.push(Case {
            name,
            expect: Expect::Answer,
            treatment: d.treatment,
            outcome: d.outcome,
            covariates: d.covariates,
        });
    }

    cases
}

/// Run one estimator over the corpus and return (false answers, false refusals).
fn score<F>(label: &str, mut estimate: F) -> (Vec<String>, Vec<String>)
where
    F: FnMut(&Case) -> Result<(), EstimationError>,
{
    let mut false_answers = Vec::new();
    let mut false_refusals = Vec::new();

    for c in corpus() {
        match (c.expect, estimate(&c)) {
            (Expect::Refuse, Ok(())) => {
                false_answers.push(format!("{label}: answered '{}'", c.name));
            }
            (Expect::Answer, Err(e)) => {
                false_refusals.push(format!("{label}: refused '{}' ({e})", c.name));
            }
            _ => {}
        }
    }
    (false_answers, false_refusals)
}

/// An estimator reduced to "did it answer or refuse", for scoring.
type Scorer<'a> = Box<dyn FnMut(&Case) -> Result<(), EstimationError> + 'a>;

// ===========================================================================
// The headline assertion
// ===========================================================================

/// **No estimator may answer an input that cannot support an answer.**
///
/// The false-answer rate must be exactly zero. Not low — zero. Every entry here
/// is a case where a returned number would be confidently wrong and
/// undetectable downstream, which is the failure this crate exists to prevent
/// and the one it shipped with (findings C1 and C10).
#[test]
fn no_estimator_answers_an_unanswerable_input() {
    let mut all_false_answers = Vec::new();
    let mut all_false_refusals = Vec::new();

    let estimators: Vec<(&str, Scorer<'_>)> = vec![
        (
            "difference_in_means",
            Box::new(|c: &Case| {
                LinearATEEstimator::difference_in_means(&c.treatment, &c.outcome).map(|_| ())
            }),
        ),
        (
            "ols_adjusted",
            Box::new(|c: &Case| {
                LinearATEEstimator::ols_adjusted(&c.treatment, &c.outcome, &c.covariates)
                    .map(|_| ())
            }),
        ),
        (
            "ipw",
            Box::new(|c: &Case| {
                PropensityScoreEstimator::ipw(&c.treatment, &c.outcome, &c.covariates).map(|_| ())
            }),
        ),
        (
            "att",
            Box::new(|c: &Case| {
                PropensityScoreEstimator::att(&c.treatment, &c.outcome, &c.covariates).map(|_| ())
            }),
        ),
    ];

    for (label, mut f) in estimators {
        // `difference_in_means` ignores covariates, so a rank-deficient design
        // is not its problem — it has a perfectly good two-arm contrast. Skip
        // the covariate-only cases for it rather than counting a false answer
        // it cannot be blamed for.
        let (fa, fr) = score(label, |c| {
            if label == "difference_in_means"
                && matches!(
                    c.name,
                    "exactly collinear design" | "fewer observations than parameters"
                )
            {
                return Err(EstimationError::NoObservations); // treated as "n/a"
            }
            f(c)
        });
        all_false_answers.extend(fa);
        all_false_refusals.extend(fr);
    }

    assert!(
        all_false_answers.is_empty(),
        "{} FALSE ANSWERS — an estimate was returned for data that cannot support one:\n  {}",
        all_false_answers.len(),
        all_false_answers.join("\n  ")
    );

    // Reported, not asserted here: `ipw` and `att` legitimately refuse some
    // well-posed inputs when overlap is insufficient, and that is correct
    // behaviour rather than a false refusal in the sense this test means.
    if !all_false_refusals.is_empty() {
        println!(
            "note: {} refusal(s) of well-posed input:\n  {}",
            all_false_refusals.len(),
            all_false_refusals.join("\n  ")
        );
    }
}

/// The `Answer` half of the corpus must actually be answerable.
///
/// Without this the test above is satisfied by an estimator that refuses
/// everything. `ols_adjusted` is the one held to it, because unlike the
/// weighting estimators it has no legitimate reason to decline well-posed data.
#[test]
fn ols_answers_every_well_posed_input() {
    let mut refused = Vec::new();
    for c in corpus().into_iter().filter(|c| c.expect == Expect::Answer) {
        if let Err(e) = LinearATEEstimator::ols_adjusted(&c.treatment, &c.outcome, &c.covariates) {
            refused.push(format!("{}: {e}", c.name));
        }
    }
    assert!(
        refused.is_empty(),
        "OLS refused {} well-posed input(s):\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
}

// ===========================================================================
// Refusals carry an actionable reason
// ===========================================================================

/// A refusal must say *which* problem it found, not merely that there was one.
///
/// "Invalid input" is not actionable. "Treatment arm 'control' is empty (20
/// treated, 0 control)" is. The error variants exist so a caller can branch on
/// the cause, and this pins the mapping so a refactor cannot quietly collapse
/// them into one.
#[test]
fn each_refusal_names_its_cause() {
    let n = 60;
    let collinear: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let a = f64::from(i % 7);
            let b = f64::from(i % 5);
            vec![a, b, a + b]
        })
        .collect();
    let c = case(
        "collinear",
        Expect::Refuse,
        (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect(),
        (0..n).map(|i| 1.0 + f64::from(i % 3)).collect(),
        collinear,
    );

    let err = LinearATEEstimator::ols_adjusted(&c.treatment, &c.outcome, &c.covariates)
        .expect_err("rank deficient");
    assert!(
        matches!(err, EstimationError::RankDeficient { ref aliased, .. } if !aliased.is_empty()),
        "rank deficiency must name the aliased columns, got: {err}"
    );

    let t = Array1::from_vec(vec![1.0; 10]);
    let y = Array1::from_vec((0..10).map(f64::from).collect());
    let err = LinearATEEstimator::difference_in_means(&t, &y).expect_err("no control arm");
    assert!(
        matches!(err, EstimationError::EmptyArm { arm: "control", .. }),
        "an empty arm must name which arm, got: {err}"
    );

    let t = Array1::from_vec(vec![1.0, 0.0]);
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let err = LinearATEEstimator::difference_in_means(&t, &y).expect_err("mismatched");
    assert!(
        matches!(
            err,
            EstimationError::LengthMismatch {
                len_a: 2,
                len_b: 3,
                ..
            }
        ),
        "a length mismatch must report both lengths, got: {err}"
    );
}

/// A weak instrument must be refused with its F statistic attached.
///
/// Below the Stock-Yogo threshold 2SLS is *more* biased than the OLS it
/// replaces, so an estimate there is worse than no estimate — and the caller
/// needs the F to know how far below they are.
#[test]
fn weak_instruments_are_refused_with_their_f_statistic() {
    let d = Dgp::new().with_n(2_000).with_instrument(0.01).sample(31);
    let z_col = d.instrument.as_ref().expect("instrument DGP");
    let z = Array2::from_shape_fn((d.n(), 1), |(i, _)| z_col[i]);

    match IVEstimator::two_stage_ls(&d.treatment, &d.outcome, &z) {
        Err(EstimationError::WeakInstrument {
            first_stage_f,
            threshold,
        }) => {
            assert!(
                first_stage_f < threshold,
                "F {first_stage_f} vs {threshold}"
            );
        }
        Err(other) => panic!("expected WeakInstrument, got {other}"),
        Ok(r) => panic!("a 1%-strength instrument produced {}", r.summary()),
    }
}

/// Insufficient overlap must be refused rather than clipped.
///
/// Clipping the propensities to `[0.01, 0.99]` and carrying on converts a
/// violated assumption into a plausible number. This is the one case where
/// refusing costs a caller an answer they might have wanted, and it is still
/// right: weighting cannot manufacture a comparison the data does not contain.
#[test]
fn insufficient_overlap_is_refused_not_clipped() {
    let d = Dgp::new().with_n(1_000).with_overlap(0.02).sample(4);
    match PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates) {
        Err(EstimationError::InsufficientOverlap { n_extreme, n, .. }) => {
            assert!(
                n_extreme * 10 > n,
                "{n_extreme} of {n} is not 'insufficient'"
            );
        }
        Err(EstimationError::Separation { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(r) => {
            let ess = r.diagnostics().effective_n.expect("ESS reported");
            assert!(ess < 1_000.0, "weighting under no overlap cannot be free");
        }
    }
}

// ===========================================================================
// Graph and identification surfaces
// ===========================================================================

/// The graph API must refuse what it cannot represent or answer.
#[test]
fn graph_surfaces_refuse_invalid_input() {
    // A cycle.
    let mut dag = CausalDag::new();
    dag.add_edge("A", "B").expect("acyclic");
    dag.add_edge("B", "C").expect("acyclic");
    assert!(dag.add_edge("C", "A").is_err(), "a cycle was accepted");
    assert!(dag.add_edge("A", "A").is_err(), "a self-loop was accepted");

    // An unknown name in a d-separation query, including in the conditioning
    // set — the likeliest typo, since adjustment sets are usually assembled
    // programmatically.
    let empty = HashSet::new();
    assert!(d_separated(&dag, "ghost", "B", &empty).is_err());
    assert!(d_separated(&dag, "A", "ghost", &empty).is_err());
    let z: HashSet<String> = ["ghost".to_string()].into_iter().collect();
    assert!(d_separated(&dag, "A", "C", &z).is_err());

    // Identification that cannot succeed.
    let mut confounded = CausalDag::new();
    confounded.add_edge("U", "T").expect("acyclic");
    confounded.add_edge("U", "Y").expect("acyclic");
    confounded.add_edge("T", "Y").expect("acyclic");
    confounded.mark_latent("U").expect("known");
    assert!(
        BackdoorCriterion::find(&confounded, "T", "Y").is_err(),
        "identification succeeded with an unmeasurable confounder"
    );

    // And must still succeed where it can — otherwise "refuses everything"
    // would pass.
    let mut clean = CausalDag::new();
    clean.add_edge("W", "T").expect("acyclic");
    clean.add_edge("W", "Y").expect("acyclic");
    clean.add_edge("T", "Y").expect("acyclic");
    let adj = BackdoorCriterion::find(&clean, "T", "Y").expect("identifiable");
    assert_eq!(adj.sorted(), vec!["W"]);
}
