//! Regression suite for the C-series correctness findings.
//!
//! # What this file is
//!
//! Most tests here were once failing specifications for confirmed defects,
//! marked `#[ignore]` and counted by `scripts/findings-ratchet.sh`. They now
//! assert the corrected behaviour and **run in the default suite**.
//!
//! ```text
//! open findings specs:  18   ->   20   ->   13   ->   2
//!                     initial  measured  re-baselined  after Tier 1
//! ```
//!
//! Eleven of the thirteen `#[ignore]`s are gone because the defects are gone.
//! The remaining two are C14, which the Tier 1 fix *created*: correcting IPW's
//! variance for the propensity being estimated removed a 3x over-coverage and
//! left a ~9% under-coverage where the weights are heavy. It is filed rather
//! than tuned away because under-coverage is the dangerous direction, and the
//! ratchet keeps it visible.
//!
//! Fixing things reveals things. A ledger that only ever shrinks is a ledger
//! that has stopped measuring.
//!
//! # Why keep them separate from the unit tests
//!
//! A unit test says what a function does. These say what it must never do
//! again, and each carries the specific way it went wrong the first time. That
//! history is why the tolerance in `c13_ipw_removes_most_of_the_confounding` is
//! tight rather than the 40% one that let a broken estimator ship.
//!
//! ```bash
//! cargo test -p cynepic-causal --test findings
//! ./scripts/findings-ratchet.sh
//! ```

use std::collections::HashSet;

use cynepic_causal::d_separated;
use cynepic_causal::dag::CausalDag;
use cynepic_causal::error::{DagError, DsepError, EstimationError, IdentificationError};
use cynepic_causal::estimand::{Estimand, StdErrorKind};
use cynepic_causal::estimate::iv::IVEstimator;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::identify::BackdoorCriterion;
use cynepic_causal::refute::{Refuter, Study};
use cynepic_testkit::{Dgp, ValidationHarness};
use ndarray::{Array1, Array2};

// ===========================================================================
// C1 — a singular design was reported with zero uncertainty
// ===========================================================================

/// A rank-deficient design must produce a named error, never a number.
///
/// Previously `solve_normal_equation` returned a zero vector on a small pivot
/// and `invert_matrix` returned a zero matrix, so the caller received a
/// plausible ATE with `std_error` of exactly 0.0 — infinite confidence, which
/// passes every downstream significance test.
#[test]
fn c1_rank_deficient_design_is_an_error_naming_the_aliased_columns() {
    let data = Dgp::new().with_p(3).collinear().with_n(500).sample(1);

    let err = LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates)
        .expect_err("an exactly collinear column must be rejected");

    match err {
        EstimationError::RankDeficient {
            rank,
            expected,
            aliased,
        } => {
            assert!(rank < expected, "rank {rank} should be below {expected}");
            assert!(!aliased.is_empty(), "the offending columns must be named");
        }
        other => panic!("expected RankDeficient, got {other}"),
    }
}

/// A standard error of exactly zero is never a legitimate output.
///
/// The more dangerous half of C1: a wrong point estimate might be noticed, but
/// zero uncertainty actively invites downstream code to trust it.
#[test]
fn c1_standard_error_is_never_exactly_zero() {
    let ok = Dgp::new().with_p(2).with_n(300).sample(2);
    let r = LinearATEEstimator::ols_adjusted(&ok.treatment, &ok.outcome, &ok.covariates)
        .expect("well-posed design");
    assert!(
        r.std_error() > 0.0 && r.std_error().is_finite(),
        "se was {}",
        r.std_error()
    );

    let bad = Dgp::new().with_p(2).collinear().with_n(300).sample(2);
    assert!(
        LinearATEEstimator::ols_adjusted(&bad.treatment, &bad.outcome, &bad.covariates).is_err(),
        "a rank-deficient design must not yield an estimate"
    );
}

// ===========================================================================
// C2 — identification returned adjustment sets you could not measure
// ===========================================================================

/// Backdoor identification must never hand back an unobservable variable.
///
/// The DAG below is the crate's own front-door fixture, where `U` is documented
/// as unobserved. Adjusting for `U` is impossible by construction.
#[test]
fn c2_adjustment_set_never_contains_a_latent_variable() {
    let mut dag = CausalDag::new();
    dag.add_edge("U", "Smoking").expect("acyclic");
    dag.add_edge("U", "Cancer").expect("acyclic");
    dag.add_edge("Smoking", "Tar").expect("acyclic");
    dag.add_edge("Tar", "Cancer").expect("acyclic");
    dag.mark_latent("U").expect("U is in the graph");

    match BackdoorCriterion::find(&dag, "Smoking", "Cancer") {
        Ok(set) => assert!(
            !set.contains("U"),
            "returned the unobservable confounder U: {:?}",
            set.sorted()
        ),
        Err(IdentificationError::RequiresLatent { latent, .. }) => {
            assert!(latent.contains(&"U".to_string()));
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}

/// Identification must be capable of failing.
///
/// A criterion that can never return "no" is not a criterion. `None` previously
/// meant "no adjustment needed" — a *successful* identification — so it could
/// not also mean "not identifiable".
#[test]
fn c2_identification_can_fail() {
    let mut dag = CausalDag::new();
    dag.add_edge("U1", "T").expect("acyclic");
    dag.add_edge("U1", "Y").expect("acyclic");
    dag.add_edge("U2", "T").expect("acyclic");
    dag.add_edge("U2", "Y").expect("acyclic");
    dag.mark_latent("U1").expect("known");
    dag.mark_latent("U2").expect("known");

    assert!(
        BackdoorCriterion::find(&dag, "T", "Y").is_err(),
        "unblockable latent confounding must not yield an adjustment set"
    );
}

/// The empty set is a success, and must stay distinguishable from failure.
#[test]
fn c2_no_adjustment_needed_is_distinct_from_not_identifiable() {
    let mut clean = CausalDag::new();
    clean.add_edge("T", "Y").expect("acyclic");
    let set = BackdoorCriterion::find(&clean, "T", "Y").expect("identifiable");
    assert!(set.is_empty(), "no confounding means no adjustment");
}

// ===========================================================================
// C5 / C13 — IPW returned confident, badly biased numbers
// ===========================================================================

/// IPW intervals must achieve nominal coverage.
///
/// Coverage is the test that catches an inconsistent variance formula; nothing
/// else does, because every individual run looks plausible. This cell measured
/// **0.0%** before the fix.
#[test]
fn c5_ipw_intervals_achieve_nominal_coverage() {
    let dgp = Dgp::new().with_n(1_000).with_confounding(1.0);

    let report = ValidationHarness::default().with_replications(200).run(
        "ipw-benign",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r = PropensityScoreEstimator::ipw(&data.treatment, &data.outcome, &data.covariates)
                .ok()?;
            let (lo, hi) = r.confidence_interval(0.95)?;
            Some((r.ate(), lo, hi))
        },
    );

    assert!(
        report.coverage_ok(0.06),
        "IPW coverage is not nominal — {}",
        report.summary()
    );
}

/// OLS on a benign, correctly specified DGP. The easy case, kept as a guard.
#[test]
fn c5_ols_intervals_achieve_nominal_coverage() {
    let dgp = Dgp::new().with_n(800);

    let report = ValidationHarness::default().with_replications(200).run(
        "ols-benign",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r =
                LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates)
                    .ok()?;
            let (lo, hi) = r.confidence_interval(0.95)?;
            Some((r.ate(), lo, hi))
        },
    );

    assert!(
        report.coverage_ok(0.05),
        "OLS coverage is not nominal on a benign DGP — {}",
        report.summary()
    );
}

/// Coverage must survive heteroskedasticity, which is what HC1 exists for.
#[test]
fn c5_coverage_survives_heteroskedasticity() {
    let dgp = Dgp::new().with_n(800).heteroskedastic();

    let report = ValidationHarness::default().with_replications(200).run(
        "ols-heteroskedastic",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r =
                LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates)
                    .ok()?;
            let (lo, hi) = r.confidence_interval(0.95)?;
            Some((r.ate(), lo, hi))
        },
    );

    assert!(
        report.coverage_ok(0.05),
        "coverage collapsed under heteroskedasticity — {}",
        report.summary()
    );
}

/// The variance must be the one that belongs to the point estimate.
#[test]
fn c5_ipw_reports_an_influence_function_variance() {
    let data = Dgp::new().with_n(1_000).sample(5);
    let r = PropensityScoreEstimator::ipw(&data.treatment, &data.outcome, &data.covariates)
        .expect("benign DGP has overlap");
    assert_eq!(r.std_error_kind(), StdErrorKind::InfluenceFunction);
}

/// IPW must remove the confounding it exists to remove.
///
/// Before the fix, the propensity model ran a fixed 100 iterations of gradient
/// descent with no convergence check and removed only 47% of the bias, while
/// reporting an interval 0.4 wide.
#[test]
fn c13_ipw_removes_most_of_the_confounding() {
    let dgp = Dgp::new().with_n(2_000);

    let mut ipw_bias = 0.0;
    let mut naive_bias = 0.0;
    let reps: u64 = 20;
    let mut counted = 0u64;

    for seed in 0..reps {
        let d = dgp.sample(seed);
        let Ok(ipw) = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates) else {
            continue;
        };
        let naive = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome)
            .expect("both arms are populated");
        ipw_bias += ipw.ate() - d.truth.ate;
        naive_bias += naive.ate() - d.truth.ate;
        counted += 1;
    }

    assert!(counted >= reps / 2, "too many refusals to judge: {counted}");
    #[allow(clippy::cast_precision_loss)]
    let n = counted as f64;
    ipw_bias /= n;
    naive_bias /= n;

    let removed = 1.0 - (ipw_bias.abs() / naive_bias.abs());
    assert!(
        removed > 0.90,
        "IPW removed only {:.0}% of the confounding bias (IPW {ipw_bias:+.3} vs naive \
         {naive_bias:+.3})",
        removed * 100.0
    );
}

/// The propensity fit must converge, and must say how.
#[test]
fn c13_propensity_fit_converges_and_reports_it() {
    let data = Dgp::new().with_n(1_000).sample(3);
    let model = PropensityScoreEstimator::fit_propensity(&data.treatment, &data.covariates)
        .expect("benign DGP is well posed");

    assert!(model.convergence.converged);
    assert!(
        model.convergence.iterations <= 15,
        "IRLS took {} iterations; gradient descent needed tens of thousands",
        model.convergence.iterations
    );
}

/// Weak overlap must be refused rather than clipped into a plausible number.
#[test]
fn c13_weak_overlap_is_refused_or_diagnosed() {
    let dgp = Dgp::new().with_n(1_000).with_overlap(0.02);
    let data = dgp.sample(4);

    match PropensityScoreEstimator::ipw(&data.treatment, &data.outcome, &data.covariates) {
        Err(EstimationError::InsufficientOverlap { .. } | EstimationError::Separation { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(r) => {
            // If it does return, the diagnostics must expose how thin the
            // effective sample is. A silent number here is the failure.
            let ess = r.diagnostics().effective_n.expect("ESS must be reported");
            assert!(
                ess < 1_000.0,
                "weighting under weak overlap cannot cost nothing (ess {ess})"
            );
        }
    }
}

// ===========================================================================
// C7 / C8 — refutation verdicts were magic constants over an unseedable LCG
// ===========================================================================

/// A refutation verdict must depend on the estimate's own uncertainty.
///
/// The old rule was `relative_change < 0.15`, which never consulted the
/// standard error: a 14% shift on a tight estimate passed, a 16% shift on one
/// spanning zero failed, and both verdicts were noise.
#[test]
fn c7_refutation_verdict_depends_on_uncertainty() {
    let data = Dgp::new().with_n(2_000).sample(11);
    let study = Study::new(
        data.treatment.clone(),
        data.outcome.clone(),
        data.covariates.clone(),
    );
    let original = study.estimate().expect("well-posed");

    let r = Refuter::new(7)
        .placebo_treatment(&study, &original)
        .expect("valid");

    assert!(
        r.discrepancy_se.is_finite(),
        "the verdict must be on the standard-error scale, got {}",
        r.discrepancy_se
    );
    assert!(r.passed, "{}", r.interpretation);
    assert!(
        r.interpretation.contains("SE"),
        "the interpretation must state the scale: {}",
        r.interpretation
    );
}

/// A placebo must re-run the estimator it was given, adjustment set included.
///
/// `placebo_treatment` used to take only `(outcome, ate, tolerance)` and always
/// re-estimate with `difference_in_means`, so it produced a verdict about an
/// estimator the caller had never run.
#[test]
fn c7_placebo_uses_the_studys_own_estimator() {
    let data = Dgp::new().with_n(1_500).with_confounding(3.0).sample(13);

    let adjusted = Study::new(
        data.treatment.clone(),
        data.outcome.clone(),
        data.covariates.clone(),
    );
    let unadjusted = Study::unadjusted(data.treatment.clone(), data.outcome.clone());

    let a = adjusted.estimate().expect("well-posed");
    let u = unadjusted.estimate().expect("well-posed");

    // Under strong confounding the adjusted and unadjusted analyses differ
    // materially. A refuter that cannot tell them apart is not refuting the
    // analysis it was handed.
    assert!(
        (a.ate() - u.ate()).abs() > 0.1,
        "fixture is not confounded enough to be a test: {} vs {}",
        a.ate(),
        u.ate()
    );

    let refuter = Refuter::new(17);
    let ra = refuter.placebo_treatment(&adjusted, &a).expect("valid");
    let ru = refuter.placebo_treatment(&unadjusted, &u).expect("valid");
    assert!(
        (ra.discrepancy_se - ru.discrepancy_se).abs() > f64::EPSILON,
        "the two studies produced identical verdicts, so the adjustment set was ignored"
    );
}

/// Refutation must be reproducible from a seed the caller controls.
#[test]
fn c8_refutation_is_seeded_and_reproducible() {
    let data = Dgp::new().with_n(800).sample(19);
    let study = Study::new(data.treatment, data.outcome, data.covariates);
    let original = study.estimate().expect("well-posed");

    let a = Refuter::new(42)
        .placebo_treatment(&study, &original)
        .expect("valid");
    let b = Refuter::new(42)
        .placebo_treatment(&study, &original)
        .expect("valid");
    assert!((a.refuted_effect - b.refuted_effect).abs() < 1e-15);

    let c = Refuter::new(43)
        .placebo_treatment(&study, &original)
        .expect("valid");
    assert!(
        (a.refuted_effect - c.refuted_effect).abs() > 1e-12,
        "different seeds must produce different draws"
    );
}

// ===========================================================================
// C10 — degenerate input panicked or invented numbers
// ===========================================================================

/// A dataset with one arm has no contrast, and must say so.
#[test]
fn c10_single_arm_data_is_an_error() {
    let treatment = Array1::from_vec(vec![1.0; 20]);
    #[allow(clippy::cast_precision_loss)]
    let outcome = Array1::from_shape_fn(20, |i| 10.0 + i as f64);

    let err = LinearATEEstimator::difference_in_means(&treatment, &outcome)
        .expect_err("no control units means no effect");
    assert!(matches!(err, EstimationError::EmptyArm { .. }), "{err}");
}

/// Mismatched lengths must be a `Result`, not a panic.
#[test]
fn c10_mismatched_lengths_do_not_panic() {
    let treatment = Array1::from_vec(vec![1.0, 0.0, 1.0]);
    let outcome = Array1::from_vec(vec![1.0, 2.0]);

    let err = LinearATEEstimator::difference_in_means(&treatment, &outcome)
        .expect_err("mismatched inputs must not be estimated");
    assert!(
        matches!(err, EstimationError::LengthMismatch { .. }),
        "{err}"
    );
}

/// An empty dataset must be an error.
#[test]
fn c10_empty_dataset_is_an_error() {
    let treatment: Array1<f64> = Array1::from_vec(vec![]);
    let outcome: Array1<f64> = Array1::from_vec(vec![]);

    let err = LinearATEEstimator::difference_in_means(&treatment, &outcome)
        .expect_err("no observations means no estimate");
    assert_eq!(err, EstimationError::NoObservations);
}

/// No public estimator entry point may panic on hostile input.
///
/// The broad version of C10: rather than enumerating known-bad cases, throw a
/// spread of degenerate shapes at every estimator and require each to return
/// rather than unwind.
#[test]
fn c10_no_estimator_panics_on_degenerate_input() {
    /// A degenerate input case: label, treatment, outcome, covariates.
    type Case = (&'static str, Array1<f64>, Array1<f64>, Array2<f64>);

    let cases: Vec<Case> = vec![
        (
            "empty",
            Array1::from_vec(vec![]),
            Array1::from_vec(vec![]),
            Array2::zeros((0, 0)),
        ),
        (
            "single unit",
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![5.0]),
            Array2::zeros((1, 1)),
        ),
        (
            "all treated",
            Array1::from_vec(vec![1.0; 10]),
            Array1::from_shape_fn(10, |i| f64::from(u8::try_from(i).unwrap_or(0))),
            Array2::zeros((10, 1)),
        ),
        (
            "constant outcome",
            Array1::from_shape_fn(10, |i| f64::from(u8::from(i % 2 == 0))),
            Array1::from_vec(vec![3.0; 10]),
            Array2::zeros((10, 1)),
        ),
        (
            "length mismatch",
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array2::zeros((2, 1)),
        ),
    ];

    for (label, t, y, x) in cases {
        let _ = LinearATEEstimator::difference_in_means(&t, &y);
        let _ = LinearATEEstimator::ols_adjusted(&t, &y, &x);
        let _ = PropensityScoreEstimator::ipw(&t, &y, &x);
        let _ = PropensityScoreEstimator::att(&t, &y, &x);
        let _ = IVEstimator::two_stage_ls(&t, &y, &x);
        // Reaching here without unwinding is the assertion.
        assert!(!label.is_empty());
    }
}

// ===========================================================================
// C11 — CausalDag did not enforce that it is a DAG
// ===========================================================================

/// Cycles must be rejected at construction, naming the cycle.
#[test]
fn c11_cycles_are_rejected_at_construction() {
    let mut dag = CausalDag::new();
    dag.add_edge("A", "B").expect("acyclic");
    dag.add_edge("B", "C").expect("acyclic");

    let err = dag.add_edge("C", "A").expect_err("this closes a cycle");
    match err {
        DagError::WouldCreateCycle { path, .. } => {
            assert!(path.len() >= 2, "the cycle must be named: {path:?}");
        }
        other => panic!("expected WouldCreateCycle, got {other}"),
    }
    assert!(dag.is_acyclic(), "the invariant must hold after rejection");
}

/// Self-loops must be rejected.
#[test]
fn c11_self_loops_are_rejected() {
    let mut dag = CausalDag::new();
    let err = dag
        .add_edge("X", "X")
        .expect_err("a variable cannot cause itself");
    assert!(matches!(err, DagError::SelfLoop { .. }), "{err}");
}

/// The invariant must hold for any sequence of insertions.
#[test]
fn c11_acyclicity_holds_under_arbitrary_insertion_orders() {
    // Every ordered pair among five variables, offered in a fixed but
    // adversarial order. Whatever is accepted, the result stays acyclic.
    let names = ["A", "B", "C", "D", "E"];
    let mut dag = CausalDag::new();
    for step in 0..40u32 {
        let i = (step * 7 % 5) as usize;
        let j = (step * 3 % 5) as usize;
        let _ = dag.add_edge(names[i], names[j]);
        assert!(
            dag.is_acyclic(),
            "cycle admitted at step {step} ({} -> {})",
            names[i],
            names[j]
        );
    }
}

// ===========================================================================
// C12 — d-separation reported independence for unknown variables
// ===========================================================================

/// An unknown variable must be an error, not a finding of independence.
///
/// `true` means "conditionally independent", so a typo produced a *positive*
/// result — the answer most likely to be acted on.
#[test]
fn c12_unknown_variable_is_an_error() {
    let mut dag = CausalDag::new();
    dag.add_edge("X", "M").expect("acyclic");
    dag.add_edge("M", "Y").expect("acyclic");

    let empty = HashSet::new();
    let err = d_separated(&dag, "Xx", "Y", &empty).expect_err("Xx is not in the graph");
    match err {
        DsepError::UnknownVariable { name, known } => {
            assert_eq!(name, "Xx");
            assert!(known.contains(&"X".to_string()), "known: {known:?}");
        }
    }
}

/// A typo in the conditioning set must be caught too.
///
/// The likeliest version in practice: the adjustment set is assembled
/// programmatically and one name does not match.
#[test]
fn c12_unknown_conditioning_variable_is_an_error() {
    let mut dag = CausalDag::new();
    dag.add_edge("X", "M").expect("acyclic");
    dag.add_edge("M", "Y").expect("acyclic");

    let z = HashSet::from(["Mm".to_string()]);
    assert!(
        d_separated(&dag, "X", "Y", &z).is_err(),
        "a misspelled conditioning variable must not silently validate the set"
    );
}

// ===========================================================================
// Estimand labelling — the ATE / ATT / LATE distinction
// ===========================================================================

/// Every estimator must name the quantity it computed.
///
/// Under effect heterogeneity, LATE and ATE are different quantities.
/// Reporting one as the other is a category error, not a rounding error.
#[test]
fn estimands_are_labelled_not_assumed() {
    let d = Dgp::new().with_n(2_000).sample(23);

    let ols = LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates)
        .expect("well-posed");
    assert_eq!(ols.estimand(), Estimand::Ate);

    let att = PropensityScoreEstimator::att(&d.treatment, &d.outcome, &d.covariates)
        .expect("benign DGP has overlap");
    assert_eq!(att.estimand(), Estimand::Att);

    let iv = Dgp::new().with_n(4_000).with_instrument(0.8).sample(29);
    let instrument = iv.instrument.as_ref().expect("instrument DGP");
    let z = Array2::from_shape_fn((iv.n(), 1), |(i, _)| instrument[i]);
    if let Ok(late) = IVEstimator::two_stage_ls(&iv.treatment, &iv.outcome, &z) {
        assert_eq!(late.estimand(), Estimand::Late);
        assert!(late.estimand().population().contains("complier"));
    }
}

/// A weak instrument must be refused, not reported.
#[test]
fn weak_instruments_are_refused() {
    let d = Dgp::new().with_n(2_000).with_instrument(0.01).sample(31);
    let instrument = d.instrument.as_ref().expect("instrument DGP");
    let z = Array2::from_shape_fn((d.n(), 1), |(i, _)| instrument[i]);

    match IVEstimator::two_stage_ls(&d.treatment, &d.outcome, &z) {
        Err(EstimationError::WeakInstrument { first_stage_f, .. }) => {
            assert!(first_stage_f < 10.0, "F was {first_stage_f}");
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(r) => panic!(
            "a 1%-strength instrument produced an estimate: {}",
            r.summary()
        ),
    }
}

// ===========================================================================
// Metamorphic relations — must hold for any correct estimator
// ===========================================================================

/// Duplicating the data leaves the estimate unchanged and shrinks the SE by √2.
///
/// Probes how a variance scales with `n`. Note this does *not* catch a variance
/// wrong by a constant factor — a uniformly mis-scaled formula still shrinks by
/// √2 — which is why coverage, not metamorphism, was what caught C5.
#[test]
fn metamorphic_duplication_shrinks_se_by_sqrt_two() {
    use cynepic_testkit::metamorphic::duplicate;

    let d = Dgp::new().with_n(1_000).sample(17);
    let base = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome).expect("valid");

    let (t2, y2, _) = duplicate(&d.treatment, &d.outcome, &d.covariates);
    let doubled = LinearATEEstimator::difference_in_means(&t2, &y2).expect("valid");

    assert!(
        (base.ate() - doubled.ate()).abs() < 1e-9,
        "duplication changed the point estimate: {} -> {}",
        base.ate(),
        doubled.ate()
    );

    let ratio = base.std_error() / doubled.std_error();
    assert!(
        (ratio - std::f64::consts::SQRT_2).abs() < 0.05,
        "SE should shrink by sqrt(2) on duplication, ratio was {ratio}"
    );
}

/// The same relation for IPW.
#[test]
fn metamorphic_ipw_duplication_shrinks_se_by_sqrt_two() {
    use cynepic_testkit::metamorphic::duplicate;

    let d = Dgp::new().with_n(1_000).sample(17);
    let base = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates)
        .expect("benign DGP has overlap");

    let (t2, y2, x2) = duplicate(&d.treatment, &d.outcome, &d.covariates);
    let doubled = PropensityScoreEstimator::ipw(&t2, &y2, &x2).expect("same data, doubled");

    assert!(
        (base.ate() - doubled.ate()).abs() < 1e-6,
        "duplication changed the IPW point estimate: {} -> {}",
        base.ate(),
        doubled.ate()
    );

    let ratio = base.std_error() / doubled.std_error();
    assert!(
        (ratio - std::f64::consts::SQRT_2).abs() < 0.05,
        "IPW SE should shrink by sqrt(2) on duplication, ratio was {ratio}"
    );
}

/// Scaling every outcome by `c` scales the effect by exactly `c`.
#[test]
fn metamorphic_scaling_outcome_scales_effect() {
    use cynepic_testkit::metamorphic::scale_outcome;

    let d = Dgp::new().with_n(500).sample(19);
    let base = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome).expect("valid");

    let scaled_y = scale_outcome(&d.outcome, 1_000.0);
    let scaled = LinearATEEstimator::difference_in_means(&d.treatment, &scaled_y).expect("valid");

    let expected = base.ate() * 1_000.0;
    assert!(
        (scaled.ate() - expected).abs() / expected.abs().max(1e-9) < 1e-9,
        "effect did not scale linearly: expected {expected}, got {}",
        scaled.ate()
    );
}

/// Shifting every outcome by a constant must not move the effect at all.
#[test]
fn metamorphic_shifting_outcome_does_not_move_effect() {
    use cynepic_testkit::metamorphic::shift_outcome;

    let d = Dgp::new().with_n(500).sample(21);
    let base =
        LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates).expect("valid");

    let shifted_y = shift_outcome(&d.outcome, 1_000.0);
    let shifted =
        LinearATEEstimator::ols_adjusted(&d.treatment, &shifted_y, &d.covariates).expect("valid");

    assert!(
        (base.ate() - shifted.ate()).abs() < 1e-6,
        "a constant shift moved the effect: {} -> {}",
        base.ate(),
        shifted.ate()
    );
}

/// A pure-noise covariate must not move the estimate.
#[test]
fn metamorphic_irrelevant_covariate_does_not_move_estimate() {
    let d = Dgp::new().with_n(2_000).sample(23);
    let base =
        LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates).expect("valid");

    let noise = Dgp::new().with_n(d.n()).sample(9_999);
    let mut wider = Array2::zeros((d.n(), d.p() + 1));
    for i in 0..d.n() {
        for j in 0..d.p() {
            wider[[i, j]] = d.covariates[[i, j]];
        }
        wider[[i, d.p()]] = noise.covariates[[i, 0]];
    }

    let widened =
        LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &wider).expect("valid");

    assert!(
        (base.ate() - widened.ate()).abs() < 0.1,
        "an irrelevant covariate moved the estimate: {} -> {}",
        base.ate(),
        widened.ate()
    );
}

/// Reversing unit order must not change anything.
#[test]
fn metamorphic_unit_order_does_not_matter() {
    use cynepic_testkit::metamorphic::reverse_units;

    let d = Dgp::new().with_n(600).sample(27);
    let base =
        LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates).expect("valid");

    let (t, y, x) = reverse_units(&d.treatment, &d.outcome, &d.covariates);
    let reversed = LinearATEEstimator::ols_adjusted(&t, &y, &x).expect("valid");

    assert!(
        (base.ate() - reversed.ate()).abs() < 1e-9,
        "unit order changed the estimate: {} -> {}",
        base.ate(),
        reversed.ate()
    );
    assert!(
        (base.std_error() - reversed.std_error()).abs() < 1e-9,
        "unit order changed the standard error"
    );
}

// ===========================================================================
// C14 — IPW under-covers when the weights are heavy (OPEN)
// ===========================================================================

/// IPW intervals must be nominal under strong confounding.
///
/// This is the residue of the C5/C13 fix, and it is a new finding rather than
/// an old one returning. The influence-function variance now subtracts the
/// projection onto the propensity model's score, which is the correct
/// asymptotic adjustment for the propensity being *estimated* rather than
/// known. Before that correction IPW over-covered at 100% with intervals about
/// three times wider than necessary; after it, coverage is nominal on most
/// cells and falls below nominal where the weights are heavy.
///
/// Measured at 300 replications, n=2000:
///
/// ```text
/// cell                    bias     MC sd   reported SE   coverage
/// benign                -0.0006    0.049      0.050         94.7%
/// strong-confounding    +0.0331    0.161      0.147         87.3%
/// moderate-overlap      +0.0170    0.151      0.141         92.3%
/// ```
///
/// The point estimate is sound — bias is under 0.04 everywhere. The standard
/// error is understated by roughly 9% at strong confounding. Two mechanisms are
/// plausible and not yet separated: the projection is estimated in-sample and
/// so removes some variation that is genuinely sampling noise, and the weight
/// distribution is heavy-tailed enough at this confounding level that the
/// symmetric normal interval is the wrong shape regardless of its width.
///
/// Under-coverage is the dangerous direction — an interval that lies about its
/// own confidence — so this is filed rather than tuned away, and the ratchet
/// keeps it visible until it is fixed.
///
/// Target: a variance that is nominal across the grid. Candidates are a
/// cross-fitted projection, or a bootstrap that refits the propensity model in
/// each resample and so captures both effects at once.
#[test]
#[ignore = "C14: IPW coverage 90.3% vs nominal 95% under strong confounding, narrowed from 87.3%"]
fn c14_ipw_coverage_is_nominal_under_strong_confounding() {
    let dgp = Dgp::new().with_n(2_000).with_confounding(3.0);

    let report = ValidationHarness::default().with_replications(300).run(
        "ipw-strong-confounding",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r = PropensityScoreEstimator::ipw(&data.treatment, &data.outcome, &data.covariates)
                .ok()?;
            let (lo, hi) = r.confidence_interval(0.95)?;
            Some((r.ate(), lo, hi))
        },
    );

    // A 3-point bar, not 5. Monte Carlo standard error at 300 replications is
    // about 1.2 points, so 3 points is a real requirement rather than noise,
    // and 5 would now be satisfied by the partial fix.
    assert!(
        report.coverage_ok(0.03),
        "IPW under-covers under strong confounding — {}",
        report.summary()
    );
}

/// The same for the ATT estimator, which shares the weighting machinery.
#[test]
#[ignore = "C14: ATT coverage below nominal under strong confounding"]
fn c14_att_coverage_is_nominal_under_strong_confounding() {
    let dgp = Dgp::new().with_n(2_000).with_confounding(3.0);

    let report = ValidationHarness::default().with_replications(300).run(
        "att-strong-confounding",
        &dgp,
        |t| Some(t.att),
        |data| {
            let r = PropensityScoreEstimator::att(&data.treatment, &data.outcome, &data.covariates)
                .ok()?;
            let (lo, hi) = r.confidence_interval(0.95)?;
            Some((r.ate(), lo, hi))
        },
    );

    assert!(
        report.coverage_ok(0.03),
        "ATT under-covers under strong confounding — {}",
        report.summary()
    );
}

/// The bootstrap estimator must agree with the analytic one on the point
/// estimate, and must report itself as a bootstrap.
///
/// Added while investigating C14. It is a guard, not a fix: measurement showed
/// the bootstrap interval is *narrower* than the analytic one exactly where the
/// analytic one was already too narrow, because resampling cannot reproduce a
/// tail event absent from the original sample. That is recorded so the option
/// is not mistaken for the remedy.
#[test]
fn ipw_bootstrap_agrees_on_the_point_estimate_and_labels_itself() {
    let d = Dgp::new().with_n(800).sample(41);

    let analytic = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates)
        .expect("benign DGP has overlap");
    let boot =
        PropensityScoreEstimator::ipw_bootstrap(&d.treatment, &d.outcome, &d.covariates, 60, 7)
            .expect("benign DGP has overlap");

    assert!(
        (analytic.ate() - boot.ate()).abs() < 1e-12,
        "the bootstrap must not re-centre the estimate: {} vs {}",
        analytic.ate(),
        boot.ate()
    );
    assert_eq!(boot.std_error_kind(), StdErrorKind::Bootstrap);
    assert!(boot.std_error() > 0.0 && boot.std_error().is_finite());
    // Degrees of freedom come from the resample count, not the sample size.
    assert_eq!(boot.diagnostics().variance_dof, Some(59.0));
}

/// A noisy variance estimate must widen the interval.
///
/// The mechanism behind C14's partial fix, asserted directly. When the
/// estimator reports few effective degrees of freedom for its variance, the
/// interval must use a Student-t quantile rather than a normal one — that is
/// what Student's t is for, and it applies to a weighted estimator dominated by
/// a few large weights as much as to a small sample.
#[test]
fn few_variance_degrees_of_freedom_widen_the_interval() {
    // Heavy weights: strong confounding drives the effective dof down.
    let heavy = Dgp::new().with_n(2_000).with_confounding(3.0).sample(43);
    let benign = Dgp::new().with_n(2_000).sample(43);

    let r_heavy =
        PropensityScoreEstimator::ipw(&heavy.treatment, &heavy.outcome, &heavy.covariates)
            .expect("has overlap");
    let r_benign =
        PropensityScoreEstimator::ipw(&benign.treatment, &benign.outcome, &benign.covariates)
            .expect("has overlap");

    let dof_heavy = r_heavy.diagnostics().variance_dof.expect("reported");
    let dof_benign = r_benign.diagnostics().variance_dof.expect("reported");

    assert!(
        dof_heavy < dof_benign,
        "heavy weights must yield fewer effective degrees of freedom: \
         {dof_heavy:.1} vs {dof_benign:.1}"
    );

    // And the widening must actually reach the interval.
    let (lo, hi) = r_heavy.confidence_interval(0.95).expect("finite se");
    let normal_width = 2.0 * 1.959_963_985 * r_heavy.std_error();
    assert!(
        (hi - lo) > normal_width,
        "the interval is no wider than a normal one despite {dof_heavy:.1} dof"
    );
}
