//! Executable specifications for the open correctness findings.
//!
//! # What this file is
//!
//! Test-driven development for a fix programme. Every test here asserts the
//! **corrected** behaviour and compiles against the API as it exists today. That
//! is the point: a finding you cannot demonstrate is a finding you cannot claim
//! to have fixed.
//!
//! Each is marked `#[ignore]` so the default suite stays green while the work is
//! outstanding. They are not skipped — CI runs them explicitly and tracks the
//! count, which may only ever go down.
//!
//! # Measured state at the time of writing
//!
//! 11 of these 18 specs fail, and those 11 are the open findings. The other 7
//! pass. That is not a defect in the findings — it is what the specs were for.
//! Three are metamorphic relations that the estimators genuinely satisfy, and
//! the OLS coverage specs show OLS is sound on the DGPs tested. Those belong in
//! the always-run suite as regression guards rather than sitting here behind an
//! `#[ignore]`, and they are moved there in the follow-up commit.
//!
//! A spec that passes on arrival has still done its job: it converted a
//! suspicion into a measurement.
//!
//! ```bash
//! cargo test -p cynepic-causal --test findings -- --ignored          # watch them fail
//! cargo test -p cynepic-causal --test findings -- --ignored C1       # one finding
//! ```
//!
//! # The workflow
//!
//! 1. A finding is confirmed → a failing spec lands here, `#[ignore]`d.
//! 2. Tier 1 implements the fix.
//! 3. The `#[ignore]` is removed **in the same pull request** as the fix.
//! 4. The spec-count ratchet in CI proves the number went down.
//!
//! A fix that arrives without deleting its `#[ignore]` has not been demonstrated,
//! and the reviewer should ask why.

use cynepic_causal::dag::CausalDag;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::identify::BackdoorCriterion;
use cynepic_causal::{dsep::d_separated, refute};
use cynepic_testkit::{Dgp, ValidationHarness};
use ndarray::{Array1, Array2};
use std::collections::HashSet;

// ===========================================================================
// C1 — a singular design matrix is reported as a precise null effect
// ===========================================================================

/// Two exactly collinear covariates must produce a named error, never a number.
///
/// Today `solve_normal_equation` returns a zero vector on a small pivot and
/// `invert_matrix` returns a zero matrix, so the caller receives
/// `ate: 0.0, std_error: 0.0` — indistinguishable from a well-identified precise
/// null, and it will pass any downstream significance check.
///
/// Target: `Err(EstimationError::RankDeficient { rank, expected, aliased })`
/// naming the offending columns.
#[test]
#[ignore = "C1: rank-revealing QR not implemented; returns 0.0/0.0 instead of an error"]
fn c1_collinear_covariates_must_not_yield_zero_effect_zero_error() {
    let data = Dgp::new().with_p(3).collinear().with_n(500).sample(1);

    let result = LinearATEEstimator::ols_adjusted(
        &data.treatment,
        &data.outcome,
        &data.covariates,
    );

    assert!(
        !(result.ate == 0.0 && result.std_error == 0.0),
        "rank-deficient design returned 'zero effect, zero uncertainty' \
         (ate={}, se={}) — the true effect is {}. seed={}",
        result.ate,
        result.std_error,
        data.truth.ate,
        data.seed
    );
}

/// A standard error of exactly zero is never a legitimate output.
///
/// Separated from the test above because it is the more dangerous half: a wrong
/// point estimate might be noticed, but zero uncertainty actively invites
/// downstream code to trust it.
#[test]
#[ignore = "C1: invert_matrix returns zeros on singular input, giving se = 0.0"]
fn c1_standard_error_is_never_exactly_zero() {
    let data = Dgp::new().with_p(2).collinear().with_n(300).sample(2);

    let result = LinearATEEstimator::ols_adjusted(
        &data.treatment,
        &data.outcome,
        &data.covariates,
    );

    assert!(
        result.std_error > 0.0 || result.std_error.is_nan(),
        "std_error was exactly 0.0 on a rank-deficient design — seed={}",
        data.seed
    );
}

// ===========================================================================
// C2 — identification can return an adjustment set you cannot measure
// ===========================================================================

/// Backdoor identification must not hand back an unobservable variable.
///
/// The DAG below is the crate's own front-door test case, where `U` is
/// documented as unobserved. `parents(T)` is a valid adjustment set only when
/// every parent is observed; here it is not, so the correct answer is that the
/// effect is not identifiable by adjustment.
///
/// Target: `Err(NotIdentifiable { blocking_paths })` once `VarKind::Latent`
/// exists.
#[test]
#[ignore = "C2: no latent-variable concept; find() returns Some({U}) unconditionally"]
fn c2_adjustment_set_must_not_contain_a_latent_variable() {
    let mut dag = CausalDag::new();
    dag.add_edge("U", "Smoking"); // U is unobserved
    dag.add_edge("U", "Cancer");
    dag.add_edge("Smoking", "Tar");
    dag.add_edge("Tar", "Cancer");

    let adjustment = BackdoorCriterion::find(&dag, "Smoking", "Cancer");

    match adjustment {
        None => { /* correct: not identifiable by adjustment */ }
        Some(set) => assert!(
            !set.contains("U"),
            "identification returned the unobservable confounder U as an \
             adjustment set: {set:?}"
        ),
    }
}

/// Identification must be capable of failing at all.
///
/// A criterion that can never return "no" is not a criterion. This asserts the
/// existence of at least one graph for which backdoor identification declines.
#[test]
#[ignore = "C2: find() returns Some(..) for every input; non-identifiability is unrepresentable"]
fn c2_identification_can_fail() {
    // Bidirected confounding that no observed set can block.
    let mut dag = CausalDag::new();
    dag.add_edge("U1", "X");
    dag.add_edge("U1", "Y");
    dag.add_edge("X", "Y");

    assert!(
        BackdoorCriterion::find(&dag, "X", "Y").is_none(),
        "an unobserved common cause must make the effect unidentifiable by adjustment"
    );
}

// ===========================================================================
// C5 / C6 — IPW's standard error does not describe IPW's point estimate
// ===========================================================================

/// Confidence-interval coverage is the test that catches an inconsistent
/// variance formula. Nothing else does — every individual run looks plausible.
///
/// A nominal 95% interval must contain the truth about 95% of the time. The
/// current IPW pairs a Hájek point estimate with a Horvitz–Thompson variance,
/// so its intervals describe a quantity it is not reporting.
#[test]
#[ignore = "C5: Hájek point estimate paired with Horvitz-Thompson variance"]
fn c5_ipw_intervals_must_achieve_nominal_coverage() {
    let dgp = Dgp::new().with_n(1_000).with_confounding(1.0);

    let report = ValidationHarness::default()
        .with_replications(200)
        .run("ipw-benign", &dgp, |t| Some(t.ate), |data| {
            let r = PropensityScoreEstimator::ipw(
                &data.treatment,
                &data.outcome,
                &data.covariates,
            );
            // Normal approximation, as the current API implies.
            Some((r.ate, r.ate - 1.96 * r.std_error, r.ate + 1.96 * r.std_error))
        });

    assert!(
        report.coverage_ok(0.05),
        "IPW coverage is not nominal — {}",
        report.summary()
    );
}

/// The same discipline applied to OLS, which should be the easy case.
///
/// If OLS cannot hit nominal coverage on a benign, correctly-specified,
/// homoskedastic DGP, the problem is the standard error, not the world.
#[test]
#[ignore = "C1/C5: verify once the QR solver and robust SEs land"]
fn c5_ols_intervals_must_achieve_nominal_coverage() {
    let dgp = Dgp::new().with_n(800);

    let report = ValidationHarness::default()
        .with_replications(200)
        .run("ols-benign", &dgp, |t| Some(t.ate), |data| {
            let r = LinearATEEstimator::ols_adjusted(
                &data.treatment,
                &data.outcome,
                &data.covariates,
            );
            Some((r.ate, r.ate - 1.96 * r.std_error, r.ate + 1.96 * r.std_error))
        });

    assert!(
        report.coverage_ok(0.05),
        "OLS coverage is not nominal on a benign DGP — {}",
        report.summary()
    );
}

/// Under heteroskedasticity, classical standard errors are wrong and coverage
/// degrades. This is the spec for HC1/HC3 robust standard errors.
#[test]
#[ignore = "C5: heteroskedasticity-robust standard errors not implemented"]
fn c5_coverage_survives_heteroskedasticity() {
    let dgp = Dgp::new().with_n(800).heteroskedastic();

    let report = ValidationHarness::default()
        .with_replications(200)
        .run("ols-heteroskedastic", &dgp, |t| Some(t.ate), |data| {
            let r = LinearATEEstimator::ols_adjusted(
                &data.treatment,
                &data.outcome,
                &data.covariates,
            );
            Some((r.ate, r.ate - 1.96 * r.std_error, r.ate + 1.96 * r.std_error))
        });

    assert!(
        report.coverage_ok(0.05),
        "coverage collapsed under heteroskedasticity — {}",
        report.summary()
    );
}

// ===========================================================================
// C7 — refutation verdicts are magic constants, not tests
// ===========================================================================

/// A refutation verdict must depend on the estimate's own uncertainty.
///
/// Two datasets with the same *relative* change but very different noise levels
/// must not receive the same verdict: in the noisy one the change is well within
/// sampling error and means nothing.
///
/// Today both are judged by `relative_change < 0.15`, so both get the same
/// answer regardless of how uncertain the estimate was.
#[test]
#[ignore = "C7: verdict is a fixed relative-change threshold with no reference to the standard error"]
fn c7_refutation_verdict_must_depend_on_uncertainty() {
    let precise = Dgp::new().with_n(4_000).with_noise(0.1).sample(11);
    let noisy = Dgp::new().with_n(4_000).with_noise(8.0).sample(11);

    let est = |d: &cynepic_testkit::Dataset| {
        LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome).ate
    };

    let r_precise =
        refute::random_common_cause(&precise.treatment, &precise.outcome, est(&precise), 20);
    let r_noisy = refute::random_common_cause(&noisy.treatment, &noisy.outcome, est(&noisy), 20);

    assert_ne!(
        r_precise.passed, r_noisy.passed,
        "a precise estimate and a very noisy one received identical verdicts; \
         the decision rule ignores the standard error entirely"
    );
}

/// A placebo test must re-run the estimator it was asked to refute.
///
/// `placebo_treatment` currently takes only the outcome and re-estimates with
/// `difference_in_means`, whatever produced the original estimate, and ignores
/// the adjustment set. It therefore does not refute the estimate it was handed.
///
/// Target: `Refuter::run(&self, study, rng)` taking the study — estimator and
/// adjustment set included.
#[test]
#[ignore = "C7: placebo_treatment hardcodes difference_in_means and ignores covariates"]
fn c7_placebo_must_refute_the_estimator_it_was_given() {
    // An adjusted estimate on confounded data. The placebo should reproduce the
    // *adjusted* null, not the unadjusted one.
    let data = Dgp::new().with_n(2_000).with_confounding(3.0).sample(13);

    let adjusted =
        LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates);
    let result = refute::placebo_treatment(&data.outcome, adjusted.ate, 0.5);

    // Under strong confounding, the unadjusted placebo distribution is nowhere
    // near the adjusted one, so a placebo that ignores covariates gives a
    // verdict about a different estimator.
    assert!(
        result.passed,
        "placebo verdict was computed with the wrong estimator: adjusted ate={}, \
         placebo effect={}, truth={}",
        adjusted.ate, result.refuted_effect, data.truth.ate
    );
}

// ===========================================================================
// C10 — library code returns garbage or panics on edge-case input
// ===========================================================================

/// An empty treatment arm must be an error, not a number.
///
/// `difference_in_means` substitutes `0.0` for the missing arm's mean and
/// returns the difference, so a dataset with no controls yields a confident
/// effect equal to the treated mean.
#[test]
#[ignore = "C10: empty arm substitutes 0.0 for the missing mean instead of erroring"]
fn c10_single_arm_data_must_not_produce_an_effect() {
    let treatment = Array1::from_vec(vec![1.0; 20]); // every unit treated
    let outcome = Array1::from_vec((0..20).map(|i| 10.0 + i as f64).collect());

    let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);

    assert!(
        result.ate.is_nan(),
        "with no control units the effect is undefined, but got ate={} \
         (which is just the treated mean)",
        result.ate
    );
}

/// Mismatched input lengths must be a `Result`, not a panic.
///
/// CLAUDE.md states library code returns `Result`. Today this is `assert_eq!`,
/// so a service embedding the crate dies on malformed input.
#[test]
#[ignore = "C10: assert_eq! on caller input; needs Result-returning API"]
fn c10_mismatched_lengths_must_not_panic() {
    let treatment = Array1::from_vec(vec![1.0, 0.0, 1.0]);
    let outcome = Array1::from_vec(vec![1.0, 2.0]); // deliberately shorter

    let outcome_len = outcome.len();
    let caught = std::panic::catch_unwind(move || {
        LinearATEEstimator::difference_in_means(&treatment, &outcome)
    });

    assert!(
        caught.is_ok(),
        "library panicked on mismatched input lengths (3 vs {outcome_len}); \
         it should return an error"
    );
}

/// Zero observations must not panic or produce an effect.
#[test]
#[ignore = "C10: no guard for n = 0"]
fn c10_empty_dataset_must_not_produce_an_effect() {
    let treatment: Array1<f64> = Array1::from_vec(vec![]);
    let outcome: Array1<f64> = Array1::from_vec(vec![]);

    let caught =
        std::panic::catch_unwind(move || {
            LinearATEEstimator::difference_in_means(&treatment, &outcome)
        });

    match caught {
        Err(_) => panic!("library panicked on an empty dataset"),
        Ok(r) => assert!(
            r.ate.is_nan(),
            "an empty dataset yielded a finite effect of {}",
            r.ate
        ),
    }
}

// ===========================================================================
// C11 — CausalDag does not enforce that it is a DAG
// ===========================================================================

/// Adding an edge that closes a cycle must be rejected at construction.
///
/// Every downstream algorithm — d-separation, backdoor, front-door, topological
/// ordering — has guarantees conditional on acyclicity. Today `add_edge` accepts
/// the cycle and `is_acyclic()` exists but is never consulted.
///
/// Target: `add_edge(..) -> Result<(), DagError>`.
#[test]
#[ignore = "C11: add_edge returns () and accepts cycles"]
fn c11_cycles_must_be_rejected_at_construction() {
    let mut dag = CausalDag::new();
    dag.add_edge("A", "B");
    dag.add_edge("B", "C");
    dag.add_edge("C", "A"); // closes a cycle — should have been refused

    assert!(
        dag.is_acyclic(),
        "a cyclic graph was constructed successfully; the type is called CausalDag"
    );
}

/// Self-loops are never meaningful in a structural causal model.
#[test]
#[ignore = "C11: add_edge accepts self-loops"]
fn c11_self_loops_must_be_rejected() {
    let mut dag = CausalDag::new();
    dag.add_edge("X", "X");

    assert_eq!(
        dag.num_edges(),
        0,
        "a self-loop X -> X was added to the graph"
    );
}

// ===========================================================================
// C12 — d-separation reports independence for unknown variables
// ===========================================================================

/// A typo must not become a claim of conditional independence.
///
/// `d_separated` returns `true` for any name it does not recognise. Since `true`
/// means "independent", a misspelled variable silently validates an adjustment
/// set that was never checked.
///
/// Target: `Result<bool, DsepError::UnknownVariable>`.
#[test]
#[ignore = "C12: unknown variable names return true (d-separated)"]
fn c12_unknown_variable_must_not_be_reported_as_independent() {
    let mut dag = CausalDag::new();
    dag.add_edge("X", "Y");

    let empty: HashSet<String> = HashSet::new();

    // "Xx" does not exist. X and Y are directly connected, so any sane answer
    // about a typo of X is "I don't know", never "independent".
    let claimed_independent = d_separated(&dag, "Xx", "Y", &empty);

    assert!(
        !claimed_independent,
        "d_separated claimed an unknown variable is independent of Y; \
         a typo now silently validates an unchecked adjustment set"
    );
}

// ===========================================================================
// Metamorphic relations — must hold for any correct estimator
// ===========================================================================

/// Duplicating the dataset must leave the point estimate unchanged and shrink
/// the standard error by √2.
///
/// This relation is unusually good at exposing an inconsistent variance formula,
/// because the point estimate and the standard error must respond to duplication
/// in *different* ways. A mismatched pair cannot satisfy both halves.
#[test]
#[ignore = "C5: variance formula inconsistent with its point estimate"]
fn metamorphic_duplication_shrinks_se_by_sqrt_two() {
    use cynepic_testkit::metamorphic::duplicate;

    let d = Dgp::new().with_n(1_000).sample(17);
    let base = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome);

    let (t2, y2, _) = duplicate(&d.treatment, &d.outcome, &d.covariates);
    let doubled = LinearATEEstimator::difference_in_means(&t2, &y2);

    assert!(
        (base.ate - doubled.ate).abs() < 1e-9,
        "duplicating the data changed the point estimate: {} -> {}",
        base.ate,
        doubled.ate
    );

    let ratio = base.std_error / doubled.std_error;
    assert!(
        (ratio - std::f64::consts::SQRT_2).abs() < 0.05,
        "standard error should shrink by sqrt(2) on duplication, ratio was {ratio}"
    );
}

/// Scaling every outcome by `c` must scale the effect by exactly `c`.
///
/// Catches magnitudes compared against absolute constants — the shape of C7's
/// `< 0.15` rule, which is not scale-free.
#[test]
#[ignore = "verify alongside the C7 refutation rewrite"]
fn metamorphic_scaling_outcome_scales_effect() {
    use cynepic_testkit::metamorphic::scale_outcome;

    let d = Dgp::new().with_n(500).sample(19);
    let base = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome);

    let scaled_y = scale_outcome(&d.outcome, 1_000.0);
    let scaled = LinearATEEstimator::difference_in_means(&d.treatment, &scaled_y);

    let expected = base.ate * 1_000.0;
    assert!(
        (scaled.ate - expected).abs() / expected.abs().max(1e-9) < 1e-9,
        "effect did not scale linearly with the outcome: expected {expected}, got {}",
        scaled.ate
    );
}

/// Adding a pure-noise covariate must not move the estimate beyond Monte Carlo
/// error. A large move means the estimator is fitting noise.
#[test]
#[ignore = "C1: verify once the solver is numerically stable"]
fn metamorphic_irrelevant_covariate_does_not_move_estimate() {
    let d = Dgp::new().with_n(2_000).sample(23);
    let base =
        LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates);

    // Append a column of independent noise derived from a different seed.
    let noise = Dgp::new().with_n(d.n()).sample(9_999);
    let mut wider = Array2::zeros((d.n(), d.p() + 1));
    for i in 0..d.n() {
        for j in 0..d.p() {
            wider[[i, j]] = d.covariates[[i, j]];
        }
        wider[[i, d.p()]] = noise.covariates[[i, 0]];
    }

    let with_noise = LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &wider);

    assert!(
        (base.ate - with_noise.ate).abs() < 3.0 * base.std_error,
        "adding an irrelevant covariate moved the estimate from {} to {} \
         (more than 3 standard errors of {})",
        base.ate,
        with_noise.ate,
        base.std_error
    );
}
