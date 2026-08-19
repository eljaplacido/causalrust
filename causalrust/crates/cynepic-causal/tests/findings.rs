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

/// Appending an exactly collinear column must not change the estimate.
///
/// The added column is the sum of two existing ones. It carries no information
/// the design did not already have, so a numerically sound solver returns the
/// same coefficient on treatment either way — or refuses, naming the aliased
/// columns.
///
/// This replaces an earlier spec that asserted `!(ate == 0.0 && se == 0.0)`.
/// That assertion passed, which was informative in the wrong direction: the
/// solver does not return `0.0/0.0`. It returns a *plausible non-zero* ATE with
/// `se == 0.0`. Requiring both halves to be zero made the spec unable to see
/// the defect it was written for, and a confident wrong number is worse than an
/// obviously broken one.
///
/// Target: `Err(EstimationError::RankDeficient { rank, expected, aliased })`.
#[test]
// Passes today, which localises C1 rather than refuting it. The aliasing is
// confined to the covariate block: the redundant column's own coefficient
// collapses while the coefficient on treatment survives intact. So the point
// estimate is fine and only the standard error is destroyed — see
// `c1_standard_error_is_never_exactly_zero`, which is the spec that fails.
fn c1_collinear_column_must_not_change_the_estimate() {
    let full_rank = Dgp::new().with_p(3).with_n(500).sample(1);
    let aliased = Dgp::new().with_p(3).collinear().with_n(500).sample(1);

    let a = LinearATEEstimator::ols_adjusted(
        &full_rank.treatment,
        &full_rank.outcome,
        &full_rank.covariates,
    );
    let b =
        LinearATEEstimator::ols_adjusted(&aliased.treatment, &aliased.outcome, &aliased.covariates);

    // Same seed, same units, same outcomes — the only difference is a redundant
    // column. Any movement is the solver reacting to rank deficiency.
    assert!(
        (a.ate - b.ate).abs() < 1e-6,
        "an exactly collinear column moved the estimate {} -> {} (true effect {}). \
         The column adds no information, so this is the solver, not the data. seed={}",
        a.ate,
        b.ate,
        aliased.truth.ate,
        aliased.seed
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

    let result = LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates);

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

    let report = ValidationHarness::default().with_replications(200).run(
        "ipw-benign",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r = PropensityScoreEstimator::ipw(&data.treatment, &data.outcome, &data.covariates);
            // Normal approximation, as the current API implies.
            Some((
                r.ate,
                r.ate - 1.96 * r.std_error,
                r.ate + 1.96 * r.std_error,
            ))
        },
    );

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
// Passes today: OLS coverage is 94.7%-97.3% across the standard grid. Kept
// in the always-run suite as the regression guard for that result.
fn c5_ols_intervals_must_achieve_nominal_coverage() {
    let dgp = Dgp::new().with_n(800);

    let report = ValidationHarness::default().with_replications(200).run(
        "ols-benign",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r =
                LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates);
            Some((
                r.ate,
                r.ate - 1.96 * r.std_error,
                r.ate + 1.96 * r.std_error,
            ))
        },
    );

    assert!(
        report.coverage_ok(0.05),
        "OLS coverage is not nominal on a benign DGP — {}",
        report.summary()
    );
}

/// Under heteroskedasticity, classical standard errors are wrong and coverage
/// degrades. This is the spec for HC1/HC3 robust standard errors.
#[test]
// Passes today at 97.0% coverage — classical SEs survive this DGP's
// heteroskedasticity. HC1/HC3 remain worth having for harsher designs, but
// this is not currently a source of wrong answers, so it guards rather than
// accuses.
fn c5_coverage_survives_heteroskedasticity() {
    let dgp = Dgp::new().with_n(800).heteroskedastic();

    let report = ValidationHarness::default().with_replications(200).run(
        "ols-heteroskedastic",
        &dgp,
        |t| Some(t.ate),
        |data| {
            let r =
                LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates);
            Some((
                r.ate,
                r.ate - 1.96 * r.std_error,
                r.ate + 1.96 * r.std_error,
            ))
        },
    );

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

/// A placebo test must be able to tell a sound estimate from a confounded one.
///
/// `placebo_treatment` takes only `(outcome, original_ate, tolerance)`. It never
/// sees the treatment, the covariates, or the estimator, and re-estimates with
/// `difference_in_means` regardless of what produced the original number. So it
/// cannot be refuting the estimate it was handed.
///
/// The consequence is checkable even though the missing parameters are not. On
/// strongly confounded data the adjusted estimate is close to the truth and the
/// unadjusted one is badly biased. A working refuter must reach different
/// verdicts about them. This one is handed only a scalar, so its verdict is a
/// function of that scalar's magnitude — not of whether the estimate is sound.
///
/// The previous version of this spec asserted `result.passed` on the adjusted
/// estimate alone, which passed and demonstrated nothing: a refuter that always
/// returns `passed` satisfies it.
///
/// Target: `Refuter::run(&self, study, rng)` taking the study — estimator and
/// adjustment set included.
#[test]
#[ignore = "C7: placebo_treatment ignores treatment, covariates and estimator"]
fn c7_placebo_must_distinguish_adjusted_from_confounded() {
    let data = Dgp::new().with_n(2_000).with_confounding(3.0).sample(13);

    let adjusted =
        LinearATEEstimator::ols_adjusted(&data.treatment, &data.outcome, &data.covariates);
    let confounded = LinearATEEstimator::difference_in_means(&data.treatment, &data.outcome);

    let on_adjusted = refute::placebo_treatment(&data.outcome, adjusted.ate, 0.5);
    let on_confounded = refute::placebo_treatment(&data.outcome, confounded.ate, 0.5);

    assert_ne!(
        on_adjusted.passed, on_confounded.passed,
        "the placebo refuter reached the same verdict ({}) for a well-adjusted \
         estimate ({}) and a badly confounded one ({}); the truth is {}. \
         A refutation that cannot separate these two is not evidence.",
        on_adjusted.passed, adjusted.ate, confounded.ate, data.truth.ate
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

    let caught = std::panic::catch_unwind(move || {
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
// Passes today. Note it exercises `difference_in_means`, whose variance IS
// consistent with its point estimate, so it never witnessed C5. The IPW arm
// below is the spec that does.
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

/// The same relation applied to IPW, which is where it bites.
///
/// Duplication is the sharpest available probe for an inconsistent variance
/// formula, because the point estimate and the standard error must respond in
/// *different* ways — the estimate unchanged, the SE down by √2. A pair drawn
/// from two different estimators cannot satisfy both halves at once.
///
/// IPW pairs a Hájek point estimate with a Horvitz–Thompson variance (C5), so
/// this was expected to be the arm that fails. It does not, and that is worth
/// recording: duplication probes how a variance scales with `n`, and IPW's
/// variance gets the `n` dependence right while getting the *scale* wrong. A
/// formula that is uniformly wrong by a constant factor still shrinks by √2.
///
/// So this relation is a genuine regression guard, but it is not a witness for
/// C5, and the file previously claimed it was. Coverage is what catches a scale
/// error, because coverage compares an interval against a known truth rather
/// than against another interval.
#[test]
fn c5_ipw_duplication_shrinks_se_by_sqrt_two() {
    use cynepic_testkit::metamorphic::duplicate;

    let d = Dgp::new().with_n(1_000).sample(17);
    let base = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates);

    let (t2, y2, x2) = duplicate(&d.treatment, &d.outcome, &d.covariates);
    let doubled = PropensityScoreEstimator::ipw(&t2, &y2, &x2);

    assert!(
        (base.ate - doubled.ate).abs() < 1e-6,
        "duplicating the data changed the IPW point estimate: {} -> {}",
        base.ate,
        doubled.ate
    );

    let ratio = base.std_error / doubled.std_error;
    assert!(
        (ratio - std::f64::consts::SQRT_2).abs() < 0.05,
        "IPW standard error should shrink by sqrt(2) on duplication, ratio was {ratio} \
         — the variance formula does not describe the reported point estimate"
    );
}

/// The propensity model must actually converge.
///
/// `ipw` runs a fixed 100 iterations of gradient descent at lr=0.1 with no
/// stopping rule. On the benign DGP it halts at roughly 55% of the true logit
/// coefficients, propensities compress toward 0.5, the weights under-correct,
/// and about 47% of the confounding survives.
///
/// This is checkable without seeing the fitted scores: an estimator that
/// adjusts for confounding must beat one that does not. Today IPW's bias is
/// +1.80 against naive difference-in-means at +3.40, so it removes only about
/// half of what it exists to remove — while reporting an interval 0.4 wide.
///
/// Target: IRLS/Newton, converging in 5–10 iterations, returning
/// `Err(NotConverged { iters, gradient_norm })` rather than a partial fit.
#[test]
#[ignore = "C13: fixed 100-iteration gradient descent, no convergence check"]
fn c13_ipw_must_remove_most_of_the_confounding() {
    let dgp = Dgp::new().with_n(2_000);

    let mut ipw_bias = 0.0;
    let mut naive_bias = 0.0;
    let reps: u64 = 20;

    for seed in 0..reps {
        let d = dgp.sample(seed);
        let ipw = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates);
        let naive = LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome);
        ipw_bias += ipw.ate - d.truth.ate;
        naive_bias += naive.ate - d.truth.ate;
    }
    #[allow(clippy::cast_precision_loss)]
    let reps_f = reps as f64;
    ipw_bias /= reps_f;
    naive_bias /= reps_f;

    let removed = 1.0 - (ipw_bias.abs() / naive_bias.abs());
    assert!(
        removed > 0.90,
        "IPW removed only {:.0}% of the confounding bias (IPW {:+.3} vs naive \
         {:+.3}). A converged fit reaches {:.0}%.",
        removed * 100.0,
        ipw_bias,
        naive_bias,
        92.0
    );
}

/// Scaling every outcome by `c` must scale the effect by exactly `c`.
///
/// Catches magnitudes compared against absolute constants — the shape of C7's
/// `< 0.15` rule, which is not scale-free.
#[test]
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
fn metamorphic_irrelevant_covariate_does_not_move_estimate() {
    let d = Dgp::new().with_n(2_000).sample(23);
    let base = LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates);

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
