//! Measured calibration of the conjugate priors and MCMC samplers.
//!
//! # What is being asserted
//!
//! A 95% credible interval must contain the parameter that generated the data
//! 95% of the time, when the parameter is drawn from the prior the model was
//! given. That is the interval's own claim, it holds exactly for a correct
//! posterior, and it is directly measurable — so it is the Bayesian counterpart
//! of the confidence-interval coverage that `cynepic-causal` is validated
//! against.
//!
//! For the samplers, coverage of one interval is a coarse instrument. These use
//! **simulation-based calibration**, whose rank histogram says *how* a
//! posterior is wrong rather than only that it is: mass at both ends means the
//! posterior is too narrow, mass in the middle means too wide, a slope means
//! biased.
//!
//! ```bash
//! cargo test -p cynepic-bayes --test calibration
//! cargo test -p cynepic-bayes --test calibration -- --ignored   # open findings
//! ```

use cynepic_bayes::priors::{BetaBinomial, GammaPoisson, NormalNormal};
use cynepic_bayes::sampler::{AdaptiveMH, MetropolisHastings, MultiDimMH};
use cynepic_testkit::calibration::{
    beta_credible_interval, credible_coverage, sample_beta, sample_gamma, sample_standard_normal,
    simulation_based_calibration,
};
use rand::Rng;

// ===========================================================================
// Conjugate priors — closed-form posteriors, so coverage should be exact
// ===========================================================================

/// Normal-Normal is conjugate with a Gaussian posterior, and its interval is
/// built from that posterior directly. It should be exactly calibrated.
#[test]
fn normal_normal_credible_intervals_are_calibrated() {
    let prior_mean = 0.0;
    let prior_var: f64 = 1.0;
    let obs_var: f64 = 1.0;

    let report = credible_coverage("NormalNormal", 2_000, 0.95, 101, |rng| {
        // Draw the truth from the model's own prior — the condition under which
        // the credible interval's claim is exact.
        let theta = prior_mean + prior_var.sqrt() * sample_standard_normal(rng);
        let obs: Vec<f64> = (0..10)
            .map(|_| theta + obs_var.sqrt() * sample_standard_normal(rng))
            .collect();

        let mut model = NormalNormal::new(prior_mean, prior_var, obs_var).expect("valid prior");
        model.update(&obs);
        let (lo, hi) = model.credible_interval_95();
        Some((theta, model.mean(), lo, hi))
    });

    assert!(
        report.coverage_ok(0.02),
        "NormalNormal is miscalibrated — {}",
        report.summary()
    );
}

/// Gamma-Poisson, same argument.
#[test]
fn gamma_poisson_posterior_mean_is_consistent() {
    // GammaPoisson exposes mean and variance but no interval, so the checkable
    // claim is that the posterior mean converges on the truth. Stated as a
    // shrinking-error test rather than a coverage one, because asserting
    // coverage without an interval would be asserting nothing.
    let mut small_error = 0.0;
    let mut large_error = 0.0;
    let reps: u32 = 200;

    for seed in 0..u64::from(reps) {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed_u64(seed);
        let lambda = sample_gamma(&mut rng, 3.0) / 1.0;

        let short: Vec<u64> = (0..5).map(|_| poisson(&mut rng, lambda)).collect();
        let long: Vec<u64> = (0..500).map(|_| poisson(&mut rng, lambda)).collect();

        let mut a = GammaPoisson::new(3.0, 1.0).expect("valid prior");
        a.update(&short);
        let mut b = GammaPoisson::new(3.0, 1.0).expect("valid prior");
        b.update(&long);

        small_error += (a.mean() - lambda).abs();
        large_error += (b.mean() - lambda).abs();
    }

    assert!(
        large_error < small_error / 3.0,
        "posterior mean should converge: 5 obs {:.3}, 500 obs {:.3}",
        small_error / f64::from(reps),
        large_error / f64::from(reps)
    );
}

// ===========================================================================
// B1 — BetaBinomial's credible interval is a normal approximation
// ===========================================================================

/// A Beta posterior's interval must contain the truth at the nominal rate.
///
/// `credible_interval_95` builds `mean ± 1.96·sd` and clamps to `[0, 1]`. A
/// Beta density is not symmetric unless `alpha == beta`, and at small counts or
/// near the boundary it is strongly skewed, so a symmetric interval sits in the
/// wrong place. The clamp then hides the overrun instead of correcting it: mass
/// that should have been in a tail is silently discarded.
///
/// This spec asks for calibration under a weak prior and few observations,
/// which is exactly the regime a tool-reliability tracker lives in — the first
/// handful of calls to a new tool.
///
/// **Closed.** `credible_interval_95` now uses the Beta quantile function.
#[test]
fn b1_beta_binomial_intervals_are_calibrated_at_small_n() {
    let report = credible_coverage("BetaBinomial n=5", 4_000, 0.95, 202, |rng| {
        let p = sample_beta(rng, 1.0, 1.0);
        let n = 5u64;
        let successes = (0..n).filter(|_| rng.random::<f64>() < p).count() as u64;

        let mut model = BetaBinomial::uniform();
        model.update(successes, n - successes);
        let (lo, hi) = model.credible_interval_95();
        Some((p, model.mean(), lo, hi))
    });

    assert!(
        report.coverage_ok(0.03),
        "BetaBinomial intervals are miscalibrated at small n — {}",
        report.summary()
    );
}

/// The same at moderate `n`, where the normal approximation should be closer.
///
/// Kept separate so the fix can be verified to help where it matters without
/// regressing where the approximation was already adequate.
#[test]
fn b1_beta_binomial_intervals_are_calibrated_at_moderate_n() {
    let report = credible_coverage("BetaBinomial n=50", 4_000, 0.95, 203, |rng| {
        let p = sample_beta(rng, 1.0, 1.0);
        let n = 50u64;
        let successes = (0..n).filter(|_| rng.random::<f64>() < p).count() as u64;

        let mut model = BetaBinomial::uniform();
        model.update(successes, n - successes);
        let (lo, hi) = model.credible_interval_95();
        Some((p, model.mean(), lo, hi))
    });

    assert!(
        report.coverage_ok(0.03),
        "BetaBinomial intervals are miscalibrated at moderate n — {}",
        report.summary()
    );
}

/// The posterior *mean* is exact even where the interval is not.
///
/// Establishes that B1 is confined to the interval. Without this the finding
/// would be "BetaBinomial is broken", which is both wider and less useful than
/// "its interval is a normal approximation".
#[test]
fn beta_binomial_posterior_mean_is_exact() {
    let mut total_error = 0.0;
    let reps: u32 = 2_000;

    for seed in 0..u64::from(reps) {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed_u64(seed + 500);
        let p = sample_beta(&mut rng, 1.0, 1.0);
        let n = 200u64;
        let successes = (0..n).filter(|_| rng.random::<f64>() < p).count() as u64;

        let mut model = BetaBinomial::uniform();
        model.update(successes, n - successes);
        total_error += model.mean() - p;
    }

    let bias = total_error / f64::from(reps);
    assert!(
        bias.abs() < 0.01,
        "posterior mean should be near-unbiased at n=200, bias was {bias:+.4}"
    );
}

// ===========================================================================
// Samplers — simulation-based calibration
// ===========================================================================

/// Metropolis-Hastings must target the posterior it is given.
///
/// The target is a conjugate normal, so the exact posterior is known and any
/// departure of the rank histogram from uniform is the sampler's fault rather
/// than the model's.
#[test]
fn metropolis_hastings_is_calibrated_on_a_conjugate_normal() {
    let report = simulation_based_calibration("MetropolisHastings", 400, 10, 303, |rng| {
        // Prior N(0,1); one observation with known variance 1.
        let theta = sample_standard_normal(rng);
        let y = theta + sample_standard_normal(rng);

        // Exact posterior: N(y/2, 1/2).
        let log_density = move |t: f64| -((t - y / 2.0).powi(2)) / (2.0 * 0.5);

        let sampler = MetropolisHastings::new(1.0, 2_000, 4_000);
        let result = sampler.sample(log_density, 0.0);

        // Thin to reduce autocorrelation: SBC assumes independent draws, and a
        // correlated chain inflates the apparent rank concentration in a way
        // that would be read as miscalibration rather than as autocorrelation.
        let draws: Vec<f64> = result.samples.iter().step_by(40).copied().collect();
        if draws.len() < 20 {
            None
        } else {
            Some((theta, draws))
        }
    });

    assert!(
        report.is_calibrated(0.001),
        "MH is miscalibrated — {}",
        report.summary()
    );
}

/// AdaptiveMH must be calibrated too — adaptation must not bias the target.
///
/// This is the risk specific to an adaptive sampler: tuning the proposal using
/// the chain's own history breaks the Markov property, and a scheme that keeps
/// adapting during the sampling phase does not target the intended
/// distribution at all. Robbins-Monro adaptation confined to warmup is safe;
/// this test is what makes that claim checkable rather than assumed.
#[test]
fn adaptive_mh_adaptation_does_not_bias_the_target() {
    let report = simulation_based_calibration("AdaptiveMH", 400, 10, 404, |rng| {
        let theta = sample_standard_normal(rng);
        let y = theta + sample_standard_normal(rng);
        let log_density = move |t: f64| -((t - y / 2.0).powi(2)) / (2.0 * 0.5);

        let sampler = AdaptiveMH::new(0.44, 2_000, 4_000);
        let result = sampler.sample(log_density, 0.0);

        let draws: Vec<f64> = result.samples.iter().step_by(40).copied().collect();
        if draws.len() < 20 {
            None
        } else {
            Some((theta, draws))
        }
    });

    assert!(
        report.is_calibrated(0.001),
        "AdaptiveMH is miscalibrated — {}",
        report.summary()
    );
}

/// The sampler's posterior mean must match the analytic one.
///
/// A coarser check than SBC, kept because it localises a failure: if this
/// passes and SBC fails, the chain is finding the right centre and the wrong
/// spread.
#[test]
fn sampler_recovers_the_analytic_posterior_mean_and_variance() {
    let y = 3.0;
    let log_density = move |t: f64| -((t - y / 2.0).powi(2)) / (2.0 * 0.5);

    let sampler = MetropolisHastings::new(1.0, 5_000, 40_000);
    let result = sampler.sample(log_density, 0.0);

    #[allow(clippy::cast_precision_loss)]
    let n = result.samples.len() as f64;
    let mean = result.samples.iter().sum::<f64>() / n;
    let var = result
        .samples
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    assert!(
        (mean - 1.5).abs() < 0.05,
        "posterior mean should be y/2 = 1.5, got {mean:.4}"
    );
    assert!(
        (var - 0.5).abs() < 0.05,
        "posterior variance should be 0.5, got {var:.4}"
    );
    // Optimal acceptance for a 1-D random walk is ~0.44, falling to ~0.234 in
    // high dimensions. The workable band is wide: what matters is that the
    // chain is neither rejecting almost everything (proposal far too large) nor
    // accepting almost everything (proposal far too small, so it barely moves).
    // With proposal_std = 1.0 against a posterior sd of 0.707 the measured rate
    // is ~0.61 — slightly conservative, and fine.
    assert!(
        (0.20..0.75).contains(&result.acceptance_rate),
        "acceptance rate {:.3} is outside a workable band for a 1-D random walk",
        result.acceptance_rate
    );
}

/// Draw from a Poisson distribution by inversion.
///
/// Only used to generate test data, so clarity beats speed. Guarded against the
/// unbounded loop that inversion invites when `lambda` is large.
fn poisson(rng: &mut rand_chacha::ChaCha8Rng, lambda: f64) -> u64 {
    let l = (-lambda).exp();
    let mut k = 0u64;
    let mut p = 1.0;
    loop {
        p *= rng.random::<f64>();
        if p <= l || k > 10_000 {
            return k;
        }
        k += 1;
    }
}

/// Local extension trait so the seeding call reads the same in every test.
trait SeedExt {
    fn from_seed_u64(seed: u64) -> Self;
}

impl SeedExt for rand_chacha::ChaCha8Rng {
    fn from_seed_u64(seed: u64) -> Self {
        use rand::SeedableRng;
        Self::seed_from_u64(seed)
    }
}

// ===========================================================================
// B1 regression guards — the shipped interval must BE the exact one
// ===========================================================================

/// The shipped Beta interval must agree with an independently written exact
/// reference across the whole parameter range.
///
/// This is the guard that actually catches a regression. Coverage tests are
/// noisy and depart from nominal at small `n` for reasons no implementation
/// controls — binomial discreteness — so a coverage assertion has to be loose
/// enough to be a weak check. Agreement with a reference is exact, so it can be
/// asserted to 1e-9.
///
/// `cynepic-bayes::special` and `cynepic-testkit::calibration` implement the
/// incomplete beta separately. Both are independently checked against the
/// closed forms for `Beta(1,1)`, `Beta(3,1)` and `Beta(1,3)`; requiring them to
/// agree here catches transcription errors that the closed-form cases would
/// miss, since those only exercise integer parameters.
#[test]
fn b1_shipped_interval_matches_the_exact_reference() {
    for successes in [0u64, 1, 3, 7, 20, 200] {
        for failures in [0u64, 1, 3, 7, 20, 200] {
            let mut model = BetaBinomial::uniform();
            model.update(successes, failures);
            let (lo, hi) = model.credible_interval_95();

            #[allow(clippy::cast_precision_loss)]
            let (a, b) = (1.0 + successes as f64, 1.0 + failures as f64);
            let (ref_lo, ref_hi) = beta_credible_interval(a, b, 0.95);

            assert!(
                (lo - ref_lo).abs() < 1e-9 && (hi - ref_hi).abs() < 1e-9,
                "Beta({a},{b}): shipped [{lo:.9}, {hi:.9}] vs reference \
                 [{ref_lo:.9}, {ref_hi:.9}]"
            );
        }
    }
}

/// The interval must stay inside `[0, 1]` without clamping.
///
/// The normal approximation satisfied this only by clamping, which discards
/// tail mass rather than placing it correctly — the interval stayed in range
/// and stopped meaning what it claimed. A quantile-based interval is in range
/// by construction.
#[test]
fn b1_interval_is_in_range_without_clamping() {
    for (s, f) in [(0u64, 50u64), (50, 0), (0, 0), (1, 1), (500, 1)] {
        let mut model = BetaBinomial::uniform();
        model.update(s, f);
        let (lo, hi) = model.credible_interval_95();
        assert!((0.0..=1.0).contains(&lo), "{s}/{f}: lo = {lo}");
        assert!((0.0..=1.0).contains(&hi), "{s}/{f}: hi = {hi}");
        assert!(lo < hi, "{s}/{f}: empty interval");
        // Clamping shows up as an endpoint sitting exactly on the boundary
        // while the posterior still has mass beyond it.
        assert!(hi < 1.0 || f == 0, "{s}/{f}: upper endpoint pinned at 1.0");
        assert!(lo > 0.0 || s == 0, "{s}/{f}: lower endpoint pinned at 0.0");
    }
}

/// The interval must be asymmetric when the counts are.
///
/// The property the normal approximation could not have: `Beta(a, b)` is
/// symmetric only when `a == b`, so an interval that is symmetric around the
/// mean for unbalanced counts is in the wrong place regardless of its width.
#[test]
fn b1_interval_is_asymmetric_when_the_posterior_is() {
    let mut model = BetaBinomial::uniform();
    model.update(2, 20);
    let (lo, hi) = model.credible_interval_95();
    let mean = model.mean();

    let left = mean - lo;
    let right = hi - mean;
    assert!(
        right > left * 1.15,
        "Beta(3,21) is right-skewed, so the upper arm should be visibly longer: \
         left {left:.4}, right {right:.4}"
    );
}

/// `GammaPoisson` gained an interval; it must be calibrated too.
#[test]
fn gamma_poisson_credible_intervals_are_calibrated() {
    let report = credible_coverage("GammaPoisson", 4_000, 0.95, 909, |rng| {
        // Draw the rate from the model's own prior, Gamma(3, 1).
        let lambda = sample_gamma(rng, 3.0);
        let obs: Vec<u64> = (0..20).map(|_| poisson(rng, lambda)).collect();

        let mut model = GammaPoisson::new(3.0, 1.0).expect("valid prior");
        model.update(&obs);
        let (lo, hi) = model.credible_interval_95();
        Some((lambda, model.mean(), lo, hi))
    });

    assert!(
        report.coverage_ok(0.03),
        "GammaPoisson intervals are miscalibrated — {}",
        report.summary()
    );
}

// ===========================================================================
// B2 — samplers were not reproducible · CLOSED
// ===========================================================================

/// The same seed must produce a byte-identical chain.
///
/// The samplers previously drew from `rand::rng()`, the thread-local
/// entropy-seeded generator, so running the same sampler twice on the same
/// log-density gave different answers with no way to recover the first. An MCMC
/// result that cannot be reproduced cannot be audited, and every other random
/// component in this workspace — `cynepic-testkit`'s DGPs, `cynepic-causal`'s
/// refuters — was already seeded.
///
/// It was also the source of RUSTSEC-2026-0097: `rand::rng()` is unsound when a
/// custom `log` logger reaches back into it during reseeding. Not using the
/// thread-local generator sidesteps the class entirely.
#[test]
fn b2_the_same_seed_reproduces_the_chain() {
    let density = |x: f64| -0.5 * x * x;

    let a = MetropolisHastings::new(1.0, 100, 500)
        .with_seed(42)
        .sample(density, 0.0);
    let b = MetropolisHastings::new(1.0, 100, 500)
        .with_seed(42)
        .sample(density, 0.0);

    assert_eq!(a.samples.len(), b.samples.len());
    for (i, (x, y)) in a.samples.iter().zip(&b.samples).enumerate() {
        assert!(
            (x - y).abs() < f64::EPSILON,
            "chains diverged at draw {i}: {x} vs {y}"
        );
    }
    assert!((a.acceptance_rate - b.acceptance_rate).abs() < f64::EPSILON);
}

/// Different seeds must produce different chains.
///
/// The other half: a "seeded" sampler that ignores its seed would satisfy the
/// test above trivially.
#[test]
fn b2_different_seeds_give_different_chains() {
    let density = |x: f64| -0.5 * x * x;

    let a = MetropolisHastings::new(1.0, 100, 500)
        .with_seed(1)
        .sample(density, 0.0);
    let b = MetropolisHastings::new(1.0, 100, 500)
        .with_seed(2)
        .sample(density, 0.0);

    let identical = a
        .samples
        .iter()
        .zip(&b.samples)
        .all(|(x, y)| (x - y).abs() < f64::EPSILON);
    assert!(!identical, "two seeds produced the same chain");
}

/// Seeding must apply to every sampler, not just the simplest one.
#[test]
fn b2_all_samplers_are_seedable() {
    let density = |x: f64| -0.5 * x * x;
    let multi = |v: &[f64]| -0.5 * (v[0] * v[0] + v[1] * v[1]);

    let a1 = AdaptiveMH::new(0.44, 100, 300)
        .with_seed(7)
        .sample(density, 0.0);
    let a2 = AdaptiveMH::new(0.44, 100, 300)
        .with_seed(7)
        .sample(density, 0.0);
    assert_eq!(a1.samples, a2.samples, "AdaptiveMH is not reproducible");

    let m1 = MultiDimMH::new(vec![1.0, 1.0], 100, 300)
        .with_seed(7)
        .sample(multi, vec![0.0, 0.0]);
    let m2 = MultiDimMH::new(vec![1.0, 1.0], 100, 300)
        .with_seed(7)
        .sample(multi, vec![0.0, 0.0]);
    assert_eq!(m1.samples, m2.samples, "MultiDimMH is not reproducible");
}

/// An unseeded sampler must still work, and must still be random.
///
/// Reproducibility is opt-in rather than mandatory: a caller running many
/// independent chains wants independent chains.
#[test]
fn b2_unseeded_samplers_remain_random() {
    let density = |x: f64| -0.5 * x * x;
    let a = MetropolisHastings::new(1.0, 50, 200).sample(density, 0.0);
    let b = MetropolisHastings::new(1.0, 50, 200).sample(density, 0.0);
    assert_eq!(a.samples.len(), 200);
    let identical = a
        .samples
        .iter()
        .zip(&b.samples)
        .all(|(x, y)| (x - y).abs() < f64::EPSILON);
    assert!(!identical, "unseeded chains were identical");
}
