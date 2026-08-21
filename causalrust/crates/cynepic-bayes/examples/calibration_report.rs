//! Print the calibration table for the conjugate priors and MCMC samplers.
//!
//! The Bayesian counterpart of `cynepic-causal`'s coverage report, and the same
//! kind of artifact: a measurement that can come back bad.
//!
//! ```bash
//! cargo run -p cynepic-bayes --example calibration_report --release
//! ```
//!
//! # Reading the output
//!
//! **Credible-interval coverage.** Draw a parameter from the prior, generate
//! data from it, form the 95% interval. The interval's own claim is that it
//! contains that parameter 95% of the time, and for a correct posterior that
//! holds exactly. `±` is the Monte Carlo standard error, so a deviation smaller
//! than about two of those is noise rather than a defect.
//!
//! **Simulation-based calibration.** For samplers, the rank of the true
//! parameter among posterior draws must be uniform. The diagnosis column reads
//! the histogram's *shape*, which names the fault: mass at both ends means the
//! posterior is too narrow (overconfident), mass in the middle means too wide,
//! a slope means biased.

use cynepic_bayes::priors::{BetaBinomial, NormalNormal};
use cynepic_bayes::sampler::{AdaptiveMH, MetropolisHastings, MultiDimMH};
use cynepic_testkit::calibration::{
    beta_credible_interval, credible_coverage, sample_beta, sample_standard_normal,
    simulation_based_calibration,
};
use rand::Rng;

fn main() {
    println!("cynepic-bayes — calibration\n");

    // ---------------------------------------------------------------- priors
    println!("── credible-interval coverage (nominal 95%) ─────────────────────");

    let normal = credible_coverage("NormalNormal (10 obs)", 4_000, 0.95, 101, |rng| {
        let theta = sample_standard_normal(rng);
        let obs: Vec<f64> = (0..10)
            .map(|_| theta + sample_standard_normal(rng))
            .collect();
        let mut model = NormalNormal::new(0.0, 1.0, 1.0).expect("valid prior");
        model.update(&obs);
        let (lo, hi) = model.credible_interval_95();
        Some((theta, model.mean(), lo, hi))
    });
    println!("  {}", normal.summary());

    // BetaBinomial across sample sizes. The normal approximation to a Beta is
    // worst where the posterior is most skewed, which is at small n and near
    // the boundary — so the sweep is the informative form, not a single cell.
    for n in [2u64, 5, 10, 50, 200] {
        let report = credible_coverage(
            &format!("BetaBinomial (n={n})"),
            8_000,
            0.95,
            200 + n,
            |rng| {
                let p = sample_beta(rng, 1.0, 1.0);
                let successes = (0..n).filter(|_| rng.random::<f64>() < p).count() as u64;
                let mut model = BetaBinomial::uniform();
                model.update(successes, n - successes);
                let (lo, hi) = model.credible_interval_95();
                Some((p, model.mean(), lo, hi))
            },
        );
        println!("  {}", report.summary());
    }

    // The boundary case the normal approximation is worst at: a prior
    // concentrated near p = 0, where a symmetric interval must overrun into
    // negative territory and be clamped.
    for (a, b, label) in [
        (1.0, 9.0, "BetaBinomial near p=0.1"),
        (9.0, 1.0, "BetaBinomial near p=0.9"),
    ] {
        let report = credible_coverage(label, 8_000, 0.95, 300, |rng| {
            let p = sample_beta(rng, a, b);
            let n = 10u64;
            let successes = (0..n).filter(|_| rng.random::<f64>() < p).count() as u64;
            let mut model = BetaBinomial::new(a, b).expect("valid prior");
            model.update(successes, n - successes);
            let (lo, hi) = model.credible_interval_95();
            Some((p, model.mean(), lo, hi))
        });
        println!("  {}", report.summary());
    }

    // Conditional coverage: hold the true p FIXED and repeat. This is the
    // question a tool-reliability monitor actually asks — the tool has one real
    // reliability, not a draw from a prior — and it is where a symmetric
    // approximation to a skewed Beta shows, because averaging over the prior
    // lets errors at low p cancel errors at high p. That cancellation is why
    // the marginal table above looked fine while the interval was wrong.
    //
    // Shown against the EXACT Beta quantile interval, because binomial interval
    // coverage oscillates with the discreteness of the data at small n — even a
    // perfect interval does not hit 95% at every p. Without the reference
    // column there is no way to tell "this approximation is wrong" from "any
    // interval behaves this way here", and claiming the former without checking
    // is exactly the kind of unsupported finding this project exists to avoid.
    println!("\n── conditional coverage at FIXED p, n=10 (nominal 95%) ──────────");
    println!(
        "       {:<22} {:>10}  {:>10}   {:>8}",
        "p", "shipped", "reference", "width"
    );
    for p_true in [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98] {
        let shipped = credible_coverage("shipped", 8_000, 0.95, 700, |rng| {
            let n = 10u64;
            let successes = (0..n).filter(|_| rng.random::<f64>() < p_true).count() as u64;
            let mut model = BetaBinomial::uniform();
            model.update(successes, n - successes);
            let (lo, hi) = model.credible_interval_95();
            Some((p_true, model.mean(), lo, hi))
        });
        let reference = credible_coverage("reference", 8_000, 0.95, 700, |rng| {
            let n = 10u64;
            let successes = (0..n).filter(|_| rng.random::<f64>() < p_true).count() as u64;
            #[allow(clippy::cast_precision_loss)]
            let (a, b) = (1.0 + successes as f64, 1.0 + (n - successes) as f64);
            let (lo, hi) = beta_credible_interval(a, b, 0.95);
            Some((p_true, a / (a + b), lo, hi))
        });

        // The comparison that matters is shipped vs reference, not shipped vs
        // 95%. Both columns depart from 95% at small n — that is the
        // discreteness of a binomial, which no interval method escapes — so
        // scoring against nominal alone would report a defect that is not one.
        let gap = (shipped.coverage - reference.coverage).abs();
        let flag = if gap < 0.02 { " ok " } else { "GAP " };
        println!(
            "  {flag} {:<22} {:>9.1}% {:>9.1}%   {:>8.4}",
            format!("p={p_true:.2}"),
            shipped.coverage * 100.0,
            reference.coverage * 100.0,
            shipped.mean_width
        );
    }

    // ------------------------------------------------------------- samplers
    println!("\n── simulation-based calibration ─────────────────────────────────");

    let mh = simulation_based_calibration("MetropolisHastings", 600, 12, 303, |rng| {
        let theta = sample_standard_normal(rng);
        let y = theta + sample_standard_normal(rng);
        let log_density = move |t: f64| -((t - y / 2.0).powi(2)) / (2.0 * 0.5);
        let result = MetropolisHastings::new(1.0, 2_000, 4_000).sample(log_density, 0.0);
        let draws: Vec<f64> = result.samples.iter().step_by(40).copied().collect();
        Some((theta, draws))
    });
    println!("  {}", mh.summary());

    let adaptive = simulation_based_calibration("AdaptiveMH", 600, 12, 404, |rng| {
        let theta = sample_standard_normal(rng);
        let y = theta + sample_standard_normal(rng);
        let log_density = move |t: f64| -((t - y / 2.0).powi(2)) / (2.0 * 0.5);
        let result = AdaptiveMH::new(0.44, 2_000, 4_000).sample(log_density, 0.0);
        let draws: Vec<f64> = result.samples.iter().step_by(40).copied().collect();
        Some((theta, draws))
    });
    println!("  {}", adaptive.summary());

    // Multi-dimensional, checked on its first coordinate. A diagonal proposal
    // on a correlated target is the case where a sampler most often looks fine
    // on its mean and is badly wrong on its spread.
    let multi = simulation_based_calibration("MultiDimMH (2D, rho=0)", 600, 12, 505, |rng| {
        let theta = [sample_standard_normal(rng), sample_standard_normal(rng)];
        let y = [
            theta[0] + sample_standard_normal(rng),
            theta[1] + sample_standard_normal(rng),
        ];
        let log_density = move |v: &[f64]| {
            -((v[0] - y[0] / 2.0).powi(2) + (v[1] - y[1] / 2.0).powi(2)) / (2.0 * 0.5)
        };
        let result =
            MultiDimMH::new(vec![1.0, 1.0], 2_000, 4_000).sample(log_density, vec![0.0, 0.0]);
        let draws: Vec<f64> = result.samples.iter().step_by(40).map(|s| s[0]).collect();
        Some((theta[0], draws))
    });
    println!("  {}", multi.summary());

    println!("\nA rank histogram that is not uniform is a correctness defect, not a");
    println!("tuning issue. The diagnosis column says which way it fails.");
}
