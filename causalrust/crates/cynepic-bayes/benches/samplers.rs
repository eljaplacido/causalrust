//! MCMC sampler throughput, and the reason raw throughput is the wrong metric.
//!
//! # Samples per second is not the number you want
//!
//! An MCMC sampler that produces 10 million samples per second and mixes badly
//! is worse than one producing 100,000 that mixes well. Successive MCMC draws
//! are correlated, so what a sampler actually delivers is **effective sample
//! size** — the number of independent draws its correlated chain is worth.
//!
//! The honest cost metric is therefore *time per effective sample*, and the
//! `ess` group below reports it. The raw `throughput` group is included because
//! it isolates per-iteration cost, which is what changes when the proposal
//! mechanism is rewritten — but it must never be quoted on its own. Tuning a
//! proposal to be cheap is trivial if you do not care whether the chain moves.
//!
//! `AdaptiveMH` is the case in point: it does strictly more work per iteration
//! than `MetropolisHastings` and should look worse on `throughput` while
//! winning on `ess`. If it ever wins on both, the adaptation is not doing
//! anything and that is a bug.
//!
//! # Targets
//!
//! Three densities, chosen because they fail different things:
//!
//! - **Standard normal** — the easy case; anything should handle it.
//! - **Bimodal mixture** — separated modes. A random-walk proposal tuned for
//!   one mode will not cross to the other, and raw throughput cannot see that
//!   the chain never left where it started.
//! - **Correlated 2D Gaussian** (rho = 0.95) — a ridge. A diagonal proposal
//!   must take tiny steps to stay on it, so ESS collapses while throughput
//!   looks unchanged. This is the shape real posteriors have.
//!
//! ```bash
//! cargo bench -p cynepic-bayes
//! cargo bench -p cynepic-bayes -- ess
//! ```
//!
//! CI builds these but does not time them; see the note in
//! `cynepic-causal/benches/estimators.rs`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cynepic_bayes::sampler::{AdaptiveMH, MetropolisHastings, MultiDimMH};
use std::hint::black_box;

fn std_normal_log_density(x: f64) -> f64 {
    -0.5 * x * x
}

/// Two well-separated modes at -4 and +4.
fn bimodal_log_density(x: f64) -> f64 {
    let a = -0.5 * (x - 4.0).powi(2);
    let b = -0.5 * (x + 4.0).powi(2);
    // log(exp(a) + exp(b)), computed stably.
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Bivariate normal with correlation 0.95 — a narrow diagonal ridge.
fn correlated_log_density(v: &[f64]) -> f64 {
    let (x, y) = (v[0], v[1]);
    let rho = 0.95;
    let d = 1.0 - rho * rho;
    -(x * x - 2.0 * rho * x * y + y * y) / (2.0 * d)
}

/// Lag-1 autocorrelation, used to approximate effective sample size.
///
/// ESS = n * (1 - r) / (1 + r) is the AR(1) approximation. It understates ESS
/// for chains with structure beyond lag 1, which is the conservative direction:
/// it never flatters a sampler.
fn ess_ar1(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean = samples.iter().sum::<f64>() / n_f;

    let mut var = 0.0;
    let mut cov = 0.0;
    for i in 0..n {
        let d = samples[i] - mean;
        var += d * d;
        if i + 1 < n {
            cov += d * (samples[i + 1] - mean);
        }
    }
    if var <= 0.0 {
        // A chain that never moved. Zero effective samples, not one.
        return 0.0;
    }
    let r = (cov / var).clamp(-0.999, 0.999);
    n_f * (1.0 - r) / (1.0 + r)
}

/// Per-iteration cost. Necessary, not sufficient — read with `ess`.
fn throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for n in [1_000usize, 10_000] {
        group.bench_with_input(BenchmarkId::new("mh_normal", n), &n, |b, &n| {
            let s = MetropolisHastings::new(1.0, n / 10, n);
            b.iter(|| s.sample(black_box(std_normal_log_density), black_box(0.0)));
        });

        group.bench_with_input(BenchmarkId::new("adaptive_normal", n), &n, |b, &n| {
            let s = AdaptiveMH::new(0.44, n / 10, n);
            b.iter(|| s.sample(black_box(std_normal_log_density), black_box(0.0)));
        });

        group.bench_with_input(BenchmarkId::new("multidim_correlated", n), &n, |b, &n| {
            let s = MultiDimMH::new(vec![1.0, 1.0], n / 10, n);
            b.iter(|| s.sample(black_box(correlated_log_density), black_box(vec![0.0, 0.0])));
        });
    }
    group.finish();
}

/// Time per effective sample — the metric that decides which sampler to use.
///
/// Criterion times the whole run; dividing by the ESS the run produced gives
/// cost per independent draw. A sampler that stalls reports a very large number
/// here even when `throughput` says it is fast, which is the entire point.
fn cost_per_effective_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("ess");
    let n = 10_000usize;

    // Report the ESS each configuration achieves before timing it, so the
    // benchmark output can be read against the mixing it is paying for.
    let probes: [(&str, f64); 3] = [
        ("mh_normal", {
            let r = MetropolisHastings::new(1.0, n / 10, n).sample(std_normal_log_density, 0.0);
            ess_ar1(&r.samples)
        }),
        ("mh_bimodal", {
            let r = MetropolisHastings::new(1.0, n / 10, n).sample(bimodal_log_density, -4.0);
            ess_ar1(&r.samples)
        }),
        ("adaptive_bimodal", {
            let r = AdaptiveMH::new(0.44, n / 10, n).sample(bimodal_log_density, -4.0);
            ess_ar1(&r.samples)
        }),
    ];
    for (name, ess) in probes {
        println!(
            "  ess/{name}: {ess:.0} effective from {n} draws ({:.1}% efficiency)",
            100.0 * ess / n as f64
        );
    }

    group.bench_function("mh_normal", |b| {
        let s = MetropolisHastings::new(1.0, n / 10, n);
        b.iter(|| {
            let r = s.sample(black_box(std_normal_log_density), black_box(0.0));
            black_box(ess_ar1(&r.samples))
        });
    });

    // The mode-separation case. A random-walk proposal started at -4 will
    // usually never reach +4, so its ESS is a fraction of its sample count.
    group.bench_function("mh_bimodal", |b| {
        let s = MetropolisHastings::new(1.0, n / 10, n);
        b.iter(|| {
            let r = s.sample(black_box(bimodal_log_density), black_box(-4.0));
            black_box(ess_ar1(&r.samples))
        });
    });

    group.bench_function("adaptive_bimodal", |b| {
        let s = AdaptiveMH::new(0.44, n / 10, n);
        b.iter(|| {
            let r = s.sample(black_box(bimodal_log_density), black_box(-4.0));
            black_box(ess_ar1(&r.samples))
        });
    });

    group.finish();
}

criterion_group!(benches, throughput, cost_per_effective_sample);
criterion_main!(benches);
