//! Estimator throughput across sample size and dimension.
//!
//! # Read this before quoting a number from here
//!
//! **Speed is only meaningful for an estimator that is correct.** Every
//! estimator benchmarked here is now validated by simulation — see
//! `docs/FINDINGS.md` and `examples/coverage_report`. That was not true when
//! this file was written: IPW then had 0.0% interval coverage, and its group
//! was named `ipw_BROKEN_C5_C13` so the label survived copy-paste.
//!
//! The rule that produced that name still stands. A benchmark row for an
//! estimator whose coverage is not measured must say so in the group name,
//! because a fast wrong answer has negative value: it arrives sooner and is
//! trusted more.
//!
//! The IPW group is now named plainly, and the numbers moved when it was
//! fixed — IRLS does more work per iteration than gradient descent and needs
//! far fewer of them. Keeping the benchmark across that change is how the
//! trade was measured rather than argued.
//!
//! # Why these are not run for time in CI
//!
//! GitHub's shared runners have neighbours. Wall-clock variance between runs on
//! identical code routinely exceeds the differences worth detecting, so a
//! timing gate there produces false alarms until people stop reading it. CI
//! builds these benchmarks (`cargo bench --no-run`) so they cannot rot, and
//! nothing more. Real numbers come from a quiet machine, recorded with the CPU
//! they were measured on.
//!
//! ```bash
//! cargo bench -p cynepic-causal
//! cargo bench -p cynepic-causal -- ols_adjusted   # one group
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cynepic_causal::dag::CausalDag;
use cynepic_causal::dsep::d_separated;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::identify::BackdoorCriterion;
use cynepic_testkit::Dgp;
use std::collections::HashSet;
use std::hint::black_box;

/// Sample sizes to sweep. Chosen to straddle the point where an O(n·p²) solve
/// stops being dominated by allocation.
const SIZES: [usize; 4] = [500, 2_000, 10_000, 50_000];

/// Covariate counts. `ols_adjusted` builds a `(p+2)²` normal-equation system,
/// so this is the axis that actually hurts.
const DIMS: [usize; 3] = [3, 10, 40];

/// Difference in means — the O(n) baseline every other estimator is paid for
/// improving on. Validated: metamorphic relations hold, variance scales in `n`.
fn difference_in_means(c: &mut Criterion) {
    let mut group = c.benchmark_group("difference_in_means");

    for n in SIZES {
        let data = Dgp::new().with_n(n).sample(1);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| {
                LinearATEEstimator::difference_in_means(
                    black_box(&d.treatment),
                    black_box(&d.outcome),
                )
                .expect("benchmark fixtures are well posed")
            });
        });
    }
    group.finish();
}

/// OLS with covariate adjustment, swept over both `n` and `p`.
///
/// Point estimates are validated across the standard DGP grid (bias < 0.02,
/// coverage 94.7%-97.3%). The rank-deficient path is not — see C1 — but no cell
/// here is rank deficient, so these timings describe the path that works.
fn ols_adjusted(c: &mut Criterion) {
    let mut group = c.benchmark_group("ols_adjusted");

    for p in DIMS {
        for n in SIZES {
            // The largest cell is a 50k x 40 solve on every iteration; skip it
            // rather than let one cell dominate the whole run.
            if n * p > 400_000 {
                continue;
            }
            let data = Dgp::new().with_n(n).with_p(p).sample(2);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(format!("p{p}"), n), &data, |b, d| {
                b.iter(|| {
                    LinearATEEstimator::ols_adjusted(
                        black_box(&d.treatment),
                        black_box(&d.outcome),
                        black_box(&d.covariates),
                    )
                    .expect("benchmark fixtures are well posed")
                });
            });
        }
    }
    group.finish();
}

/// IPW over an IRLS-fitted propensity model.
///
/// Cost is dominated by the propensity fit. IRLS solves a weighted least
/// squares problem per iteration — far more expensive than a gradient step —
/// but converges in under ten iterations rather than never, so it is both
/// faster in wall-clock and correct. That is an unusually clean trade and it is
/// worth keeping a benchmark on it.
fn ipw(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipw");

    for n in [500, 2_000, 10_000] {
        let data = Dgp::new().with_n(n).sample(3);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| {
                PropensityScoreEstimator::ipw(
                    black_box(&d.treatment),
                    black_box(&d.outcome),
                    black_box(&d.covariates),
                )
                .expect("benchmark fixtures have adequate overlap")
            });
        });
    }
    group.finish();
}

/// Build a chain DAG `X0 -> X1 -> ... -> Xn`, with a confounder over each pair.
/// Deliberately the shape that maximises path enumeration.
fn chain_dag(nodes: usize) -> CausalDag {
    let mut dag = CausalDag::new();
    for i in 0..nodes.saturating_sub(1) {
        // Chain plus a confounder over each consecutive pair. Acyclic by
        // construction, so these cannot fail.
        dag.add_edge(&format!("X{i}"), &format!("X{}", i + 1))
            .expect("chain is acyclic");
        dag.add_edge(&format!("U{i}"), &format!("X{i}"))
            .expect("confounder is acyclic");
        dag.add_edge(&format!("U{i}"), &format!("X{}", i + 1))
            .expect("confounder is acyclic");
    }
    dag
}

/// Graph algorithms. These scale with graph size, not sample size, and are the
/// operations an interactive tool calls on every edit — so latency here is
/// user-visible in a way the estimators' is not.
///
/// Both now return `Result`, so the benchmark measures the checked path —
/// including the name validation that closed finding C12.
fn graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph");

    for nodes in [10, 50, 200] {
        let dag = chain_dag(nodes);
        let last = format!("X{}", nodes - 1);
        let conditioning: HashSet<String> = (0..nodes / 2).map(|i| format!("U{i}")).collect();

        group.bench_with_input(BenchmarkId::new("d_separated", nodes), &dag, |b, d| {
            b.iter(|| {
                d_separated(
                    black_box(d),
                    black_box("X0"),
                    black_box(&last),
                    black_box(&conditioning),
                )
                .expect("all names exist in the fixture")
            });
        });

        group.bench_with_input(BenchmarkId::new("backdoor_find", nodes), &dag, |b, d| {
            // Identification can legitimately fail on this graph shape; the
            // benchmark measures the search either way.
            b.iter(|| {
                let _ = BackdoorCriterion::find(black_box(d), black_box("X0"), black_box(&last));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    difference_in_means,
    ols_adjusted,
    ipw,
    graph_operations
);
criterion_main!(benches);
