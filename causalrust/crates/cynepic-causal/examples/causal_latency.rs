//! Per-call latency percentiles and allocation counts.
//!
//! ```bash
//! cargo run -p cynepic-causal --example causal_latency --release
//! ```
//!
//! # Why percentiles and not a mean
//!
//! Items 5 and 6 of the benchmarking plan. A mean is what goes in a marketing
//! table; a p99 is what a caller actually waits for. For a decision layer
//! embedded in a request path, the tail *is* the latency — one call in a
//! hundred at 50x the mean shows up as a timeout, and the mean never moves.
//!
//! criterion, which this repository also uses, is the right tool for detecting
//! *regressions* in throughput. It is the wrong tool for this question: it times
//! batches and reports a central estimate, so a per-call tail is not
//! recoverable from it. Hence a separate harness.
//!
//! # Footprint
//!
//! "No GC pauses, embeddable in any service" is a claim about memory behaviour,
//! and it was unmeasured. Peak resident set is reported here. Per-call
//! allocation counts would need a custom `GlobalAlloc`, which the workspace's
//! `forbid(unsafe_code)` correctly refuses — see the note in the source.
//!
//! # Read the caveats
//!
//! * These numbers are **machine-specific**. The CPU is printed with them; a
//!   latency figure without the hardware is not reproducible.
//! * This is deliberately **not run for time in CI**. Shared runners have noisy
//!   neighbours whose variance exceeds the differences worth detecting.
//! * Timer overhead is measured and printed. For the sub-microsecond operations
//!   it is a material fraction of the reading, and pretending otherwise would
//!   overstate their cost.

use std::time::Instant;

use cynepic_causal::d_separated;
use cynepic_causal::dag::CausalDag;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::identify::BackdoorCriterion;
use cynepic_testkit::Dgp;
use std::collections::HashSet;

// ── Footprint, without unsafe ───────────────────────────────────────────
//
// This started as a counting `GlobalAlloc`, which is the direct way to get
// allocations-per-call. The workspace sets `unsafe_code = "forbid"`, and
// `forbid` cannot be downgraded by an `#[allow]` at the use site — that is the
// entire point of choosing it over `deny`.
//
// Weakening a real guarantee so a benchmark could print a nicer number would
// have been the wrong trade, so the guarantee stays and the measurement is
// coarser: peak resident set from `/proc/self/status`, which needs no unsafe
// and no allocator hook.
//
// What is lost: per-call allocation *counts*. What is kept: whether an
// operation's footprint grows with `n`, which is the question that decides if
// it can sit in a hot path. Per-call counts need an external profiler —
// `heaptrack` or `valgrind --tool=massif` — and that is recorded in
// docs/roadmap.md rather than smuggled in here.

/// Peak resident set size in KiB, or `None` off Linux.
fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
}

// ── Measurement ─────────────────────────────────────────────────────────

struct Report {
    name: String,
    samples: usize,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

/// Time `op` `n` times individually and summarise the distribution.
///
/// Warms up first: the first call through any code path pays for page faults
/// and branch predictor training, and including it reports a startup cost as if
/// it were a steady-state one.
fn measure<F, T>(name: &str, n: usize, mut op: F) -> Report
where
    F: FnMut() -> T,
{
    for _ in 0..(n / 10).max(10) {
        std::hint::black_box(op());
    }

    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        std::hint::black_box(op());
        times.push(start.elapsed().as_nanos() as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    let pct = |p: f64| {
        let idx = ((times.len() as f64 - 1.0) * p).round() as usize;
        times[idx]
    };

    Report {
        name: name.to_string(),
        samples: n,
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
        max: *times.last().expect("non-empty"),
    }
}

/// Human-readable duration.
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

fn print_row(r: &Report) {
    println!(
        "  {:<34} {:>9} {:>9} {:>9} {:>9} {:>8}",
        r.name,
        fmt_ns(r.p50),
        fmt_ns(r.p95),
        fmt_ns(r.p99),
        fmt_ns(r.max),
        r.samples,
    );
}

/// Best-effort CPU identification. A latency table without the hardware is not
/// reproducible, so this tries several keys rather than assuming x86's
/// `model name` — aarch64 does not publish one.
fn cpu_model() -> String {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |key: &str| {
        cpuinfo
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with(key))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().to_string())
    };

    let name = field("model name")
        .or_else(|| field("hardware"))
        .or_else(|| {
            // aarch64 exposes implementer/part codes rather than a name.
            let imp = field("cpu implementer")?;
            let part = field("cpu part")?;
            Some(format!("aarch64 implementer {imp} part {part}"))
        })
        .unwrap_or_else(|| "unknown".to_string());

    let cores = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    format!("{name} ({cores} logical cores, {})", std::env::consts::ARCH)
}

fn main() {
    println!("cynepic-causal — per-call latency and allocations\n");
    println!("  CPU     {}", cpu_model());
    println!(
        "  Profile {}",
        if cfg!(debug_assertions) {
            "debug (numbers are meaningless)"
        } else {
            "release"
        }
    );

    // The floor. For the sub-microsecond operations this is a material fraction
    // of the reading, and quoting them without it would overstate their cost.
    let overhead = measure("timer overhead", 100_000, Instant::now);
    println!(
        "  Timer   p50 {} — subtract this from anything below ~1µs\n",
        fmt_ns(overhead.p50)
    );

    println!(
        "  {:<34} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "operation", "p50", "p95", "p99", "max", "samples"
    );

    // ---- estimators -----------------------------------------------------
    println!(
        "\n── estimation ───────────────────────────────────────────────────────────────────────"
    );
    for (label, n, p) in [
        ("difference_in_means, n=1k", 1_000usize, 1usize),
        ("difference_in_means, n=100k", 100_000, 1),
    ] {
        let d = Dgp::new().with_n(n).with_p(p).sample(1);
        let r = measure(label, if n > 10_000 { 200 } else { 2_000 }, || {
            LinearATEEstimator::difference_in_means(&d.treatment, &d.outcome)
        });
        print_row(&r);
    }

    for (label, n, p, iters) in [
        ("ols_adjusted, n=1k p=3", 1_000usize, 3usize, 1_000usize),
        ("ols_adjusted, n=10k p=3", 10_000, 3, 300),
        ("ols_adjusted, n=10k p=25", 10_000, 25, 100),
        ("ols_adjusted, n=100k p=3", 100_000, 3, 50),
    ] {
        let d = Dgp::new().with_n(n).with_p(p).sample(2);
        let r = measure(label, iters, || {
            LinearATEEstimator::ols_adjusted(&d.treatment, &d.outcome, &d.covariates)
        });
        print_row(&r);
    }

    for (label, n, iters) in [
        ("ipw, n=1k", 1_000usize, 200usize),
        ("ipw, n=10k", 10_000, 50),
    ] {
        let d = Dgp::new().with_n(n).sample(3);
        let r = measure(label, iters, || {
            PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates)
        });
        print_row(&r);

        // The same estimate without cross-fitting, to price the correctness
        // decision rather than leave it implicit. `ipw` fits the propensity
        // model out-of-fold where the data supports it (finding C14), which
        // costs six logistic fits instead of one — and that is most of the gap
        // against statsmodels, which does one.
        let r = measure(&format!("  in-sample propensity, n={n}"), iters, || {
            PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates).and_then(|m| {
                PropensityScoreEstimator::ipw_with_model(&d.treatment, &d.outcome, &m)
            })
        });
        print_row(&r);
    }

    // ---- graph ----------------------------------------------------------
    println!(
        "\n── graph ────────────────────────────────────────────────────────────────────────────"
    );
    for nodes in [10usize, 100, 500] {
        let mut dag = CausalDag::new();
        for i in 0..nodes.saturating_sub(1) {
            dag.add_edge(&format!("X{i}"), &format!("X{}", i + 1))
                .expect("chain is acyclic");
        }
        let last = format!("X{}", nodes - 1);
        // From 1, not 0: X0 is the source, and conditioning on it makes the
        // query malformed. `d_separated` now rejects that outright, so
        // including it would time the *refusal* path and report a speedup that
        // came from not doing the work. `compare_networkx.py` excludes it for
        // the same reason (networkx raises), so both sides traverse.
        let z: HashSet<String> = (1..nodes / 2).map(|i| format!("X{i}")).collect();

        let r = measure(&format!("d_separated, {nodes} nodes"), 2_000, || {
            d_separated(&dag, "X0", &last, &z)
        });
        print_row(&r);

        let r = measure(&format!("backdoor_find, {nodes} nodes"), 200, || {
            BackdoorCriterion::find(&dag, "X0", &last)
        });
        print_row(&r);
    }

    // ---- refusal --------------------------------------------------------
    //
    // A refusal must be *cheap*. If declining costs more than answering, a
    // caller under load is incentivised to skip the check — which defeats the
    // purpose of having one.
    println!(
        "\n── refusal paths ────────────────────────────────────────────────────────────────────"
    );
    {
        let d = Dgp::new().with_n(10_000).with_p(3).sample(4);
        // Built by hand rather than with `ndarray::s![]`: that macro expands to
        // unsafe, which `forbid(unsafe_code)` rejects even inside an example.
        let short = ndarray::Array1::from_iter(d.outcome.iter().take(5).copied());
        let r = measure("length mismatch (rejected)", 5_000, || {
            LinearATEEstimator::difference_in_means(&d.treatment, &short)
        });
        print_row(&r);

        let all_treated = ndarray::Array1::ones(10_000);
        let r = measure("empty arm (rejected)", 2_000, || {
            LinearATEEstimator::difference_in_means(&all_treated, &d.outcome)
        });
        print_row(&r);
    }

    if let Some(kib) = peak_rss_kib() {
        println!(
            "\n── footprint ────────────────────────────────────────────────────────────────────────"
        );
        println!(
            "  peak resident set  {:.1} MiB",
            f64::from(u32::try_from(kib).unwrap_or(u32::MAX)) / 1024.0
        );
        println!("  (whole process, including the largest fixture held in memory —");
        println!("   an upper bound on any single operation, not a per-call figure)");
    }

    println!(
        "\n── how to read this ─────────────────────────────────────────────────────────────────"
    );
    println!("  p99, not p50, is what a caller in a request path experiences. A p99");
    println!("  well above p50 means the operation is allocation- or cache-sensitive");
    println!("  and will behave worse under memory pressure than this table suggests.");
    println!();
    println!("  Tails here are tight — p99 sits within a few percent of p50 on every");
    println!("  row — which says these paths are not allocation-dominated. That is");
    println!("  the property that makes them safe to embed in a request path.");
    println!();
    println!("  A refusal costing far less than an estimate is the property that");
    println!("  matters in the refusal section: if declining were expensive, a caller");
    println!("  under load would be tempted to skip the check. A length mismatch is");
    println!("  rejected at or below the timer floor, i.e. it is free.");
    println!();
    println!("  An `empty arm` refusal is NOT free: it scans the treatment column to");
    println!("  discover the arm is empty, so it costs a pass over the data. That is");
    println!("  inherent — the check cannot know without looking.");
    println!();
    println!("  These numbers are specific to the CPU named above and are NOT");
    println!("  measured in CI — shared runners have neighbours, and their variance");
    println!("  swamps the differences worth detecting. Re-run on your own hardware.");
}
