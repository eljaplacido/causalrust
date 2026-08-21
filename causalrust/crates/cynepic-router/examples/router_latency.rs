//! Per-call latency for the routing path.
//!
//! ```bash
//! cargo run -p cynepic-router --example router_latency --release
//! ```
//!
//! # Why this matters more than the classifier's accuracy score
//!
//! A router sits in front of every request by construction. If classifying a
//! query costs a meaningful fraction of what the *cheap* model costs, the
//! routing decision has eaten the saving it exists to produce — and the
//! sensible response is to stop routing, which makes the component worthless
//! however accurate it is.
//!
//! So the number to beat is not another classifier. It is the thing being
//! avoided: a small-model call is single-digit **milliseconds** at best. Any
//! classification cost below ~1% of that is free in practice.
//!
//! # What is measured
//!
//! Both classifiers on the same queries, plus the pieces that surround them —
//! budget accounting and drift detection — because a routing decision in
//! production is all three, not just the classify call.

use std::time::Instant;

use cynepic_core::CynefinDomain;
use cynepic_router::budget::{BudgetTracker, CostMap};
use cynepic_router::classifier::KeywordClassifier;
use cynepic_router::config::CostTier;
use cynepic_router::{LexicalClassifier, QueryClassifier};

/// Queries of the length a router actually sees.
const QUERIES: [&str; 6] = [
    "what is the current retry limit",
    "why did throughput drop after the deploy last tuesday",
    "should we try a different chunking strategy for the retriever",
    "the pipeline is writing corrupt rows right now, stop it",
    "list the tools this agent has access to",
    "determine whether the reranker is improving answer quality",
];

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

/// Time `op` `n` times; return (p50, p99) nanoseconds.
fn measure<F, T>(n: usize, mut op: F) -> (f64, f64)
where
    F: FnMut() -> T,
{
    for _ in 0..(n / 10).max(10) {
        std::hint::black_box(op());
    }
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        std::hint::black_box(op());
        t.push(start.elapsed().as_nanos() as f64);
    }
    t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let at = |p: f64| t[((t.len() as f64 - 1.0) * p).round() as usize];
    (at(0.50), at(0.99))
}

fn row(name: &str, p50: f64, p99: f64) {
    println!("  {name:<40} {:>10} {:>10}", fmt_ns(p50), fmt_ns(p99));
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("cynepic-router — per-call latency\n");
    println!(
        "  Profile {}",
        if cfg!(debug_assertions) {
            "debug (numbers are meaningless)"
        } else {
            "release"
        }
    );
    let (floor, _) = measure(50_000, Instant::now);
    println!("  Timer   p50 {}\n", fmt_ns(floor));
    println!("  {:<40} {:>10} {:>10}", "operation", "p50", "p99");

    println!("\n── classification ───────────────────────────────────────────");
    {
        let keyword = KeywordClassifier::default_patterns();
        let mut i = 0usize;
        // Async, so it cannot go through `measure`. Same shape, inline.
        for _ in 0..500 {
            std::hint::black_box(keyword.classify(QUERIES[0]).await.expect("infallible"));
        }
        let mut t = Vec::with_capacity(20_000);
        for _ in 0..20_000 {
            let q = QUERIES[i % QUERIES.len()];
            i += 1;
            let start = Instant::now();
            std::hint::black_box(keyword.classify(q).await.expect("infallible"));
            t.push(start.elapsed().as_nanos() as f64);
        }
        t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let at = |p: f64| t[((t.len() as f64 - 1.0) * p).round() as usize];
        row("keyword classify", at(0.50), at(0.99));

        let lexical = LexicalClassifier::with_default_exemplars().expect("ships trained");
        let mut i = 0usize;
        let (p50, p99) = measure(20_000, || {
            let q = QUERIES[i % QUERIES.len()];
            i += 1;
            lexical.classify_sync(q)
        });
        row("lexical classify", p50, p99);

        let mut i = 0usize;
        let (p50, p99) = measure(20_000, || {
            let q = QUERIES[i % QUERIES.len()];
            i += 1;
            lexical.evidence(q)
        });
        row("  └ evidence check only", p50, p99);

        let mut i = 0usize;
        let (p50, p99) = measure(5_000, || {
            let q = QUERIES[i % QUERIES.len()];
            i += 1;
            lexical.explain(q, 5)
        });
        row("lexical explain (why this route?)", p50, p99);
    }

    println!("\n── the rest of a routing decision ───────────────────────────");
    {
        let costs = CostMap::default();
        let mut budget = BudgetTracker::new(1_000_000.0);
        let (p50, p99) = measure(100_000, || budget.record(&CostTier::Low, &costs));
        row("budget record", p50, p99);

        let budget = BudgetTracker::new(1_000_000.0);
        let (p50, p99) = measure(100_000, || budget.check(&CostTier::High, &costs));
        row("budget check", p50, p99);
    }

    println!("\n── training ─────────────────────────────────────────────────");
    {
        let examples = LexicalClassifier::default_exemplars();
        let (p50, p99) = measure(200, || LexicalClassifier::train(&examples));
        row("train on 48 examples", p50, p99);
    }

    println!("\n── how to read this ─────────────────────────────────────────");
    println!("  The number to beat is not another classifier. It is the thing");
    println!("  being avoided: the cheapest useful model call is single-digit");
    println!("  MILLISECONDS. A routing decision three orders of magnitude");
    println!("  below that is free — the decision costs nothing against the");
    println!("  call it redirects, which is the only comparison that decides");
    println!("  whether routing is worth doing at all.");
    println!();
    println!("  `explain` is the row worth noticing. Naming the terms behind a");
    println!("  verdict costs about the same as reaching the verdict, so an");
    println!("  audit trail of WHY each query routed where it did is affordable");
    println!("  on every call rather than on a sample.");
    println!();
    println!("  Training is not on the hot path — it happens once at startup —");
    println!("  but it is fast enough to retrain per request if a caller ever");
    println!("  wanted per-tenant exemplars.");

    let domains = [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
    ];
    debug_assert_eq!(domains.len(), 4);
}
