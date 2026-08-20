//! Per-step dispatch latency, and our half of the README's LangGraph row.
//!
//! ```bash
//! cargo run -p cynepic-graph --example graph_latency --release
//! ```
//!
//! # What makes this comparison fair, and what would not have
//!
//! Of the four rows in the README's speedup table this is the one that survives
//! scrutiny as written: both sides build a graph of trivial nodes, compile it
//! once, and are then asked to dispatch. Neither side is doing user work, so
//! what is left is the framework's own overhead — which is the thing the row
//! claims to compare.
//!
//! Three ways it could have been rigged, and how each is avoided:
//!
//! * **Building the graph inside the timed region.** LangGraph's `compile()`
//!   does real work; calling it per iteration would report a setup cost as a
//!   per-step cost. Both sides build once and invoke many times.
//! * **Nodes that do something.** Any real work inside a node dilutes the
//!   dispatch difference toward 1x, which flatters whichever side is slower.
//!   Both sides increment an integer and nothing else.
//! * **Quoting per-graph rather than per-step.** A 100-node graph makes any
//!   framework look slow in absolute terms. The comparable unit is one step.
//!
//! The Python side lives in `scripts/compare_langgraph.py`, which prints the
//! versions it used. Run both on one machine or neither; a cross-machine ratio
//! is not a result.
//!
//! # The honest caveat
//!
//! `cynepic-graph` and LangGraph are not the same product. LangGraph carries
//! checkpointing middleware, a channel-based reducer model and interrupt
//! support through every step; some of what is measured here as overhead is
//! that machinery rather than waste. A dispatch-speed win is not a claim that
//! one replaces the other.

use std::sync::Arc;
use std::time::Instant;

use cynepic_graph::graph::StateGraph;
use cynepic_graph::node::{FnNode, NodeId};

/// Matches the LangGraph fixture: one integer, incremented once per node.
#[derive(Debug, Clone, Default)]
struct Counter {
    counter: u64,
}

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

/// Build a linear chain of `n` incrementing nodes.
fn chain(n: usize) -> StateGraph<Counter> {
    let mut g = StateGraph::new();
    for i in 0..n {
        g = g.add_node(Arc::new(FnNode::new(
            format!("n{i}"),
            |mut s: Counter| async move {
                s.counter += 1;
                Ok(s)
            },
        )));
    }
    for i in 0..n - 1 {
        g = g.add_edge(
            NodeId::new(format!("n{i}")),
            NodeId::new(format!("n{}", i + 1)),
        );
    }
    g.set_entry(NodeId::new("n0"))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    println!("cynepic-graph — per-step dispatch latency\n");
    println!(
        "  Profile {}",
        if cfg!(debug_assertions) {
            "debug (numbers are meaningless)"
        } else {
            "release"
        }
    );
    println!(
        "  Runtime tokio multi-thread, {} workers\n",
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
    );

    println!(
        "  {:>6} {:>11} {:>11} {:>14}",
        "nodes", "graph p50", "graph p99", "per step p50"
    );

    for (n, reps) in [(5usize, 4_000usize), (20, 2_000), (100, 500)] {
        let g = chain(n);
        g.validate().expect("chain is a valid graph");

        for _ in 0..20 {
            let out = g
                .execute(Counter::default(), n + 1)
                .await
                .expect("chain completes");
            std::hint::black_box(out);
        }

        let mut t = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let out = g
                .execute(Counter::default(), n + 1)
                .await
                .expect("chain completes");
            t.push(start.elapsed().as_nanos() as f64);
            assert_eq!(out.counter, n as u64, "every node must have run");
        }
        t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let at = |p: f64| t[((t.len() as f64 - 1.0) * p).round() as usize];
        let (p50, p99) = (at(0.50), at(0.99));

        println!(
            "  {n:>6} {:>11} {:>11} {:>14}",
            fmt_ns(p50),
            fmt_ns(p99),
            fmt_ns(p50 / n as f64)
        );
    }

    println!("\n── measured against LangGraph ───────────────────────────────");
    println!("  Same workload, same machine (aarch64), langgraph 1.2.11:");
    println!();
    println!("    nodes    langgraph/step    cynepic/step    speedup");
    println!("        5            66.3µs           550ns        120x");
    println!("       20            58.3µs           572ns        102x");
    println!("      100            74.9µs           392ns        191x");
    println!();
    println!("  The README assumed ~10x for this row. It is 100-191x, so this");
    println!("  one was UNDERSTATED by an order of magnitude — the opposite");
    println!("  direction from the NetworkX row, which was overstated by 50x.");
    println!("  Two plausible assumptions, both wrong, in opposite directions.");
    println!("  That is what un-instrumented figures look like in aggregate:");
    println!("  not biased, just noise dressed up as a claim.");
    println!();
    println!("  Note the shapes differ too. LangGraph's per-step cost is flat");
    println!("  in n; ours falls, because we amortise fixed setup over more");
    println!("  steps. Theirs is genuinely per-step overhead.");
    println!();
    println!("  Re-run scripts/compare_langgraph.py on this machine to refresh");
    println!("  the left column. It prints the versions it used.");

    println!("\n── what this does not say ───────────────────────────────────");
    println!("  LangGraph carries checkpointing, a channel-based reducer model");
    println!("  and interrupt support through every step. Some of the gap is");
    println!("  that machinery rather than waste, and a dispatch win is not a");
    println!("  claim that one replaces the other.");
}
