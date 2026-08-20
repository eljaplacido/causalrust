//! Guardrail latency, and our half of the README's OPA row.
//!
//! ```bash
//! cargo run -p cynepic-guardian --example latency_report --release --features rego
//! ```
//!
//! # The row this measures is real, but it does not measure what it looks like
//!
//! The README claims ~100x for "policy evaluation, OPA sidecar vs in-process
//! regorus". That is two claims wearing one number:
//!
//! 1. **Transport.** A sidecar is an HTTP round trip. Deleting it is a win, and
//!    it has nothing to do with which policy engine is better.
//! 2. **Engine.** regorus (Rust) against OPA's Go evaluator, on the same policy
//!    and the same input.
//!
//! Quoting only the combined figure credits the engine for a win that belongs
//! to the deployment topology. So `scripts/compare_opa.py` asks OPA for its own
//! `timer_rego_query_eval_ns`, which separates the two, and reports both.
//!
//! # The guardrail rows matter more than the policy row
//!
//! A circuit breaker, a rate limiter and a loop detector sit on *every* call by
//! construction — that is what makes them guardrails. If checking costs more
//! than the thing being checked, callers start sampling the check, and a
//! guardrail applied to 1 in 10 calls is not a guardrail.
//!
//! That is the question this file exists to answer, and it is a different one
//! from "are we faster than OPA".

use std::time::{Duration, Instant};

use cynepic_guardian::circuit_breaker::CircuitBreaker;
use cynepic_guardian::rate_limiter::RateLimiter;

#[cfg(feature = "rego")]
use cynepic_guardian::PolicyEvaluator;

/// The policy both sides evaluate. Kept in sync with `scripts/compare_opa.py`
/// by hand — if they drift, the comparison is meaningless, so it is short.
#[cfg(feature = "rego")]
const POLICY: &str = r#"
package policy

default allow := false

allow if {
    input.role == "admin"
}

allow if {
    input.role == "operator"
    input.amount < 1000
}
"#;

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

/// Time `op` `n` times individually; return (p50, p99) nanoseconds.
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

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    println!("cynepic-guardian — guardrail and policy latency\n");
    println!(
        "  Profile {}",
        if cfg!(debug_assertions) {
            "debug (numbers are meaningless)"
        } else {
            "release"
        }
    );
    let (floor, _) = measure(50_000, Instant::now);
    println!(
        "  Timer   p50 {} — subtract from anything below ~1µs\n",
        fmt_ns(floor)
    );
    println!("  {:<40} {:>10} {:>10}", "operation", "p50", "p99");

    println!("\n── always-on guardrails ─────────────────────────────────────────");
    {
        let cb = CircuitBreaker::new(5, Duration::from_secs(30));
        let (p50, p99) = measure(200_000, || cb.is_open());
        row("circuit_breaker is_open (closed)", p50, p99);

        let (p50, p99) = measure(200_000, || cb.record_success());
        row("circuit_breaker record_success", p50, p99);

        let (p50, p99) = measure(200_000, || cb.state());
        row("circuit_breaker state", p50, p99);

        let mut rl = RateLimiter::new(u32::MAX, 1e9);
        let (p50, p99) = measure(100_000, || rl.check("actor-1"));
        row("rate_limiter check (allowed)", p50, p99);
    }

    println!("\n  These sit on every call by construction. At the timer floor");
    println!("  they are free relative to anything they guard, which is the");
    println!("  property that lets them stay on rather than be sampled.");

    #[cfg(feature = "rego")]
    {
        use cynepic_guardian::policy::RegoPolicyEvaluator;
        use serde_json::json;

        println!("\n── rego policy evaluation ───────────────────────────────────────");
        let ev = RegoPolicyEvaluator::from_policy(POLICY).expect("policy compiles");
        let input = json!({"role": "operator", "amount": 250});

        // Async, so it cannot go through `measure`. Same shape, inline.
        for _ in 0..200 {
            std::hint::black_box(ev.evaluate("transfer", &input).await.expect("evaluates"));
        }
        let mut t = Vec::with_capacity(5_000);
        for _ in 0..5_000 {
            let start = Instant::now();
            std::hint::black_box(ev.evaluate("transfer", &input).await.expect("evaluates"));
            t.push(start.elapsed().as_nanos() as f64);
        }
        t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let at = |p: f64| t[((t.len() as f64 - 1.0) * p).round() as usize];
        row("rego evaluate (in-process)", at(0.50), at(0.99));

        println!("\n  Note this figure includes an `Engine::clone()` per call, which");
        println!("  the evaluator does so that concurrent callers cannot observe");
        println!("  each other's `set_input`. That is a correctness requirement,");
        println!("  not an oversight — but it means the number above is the cost");
        println!("  of a *safe* evaluation, not of regorus's evaluator alone.");
    }

    #[cfg(not(feature = "rego"))]
    println!("\n  (rego feature off — re-run with --features rego for the policy row)");

    println!("\n── measured against OPA ─────────────────────────────────────────");
    println!("  Same policy, same input, same machine (aarch64), opa 1.19.1:");
    println!();
    println!("    opa sidecar, round trip              320.1µs");
    println!("    opa timer_rego_query_eval_ns          17.6µs");
    println!("    └ transport + serialisation          302.4µs   (94.5%)");
    println!("    cynepic-guardian, in-process           5.6µs");
    println!();
    println!("  Which gives TWO ratios, and they are different claims:");
    println!();
    println!("    deployment  sidecar -> in-process       57x");
    println!("    engine      OPA's evaluator -> regorus   3.1x");
    println!();
    println!("  The README claims ~100x. 57x is the right order of magnitude —");
    println!("  the first assumption in that table that was — but almost none of");
    println!("  it is about the policy engine. 94.5% of what a caller waits for");
    println!("  is HTTP, JSON and a loopback hop. Deleting a sidecar is a real");
    println!("  and useful win; it is just not a claim about regorus.");
    println!();
    println!("  3.1x is the engine figure, and it is a LOWER bound in our favour:");
    println!("  our 5.6µs includes the Engine::clone() noted above, while OPA's");
    println!("  timer excludes its own request handling. Quoting 100x for the");
    println!("  engine would overstate a 3.1x result by more than thirty times.");
    println!();
    println!("  Reproduce with scripts/compare_opa.py --opa <path>.");
}
