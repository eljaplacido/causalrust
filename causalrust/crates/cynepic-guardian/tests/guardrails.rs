//! Measured guardrail behaviour.
//!
//! # Why a guardrail needs a different kind of test
//!
//! A guardrail's job is to say no. Its unit tests almost always check that it
//! says no in the obvious case, which is the case it was written for. What
//! matters in production is the shape of its *failure*: when it lets something
//! through that it should have stopped, and how much.
//!
//! So these tests are stated as properties over sequences of operations rather
//! than as single-scenario checks, and each one names the production
//! consequence of its failing. A guardrail that is right 95% of the time is not
//! 95% of a guardrail — the 5% is the entire reason it exists.
//!
//! # Asymmetry
//!
//! False-accept and false-reject are not equally bad and are never traded
//! evenly. A rate limiter that occasionally denies a legitimate call costs
//! latency; one that occasionally admits a call past its budget costs money
//! with no bound. A circuit breaker that stays open too long costs availability;
//! one that reopens too early costs the outage it exists to contain.
//!
//! Where the two conflict, these tests assert the conservative direction.

use std::sync::Arc;
use std::time::Duration;

use cynepic_guardian::circuit_breaker::CircuitBreaker;
use cynepic_guardian::loop_detector::{LoopDetector, LoopViolation};
use cynepic_guardian::rate_limiter::{RateLimitDecision, RateLimiter};

// ===========================================================================
// Circuit breaker
// ===========================================================================

/// The breaker must open at exactly the configured threshold, not before.
///
/// Opening early is a false positive that takes down a healthy dependency.
#[tokio::test]
async fn breaker_opens_at_the_threshold_and_not_before() {
    for threshold in [1u64, 2, 5, 10] {
        let breaker = CircuitBreaker::new(threshold, Duration::from_secs(60));

        for i in 1..threshold {
            breaker.record_failure().await;
            assert!(
                !breaker.is_open(),
                "threshold {threshold}: opened after {i} failures"
            );
            assert!(
                breaker.allow().await,
                "threshold {threshold}: blocked at {i}"
            );
        }

        breaker.record_failure().await;
        assert!(
            breaker.is_open(),
            "threshold {threshold}: did not open at {threshold} failures"
        );
    }
}

/// A success must clear accumulated failures.
///
/// Without this, a service with a 1% error rate trips eventually no matter how
/// healthy it is — the counter only ever goes up.
#[tokio::test]
async fn success_resets_the_failure_count() {
    let breaker = CircuitBreaker::new(3, Duration::from_secs(60));

    // Two failures, then a success, repeated. Never reaches three in a row.
    for _ in 0..20 {
        breaker.record_failure().await;
        breaker.record_failure().await;
        breaker.record_success();
        assert!(
            !breaker.is_open(),
            "breaker tripped on an intermittent failure pattern"
        );
    }
}

/// While open and inside the timeout, the breaker must block everything.
#[tokio::test]
async fn open_breaker_blocks_every_call_within_the_timeout() {
    let breaker = CircuitBreaker::new(2, Duration::from_secs(60));
    breaker.record_failure().await;
    breaker.record_failure().await;
    assert!(breaker.is_open());

    for i in 0..100 {
        assert!(
            !breaker.allow().await,
            "call {i} was admitted through an open breaker"
        );
    }
}

/// The breaker must be safe to share across tasks.
///
/// It is `Clone` and holds an `Arc`, so it is meant to be. A guardrail that
/// loses failures under concurrency under-counts exactly when load is highest.
#[tokio::test]
async fn concurrent_failures_are_all_counted() {
    let breaker = Arc::new(CircuitBreaker::new(50, Duration::from_secs(60)));

    let mut handles = Vec::new();
    for _ in 0..49 {
        let b = Arc::clone(&breaker);
        handles.push(tokio::spawn(async move { b.record_failure().await }));
    }
    for h in handles {
        h.await.expect("task panicked");
    }

    assert!(
        !breaker.is_open(),
        "49 concurrent failures tripped a threshold of 50; failures are being over-counted"
    );

    breaker.record_failure().await;
    assert!(
        breaker.is_open(),
        "50 failures did not trip a threshold of 50; failures are being lost under concurrency"
    );
}

// ===========================================================================
// G1 — half-open state · CLOSED
// ===========================================================================

/// After the reset timeout, the breaker must admit exactly ONE probe.
///
/// That is what half-open means, and it is the whole value of the pattern. The
/// implementation's comment says "Allow one attempt (half-open)", but nothing
/// counts the attempts: once `reset_timeout` has elapsed, `allow()` returns
/// `true` to every caller while `is_open()` still reports `true`.
///
/// The production consequence is precise and bad. A dependency goes down, the
/// breaker trips and shields it. The timeout elapses. Instead of one probe
/// deciding whether the dependency has recovered, **the entire backed-up load
/// arrives at once** — which is the thundering-herd failure the breaker was
/// installed to prevent, now delivered on a timer.
///
/// Measured before the fix: **100 of 100** calls admitted.
///
/// Fixed with a three-valued state atomic and a compare-and-exchange on the
/// `Open -> HalfOpen` transition, so exactly one caller can claim the probe. A
/// boolean cannot express this: two concurrent callers both read "timeout
/// elapsed" and, with nothing to claim, both proceed.
#[tokio::test]
async fn g1_half_open_admits_exactly_one_probe() {
    let breaker = CircuitBreaker::new(2, Duration::from_millis(50));
    breaker.record_failure().await;
    breaker.record_failure().await;
    assert!(breaker.is_open(), "precondition: the breaker is open");

    tokio::time::sleep(Duration::from_millis(80)).await;

    let mut admitted = 0;
    for _ in 0..100 {
        if breaker.allow().await {
            admitted += 1;
        }
    }

    assert_eq!(
        admitted, 1,
        "{admitted} of 100 calls were admitted after the reset timeout; \
         half-open must admit exactly one probe, or the whole backed-up load \
         arrives at once"
    );
}

/// A failed probe must re-open the breaker for another full timeout.
///
/// The other half of G1. This one passed even before the fix — `record_failure`
/// already restarted the trip clock — so it was never a witness for the
/// finding, and it runs as a guard rather than an accusation. Worth keeping:
/// the state machine could easily have broken it.
#[tokio::test]
async fn g1_failed_probe_reopens_the_breaker() {
    let breaker = CircuitBreaker::new(2, Duration::from_millis(50));
    breaker.record_failure().await;
    breaker.record_failure().await;

    tokio::time::sleep(Duration::from_millis(80)).await;

    // The probe is admitted, and fails.
    assert!(breaker.allow().await, "the probe should be admitted");
    breaker.record_failure().await;

    // Everything after it must be blocked again immediately.
    let mut admitted = 0;
    for _ in 0..50 {
        if breaker.allow().await {
            admitted += 1;
        }
    }
    assert_eq!(
        admitted, 0,
        "{admitted} calls admitted after the probe failed; the breaker must \
         re-open for a full timeout"
    );
}

// ===========================================================================
// Rate limiter
// ===========================================================================

/// Burst capacity must be exactly the configured maximum.
///
/// One token over is a budget breach; one under is a wrongly rejected call.
#[test]
fn burst_capacity_is_exactly_the_configured_maximum() {
    for max in [1u32, 5, 20, 100] {
        let mut limiter = RateLimiter::new(max, 0.0);
        let mut allowed = 0u32;
        for _ in 0..(max * 2 + 10) {
            if matches!(limiter.check("actor"), RateLimitDecision::Allowed { .. }) {
                allowed += 1;
            }
        }
        assert_eq!(
            allowed, max,
            "max_tokens {max} admitted {allowed} calls with no refill"
        );
    }
}

/// Buckets must be isolated per key.
///
/// A shared bucket means one noisy actor can exhaust everyone else's budget,
/// which is a denial of service delivered by the component meant to prevent it.
#[test]
fn keys_do_not_share_a_budget() {
    let mut limiter = RateLimiter::new(3, 0.0);

    for _ in 0..3 {
        assert!(matches!(
            limiter.check("noisy"),
            RateLimitDecision::Allowed { .. }
        ));
    }
    assert!(matches!(
        limiter.check("noisy"),
        RateLimitDecision::Denied { .. }
    ));

    // A different actor must be unaffected.
    for i in 0..3 {
        assert!(
            matches!(limiter.check("quiet"), RateLimitDecision::Allowed { .. }),
            "call {i} from an unrelated key was denied"
        );
    }
}

/// A denial must carry a usable retry hint.
///
/// A limiter that says "no" without saying "when" forces the caller to poll,
/// which converts a rate limit into a busy loop.
#[test]
fn denial_reports_a_finite_retry_delay() {
    let mut limiter = RateLimiter::new(1, 2.0);
    assert!(matches!(
        limiter.check("k"),
        RateLimitDecision::Allowed { .. }
    ));

    match limiter.check("k") {
        RateLimitDecision::Denied { retry_after_ms } => {
            assert!(
                retry_after_ms > 0 && retry_after_ms < 10_000,
                "retry_after_ms was {retry_after_ms}; at 2 tokens/sec it should be ~500"
            );
        }
        RateLimitDecision::Allowed { .. } => panic!("second call admitted with capacity 1"),
    }
}

/// `peek` must not consume.
///
/// If it did, a caller checking whether it may proceed would spend the budget
/// it was asking about.
#[test]
fn peek_does_not_consume_a_token() {
    let mut limiter = RateLimiter::new(2, 0.0);
    for _ in 0..10 {
        let _ = limiter.peek("k");
    }
    assert_eq!(
        limiter.remaining("k"),
        2,
        "peek consumed tokens; a check must not spend the budget it is checking"
    );
}

/// Refill must actually restore capacity, at roughly the configured rate.
#[test]
fn tokens_refill_at_approximately_the_configured_rate() {
    let mut limiter = RateLimiter::new(10, 100.0);
    for _ in 0..10 {
        let _ = limiter.check("k");
    }
    assert!(matches!(
        limiter.check("k"),
        RateLimitDecision::Denied { .. }
    ));

    std::thread::sleep(Duration::from_millis(60));

    // 100 tokens/sec for 60ms is about 6 tokens. Bounded generously on both
    // sides: the point is that refill happens and is neither absent nor
    // unbounded, not that the clock is precise.
    let mut allowed = 0;
    for _ in 0..10 {
        if matches!(limiter.check("k"), RateLimitDecision::Allowed { .. }) {
            allowed += 1;
        }
    }
    assert!(
        (2..=10).contains(&allowed),
        "after 60ms at 100 tokens/sec, {allowed} calls were admitted; expected ~6"
    );
}

// ===========================================================================
// Loop detector
// ===========================================================================

/// Overvisiting a node must be detected at the configured limit.
///
/// This is the guard against an agent burning a budget in a loop, so a missed
/// detection is measured in money.
#[test]
fn overvisit_is_detected_at_the_limit() {
    for limit in [2usize, 3, 10] {
        let mut detector = LoopDetector::new(limit, 100);
        let mut violation = None;
        for i in 0..=limit {
            if let Some(v) = detector.record_visit("stuck") {
                violation = Some((i, v));
                break;
            }
        }
        match violation {
            Some((i, LoopViolation::NodeOvervisited { .. })) => {
                assert!(
                    i + 1 >= limit,
                    "limit {limit} fired after only {} visits",
                    i + 1
                );
            }
            Some((_, other)) => panic!("limit {limit}: wrong violation {other:?}"),
            None => panic!("limit {limit}: {limit} visits to one node were not detected"),
        }
    }
}

/// A legitimate long path must not be flagged.
///
/// False positives here abort valid work, so the detector must distinguish a
/// long workflow from a stuck one.
#[test]
fn a_long_path_of_distinct_nodes_is_not_a_loop() {
    let mut detector = LoopDetector::new(3, 10);
    for i in 0..200 {
        let node = format!("step_{i}");
        assert!(
            detector.record_visit(&node).is_none(),
            "distinct node {node} was reported as a loop"
        );
    }
}

/// Two nodes alternating must be caught even though neither is overvisited
/// quickly.
///
/// The failure mode a naive visit counter misses: A-B-A-B-A-B trips no
/// per-node limit for a long time, but the agent is plainly stuck.
#[test]
fn alternation_between_two_nodes_is_detected() {
    let mut detector = LoopDetector::new(1_000, 4);
    let mut violation = None;
    for i in 0..40 {
        let node = if i % 2 == 0 { "a" } else { "b" };
        if let Some(v) = detector.record_visit(node) {
            violation = Some(v);
            break;
        }
    }
    assert!(
        matches!(violation, Some(LoopViolation::AlternationDetected { .. })),
        "A-B-A-B thrashing was not detected: {violation:?}"
    );
}

/// Reset must actually clear state.
///
/// A detector that keeps history across runs eventually flags every workflow.
#[test]
fn reset_clears_accumulated_history() {
    let mut detector = LoopDetector::new(3, 100);
    let _ = detector.record_visit("a");
    let _ = detector.record_visit("a");
    detector.reset();

    assert_eq!(detector.visit_count("a"), 0);
    assert!(detector.history().is_empty());
    // And the budget is genuinely restored.
    assert!(detector.record_visit("a").is_none());
    assert!(detector.record_visit("a").is_none());
}
