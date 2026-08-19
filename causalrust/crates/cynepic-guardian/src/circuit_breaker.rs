//! Circuit breaker with a real half-open state.
//!
//! # What changed and why
//!
//! The previous implementation held a single `is_open` boolean. Once
//! `reset_timeout` had elapsed, `allow()` returned `true` to **every** caller
//! while `is_open()` still reported `true`. Its comment said "Allow one attempt
//! (half-open)", but nothing counted the attempts — measured, 100 of 100 calls
//! were admitted (finding G1).
//!
//! The consequence is precise and bad. A dependency goes down, the breaker
//! trips and shields it, the timeout elapses — and instead of one probe
//! deciding whether the dependency has recovered, the entire backed-up load
//! arrives at once. That is the thundering herd the breaker exists to prevent,
//! now delivered on a timer.
//!
//! # The fix is a state machine, not a patch
//!
//! Admitting exactly one probe cannot be done with a boolean, because two
//! concurrent callers both read "timeout elapsed" and both proceed. It needs an
//! atomic claim: the state is a three-valued atomic, and the transition from
//! `Open` to `HalfOpen` is a compare-and-exchange. Exactly one caller wins that
//! exchange and becomes the probe; every other caller sees `HalfOpen` and is
//! turned away.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Which state the breaker is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation. All calls are admitted.
    Closed,
    /// Tripped. All calls are refused until `reset_timeout` elapses.
    Open,
    /// One probe is in flight. Every other call is refused until it reports.
    ///
    /// This is the state whose absence was finding G1. Without it there is
    /// nothing to distinguish "the timeout elapsed, let one through" from "the
    /// timeout elapsed, let everything through".
    HalfOpen,
}

// Encodings for the state atomic. Kept private: `BreakerState` is the API.
const CLOSED: u8 = 0;
const OPEN: u8 = 1;
const HALF_OPEN: u8 = 2;

/// Circuit breaker for guarding a failure-prone dependency.
///
/// Cheap to clone; clones share one state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    inner: Arc<CircuitBreakerInner>,
}

#[derive(Debug)]
struct CircuitBreakerInner {
    /// Current state, encoded as `CLOSED` / `OPEN` / `HALF_OPEN`.
    state: AtomicU8,
    /// Consecutive failures since the last success.
    failure_count: AtomicU64,
    /// Failures required to trip.
    failure_threshold: u64,
    /// How long to stay open before admitting a probe.
    reset_timeout: Duration,
    /// When the breaker last entered `Open`.
    last_tripped: Mutex<Option<Instant>>,
    /// Probes admitted since construction, for observability and tests.
    probes_admitted: AtomicU64,
}

impl CircuitBreaker {
    /// Create a breaker.
    ///
    /// - `failure_threshold` — consecutive failures before tripping.
    /// - `reset_timeout` — how long to stay open before admitting one probe.
    pub fn new(failure_threshold: u64, reset_timeout: Duration) -> Self {
        Self {
            inner: Arc::new(CircuitBreakerInner {
                state: AtomicU8::new(CLOSED),
                failure_count: AtomicU64::new(0),
                // A threshold of zero would trip before any failure and pin the
                // breaker open forever, which is never what a caller means.
                failure_threshold: failure_threshold.max(1),
                reset_timeout,
                last_tripped: Mutex::new(None),
                probes_admitted: AtomicU64::new(0),
            }),
        }
    }

    /// Whether this call may proceed.
    ///
    /// In `Open`, once `reset_timeout` has elapsed, **exactly one** caller is
    /// admitted as a probe and the breaker moves to `HalfOpen`. The claim is a
    /// compare-and-exchange, so concurrent callers cannot both win it.
    pub async fn allow(&self) -> bool {
        match self.inner.state.load(Ordering::SeqCst) {
            CLOSED => true,
            HALF_OPEN => false, // a probe is already in flight
            _ => {
                // Open. Admit a probe only if the timeout has elapsed.
                let elapsed = {
                    let last = self.inner.last_tripped.lock().await;
                    last.is_some_and(|t| t.elapsed() >= self.inner.reset_timeout)
                };
                if !elapsed {
                    return false;
                }

                // Claim the probe. Exactly one caller can win this.
                let won = self
                    .inner
                    .state
                    .compare_exchange(OPEN, HALF_OPEN, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok();
                if won {
                    self.inner.probes_admitted.fetch_add(1, Ordering::SeqCst);
                }
                won
            }
        }
    }

    /// Record a success.
    ///
    /// From `HalfOpen` this closes the breaker: the probe found the dependency
    /// healthy, so normal traffic resumes.
    pub fn record_success(&self) {
        self.inner.failure_count.store(0, Ordering::SeqCst);
        self.inner.state.store(CLOSED, Ordering::SeqCst);
    }

    /// Record a failure.
    ///
    /// From `HalfOpen` this re-opens the breaker for a **full** timeout: the
    /// probe found the dependency still down, so the load must stay shielded.
    /// From `Closed` it trips once the threshold is reached.
    pub async fn record_failure(&self) {
        let count = self.inner.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        let was_half_open = self.inner.state.load(Ordering::SeqCst) == HALF_OPEN;

        if was_half_open || count >= self.inner.failure_threshold {
            self.inner.state.store(OPEN, Ordering::SeqCst);
            // Restart the clock. Without this a failed probe would be followed
            // immediately by another, since the original trip time is already
            // older than the timeout.
            let mut last_tripped = self.inner.last_tripped.lock().await;
            *last_tripped = Some(Instant::now());

            if was_half_open {
                tracing::warn!(
                    failure_count = count,
                    "Circuit breaker probe failed; re-opening for a full timeout"
                );
            } else {
                tracing::warn!(
                    failure_count = count,
                    threshold = self.inner.failure_threshold,
                    "Circuit breaker tripped"
                );
            }
        }
    }

    /// The current state.
    pub fn state(&self) -> BreakerState {
        match self.inner.state.load(Ordering::SeqCst) {
            CLOSED => BreakerState::Closed,
            HALF_OPEN => BreakerState::HalfOpen,
            _ => BreakerState::Open,
        }
    }

    /// Whether the breaker is not admitting normal traffic.
    ///
    /// True in both `Open` and `HalfOpen`. `HalfOpen` counts as open because
    /// normal traffic is still refused there — only the single probe proceeds.
    pub fn is_open(&self) -> bool {
        self.inner.state.load(Ordering::SeqCst) != CLOSED
    }

    /// How many probes have been admitted since construction.
    ///
    /// Exposed because "how often did we retry a dead dependency" is the
    /// question this component exists to answer, and it was previously
    /// unanswerable.
    pub fn probes_admitted(&self) -> u64 {
        self.inner.probes_admitted.load(Ordering::SeqCst)
    }

    /// Consecutive failures since the last success.
    pub fn failure_count(&self) -> u64 {
        self.inner.failure_count.load(Ordering::SeqCst)
    }

    /// Force the breaker closed.
    pub fn reset(&self) {
        self.inner.failure_count.store(0, Ordering::SeqCst);
        self.inner.state.store(CLOSED, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn trips_after_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert!(cb.allow().await);

        cb.record_failure().await;
        cb.record_failure().await;
        assert!(cb.allow().await); // still under threshold

        cb.record_failure().await;
        assert!(cb.is_open());
        assert!(!cb.allow().await); // tripped
    }

    #[tokio::test]
    async fn resets_on_success() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(5));
        cb.record_failure().await;
        cb.record_failure().await;
        assert!(cb.is_open());

        cb.reset();
        assert!(!cb.is_open());
        assert!(cb.allow().await);
    }

    #[tokio::test]
    async fn half_open_admits_exactly_one_probe() {
        // G1. Previously every caller was admitted once the timeout elapsed.
        let cb = CircuitBreaker::new(2, Duration::from_millis(20));
        cb.record_failure().await;
        cb.record_failure().await;
        assert_eq!(cb.state(), BreakerState::Open);

        tokio::time::sleep(Duration::from_millis(40)).await;

        let mut admitted = 0;
        for _ in 0..100 {
            if cb.allow().await {
                admitted += 1;
            }
        }
        assert_eq!(admitted, 1, "{admitted} of 100 admitted");
        assert_eq!(cb.state(), BreakerState::HalfOpen);
        assert_eq!(cb.probes_admitted(), 1);
    }

    #[tokio::test]
    async fn concurrent_callers_cannot_both_win_the_probe() {
        // The reason a boolean cannot express this: two callers both observe
        // "timeout elapsed" and, without an atomic claim, both proceed.
        let cb = Arc::new(CircuitBreaker::new(1, Duration::from_millis(10)));
        cb.record_failure().await;
        tokio::time::sleep(Duration::from_millis(25)).await;

        let mut handles = Vec::new();
        for _ in 0..64 {
            let c = Arc::clone(&cb);
            handles.push(tokio::spawn(async move { c.allow().await }));
        }
        let mut admitted = 0;
        for h in handles {
            if h.await.expect("task panicked") {
                admitted += 1;
            }
        }
        assert_eq!(admitted, 1, "{admitted} of 64 concurrent callers admitted");
    }

    #[tokio::test]
    async fn successful_probe_closes_the_breaker() {
        let cb = CircuitBreaker::new(2, Duration::from_millis(20));
        cb.record_failure().await;
        cb.record_failure().await;
        tokio::time::sleep(Duration::from_millis(40)).await;

        assert!(cb.allow().await, "probe should be admitted");
        cb.record_success();

        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow().await);
        assert_eq!(cb.failure_count(), 0);
    }

    #[tokio::test]
    async fn failed_probe_reopens_for_a_full_timeout() {
        let cb = CircuitBreaker::new(2, Duration::from_millis(30));
        cb.record_failure().await;
        cb.record_failure().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(cb.allow().await, "probe should be admitted");
        cb.record_failure().await;
        assert_eq!(cb.state(), BreakerState::Open);

        // Immediately after, nothing is admitted: the clock restarted.
        for _ in 0..20 {
            assert!(!cb.allow().await);
        }

        // And after another full timeout, exactly one probe again.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut admitted = 0;
        for _ in 0..20 {
            if cb.allow().await {
                admitted += 1;
            }
        }
        assert_eq!(admitted, 1);
        assert_eq!(cb.probes_admitted(), 2);
    }

    #[tokio::test]
    async fn a_zero_threshold_does_not_pin_the_breaker_open() {
        // Clamped to 1. A literal zero would trip before any failure and never
        // admit anything, which is never what a caller means.
        let cb = CircuitBreaker::new(0, Duration::from_secs(5));
        assert!(cb.allow().await, "a fresh breaker must admit traffic");
        cb.record_failure().await;
        assert!(cb.is_open());
    }
}
