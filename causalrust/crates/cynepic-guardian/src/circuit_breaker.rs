use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Circuit breaker for LLM output interception.
///
/// Implements the circuit breaker pattern to prevent cascading failures
/// when an LLM or downstream service is misbehaving. The breaker has
/// three states: Closed (normal), Open (tripped), Half-Open (testing).
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    inner: Arc<CircuitBreakerInner>,
}

#[derive(Debug)]
struct CircuitBreakerInner {
    /// Whether the breaker is currently open (tripped).
    is_open: AtomicBool,
    /// Number of consecutive failures.
    failure_count: AtomicU64,
    /// Threshold before tripping.
    failure_threshold: u64,
    /// How long to stay open before retrying.
    reset_timeout: Duration,
    /// When the breaker was last tripped.
    last_tripped: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// - `failure_threshold`: number of consecutive failures before tripping
    /// - `reset_timeout`: how long the breaker stays open before allowing a retry
    pub fn new(failure_threshold: u64, reset_timeout: Duration) -> Self {
        Self {
            inner: Arc::new(CircuitBreakerInner {
                is_open: AtomicBool::new(false),
                failure_count: AtomicU64::new(0),
                failure_threshold,
                reset_timeout,
                last_tripped: Mutex::new(None),
            }),
        }
    }

    /// Check if the circuit breaker allows the operation.
    pub async fn allow(&self) -> bool {
        if !self.inner.is_open.load(Ordering::Relaxed) {
            return true;
        }

        // Check if reset timeout has elapsed (half-open state)
        let last_tripped = self.inner.last_tripped.lock().await;
        if let Some(tripped_at) = *last_tripped {
            if tripped_at.elapsed() >= self.inner.reset_timeout {
                return true; // Allow one attempt (half-open)
            }
        }

        false
    }

    /// Record a successful operation, resetting the failure count.
    pub fn record_success(&self) {
        self.inner.failure_count.store(0, Ordering::Relaxed);
        self.inner.is_open.store(false, Ordering::Relaxed);
    }

    /// Record a failed operation. Trips the breaker if threshold is exceeded.
    pub async fn record_failure(&self) {
        let count = self.inner.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= self.inner.failure_threshold {
            self.inner.is_open.store(true, Ordering::Relaxed);
            let mut last_tripped = self.inner.last_tripped.lock().await;
            *last_tripped = Some(Instant::now());
            tracing::warn!(
                failure_count = count,
                threshold = self.inner.failure_threshold,
                "Circuit breaker tripped"
            );
        }
    }

    /// Whether the breaker is currently open (tripped).
    pub fn is_open(&self) -> bool {
        self.inner.is_open.load(Ordering::Relaxed)
    }

    /// Reset the breaker to closed state.
    pub fn reset(&self) {
        self.inner.failure_count.store(0, Ordering::Relaxed);
        self.inner.is_open.store(false, Ordering::Relaxed);
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
}
