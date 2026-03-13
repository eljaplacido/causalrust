use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Token-bucket rate limiter for policy-controlled actions.
///
/// Limits how frequently a specific action or actor can perform operations.
/// Each key gets its own independent token bucket.
#[derive(Debug)]
pub struct RateLimiter {
    /// Max tokens (burst capacity).
    max_tokens: u32,
    /// Token refill rate (tokens per second).
    refill_rate: f64,
    /// Per-key buckets.
    buckets: HashMap<String, TokenBucket>,
}

#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    max_tokens: u32,
    refill_rate: f64,
}

/// Result of a rate limit check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitDecision {
    /// The action is allowed. Shows remaining tokens after consumption.
    Allowed { remaining_tokens: u32 },
    /// The action is denied. Shows how long to wait before retrying.
    Denied { retry_after_ms: u64 },
}

impl TokenBucket {
    fn new(max_tokens: u32, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens as f64,
            last_refill: Instant::now(),
            max_tokens,
            refill_rate,
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed();
        let new_tokens = elapsed.as_secs_f64() * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.max_tokens as f64);
        self.last_refill = Instant::now();
    }

    /// Try to consume one token. Returns true if successful.
    fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Peek at remaining tokens without consuming.
    fn remaining(&mut self) -> u32 {
        self.refill();
        self.tokens as u32
    }

    /// Compute how long until one token is available (in milliseconds).
    fn retry_after_ms(&self) -> u64 {
        if self.tokens >= 1.0 {
            return 0;
        }
        let needed = 1.0 - self.tokens;
        let seconds = needed / self.refill_rate;
        (seconds * 1000.0).ceil() as u64
    }
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// - `max_tokens`: burst capacity per key
    /// - `refill_rate`: tokens added per second
    pub fn new(max_tokens: u32, refill_rate: f64) -> Self {
        Self {
            max_tokens,
            refill_rate,
            buckets: HashMap::new(),
        }
    }

    /// Get or create a bucket for the given key.
    fn bucket(&mut self, key: &str) -> &mut TokenBucket {
        self.buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(self.max_tokens, self.refill_rate))
    }

    /// Check if an action is allowed for the given key.
    /// Consumes one token if allowed.
    pub fn check(&mut self, key: &str) -> RateLimitDecision {
        let max_tokens = self.max_tokens;
        let refill_rate = self.refill_rate;
        let bucket = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(max_tokens, refill_rate));

        if bucket.try_consume() {
            RateLimitDecision::Allowed {
                remaining_tokens: bucket.tokens as u32,
            }
        } else {
            RateLimitDecision::Denied {
                retry_after_ms: bucket.retry_after_ms(),
            }
        }
    }

    /// Check without consuming a token (peek).
    pub fn peek(&mut self, key: &str) -> RateLimitDecision {
        let bucket = self.bucket(key);
        let remaining = bucket.remaining();
        if remaining >= 1 {
            RateLimitDecision::Allowed {
                remaining_tokens: remaining,
            }
        } else {
            RateLimitDecision::Denied {
                retry_after_ms: bucket.retry_after_ms(),
            }
        }
    }

    /// Get remaining tokens for a key.
    pub fn remaining(&mut self, key: &str) -> u32 {
        self.bucket(key).remaining()
    }

    /// Reset a specific key's bucket.
    pub fn reset_key(&mut self, key: &str) {
        self.buckets.remove(key);
    }

    /// Reset all buckets.
    pub fn reset_all(&mut self) {
        self.buckets.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn basic_allow() {
        let mut limiter = RateLimiter::new(3, 1.0);
        match limiter.check("user_1") {
            RateLimitDecision::Allowed { remaining_tokens } => {
                assert_eq!(remaining_tokens, 2);
            }
            _ => panic!("Expected Allowed"),
        }
    }

    #[test]
    fn deny_after_exhaustion() {
        let mut limiter = RateLimiter::new(2, 1.0);
        // Consume both tokens
        assert!(matches!(limiter.check("user_1"), RateLimitDecision::Allowed { .. }));
        assert!(matches!(limiter.check("user_1"), RateLimitDecision::Allowed { .. }));
        // Third request should be denied
        match limiter.check("user_1") {
            RateLimitDecision::Denied { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            _ => panic!("Expected Denied"),
        }
    }

    #[test]
    fn refill_after_time() {
        let mut limiter = RateLimiter::new(1, 1000.0); // very fast refill: 1000/sec
        // Consume the single token
        assert!(matches!(limiter.check("user_1"), RateLimitDecision::Allowed { .. }));
        // Should be denied immediately
        assert!(matches!(limiter.check("user_1"), RateLimitDecision::Denied { .. }));
        // Sleep a small amount to allow refill (1000 tokens/sec => 1ms = 1 token)
        std::thread::sleep(Duration::from_millis(5));
        // Should be allowed again after refill
        match limiter.check("user_1") {
            RateLimitDecision::Allowed { .. } => {}
            other => panic!("Expected Allowed after refill, got {:?}", other),
        }
    }
}
