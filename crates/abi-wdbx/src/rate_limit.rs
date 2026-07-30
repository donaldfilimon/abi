//! Thread-safe token-bucket limiting for the loopback WDBX REST service.

use abi_foundation::env::{WDBX_RATE_LIMIT_CAPACITY, WDBX_RATE_LIMIT_REFILL};
use serde::Serialize;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Default maximum burst.
pub const DEFAULT_CAPACITY: u64 = 100;
/// Default refill rate in tokens per second.
pub const DEFAULT_REFILL_RATE: u64 = 10;

#[derive(Debug, Clone, Copy)]
struct State {
    tokens: u64,
    last_refill_ms: i64,
    allowed: u64,
    denied: u64,
}

/// Observable token-bucket counters embedded in `/stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RateLimitStats {
    /// Maximum burst.
    pub capacity: u64,
    /// Whole tokens currently available.
    pub tokens: u64,
    /// Tokens added per second.
    pub refill_rate: u64,
    /// Requests admitted since startup.
    pub allowed: u64,
    /// Requests rejected since startup.
    pub denied: u64,
}

/// A mutex-protected, whole-token bucket.
#[derive(Debug)]
pub struct RateLimiter {
    capacity: u64,
    refill_rate: u64,
    state: Mutex<State>,
}

impl RateLimiter {
    /// Create a full bucket at `now_ms`.
    #[must_use]
    pub fn new(capacity: u64, refill_rate: u64, now_ms: i64) -> Self {
        Self {
            capacity,
            refill_rate,
            state: Mutex::new(State {
                tokens: capacity,
                last_refill_ms: now_ms,
                allowed: 0,
                denied: 0,
            }),
        }
    }

    /// Create a bucket from the two canonical environment variables.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(
            env_u64(WDBX_RATE_LIMIT_CAPACITY, DEFAULT_CAPACITY),
            env_u64(WDBX_RATE_LIMIT_REFILL, DEFAULT_REFILL_RATE),
            unix_ms(),
        )
    }

    /// Consume one token using the system clock.
    pub fn acquire(&self) -> bool {
        self.acquire_at(unix_ms())
    }

    /// Consume one token at an injected timestamp.
    ///
    /// This is public so transport tests can prove refill semantics without
    /// sleeping.
    pub fn acquire_at(&self, now_ms: i64) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        refill(&mut state, self.capacity, self.refill_rate, now_ms);
        if state.tokens > 0 {
            state.tokens -= 1;
            state.allowed = state.allowed.saturating_add(1);
            true
        } else {
            state.denied = state.denied.saturating_add(1);
            false
        }
    }

    /// Snapshot the counters without refilling.
    #[must_use]
    pub fn stats(&self) -> RateLimitStats {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        RateLimitStats {
            capacity: self.capacity,
            tokens: state.tokens,
            refill_rate: self.refill_rate,
            allowed: state.allowed,
            denied: state.denied,
        }
    }
}

fn refill(state: &mut State, capacity: u64, refill_rate: u64, now_ms: i64) {
    let elapsed_ms = now_ms.saturating_sub(state.last_refill_ms);
    if elapsed_ms <= 0 {
        return;
    }
    let elapsed = u64::try_from(elapsed_ms).unwrap_or(u64::MAX);
    let to_add = elapsed.saturating_mul(refill_rate) / 1_000;
    if to_add > 0 {
        state.tokens = capacity.min(state.tokens.saturating_add(to_add));
        state.last_refill_ms = now_ms;
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_denial_and_counters_match_the_oracle() {
        let limiter = RateLimiter::new(2, 10, 1_000);
        assert!(limiter.acquire_at(1_000));
        assert!(limiter.acquire_at(1_000));
        assert!(!limiter.acquire_at(1_000));
        assert_eq!(
            limiter.stats(),
            RateLimitStats {
                capacity: 2,
                tokens: 0,
                refill_rate: 10,
                allowed: 2,
                denied: 1,
            }
        );
    }

    #[test]
    fn refill_advances_only_after_a_whole_token_is_available() {
        let limiter = RateLimiter::new(2, 10, 1_000);
        assert!(limiter.acquire_at(1_000));
        assert!(limiter.acquire_at(1_000));
        assert!(!limiter.acquire_at(1_099));
        assert!(limiter.acquire_at(1_100));
        assert!(!limiter.acquire_at(1_100));
    }
}
