//! Resilience patterns for fault-tolerant systems.
//!
//! Provides shared implementations of common resilience patterns used
//! across feature modules (network, streaming, gateway).

pub const circuit_breaker = @import("circuit_breaker.zig");
pub const rate_limiter = @import("rate_limiter.zig");

pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const CircuitState = circuit_breaker.CircuitState;
pub const CircuitBreakerConfig = circuit_breaker.Config;
pub const CircuitBreakerStats = circuit_breaker.Stats;
pub const SyncStrategy = circuit_breaker.SyncStrategy;

pub const AtomicCircuitBreaker = circuit_breaker.AtomicCircuitBreaker;
pub const MutexCircuitBreaker = circuit_breaker.MutexCircuitBreaker;
pub const SimpleCircuitBreaker = circuit_breaker.SimpleCircuitBreaker;

pub const RateLimiter = rate_limiter.RateLimiter;
pub const RateLimiterAlgorithm = rate_limiter.Algorithm;
pub const RateLimiterConfig = rate_limiter.Config;
pub const RateLimiterResult = rate_limiter.Result;

pub const AtomicRateLimiter = rate_limiter.AtomicRateLimiter;
pub const MutexRateLimiter = rate_limiter.MutexRateLimiter;
pub const SimpleRateLimiter = rate_limiter.SimpleRateLimiter;

test {
    _ = circuit_breaker;
    _ = rate_limiter;
}
