//! Resilience patterns for fault-tolerant systems.
//!
//! Provides shared implementations of common resilience patterns used
//! across feature modules (network, streaming, gateway).

pub const circuit_breaker = @import("circuit_breaker.zig");

pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const CircuitState = circuit_breaker.CircuitState;
pub const CircuitBreakerConfig = circuit_breaker.Config;
pub const CircuitBreakerStats = circuit_breaker.Stats;
pub const SyncStrategy = circuit_breaker.SyncStrategy;

pub const AtomicCircuitBreaker = circuit_breaker.AtomicCircuitBreaker;
pub const MutexCircuitBreaker = circuit_breaker.MutexCircuitBreaker;
pub const SimpleCircuitBreaker = circuit_breaker.SimpleCircuitBreaker;

test {
    _ = circuit_breaker;
}
