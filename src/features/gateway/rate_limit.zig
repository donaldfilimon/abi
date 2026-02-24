//! Gateway-specific rate limiting (nanosecond-precision, histogram-based sliding window).
//!
//! Delegates to the shared `services/shared/resilience/rate_limiter.zig` for core
//! algorithms (token bucket, sliding window, fixed window). This module adapts
//! the gateway-specific `RateLimitConfig` to the shared `Config` format.
//!
//! For HTTP/API-level rate limiting with per-key tracking, bans, whitelist,
//! and auth integration, see `services/shared/security/rate_limit.zig`.
//! For per-connection rate limiting, see `features/network/rate_limiter.zig`.

const std = @import("std");
const types = @import("types.zig");
const shared_rl = @import("../../services/shared/resilience/rate_limiter.zig");

pub const RateLimitConfig = types.RateLimitConfig;
pub const RateLimitAlgorithm = types.RateLimitAlgorithm;
pub const RateLimitResult = types.RateLimitResult;

/// Shared core type (single-threaded / no sync — gateway is lock-guarded externally).
const CoreLimiter = shared_rl.SimpleRateLimiter;

/// Convert gateway-specific algorithm enum to shared algorithm enum.
fn toSharedAlgorithm(algo: RateLimitAlgorithm) shared_rl.Algorithm {
    return switch (algo) {
        .token_bucket => .token_bucket,
        .sliding_window => .sliding_window,
        .fixed_window => .fixed_window,
    };
}

/// Convert gateway RateLimitConfig to shared Config.
fn toSharedConfig(config: RateLimitConfig) shared_rl.Config {
    return .{
        .max_requests = config.requests_per_second,
        .window_ns = std.time.ns_per_s, // gateway always uses 1-second windows
        .algorithm = toSharedAlgorithm(config.algorithm),
        .burst_capacity = config.burst_size,
    };
}

/// Convert shared Result to gateway RateLimitResult.
fn toGatewayResult(result: shared_rl.Result) RateLimitResult {
    return switch (result) {
        .allowed => |info| .{
            .allowed = true,
            .remaining = info.remaining,
            .reset_after_ms = 0,
        },
        .denied => |info| .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intCast(info.retry_after_ns / std.time.ns_per_ms),
        },
    };
}

/// Rate limiter wrapping the 3 algorithm variants.
/// Preserves the existing gateway public API while delegating to shared core.
pub const RateLimiter = struct {
    core: CoreLimiter,

    pub fn init(config: RateLimitConfig, now_ns: u128) RateLimiter {
        return .{
            .core = CoreLimiter.init(toSharedConfig(config), now_ns),
        };
    }

    pub fn tryConsume(self: *RateLimiter, now_ns: u128) RateLimitResult {
        const result = self.core.acquire(now_ns);
        return toGatewayResult(result);
    }
};

test {
    std.testing.refAllDecls(@This());
}
