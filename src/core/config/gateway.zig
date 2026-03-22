pub const RateLimitAlgorithm = enum { token_bucket, sliding_window, fixed_window };
pub const CircuitBreakerState = enum { closed, open, half_open };

pub const RateLimitConfig = struct {
    requests_per_second: u32 = 100,
    burst_size: u32 = 200,
    algorithm: RateLimitAlgorithm = .token_bucket,
};

pub const CircuitBreakerConfig = struct {
    failure_threshold: u32 = 5,
    reset_timeout_ms: u64 = 30_000,
    half_open_max_requests: u32 = 3,
};

pub const GatewayConfig = struct {
    max_routes: u32 = 256,
    default_timeout_ms: u64 = 30_000,
    rate_limit: RateLimitConfig = .{},
    circuit_breaker: CircuitBreakerConfig = .{},
    enable_access_log: bool = true,
    enable_response_transform: bool = false,

    pub fn defaults() GatewayConfig {
        return .{};
    }
};
