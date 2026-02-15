const std = @import("std");
const types = @import("types.zig");

pub const CircuitBreakerState = types.CircuitBreakerState;
pub const CircuitBreakerConfig = types.CircuitBreakerConfig;

/// Circuit breaker state machine.
pub const CircuitBreaker = struct {
    cb_state: CircuitBreakerState = .closed,
    failure_count: u32 = 0,
    success_count: u32 = 0,
    open_until_ns: u128 = 0,
    config: CircuitBreakerConfig,

    pub fn init(config: CircuitBreakerConfig) CircuitBreaker {
        return .{ .config = config };
    }

    pub fn recordSuccess(self: *CircuitBreaker) void {
        switch (self.cb_state) {
            .closed => {
                self.failure_count = 0;
            },
            .half_open => {
                self.success_count += 1;
                if (self.success_count >= self.config.half_open_max_requests) {
                    self.cb_state = .closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            },
            .open => {},
        }
    }

    pub fn recordFailure(self: *CircuitBreaker, now_ns: u128) void {
        self.failure_count += 1;
        switch (self.cb_state) {
            .closed => {
                if (self.failure_count >= self.config.failure_threshold) {
                    self.cb_state = .open;
                    self.open_until_ns = now_ns +
                        @as(u128, self.config.reset_timeout_ms) * std.time.ns_per_ms;
                }
            },
            .half_open => {
                self.cb_state = .open;
                self.open_until_ns = now_ns +
                    @as(u128, self.config.reset_timeout_ms) * std.time.ns_per_ms;
                self.success_count = 0;
            },
            .open => {},
        }
    }

    pub fn isAllowed(self: *CircuitBreaker, now_ns: u128) bool {
        switch (self.cb_state) {
            .closed => return true,
            .open => {
                if (now_ns >= self.open_until_ns) {
                    self.cb_state = .half_open;
                    self.success_count = 0;
                    return true;
                }
                return false;
            },
            .half_open => return true,
        }
    }

    pub fn reset(self: *CircuitBreaker) void {
        self.cb_state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.open_until_ns = 0;
    }
};
