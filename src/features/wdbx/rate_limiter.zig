//! Token-bucket rate limiter for the WDBX REST API.
//!
//! Configurable capacity (max burst) and refill rate (tokens per second).
//! Thread-safe via internal mutex. Reads environment variables
//! `ABI_WDBX_RATE_LIMIT_CAPACITY` and `ABI_WDBX_RATE_LIMIT_REFILL` for
//! configuration, falling back to 100-cap / 10-tokens-per-second defaults.

const std = @import("std");
const foundation_time = @import("../../foundation/time.zig");
const env = @import("../../foundation/env.zig");

/// Simple spin-lock wrapping std.atomic.Mutex.
/// In this Zig version std.Thread.Mutex is not available, so we build
/// a minimal lock from the atomic tryLock/unlock primitives.
const Lock = struct {
    state: std.atomic.Mutex = .unlocked,

    pub fn lock(l: *Lock) void {
        while (!l.state.tryLock()) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(l: *Lock) void {
        l.state.unlock();
    }
};

pub const RATE_LIMIT_CAPACITY_ENV = env.WDBX_RATE_LIMIT_CAPACITY_ENV;
pub const RATE_LIMIT_REFILL_ENV = env.WDBX_RATE_LIMIT_REFILL_ENV;

/// Token-bucket rate limiter.
///
/// Each call to `acquire` consumes one token. Tokens refill continuously at
/// `refill_rate` per second up to `capacity`. If no tokens remain, the caller
/// is rate-limited (returns `false`).
pub const RateLimiter = struct {
    capacity: u64,
    tokens: u64,
    refill_rate: u64, // tokens per second
    last_refill: i64, // unix ms timestamp
    lock: Lock = .{},
    requests_allowed: u64 = 0,
    requests_denied: u64 = 0,

    pub fn init(capacity: u64, refill_rate: u64) RateLimiter {
        const now = foundation_time.unixMs();
        return .{
            .capacity = capacity,
            .tokens = capacity,
            .refill_rate = refill_rate,
            .last_refill = now,
        };
    }

    /// Initialize from environment variables, falling back to defaults.
    pub fn initFromEnv() RateLimiter {
        const cap = parseEnvInt(RATE_LIMIT_CAPACITY_ENV, 100);
        const rate = parseEnvInt(RATE_LIMIT_REFILL_ENV, 10);
        return init(cap, rate);
    }

    /// Attempt to acquire one token.
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn acquire(self: *RateLimiter) bool {
        self.lock.lock();
        defer self.lock.unlock();
        self.refillLocked();
        if (self.tokens > 0) {
            self.tokens -= 1;
            self.requests_allowed += 1;
            return true;
        }
        self.requests_denied += 1;
        return false;
    }

    /// Add tokens based on wall-clock time elapsed since last refill.
    fn refillLocked(self: *RateLimiter) void {
        const now = foundation_time.unixMs();
        const elapsed_ms = now - self.last_refill;
        if (elapsed_ms <= 0) return;
        const tokens_to_add = @as(u64, @intCast(elapsed_ms)) * self.refill_rate / 1000;
        if (tokens_to_add > 0) {
            self.tokens = @min(self.capacity, self.tokens + tokens_to_add);
            self.last_refill = now;
        }
    }

    /// Return a JSON key-value snippet for embedding in a stats response.
    /// Produces: `"rate_limit":{...}` (no outer braces).
    pub fn statsJson(self: *RateLimiter, allocator: std.mem.Allocator) ![]u8 {
        self.lock.lock();
        defer self.lock.unlock();
        return std.fmt.allocPrint(
            allocator,
            "\"rate_limit\":{{\"capacity\":{d},\"tokens\":{d},\"refill_rate\":{d},\"allowed\":{d},\"denied\":{d}}}",
            .{
                self.capacity,
                self.tokens,
                self.refill_rate,
                self.requests_allowed,
                self.requests_denied,
            },
        );
    }
};

fn parseEnvInt(key: []const u8, default: u64) u64 {
    const raw = env.get(key) orelse return default;
    return std.fmt.parseInt(u64, std.mem.trim(u8, raw, " \t\r\n"), 10) catch default;
}

test "RateLimiter: allows requests within capacity" {
    var rl = RateLimiter.init(5, 10);
    for (0..5) |_| try std.testing.expect(rl.acquire());
    // 6th request exceeds capacity (no time has elapsed for refill)
    try std.testing.expect(!rl.acquire());
}

test "RateLimiter: tracks allowed and denied counts" {
    var rl = RateLimiter.init(3, 10);
    _ = rl.acquire();
    _ = rl.acquire();
    _ = rl.acquire();
    _ = rl.acquire(); // denied — tokens exhausted
    try std.testing.expectEqual(@as(u64, 3), rl.requests_allowed);
    try std.testing.expectEqual(@as(u64, 1), rl.requests_denied);
}

test "RateLimiter: refills tokens over elapsed time" {
    var rl = RateLimiter.init(10, 10);
    for (0..10) |_| _ = rl.acquire();
    try std.testing.expect(!rl.acquire()); // drained

    // Simulate 500ms passing by winding last_refill back.
    rl.lock.lock();
    rl.last_refill -= 500;
    rl.lock.unlock();

    // Should have 5 tokens back (500 ms * 10/s = 5).
    for (0..5) |_| try std.testing.expect(rl.acquire());
    try std.testing.expect(!rl.acquire());
}

test "RateLimiter: env parsing uses defaults when variables are unset" {
    const rl = RateLimiter.initFromEnv();
    try std.testing.expect(rl.capacity > 0);
    try std.testing.expect(rl.refill_rate > 0);
}

test {
    std.testing.refAllDecls(@This());
}
