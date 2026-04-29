//! Token-bucket rate limiter for Discord API
const std = @import("std");
const foundation = @import("../../../foundation/mod.zig");

pub const TokenBucket = struct {
    capacity: f64,
    tokens: f64,
    refill_rate_per_sec: f64,
    last_update: i64,
    mutex: foundation.sync.Mutex = .{},

    pub fn init(capacity: f64, refill_rate: f64) TokenBucket {
        return .{
            .capacity = capacity,
            .tokens = capacity,
            .refill_rate_per_sec = refill_rate,
            .last_update = foundation.time.unixMs(),
        };
    }

    pub fn consume(self: *TokenBucket, amount: f64) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = foundation.time.unixMs();
        const delta = @as(f64, @floatFromInt(now - self.last_update)) / 1000.0;
        self.last_update = now;

        self.tokens = @min(self.capacity, self.tokens + delta * self.refill_rate_per_sec);

        if (self.tokens >= amount) {
            self.tokens -= amount;
            return true;
        }
        return false;
    }
};
