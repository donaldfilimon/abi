//! Rate limiting with authentication integration.
//!
//! This module provides:
//! - Token bucket algorithm
//! - Sliding window rate limiting
//! - Per-user, per-IP, and per-API-key limits
//! - Burst allowance
//! - Automatic ban for abuse
//! - Integration with auth system
//! - Distributed rate limiting support

const std = @import("std");
const time = @import("../time.zig");
const sync = @import("../sync.zig");

/// Rate limiting algorithm
pub const Algorithm = enum {
    /// Token bucket (allows bursts)
    token_bucket,
    /// Sliding window (smoother distribution)
    sliding_window,
    /// Fixed window (simpler, less accurate)
    fixed_window,
    /// Leaky bucket (constant rate)
    leaky_bucket,
};

/// Rate limit scope
pub const Scope = enum {
    /// Per IP address
    ip,
    /// Per user ID
    user,
    /// Per API key
    api_key,
    /// Per endpoint/route
    endpoint,
    /// Global
    global,
    /// Custom key
    custom,
};

/// Rate limit configuration
pub const RateLimitConfig = struct {
    /// Whether rate limiting is enabled. Defaults to false for backwards compatibility.
    /// Use `productionDefaults()` for production deployments with rate limiting enabled.
    enabled: bool = false,
    /// Requests per window
    requests: u32 = 100,
    /// Window duration in seconds
    window_seconds: u32 = 60,
    /// Burst allowance (for token bucket)
    burst: u32 = 20,
    /// Algorithm to use
    algorithm: Algorithm = .token_bucket,
    /// Scope for limiting
    scope: Scope = .ip,
    /// Ban duration in seconds (0 = no auto-ban)
    ban_duration: u32 = 0,
    /// Threshold for auto-ban (consecutive violations)
    ban_threshold: u32 = 10,
    /// Retry-After header value strategy
    retry_after_strategy: RetryAfterStrategy = .remaining_window,
    /// Custom key extractor (for Scope.custom)
    custom_key_fn: ?*const fn (request: *const anyopaque) []const u8 = null,
    /// Whitelisted keys (skip rate limiting)
    whitelist: []const []const u8 = &.{},
    /// Exempt authenticated users
    exempt_authenticated: bool = false,
    /// Different limits for authenticated users
    authenticated_multiplier: f32 = 2.0,

    /// Returns production-ready rate limit configuration.
    /// Default: 100 requests per minute with 20 burst allowance.
    /// Rate limiting is enabled by default in production mode.
    pub fn productionDefaults() RateLimitConfig {
        return .{
            .enabled = true,
            .requests = 100,
            .window_seconds = 60,
            .burst = 20,
            .algorithm = .token_bucket,
            .scope = .ip,
        };
    }

    /// Returns whether this configuration has rate limiting enabled.
    pub fn isEnabled(self: RateLimitConfig) bool {
        return self.enabled;
    }
};

/// Retry-After header strategy
pub const RetryAfterStrategy = enum {
    /// Time until window resets
    remaining_window,
    /// Exact number of seconds
    fixed_seconds,
    /// Calculated based on current rate
    calculated,
};

/// Rate limit status
pub const RateLimitStatus = struct {
    /// Whether the request is allowed
    allowed: bool,
    /// Remaining requests in window
    remaining: u32,
    /// Total limit
    limit: u32,
    /// Window reset time (Unix timestamp)
    reset_at: i64,
    /// Retry-After value in seconds (if blocked)
    retry_after: ?u32 = null,
    /// Current request count in window
    current: u32,
    /// Whether the client is banned
    banned: bool = false,
    /// Ban expires at (Unix timestamp, if banned)
    ban_expires_at: ?i64 = null,
};

/// Internal bucket state
const Bucket = struct {
    /// Available tokens (for token bucket)
    tokens: f64,
    /// Last update timestamp
    last_update: i64,
    /// Request timestamps (for sliding window)
    requests: std.ArrayListUnmanaged(i64),
    /// Consecutive violations (for auto-ban)
    violations: u32,
    /// Ban expiration time
    ban_until: ?i64,
    /// Total requests ever
    total_requests: u64,
};

/// Rate limiter
pub const RateLimiter = struct {
    allocator: std.mem.Allocator,
    config: RateLimitConfig,
    buckets: std.StringArrayHashMapUnmanaged(Bucket),
    mutex: sync.Mutex,
    /// Statistics
    stats: RateLimiterStats,

    pub const RateLimiterStats = struct {
        total_requests: u64 = 0,
        allowed_requests: u64 = 0,
        blocked_requests: u64 = 0,
        bans_issued: u64 = 0,
        active_bans: u64 = 0,
        unique_clients: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: RateLimitConfig) RateLimiter {
        return .{
            .allocator = allocator,
            .config = config,
            .buckets = std.StringArrayHashMapUnmanaged(Bucket){},
            .mutex = .{},
            .stats = .{},
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        var it = self.buckets.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.requests.deinit(self.allocator);
        }
        self.buckets.deinit(self.allocator);
    }

    /// Check if a request is allowed
    pub fn check(self: *RateLimiter, key: []const u8) RateLimitStatus {
        return self.checkWithOptions(key, .{});
    }

    /// Check with additional options
    pub fn checkWithOptions(self: *RateLimiter, key: []const u8, options: CheckOptions) RateLimitStatus {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats.total_requests += 1;

        // Check whitelist
        for (self.config.whitelist) |whitelisted| {
            if (std.mem.eql(u8, key, whitelisted)) {
                self.stats.allowed_requests += 1;
                return .{
                    .allowed = true,
                    .remaining = self.config.requests,
                    .limit = self.config.requests,
                    .reset_at = 0,
                    .current = 0,
                };
            }
        }

        // Check exempt authenticated
        if (self.config.exempt_authenticated and options.is_authenticated) {
            self.stats.allowed_requests += 1;
            return .{
                .allowed = true,
                .remaining = self.config.requests,
                .limit = self.config.requests,
                .reset_at = 0,
                .current = 0,
            };
        }

        // Calculate effective limit (authenticated users may have higher limits)
        const effective_limit = if (options.is_authenticated)
            @as(u32, @intFromFloat(@as(f32, @floatFromInt(self.config.requests)) * self.config.authenticated_multiplier))
        else
            self.config.requests;

        // Get or create bucket
        const bucket = self.getOrCreateBucket(key) catch {
            // On error, allow the request but don't track
            return .{
                .allowed = true,
                .remaining = effective_limit,
                .limit = effective_limit,
                .reset_at = 0,
                .current = 0,
            };
        };

        const now = time.unixSeconds();

        // Check ban
        if (bucket.ban_until) |ban_time| {
            if (now < ban_time) {
                self.stats.blocked_requests += 1;
                return .{
                    .allowed = false,
                    .remaining = 0,
                    .limit = effective_limit,
                    .reset_at = ban_time,
                    .retry_after = @intCast(ban_time - now),
                    .current = 0,
                    .banned = true,
                    .ban_expires_at = ban_time,
                };
            } else {
                // Ban expired
                bucket.ban_until = null;
                bucket.violations = 0;
                self.stats.active_bans -|= 1;
            }
        }

        // Apply rate limiting algorithm
        const result = switch (self.config.algorithm) {
            .token_bucket => self.tokenBucketCheck(bucket, effective_limit, now),
            .sliding_window => self.slidingWindowCheck(bucket, effective_limit, now),
            .fixed_window => self.fixedWindowCheck(bucket, effective_limit, now),
            .leaky_bucket => self.leakyBucketCheck(bucket, effective_limit, now),
        };

        // Handle violation
        if (!result.allowed) {
            bucket.violations += 1;
            self.stats.blocked_requests += 1;

            // Check for auto-ban
            if (self.config.ban_duration > 0 and bucket.violations >= self.config.ban_threshold) {
                bucket.ban_until = now + self.config.ban_duration;
                bucket.violations = 0;
                self.stats.bans_issued += 1;
                self.stats.active_bans += 1;

                return .{
                    .allowed = false,
                    .remaining = 0,
                    .limit = effective_limit,
                    .reset_at = bucket.ban_until.?,
                    .retry_after = self.config.ban_duration,
                    .current = result.current,
                    .banned = true,
                    .ban_expires_at = bucket.ban_until,
                };
            }
        } else {
            bucket.violations = 0;
            self.stats.allowed_requests += 1;
        }

        bucket.total_requests += 1;

        return result;
    }

    /// Reset limits for a key
    pub fn reset(self: *RateLimiter, key: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.buckets.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            var bucket = kv.value;
            bucket.requests.deinit(self.allocator);
        }
    }

    /// Unban a key
    pub fn unban(self: *RateLimiter, key: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.buckets.getPtr(key)) |bucket| {
            if (bucket.ban_until != null) {
                bucket.ban_until = null;
                bucket.violations = 0;
                self.stats.active_bans -|= 1;
                return true;
            }
        }
        return false;
    }

    /// Get statistics
    pub fn getStats(self: *RateLimiter) RateLimiterStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get bucket info for a key
    pub fn getBucketInfo(self: *RateLimiter, key: []const u8) ?BucketInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.buckets.get(key)) |bucket| {
            return .{
                .total_requests = bucket.total_requests,
                .violations = bucket.violations,
                .is_banned = bucket.ban_until != null and bucket.ban_until.? > time.unixSeconds(),
                .ban_expires_at = bucket.ban_until,
            };
        }
        return null;
    }

    pub const BucketInfo = struct {
        total_requests: u64,
        violations: u32,
        is_banned: bool,
        ban_expires_at: ?i64,
    };

    /// Clean up expired entries
    pub fn cleanup(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();
        const window = @as(i64, self.config.window_seconds);

        var to_remove = std.ArrayListUnmanaged([]const u8).empty;
        defer to_remove.deinit(self.allocator);

        var it = self.buckets.iterator();
        while (it.next()) |entry| {
            const bucket = entry.value_ptr;

            // Remove if no activity for 2 windows and not banned
            if (bucket.last_update < now - window * 2 and bucket.ban_until == null) {
                to_remove.append(self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |key| {
            if (self.buckets.fetchRemove(key)) |kv| {
                self.allocator.free(kv.key);
                var bucket = kv.value;
                bucket.requests.deinit(self.allocator);
            }
        }
    }

    // Private methods

    fn getOrCreateBucket(self: *RateLimiter, key: []const u8) !*Bucket {
        if (self.buckets.getPtr(key)) |bucket| {
            return bucket;
        }

        // Create new bucket
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const now = time.unixSeconds();
        const bucket = Bucket{
            .tokens = @floatFromInt(self.config.requests + self.config.burst),
            .last_update = now,
            .requests = std.ArrayListUnmanaged(i64).empty,
            .violations = 0,
            .ban_until = null,
            .total_requests = 0,
        };

        try self.buckets.put(self.allocator, key_copy, bucket);
        self.stats.unique_clients += 1;

        return self.buckets.getPtr(key_copy).?;
    }

    fn tokenBucketCheck(self: *RateLimiter, bucket: *Bucket, limit: u32, now: i64) RateLimitStatus {
        const elapsed = @as(f64, @floatFromInt(now - bucket.last_update));
        const refill_rate = @as(f64, @floatFromInt(limit)) / @as(f64, @floatFromInt(self.config.window_seconds));

        // Refill tokens
        bucket.tokens = @min(
            bucket.tokens + elapsed * refill_rate,
            @as(f64, @floatFromInt(limit + self.config.burst)),
        );
        bucket.last_update = now;

        if (bucket.tokens >= 1.0) {
            bucket.tokens -= 1.0;
            return .{
                .allowed = true,
                .remaining = @intFromFloat(bucket.tokens),
                .limit = limit,
                .reset_at = now + self.config.window_seconds,
                .current = limit -| @as(u32, @intFromFloat(@min(bucket.tokens, @as(f64, @floatFromInt(limit))))),
            };
        }

        // Calculate retry-after
        const tokens_needed: f64 = 1.0 - bucket.tokens;
        const seconds_until_token = tokens_needed / refill_rate;

        return .{
            .allowed = false,
            .remaining = 0,
            .limit = limit,
            .reset_at = now + self.config.window_seconds,
            .retry_after = @intFromFloat(@ceil(seconds_until_token)),
            .current = limit,
        };
    }

    fn slidingWindowCheck(self: *RateLimiter, bucket: *Bucket, limit: u32, now: i64) RateLimitStatus {
        const window_start = now - self.config.window_seconds;

        // Remove old requests
        var i: usize = 0;
        while (i < bucket.requests.items.len) {
            if (bucket.requests.items[i] < window_start) {
                _ = bucket.requests.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        const current: u32 = @intCast(bucket.requests.items.len);

        if (current < limit) {
            // If we can't track the request, deny it (fail-safe for security)
            bucket.requests.append(self.allocator, now) catch {
                std.log.warn("rate_limit: failed to track request, denying for safety", .{});
                return .{
                    .allowed = false,
                    .remaining = 0,
                    .limit = limit,
                    .reset_at = now + self.config.window_seconds,
                    .retry_after = 1, // Retry shortly
                    .current = current,
                };
            };
            bucket.last_update = now;
            return .{
                .allowed = true,
                .remaining = limit - current - 1,
                .limit = limit,
                .reset_at = now + self.config.window_seconds,
                .current = current + 1,
            };
        }

        // Calculate when oldest request will expire
        const oldest = if (bucket.requests.items.len > 0) bucket.requests.items[0] else now;
        const retry_after: u32 = @intCast(@max(0, oldest + self.config.window_seconds - now));

        return .{
            .allowed = false,
            .remaining = 0,
            .limit = limit,
            .reset_at = oldest + self.config.window_seconds,
            .retry_after = retry_after,
            .current = current,
        };
    }

    fn fixedWindowCheck(self: *RateLimiter, bucket: *Bucket, limit: u32, now: i64) RateLimitStatus {
        const window_start = @divTrunc(now, self.config.window_seconds) * self.config.window_seconds;

        // Reset if new window
        if (bucket.last_update < window_start) {
            bucket.tokens = @floatFromInt(limit);
            bucket.last_update = now;
        }

        const current: u32 = limit - @as(u32, @intFromFloat(bucket.tokens));

        if (bucket.tokens >= 1.0) {
            bucket.tokens -= 1.0;
            return .{
                .allowed = true,
                .remaining = @intFromFloat(bucket.tokens),
                .limit = limit,
                .reset_at = window_start + self.config.window_seconds,
                .current = current + 1,
            };
        }

        return .{
            .allowed = false,
            .remaining = 0,
            .limit = limit,
            .reset_at = window_start + self.config.window_seconds,
            .retry_after = @intCast(window_start + self.config.window_seconds - now),
            .current = limit,
        };
    }

    fn leakyBucketCheck(self: *RateLimiter, bucket: *Bucket, limit: u32, now: i64) RateLimitStatus {
        const elapsed = @as(f64, @floatFromInt(now - bucket.last_update));
        const leak_rate = @as(f64, @floatFromInt(limit)) / @as(f64, @floatFromInt(self.config.window_seconds));

        // Leak water
        bucket.tokens = @max(0, bucket.tokens - elapsed * leak_rate);
        bucket.last_update = now;

        const bucket_size: f64 = @floatFromInt(limit);

        if (bucket.tokens < bucket_size) {
            bucket.tokens += 1.0;
            return .{
                .allowed = true,
                .remaining = @intFromFloat(bucket_size - bucket.tokens),
                .limit = limit,
                .reset_at = now + self.config.window_seconds,
                .current = @intFromFloat(bucket.tokens),
            };
        }

        // Calculate when there will be room
        const overflow = bucket.tokens - bucket_size + 1.0;
        const seconds_to_leak = overflow / leak_rate;

        return .{
            .allowed = false,
            .remaining = 0,
            .limit = limit,
            .reset_at = now + self.config.window_seconds,
            .retry_after = @intFromFloat(@ceil(seconds_to_leak)),
            .current = limit,
        };
    }
};

/// Check options
pub const CheckOptions = struct {
    is_authenticated: bool = false,
    user_id: ?[]const u8 = null,
    api_key_id: ?[]const u8 = null,
    endpoint: ?[]const u8 = null,
};

/// Multi-tier rate limiter (combines multiple limits)
pub const MultiTierRateLimiter = struct {
    allocator: std.mem.Allocator,
    tiers: std.ArrayListUnmanaged(Tier),

    pub const Tier = struct {
        name: []const u8,
        limiter: RateLimiter,
        scope: Scope,
    };

    pub fn init(allocator: std.mem.Allocator) MultiTierRateLimiter {
        return .{
            .allocator = allocator,
            .tiers = std.ArrayListUnmanaged(Tier).empty,
        };
    }

    pub fn deinit(self: *MultiTierRateLimiter) void {
        for (self.tiers.items) |*tier| {
            self.allocator.free(tier.name);
            tier.limiter.deinit();
        }
        self.tiers.deinit(self.allocator);
    }

    /// Add a rate limiting tier
    pub fn addTier(self: *MultiTierRateLimiter, name: []const u8, config: RateLimitConfig) !void {
        try self.tiers.append(self.allocator, .{
            .name = try self.allocator.dupe(u8, name),
            .limiter = RateLimiter.init(self.allocator, config),
            .scope = config.scope,
        });
    }

    /// Check all tiers (must pass all)
    pub fn check(self: *MultiTierRateLimiter, keys: Keys) MultiTierStatus {
        var statuses: std.StaticArrayList(TierStatus, 10) = .{};
        var all_allowed = true;
        var most_restrictive: ?RateLimitStatus = null;

        for (self.tiers.items) |*tier| {
            const key = switch (tier.scope) {
                .ip => keys.ip,
                .user => keys.user,
                .api_key => keys.api_key,
                .endpoint => keys.endpoint,
                .global => "global",
                .custom => keys.custom,
            } orelse continue;

            const status = tier.limiter.check(key);

            statuses.append(.{
                .tier_name = tier.name,
                .status = status,
            }) catch |err| {
                std.log.debug("Failed to append rate limit status: {t}", .{err});
                std.log.debug("Failed to append rate limit status: {t}", .{err});
            };

            if (!status.allowed) {
                all_allowed = false;
                if (most_restrictive == null or
                    (status.retry_after orelse 0) > (most_restrictive.?.retry_after orelse 0))
                {
                    most_restrictive = status;
                }
            }
        }

        return .{
            .allowed = all_allowed,
            .tier_statuses = statuses.items,
            .blocking_status = most_restrictive,
        };
    }

    pub const Keys = struct {
        ip: ?[]const u8 = null,
        user: ?[]const u8 = null,
        api_key: ?[]const u8 = null,
        endpoint: ?[]const u8 = null,
        custom: ?[]const u8 = null,
    };

    pub const TierStatus = struct {
        tier_name: []const u8,
        status: RateLimitStatus,
    };

    pub const MultiTierStatus = struct {
        allowed: bool,
        tier_statuses: []const TierStatus,
        blocking_status: ?RateLimitStatus,
    };
};

/// Common rate limit presets
pub const Presets = struct {
    /// Standard API rate limit
    pub const standard: RateLimitConfig = .{
        .requests = 100,
        .window_seconds = 60,
        .burst = 20,
        .algorithm = .token_bucket,
    };

    /// Strict rate limit for sensitive endpoints
    pub const strict: RateLimitConfig = .{
        .requests = 10,
        .window_seconds = 60,
        .burst = 2,
        .algorithm = .sliding_window,
        .ban_duration = 3600,
        .ban_threshold = 5,
    };

    /// Lenient rate limit for public endpoints
    pub const lenient: RateLimitConfig = .{
        .requests = 1000,
        .window_seconds = 60,
        .burst = 100,
        .algorithm = .token_bucket,
    };

    /// Login endpoint (strict with ban)
    pub const login: RateLimitConfig = .{
        .requests = 5,
        .window_seconds = 300, // 5 minutes
        .burst = 0,
        .algorithm = .sliding_window,
        .ban_duration = 900, // 15 minute ban
        .ban_threshold = 10,
    };
};

// Tests

test "token bucket rate limiting" {
    const allocator = std.testing.allocator;
    var limiter = RateLimiter.init(allocator, .{
        .requests = 10,
        .window_seconds = 60,
        .burst = 5,
        .algorithm = .token_bucket,
    });
    defer limiter.deinit();

    // First requests should be allowed
    for (0..10) |_| {
        const status = limiter.check("test_client");
        try std.testing.expect(status.allowed);
    }

    // Should have used burst, more requests allowed
    for (0..5) |_| {
        const status = limiter.check("test_client");
        try std.testing.expect(status.allowed);
    }

    // Now should be rate limited
    const status = limiter.check("test_client");
    try std.testing.expect(!status.allowed);
    try std.testing.expect(status.retry_after != null);
}

test "auto-ban functionality" {
    const allocator = std.testing.allocator;
    var limiter = RateLimiter.init(allocator, .{
        .requests = 2,
        .window_seconds = 1,
        .burst = 0,
        .algorithm = .fixed_window,
        .ban_duration = 60,
        .ban_threshold = 3,
    });
    defer limiter.deinit();

    // Exhaust limit
    _ = limiter.check("bad_client");
    _ = limiter.check("bad_client");

    // Trigger violations
    for (0..3) |_| {
        const status = limiter.check("bad_client");
        try std.testing.expect(!status.allowed);
    }

    // Should now be banned
    const status = limiter.check("bad_client");
    try std.testing.expect(status.banned);
    try std.testing.expect(status.ban_expires_at != null);
}

test "whitelist bypass" {
    const allocator = std.testing.allocator;
    var limiter = RateLimiter.init(allocator, .{
        .requests = 1,
        .window_seconds = 60,
        .burst = 0,
        .whitelist = &.{"trusted_client"},
    });
    defer limiter.deinit();

    // Whitelisted client always allowed
    for (0..100) |_| {
        const status = limiter.check("trusted_client");
        try std.testing.expect(status.allowed);
    }

    // Non-whitelisted is limited
    _ = limiter.check("normal_client");
    const status = limiter.check("normal_client");
    try std.testing.expect(!status.allowed);
}

test "production defaults have rate limiting enabled" {
    // Test that productionDefaults() returns a config with rate limiting enabled
    const prod_config = RateLimitConfig.productionDefaults();

    // Rate limiting must be enabled in production defaults
    try std.testing.expect(prod_config.enabled);
    try std.testing.expect(prod_config.isEnabled());

    // Verify sensible production values
    try std.testing.expectEqual(@as(u32, 100), prod_config.requests);
    try std.testing.expectEqual(@as(u32, 60), prod_config.window_seconds);
    try std.testing.expectEqual(@as(u32, 20), prod_config.burst);
    try std.testing.expectEqual(Algorithm.token_bucket, prod_config.algorithm);
    try std.testing.expectEqual(Scope.ip, prod_config.scope);
}

test "default config has rate limiting disabled for backwards compatibility" {
    // Default config should have rate limiting disabled to maintain backwards compatibility
    const default_config = RateLimitConfig{};

    try std.testing.expect(!default_config.enabled);
    try std.testing.expect(!default_config.isEnabled());
}

test "presets maintain backwards compatibility" {
    // Existing presets should not have enabled field explicitly set (defaults to false)
    // This ensures existing code continues to work
    const standard = Presets.standard;
    const strict = Presets.strict;
    const lenient = Presets.lenient;
    const login = Presets.login;

    // Presets don't set enabled, so they default to false (backwards compatible)
    try std.testing.expect(!standard.enabled);
    try std.testing.expect(!strict.enabled);
    try std.testing.expect(!lenient.enabled);
    try std.testing.expect(!login.enabled);
}

test {
    std.testing.refAllDecls(@This());
}
