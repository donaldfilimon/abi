//! Web Configuration
//!
//! Configuration for HTTP server and web utilities.

const std = @import("std");
const rate_limit = @import("../../services/shared/security/rate_limit.zig");

/// Web/HTTP utilities configuration.
pub const WebConfig = struct {
    /// HTTP server bind address.
    bind_address: []const u8 = "127.0.0.1",

    /// HTTP server port.
    port: u16 = 3000,

    /// Enable CORS.
    cors_enabled: bool = true,

    /// Request timeout in milliseconds.
    timeout_ms: u64 = 30000,

    /// Maximum request body size.
    max_body_size: usize = 10 * 1024 * 1024, // 10MB

    /// Rate limiting configuration.
    /// Defaults to disabled for backwards compatibility.
    /// Use `productionDefaults()` for production deployments with rate limiting enabled.
    rate_limit: rate_limit.RateLimitConfig = .{},

    /// Returns default development configuration.
    /// Rate limiting is disabled by default for development environments.
    pub fn defaults() WebConfig {
        return .{};
    }

    /// Returns production-ready configuration with secure defaults.
    /// Rate limiting is enabled with sensible defaults (100 req/min, 20 burst).
    pub fn productionDefaults() WebConfig {
        return .{
            .bind_address = "0.0.0.0", // Listen on all interfaces in production
            .rate_limit = rate_limit.RateLimitConfig.productionDefaults(),
        };
    }

    /// Returns whether rate limiting is enabled in this configuration.
    pub fn isRateLimitingEnabled(self: WebConfig) bool {
        return self.rate_limit.isEnabled();
    }
};

test "WebConfig.defaults has rate limiting disabled" {
    const config = WebConfig.defaults();
    try std.testing.expect(!config.isRateLimitingEnabled());
}

test "WebConfig.productionDefaults has rate limiting enabled" {
    const config = WebConfig.productionDefaults();
    try std.testing.expect(config.isRateLimitingEnabled());
    try std.testing.expectEqual(@as(u32, 100), config.rate_limit.requests);
    try std.testing.expectEqual(@as(u32, 60), config.rate_limit.window_seconds);
    try std.testing.expectEqual(@as(u32, 20), config.rate_limit.burst);
}
