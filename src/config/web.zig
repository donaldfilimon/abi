//! Web Configuration
//!
//! Configuration for HTTP server and web utilities.

const std = @import("std");

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

    pub fn defaults() WebConfig {
        return .{};
    }
};
