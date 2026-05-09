//! Server Configuration and Error Types
//!
//! Defines the streaming server configuration struct, error set, and
//! constants used across server submodules.

const std = @import("std");
const recovery = @import("../recovery.zig");
const backends = @import("../backends/mod.zig");

pub const RecoveryConfig = recovery.RecoveryConfig;

// ── Constants ────────────────────────────────────────────────────────────────

pub const MAX_BODY_BYTES = 1024 * 1024; // 1MB max request body
pub const HEARTBEAT_INTERVAL_MS: u64 = 15000; // 15 second heartbeats
pub const ADMIN_RELOAD_DRAIN_TIMEOUT_NS: u64 = 30_000_000_000; // 30 second drain timeout
pub const ADMIN_RELOAD_POLL_INTERVAL_NS: u64 = 100_000_000; // 100ms poll interval

// ── Error Set ────────────────────────────────────────────────────────────────

pub const StreamingServerError = std.mem.Allocator.Error || error{
    InvalidAddress,
    InvalidRequest,
    Unauthorized,
    BackendError,
    StreamError,
    WebSocketError,
    RequestTooLarge,
    UnsupportedBackend,
    ModelReloadFailed,
    ModelReloadTimeout,
    CircuitBreakerOpen,
};

// ── Server Configuration ─────────────────────────────────────────────────────

/// Server configuration
pub const ServerConfig = struct {
    /// Listen address (e.g., "127.0.0.1:8080")
    address: []const u8 = "127.0.0.1:8080",
    /// Bearer token for authentication (null = no auth required)
    auth_token: ?[]const u8 = null,
    /// Allow health endpoint without auth
    allow_health_without_auth: bool = true,
    /// Default backend for inference
    default_backend: backends.BackendType = .local,
    /// Heartbeat interval in milliseconds (0 = disabled)
    heartbeat_interval_ms: u64 = HEARTBEAT_INTERVAL_MS,
    /// Maximum concurrent streams
    max_concurrent_streams: u32 = 100,
    /// Enable OpenAI-compatible endpoints
    enable_openai_compat: bool = true,
    /// Enable WebSocket support
    enable_websocket: bool = true,
    /// Path to default local model (optional, for local backend)
    default_model_path: ?[]const u8 = null,
    /// Pre-load model on server start (reduces first-request latency)
    preload_model: bool = false,
    /// Enable error recovery (circuit breakers, retry, session caching)
    enable_recovery: bool = true,
    /// Recovery configuration (only used if enable_recovery is true)
    recovery_config: RecoveryConfig = .{},
};

test {
    std.testing.refAllDecls(@This());
}
