//! API Module
//!
//! This module provides various API interfaces for the WDBX database,
//! including CLI, HTTP, TCP, and WebSocket servers.

pub const cli = @import("cli/mod.zig");
pub const http = @import("http/mod.zig");
pub const tcp = @import("tcp/mod.zig");

// Re-export main types
pub const CLI = cli.CLI;
pub const HttpServer = http.HttpServer;
pub const TcpServer = tcp.TcpServer;

// API configuration
pub const ApiConfig = struct {
    /// Enable authentication
    enable_auth: bool = true,
    /// Enable rate limiting
    enable_rate_limit: bool = true,
    /// Maximum requests per minute
    rate_limit_rpm: u32 = 1000,
    /// Enable metrics collection
    enable_metrics: bool = true,
    /// Enable request logging
    enable_logging: bool = true,
    /// Maximum request body size
    max_body_size: usize = 10 * 1024 * 1024, // 10MB
    /// Request timeout in milliseconds
    timeout_ms: u32 = 30000,
};