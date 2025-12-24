//! Web Feature Module
//!
//! HTTP servers, clients, and web services

const std = @import("std");

// Web servers
pub const enhanced_web_server = @import("enhanced_web_server.zig");
pub const WdbxHttpServer = wdbx_http.WdbxHttpServer;

// HTTP clients and utilities
pub const http_client = @import("http_client.zig");
pub const curl_wrapper = @import("curl_wrapper.zig");
pub const python_bindings = @import("python_bindings.zig");

// Specialized web services
pub const wdbx_http = @import("wdbx_http.zig");
pub const weather = @import("weather.zig");

// API interfaces
pub const c_api = @import("c_api.zig");

// Re-export key types for convenience
pub const HttpServer = WdbxHttpServer;
pub const HttpError = wdbx_http.HttpError;
pub const Response = wdbx_http.Response;
pub const ServerConfig = wdbx_http.ServerConfig;

/// Initialize the web feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global web state to initialize
}

/// Deinitialize the web feature module
pub fn deinit() void {
    // Currently no global web state to cleanup
}

// Legacy compatibility removed - circular import fixed

test {
    std.testing.refAllDecls(@This());
}
