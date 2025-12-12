//! Web Feature Module
//!
//! HTTP servers, clients, and web services

const std = @import("std");

// Web servers
// pub const enhanced_web_server = @import("enhanced_web_server.zig"); // TODO: implement

// HTTP clients and utilities
pub const http_client = @import("http_client.zig");
pub const curl_wrapper = @import("curl_wrapper.zig");

// Specialized web services
pub const wdbx_http = @import("wdbx_http.zig");
pub const weather = @import("weather.zig");

// API interfaces
pub const c_api = @import("c_api.zig");

// Legacy compatibility removed - circular import fixed

test {
    std.testing.refAllDecls(@This());
}
