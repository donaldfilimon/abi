//! Web Server Module
//!
//! Provides HTTP server functionality for the ABI Framework.
//! Includes request parsing, response building, and connection management.
//!
//! ## Usage
//!
//! ```zig
//! const web = @import("abi").web;
//!
//! var server = web.server.Server.init(allocator, .{
//!     .host = "0.0.0.0",
//!     .port = 8080,
//! });
//! defer server.deinit();
//!
//! server.setHandler(&myHandler);
//! try server.run();
//! ```

const std = @import("std");

pub const types = @import("types.zig");
pub const http_server = @import("http_server.zig");
pub const request_parser = @import("request_parser.zig");
pub const response_builder = @import("response_builder.zig");

// Re-export main types for convenience
pub const ServerConfig = types.ServerConfig;
pub const ServerState = types.ServerState;
pub const ServerStats = types.ServerStats;
pub const ServerError = types.ServerError;
pub const Connection = types.Connection;
pub const Method = types.Method;
pub const Status = types.Status;
pub const Header = types.Header;
pub const MimeType = types.MimeType;

pub const Server = http_server.Server;
pub const Request = http_server.Server.Request;
pub const Response = http_server.Server.Response;
pub const RequestHandler = http_server.Server.RequestHandler;

pub const ParsedRequest = request_parser.ParsedRequest;
pub const HttpVersion = request_parser.HttpVersion;
pub const ParseError = request_parser.ParseError;
pub const parseRequest = request_parser.parseRequest;
pub const extractPathParams = request_parser.extractPathParams;
pub const matchesPattern = request_parser.matchesPattern;
pub const urlDecode = request_parser.urlDecode;

pub const ResponseBuilder = response_builder.ResponseBuilder;
pub const CookieOptions = response_builder.CookieOptions;
pub const errorResponse = response_builder.errorResponse;
pub const successResponse = response_builder.successResponse;

/// Creates a new server with the given configuration.
pub fn createServer(allocator: std.mem.Allocator, config: ServerConfig) Server {
    return Server.init(allocator, config);
}

/// Creates a server with default configuration.
pub fn createDefaultServer(allocator: std.mem.Allocator) Server {
    return Server.init(allocator, .{});
}

/// Creates a server configured for development (localhost, debug logging).
pub fn createDevServer(allocator: std.mem.Allocator, port: u16) Server {
    return Server.init(allocator, .{
        .host = "127.0.0.1",
        .port = port,
        .keep_alive = true,
        .keep_alive_timeout_ms = 30000,
    });
}

/// Creates a server configured for production.
pub fn createProdServer(allocator: std.mem.Allocator, host: []const u8, port: u16) Server {
    return Server.init(allocator, .{
        .host = host,
        .port = port,
        .max_connections = 10000,
        .keep_alive = true,
        .keep_alive_timeout_ms = 5000,
        .read_timeout_ms = 30000,
        .write_timeout_ms = 30000,
    });
}

test "module exports" {
    // Verify all exports are accessible
    _ = ServerConfig{};
    _ = ServerState.stopped;
    _ = ServerStats{};
    _ = Method.GET;
    _ = Status.ok;
    _ = Header.content_type;
    _ = MimeType.json;
}

test "server creation helpers" {
    const allocator = std.testing.allocator;

    var server1 = createDefaultServer(allocator);
    defer server1.deinit();
    try std.testing.expectEqual(@as(u16, 8080), server1.config.port);

    var server2 = createDevServer(allocator, 3000);
    defer server2.deinit();
    try std.testing.expectEqual(@as(u16, 3000), server2.config.port);

    var server3 = createProdServer(allocator, "0.0.0.0", 80);
    defer server3.deinit();
    try std.testing.expectEqual(@as(u16, 80), server3.config.port);
    try std.testing.expectEqual(@as(usize, 10000), server3.config.max_connections);
}

test {
    // Run all submodule tests
    std.testing.refAllDecls(@This());
}
