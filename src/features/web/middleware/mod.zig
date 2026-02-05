//! Web Middleware Module
//!
//! Provides middleware components for the HTTP server pipeline.
//!
//! ## Available Middleware
//!
//! - **Logging**: Request/response logging in various formats
//! - **CORS**: Cross-Origin Resource Sharing headers
//! - **Auth**: JWT and API key authentication
//! - **Error Handler**: Converts errors to HTTP responses
//!
//! ## Usage
//!
//! ```zig
//! const middleware = @import("abi").web.middleware;
//!
//! var chain = middleware.MiddlewareChain.init(allocator);
//! try chain.use(&middleware.logging.minimalLogger);
//! try chain.use(&middleware.cors.permissiveCors);
//! try chain.use(&middleware.auth.optionalAuth);
//!
//! var ctx = middleware.MiddlewareContext.init(allocator, &request, &response);
//! try chain.execute(&ctx);
//! ```

const std = @import("std");

pub const types = @import("types.zig");
pub const logging = @import("logging.zig");
pub const cors = @import("cors.zig");
pub const auth = @import("auth.zig");
pub const error_handler = @import("error_handler.zig");

// Re-export main types
pub const MiddlewareContext = types.MiddlewareContext;
pub const MiddlewareFn = types.MiddlewareFn;
pub const MiddlewareChain = types.MiddlewareChain;
pub const HandlerFn = types.HandlerFn;
pub const RouteHandler = types.RouteHandler;

// Re-export configurations
pub const LogConfig = logging.LogConfig;
pub const LogFormat = logging.LogFormat;
pub const CorsConfig = cors.CorsConfig;
pub const AuthConfig = auth.AuthConfig;
pub const AuthResult = auth.AuthResult;
pub const ErrorConfig = error_handler.ErrorConfig;
pub const ErrorFormat = error_handler.ErrorFormat;
pub const ErrorResponse = error_handler.ErrorResponse;

/// Creates a default middleware chain with common middleware.
pub fn defaultChain(allocator: std.mem.Allocator) !MiddlewareChain {
    var chain = MiddlewareChain.init(allocator);
    errdefer chain.deinit();

    // Add common middleware in order
    try chain.use(&logging.minimalLogger);
    try chain.use(&cors.permissiveCors);
    try chain.use(&error_handler.jsonErrorHandler);

    return chain;
}

/// Creates a secure middleware chain with authentication.
pub fn secureChain(allocator: std.mem.Allocator) !MiddlewareChain {
    var chain = MiddlewareChain.init(allocator);
    errdefer chain.deinit();

    try chain.use(&logging.minimalLogger);
    try chain.use(&cors.permissiveCors);
    try chain.use(&auth.requireAuth);
    try chain.use(&error_handler.jsonErrorHandler);

    return chain;
}

/// Creates a minimal middleware chain (just logging).
pub fn minimalChain(allocator: std.mem.Allocator) !MiddlewareChain {
    var chain = MiddlewareChain.init(allocator);
    errdefer chain.deinit();

    try chain.use(&logging.minimalLogger);

    return chain;
}

/// Creates an API middleware chain (CORS + auth + error handling).
pub fn apiChain(allocator: std.mem.Allocator) !MiddlewareChain {
    var chain = MiddlewareChain.init(allocator);
    errdefer chain.deinit();

    try chain.use(&logging.minimalLogger);
    try chain.use(&cors.permissiveCors);
    try chain.use(&auth.optionalAuth);
    try chain.use(&error_handler.jsonErrorHandler);

    return chain;
}

test "defaultChain creation" {
    const allocator = std.testing.allocator;

    var chain = try defaultChain(allocator);
    defer chain.deinit();

    try std.testing.expectEqual(@as(usize, 3), chain.len());
}

test "secureChain creation" {
    const allocator = std.testing.allocator;

    var chain = try secureChain(allocator);
    defer chain.deinit();

    try std.testing.expectEqual(@as(usize, 4), chain.len());
}

test "minimalChain creation" {
    const allocator = std.testing.allocator;

    var chain = try minimalChain(allocator);
    defer chain.deinit();

    try std.testing.expectEqual(@as(usize, 1), chain.len());
}

test {
    // Run all submodule tests
    std.testing.refAllDecls(@This());
}
