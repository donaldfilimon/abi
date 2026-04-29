//! Cloud Response Builder
//!
//! Fluent API for building cloud function responses with common patterns
//! like JSON content type, CORS headers, and custom headers.

const std = @import("std");
const types = @import("types.zig");
const CloudResponse = types.CloudResponse;

/// Create a response helper for common response patterns.
pub const ResponseBuilder = struct {
    allocator: std.mem.Allocator,
    response: CloudResponse,

    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{
            .allocator = allocator,
            .response = CloudResponse.init(allocator),
        };
    }

    /// Set HTTP status code.
    pub fn status(self: *ResponseBuilder, code: u16) *ResponseBuilder {
        self.response.status_code = code;
        return self;
    }

    /// Add a header.
    pub fn header(self: *ResponseBuilder, key: []const u8, value: []const u8) *ResponseBuilder {
        self.response.headers.put(self.allocator, key, value) catch |err| {
            std.log.warn("Failed to set header '{s}': {t}", .{ key, err });
        };
        return self;
    }

    /// Set content type to JSON.
    pub fn json(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "application/json");
    }

    /// Set content type to plain text.
    pub fn text(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "text/plain");
    }

    /// Set content type to HTML.
    pub fn html(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "text/html");
    }

    /// Set the response body.
    pub fn body(self: *ResponseBuilder, content: []const u8) *ResponseBuilder {
        self.response.body = content;
        return self;
    }

    /// Add CORS headers.
    pub fn cors(self: *ResponseBuilder, origin: []const u8) *ResponseBuilder {
        _ = self.header("Access-Control-Allow-Origin", origin);
        _ = self.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        _ = self.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        return self;
    }

    /// Build the final response.
    pub fn build(self: *ResponseBuilder) CloudResponse {
        return self.response;
    }
};
