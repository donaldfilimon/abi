//! CORS (Cross-Origin Resource Sharing) configuration.
//!
//! This module provides:
//! - CORS policy configuration
//! - Origin validation
//! - Preflight request handling
//! - Credential support
//! - Header generation

const std = @import("std");

/// CORS configuration
pub const CorsConfig = struct {
    /// Enable CORS handling
    enabled: bool = true,
    /// Allowed origins (empty = all, "*" = wildcard)
    allowed_origins: []const []const u8 = &.{"*"},
    /// Allow credentials (cookies, authorization headers)
    allow_credentials: bool = false,
    /// Allowed HTTP methods
    allowed_methods: []const Method = &.{ .GET, .POST, .PUT, .DELETE, .PATCH, .OPTIONS },
    /// Allowed request headers
    allowed_headers: []const []const u8 = &.{
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
    },
    /// Headers to expose to the client
    exposed_headers: []const []const u8 = &.{
        "Content-Length",
        "Content-Type",
        "X-Request-Id",
    },
    /// Max age for preflight cache (seconds)
    max_age: u32 = 86400, // 24 hours
    /// Allow origin patterns (supports wildcards like "*.example.com")
    allowed_origin_patterns: []const []const u8 = &.{},
    /// Vary header behavior
    vary_origin: bool = true,
    /// Handle private network access (Chrome feature)
    private_network_access: bool = false,

    pub const Method = enum {
        GET,
        POST,
        PUT,
        DELETE,
        PATCH,
        HEAD,
        OPTIONS,
        TRACE,
        CONNECT,

        pub fn toString(self: Method) []const u8 {
            return switch (self) {
                .GET => "GET",
                .POST => "POST",
                .PUT => "PUT",
                .DELETE => "DELETE",
                .PATCH => "PATCH",
                .HEAD => "HEAD",
                .OPTIONS => "OPTIONS",
                .TRACE => "TRACE",
                .CONNECT => "CONNECT",
            };
        }

        pub fn fromString(s: []const u8) ?Method {
            if (std.mem.eql(u8, s, "GET")) return .GET;
            if (std.mem.eql(u8, s, "POST")) return .POST;
            if (std.mem.eql(u8, s, "PUT")) return .PUT;
            if (std.mem.eql(u8, s, "DELETE")) return .DELETE;
            if (std.mem.eql(u8, s, "PATCH")) return .PATCH;
            if (std.mem.eql(u8, s, "HEAD")) return .HEAD;
            if (std.mem.eql(u8, s, "OPTIONS")) return .OPTIONS;
            if (std.mem.eql(u8, s, "TRACE")) return .TRACE;
            if (std.mem.eql(u8, s, "CONNECT")) return .CONNECT;
            return null;
        }
    };
};

/// CORS handler
pub const CorsHandler = struct {
    allocator: std.mem.Allocator,
    config: CorsConfig,

    pub fn init(allocator: std.mem.Allocator, config: CorsConfig) CorsHandler {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Check if an origin is allowed
    pub fn isOriginAllowed(self: *CorsHandler, origin: []const u8) bool {
        if (!self.config.enabled) return true;

        // Check for wildcard
        for (self.config.allowed_origins) |allowed| {
            if (std.mem.eql(u8, allowed, "*")) return true;
            if (std.mem.eql(u8, allowed, origin)) return true;
        }

        // Check patterns
        for (self.config.allowed_origin_patterns) |pattern| {
            if (matchesOriginPattern(origin, pattern)) return true;
        }

        return false;
    }

    /// Check if a method is allowed
    pub fn isMethodAllowed(self: *CorsHandler, method: []const u8) bool {
        const parsed = CorsConfig.Method.fromString(method) orelse return false;

        for (self.config.allowed_methods) |allowed| {
            if (allowed == parsed) return true;
        }

        return false;
    }

    /// Check if all requested headers are allowed
    pub fn areHeadersAllowed(self: *CorsHandler, requested_headers: []const u8) bool {
        // Parse comma-separated headers
        var it = std.mem.splitScalar(u8, requested_headers, ',');
        while (it.next()) |header_raw| {
            const header = std.mem.trim(u8, header_raw, " ");
            if (header.len == 0) continue;

            var found = false;
            for (self.config.allowed_headers) |allowed| {
                if (std.ascii.eqlIgnoreCase(header, allowed)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }

        return true;
    }

    /// Handle a preflight (OPTIONS) request
    pub fn handlePreflight(self: *CorsHandler, origin: []const u8, method: ?[]const u8, headers: ?[]const u8) PreflightResult {
        if (!self.config.enabled) {
            return .{ .allowed = true, .headers = &.{} };
        }

        // Check origin
        if (!self.isOriginAllowed(origin)) {
            return .{ .allowed = false, .reason = "Origin not allowed" };
        }

        // Check method
        if (method) |m| {
            if (!self.isMethodAllowed(m)) {
                return .{ .allowed = false, .reason = "Method not allowed" };
            }
        }

        // Check headers
        if (headers) |h| {
            if (!self.areHeadersAllowed(h)) {
                return .{ .allowed = false, .reason = "Headers not allowed" };
            }
        }

        return .{ .allowed = true };
    }

    /// Get CORS response headers
    pub fn getResponseHeaders(self: *CorsHandler, origin: []const u8, is_preflight: bool) ![]CorsHeader {
        var headers = std.ArrayListUnmanaged(CorsHeader).empty;
        errdefer headers.deinit(self.allocator);

        if (!self.config.enabled) {
            return headers.toOwnedSlice(self.allocator);
        }

        // Access-Control-Allow-Origin
        if (self.isOriginAllowed(origin)) {
            const allow_origin = if (self.config.allow_credentials)
                // Can't use "*" with credentials
                try self.allocator.dupe(u8, origin)
            else if (self.hasWildcard())
                try self.allocator.dupe(u8, "*")
            else
                try self.allocator.dupe(u8, origin);
            errdefer self.allocator.free(allow_origin);

            try headers.append(self.allocator, .{
                .name = "Access-Control-Allow-Origin",
                .value = allow_origin,
            });
        }

        // Vary header
        if (self.config.vary_origin and !self.hasWildcard()) {
            try headers.append(self.allocator, .{
                .name = "Vary",
                .value = try self.allocator.dupe(u8, "Origin"),
            });
        }

        // Allow credentials
        if (self.config.allow_credentials) {
            try headers.append(self.allocator, .{
                .name = "Access-Control-Allow-Credentials",
                .value = try self.allocator.dupe(u8, "true"),
            });
        }

        // Preflight-specific headers
        if (is_preflight) {
            // Allow methods
            try headers.append(self.allocator, .{
                .name = "Access-Control-Allow-Methods",
                .value = try self.buildMethodsList(),
            });

            // Allow headers
            try headers.append(self.allocator, .{
                .name = "Access-Control-Allow-Headers",
                .value = try self.buildHeadersList(),
            });

            // Max age
            try headers.append(self.allocator, .{
                .name = "Access-Control-Max-Age",
                .value = try std.fmt.allocPrint(self.allocator, "{d}", .{self.config.max_age}),
            });

            // Private network access
            if (self.config.private_network_access) {
                try headers.append(self.allocator, .{
                    .name = "Access-Control-Allow-Private-Network",
                    .value = try self.allocator.dupe(u8, "true"),
                });
            }
        }

        // Exposed headers (for actual requests)
        if (!is_preflight and self.config.exposed_headers.len > 0) {
            try headers.append(self.allocator, .{
                .name = "Access-Control-Expose-Headers",
                .value = try self.buildExposedHeadersList(),
            });
        }

        return headers.toOwnedSlice(self.allocator);
    }

    /// Free headers allocated by getResponseHeaders
    pub fn freeHeaders(self: *CorsHandler, headers: []CorsHeader) void {
        for (headers) |header| {
            self.allocator.free(header.value);
        }
        self.allocator.free(headers);
    }

    // Private helpers

    fn hasWildcard(self: *CorsHandler) bool {
        for (self.config.allowed_origins) |origin| {
            if (std.mem.eql(u8, origin, "*")) return true;
        }
        return false;
    }

    fn buildMethodsList(self: *CorsHandler) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        for (self.config.allowed_methods, 0..) |method, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ", ");
            try buffer.appendSlice(self.allocator, method.toString());
        }

        return buffer.toOwnedSlice(self.allocator);
    }

    fn buildHeadersList(self: *CorsHandler) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        for (self.config.allowed_headers, 0..) |header, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ", ");
            try buffer.appendSlice(self.allocator, header);
        }

        return buffer.toOwnedSlice(self.allocator);
    }

    fn buildExposedHeadersList(self: *CorsHandler) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        for (self.config.exposed_headers, 0..) |header, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ", ");
            try buffer.appendSlice(self.allocator, header);
        }

        return buffer.toOwnedSlice(self.allocator);
    }
};

/// CORS response header
pub const CorsHeader = struct {
    name: []const u8,
    value: []const u8,
};

/// Preflight request result
pub const PreflightResult = struct {
    allowed: bool,
    reason: ?[]const u8 = null,
    headers: []const CorsHeader = &.{},
};

/// Check if origin matches a pattern (supports subdomain wildcards)
fn matchesOriginPattern(origin: []const u8, pattern: []const u8) bool {
    // Handle *.example.com pattern
    if (std.mem.startsWith(u8, pattern, "*.")) {
        const domain = pattern[2..];

        // Parse origin to get host
        // Origin format: scheme://host[:port]
        const host_start = if (std.mem.indexOf(u8, origin, "://")) |idx| idx + 3 else 0;
        const host_end = if (std.mem.indexOfScalar(u8, origin[host_start..], ':')) |idx|
            host_start + idx
        else if (std.mem.indexOfScalar(u8, origin[host_start..], '/')) |idx|
            host_start + idx
        else
            origin.len;

        const host = origin[host_start..host_end];

        // Check if host ends with domain
        if (std.mem.endsWith(u8, host, domain)) {
            // Ensure it's a proper subdomain (has a dot before the domain)
            if (host.len > domain.len) {
                const prefix = host[0 .. host.len - domain.len];
                if (std.mem.endsWith(u8, prefix, ".")) return true;
            }
        }

        return false;
    }

    // Exact match
    return std.mem.eql(u8, origin, pattern);
}

/// Common CORS presets
pub const Presets = struct {
    /// Development preset (allow all)
    pub const development: CorsConfig = .{
        .allowed_origins = &.{"*"},
        .allow_credentials = false,
        .max_age = 3600,
    };

    /// Strict preset (no CORS)
    pub const strict: CorsConfig = .{
        .enabled = false,
    };

    /// Single origin preset (customize allowed_origins)
    pub const single_origin: CorsConfig = .{
        .allowed_origins = &.{}, // Set your origin
        .allow_credentials = true,
        .max_age = 86400,
    };

    /// API preset (common for REST APIs)
    pub const api: CorsConfig = .{
        .allowed_origins = &.{"*"},
        .allow_credentials = false,
        .allowed_methods = &.{ .GET, .POST, .PUT, .DELETE, .PATCH },
        .allowed_headers = &.{
            "Accept",
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-API-Key",
        },
        .exposed_headers = &.{
            "X-Request-Id",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
        },
        .max_age = 86400,
    };
};

// Tests

test "origin matching" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_origins = &.{ "https://example.com", "https://api.example.com" },
    });

    try std.testing.expect(handler.isOriginAllowed("https://example.com"));
    try std.testing.expect(handler.isOriginAllowed("https://api.example.com"));
    try std.testing.expect(!handler.isOriginAllowed("https://other.com"));
    try std.testing.expect(!handler.isOriginAllowed("https://example.com.evil.com"));
}

test "wildcard origin" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_origins = &.{"*"},
    });

    try std.testing.expect(handler.isOriginAllowed("https://any-origin.com"));
    try std.testing.expect(handler.isOriginAllowed("http://localhost:3000"));
}

test "pattern matching" {
    try std.testing.expect(matchesOriginPattern("https://sub.example.com", "*.example.com"));
    try std.testing.expect(matchesOriginPattern("https://deep.sub.example.com", "*.example.com"));
    try std.testing.expect(!matchesOriginPattern("https://example.com", "*.example.com"));
    try std.testing.expect(!matchesOriginPattern("https://notexample.com", "*.example.com"));
}

test "method checking" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_methods = &.{ .GET, .POST },
    });

    try std.testing.expect(handler.isMethodAllowed("GET"));
    try std.testing.expect(handler.isMethodAllowed("POST"));
    try std.testing.expect(!handler.isMethodAllowed("DELETE"));
    try std.testing.expect(!handler.isMethodAllowed("PUT"));
}

test "header checking" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_headers = &.{ "Content-Type", "Authorization" },
    });

    try std.testing.expect(handler.areHeadersAllowed("Content-Type"));
    try std.testing.expect(handler.areHeadersAllowed("content-type")); // Case insensitive
    try std.testing.expect(handler.areHeadersAllowed("Content-Type, Authorization"));
    try std.testing.expect(!handler.areHeadersAllowed("X-Custom-Header"));
}

test "preflight handling" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_origins = &.{"https://example.com"},
        .allowed_methods = &.{ .GET, .POST },
    });

    // Allowed preflight
    const result1 = handler.handlePreflight("https://example.com", "GET", null);
    try std.testing.expect(result1.allowed);

    // Disallowed origin
    const result2 = handler.handlePreflight("https://other.com", "GET", null);
    try std.testing.expect(!result2.allowed);

    // Disallowed method
    const result3 = handler.handlePreflight("https://example.com", "DELETE", null);
    try std.testing.expect(!result3.allowed);
}

test "response headers generation" {
    const allocator = std.testing.allocator;
    var handler = CorsHandler.init(allocator, .{
        .allowed_origins = &.{"https://example.com"},
        .allow_credentials = true,
    });

    const headers = try handler.getResponseHeaders("https://example.com", false);
    defer handler.freeHeaders(headers);

    try std.testing.expect(headers.len > 0);

    // Find Access-Control-Allow-Origin
    var found_origin = false;
    for (headers) |header| {
        if (std.mem.eql(u8, header.name, "Access-Control-Allow-Origin")) {
            try std.testing.expectEqualStrings("https://example.com", header.value);
            found_origin = true;
        }
    }
    try std.testing.expect(found_origin);
}

test {
    std.testing.refAllDecls(@This());
}
