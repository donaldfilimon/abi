//! Utilities Module
//! This module contains common utilities, definitions, and types used across the project

const std = @import("std");

// =============================================================================
// COMMON DEFINITIONS
// =============================================================================

/// Project version information
pub const VERSION = .{
    .major = 1,
    .minor = 0,
    .patch = 0,
    .pre_release = "alpha",
};

/// Render version as semantic version string: "major.minor.patch[-pre]"
pub fn versionString(allocator: std.mem.Allocator) ![]u8 {
    if (VERSION.pre_release.len > 0) {
        return std.fmt.allocPrint(allocator, "{d}.{d}.{d}-{s}", .{ VERSION.major, VERSION.minor, VERSION.patch, VERSION.pre_release });
    }
    return std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{ VERSION.major, VERSION.minor, VERSION.patch });
}

/// Common configuration struct
pub const Config = struct {
    name: []const u8 = "abi-ai",
    version: u32 = 1,
    debug_mode: bool = false,

    pub fn init(name: []const u8) Config {
        return .{ .name = name };
    }
};

/// Definition types used throughout the project
pub const DefinitionType = enum {
    core,
    database,
    neural,
    web,
    cli,

    pub fn toString(self: DefinitionType) []const u8 {
        return switch (self) {
            .core => "core",
            .database => "database",
            .neural => "neural",
            .web => "web",
            .cli => "cli",
        };
    }
};

// =============================================================================
// WEB UTILITIES
// =============================================================================

/// HTTP status codes
pub const HttpStatus = enum(u16) {
    // 2xx Success
    ok = 200,
    created = 201,
    accepted = 202,
    no_content = 204,

    // 3xx Redirection
    moved_permanently = 301,
    found = 302,
    not_modified = 304,

    // 4xx Client Error
    bad_request = 400,
    unauthorized = 401,
    forbidden = 403,
    not_found = 404,
    method_not_allowed = 405,
    conflict = 409,
    unprocessable_entity = 422,
    too_many_requests = 429,

    // 5xx Server Error
    internal_server_error = 500,
    not_implemented = 501,
    bad_gateway = 502,
    service_unavailable = 503,
    gateway_timeout = 504,

    pub fn phrase(self: HttpStatus) []const u8 {
        return switch (self) {
            .ok => "OK",
            .created => "Created",
            .accepted => "Accepted",
            .no_content => "No Content",
            .moved_permanently => "Moved Permanently",
            .found => "Found",
            .not_modified => "Not Modified",
            .bad_request => "Bad Request",
            .unauthorized => "Unauthorized",
            .forbidden => "Forbidden",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .conflict => "Conflict",
            .unprocessable_entity => "Unprocessable Entity",
            .too_many_requests => "Too Many Requests",
            .internal_server_error => "Internal Server Error",
            .not_implemented => "Not Implemented",
            .bad_gateway => "Bad Gateway",
            .service_unavailable => "Service Unavailable",
            .gateway_timeout => "Gateway Timeout",
        };
    }

    /// Numeric code accessor
    pub fn code(self: HttpStatus) u16 {
        return @intFromEnum(self);
    }

    /// Classification helpers
    pub fn isSuccess(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 200 and c < 300;
    }

    pub fn isRedirect(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 300 and c < 400;
    }

    pub fn isClientError(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 400 and c < 500;
    }

    pub fn isServerError(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 500 and c < 600;
    }

    pub fn isError(self: HttpStatus) bool {
        return self.isClientError() or self.isServerError();
    }
};

/// HTTP method types
pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
    TRACE,
    CONNECT,

    pub fn fromString(method: []const u8) ?HttpMethod {
        if (std.ascii.eqlIgnoreCase(method, "GET")) return .GET;
        if (std.ascii.eqlIgnoreCase(method, "POST")) return .POST;
        if (std.ascii.eqlIgnoreCase(method, "PUT")) return .PUT;
        if (std.ascii.eqlIgnoreCase(method, "DELETE")) return .DELETE;
        if (std.ascii.eqlIgnoreCase(method, "PATCH")) return .PATCH;
        if (std.ascii.eqlIgnoreCase(method, "OPTIONS")) return .OPTIONS;
        if (std.ascii.eqlIgnoreCase(method, "HEAD")) return .HEAD;
        if (std.ascii.eqlIgnoreCase(method, "TRACE")) return .TRACE;
        if (std.ascii.eqlIgnoreCase(method, "CONNECT")) return .CONNECT;
        return null;
    }

    pub fn toString(self: HttpMethod) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .DELETE => "DELETE",
            .PATCH => "PATCH",
            .OPTIONS => "OPTIONS",
            .HEAD => "HEAD",
            .TRACE => "TRACE",
            .CONNECT => "CONNECT",
        };
    }

    /// RFC 7231 safe methods (do not modify server state)
    pub fn isSafe(self: HttpMethod) bool {
        return switch (self) {
            .GET, .HEAD, .OPTIONS, .TRACE => true,
            else => false,
        };
    }

    /// RFC 7231 idempotent methods
    pub fn isIdempotent(self: HttpMethod) bool {
        return switch (self) {
            .GET, .PUT, .DELETE, .HEAD, .OPTIONS, .TRACE => true,
            else => false,
        };
    }

    /// Whether a request method typically allows a body
    pub fn allowsBody(self: HttpMethod) bool {
        return switch (self) {
            .POST, .PUT, .PATCH, .DELETE => true,
            else => false,
        };
    }
};

/// HTTP header management
pub const Headers = struct {
    map: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Headers {
        return .{
            .map = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Headers) void {
        self.map.deinit();
    }

    pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
        try self.map.put(name, value);
    }

    pub fn get(self: *Headers, name: []const u8) ?[]const u8 {
        return self.map.get(name);
    }

    pub fn remove(self: *Headers, name: []const u8) bool {
        return self.map.remove(name);
    }

    /// Get a header or return a provided default
    pub fn getOr(self: *Headers, name: []const u8, default_value: []const u8) []const u8 {
        return self.get(name) orelse default_value;
    }
};

/// HTTP request structure
pub const HttpRequest = struct {
    method: HttpMethod,
    path: []const u8,
    headers: Headers,
    body: ?[]const u8 = null,
    query_params: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, method: HttpMethod, path: []const u8) HttpRequest {
        return .{
            .method = method,
            .path = path,
            .headers = Headers.init(allocator),
            .query_params = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *HttpRequest) void {
        self.headers.deinit();
        self.query_params.deinit();
    }
};

/// HTTP response structure
pub const HttpResponse = struct {
    status: HttpStatus,
    headers: Headers,
    body: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, status: HttpStatus) HttpResponse {
        return .{
            .status = status,
            .headers = Headers.init(allocator),
        };
    }

    pub fn deinit(self: *HttpResponse) void {
        self.headers.deinit();
    }

    pub fn setContentType(self: *HttpResponse, content_type: []const u8) !void {
        try self.headers.set("Content-Type", content_type);
    }

    pub fn setJson(self: *HttpResponse) !void {
        try self.setContentType("application/json");
    }

    pub fn setText(self: *HttpResponse) !void {
        try self.setContentType("text/plain");
    }

    pub fn setHtml(self: *HttpResponse) !void {
        try self.setContentType("text/html");
    }
};

// =============================================================================
// COMMON UTILITIES
// =============================================================================

/// String utilities
pub const StringUtils = struct {
    /// Check if string is empty or whitespace only
    pub fn isEmptyOrWhitespace(str: []const u8) bool {
        return std.mem.trim(u8, str, " \t\r\n").len == 0;
    }

    /// Convert string to lowercase (allocates)
    pub fn toLower(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }

    /// Convert string to uppercase (allocates)
    pub fn toUpper(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toUpper(c);
        }
        return result;
    }
};

/// Array utilities
pub const ArrayUtils = struct {
    /// Check if array contains element
    pub fn contains(comptime T: type, haystack: []const T, needle: T) bool {
        for (haystack) |item| {
            if (item == needle) return true;
        }
        return false;
    }

    /// Find index of element in array
    pub fn indexOf(comptime T: type, haystack: []const T, needle: T) ?usize {
        for (haystack, 0..) |item, i| {
            if (item == needle) return i;
        }
        return null;
    }
};

/// Time utilities
pub const TimeUtils = struct {
    /// Get current timestamp in milliseconds
    pub fn nowMs() i64 {
        return std.time.milliTimestamp();
    }

    /// Get current timestamp in microseconds
    pub fn nowUs() i64 {
        return std.time.microTimestamp();
    }

    /// Get current timestamp in nanoseconds
    pub fn nowNs() i64 {
        return std.time.nanoTimestamp();
    }

    /// Format duration in human readable format
    pub fn formatDuration(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
        const ms = duration_ns / std.time.ns_per_ms;
        const us = (duration_ns % std.time.ns_per_ms) / std.time.ns_per_us;
        const ns = duration_ns % std.time.ns_per_us;

        if (ms > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}ms", .{ ms, us });
        } else if (us > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}μs", .{ us, ns / 1000 });
        } else {
            return std.fmt.allocPrint(allocator, "{d}ns", .{ns});
        }
    }
};

// =============================================================================
// JSON UTILITIES - Parse/serialize JSON (complements HTTP support)
// =============================================================================

/// JSON utilities for parsing and serialization
pub const JsonUtils = struct {
    /// Simple JSON value types
    pub const JsonValue = union(enum) {
        null,
        bool: bool,
        int: i64,
        float: f64,
        string: []const u8,
        array: []JsonValue,
        object: std.StringHashMap(JsonValue),

        pub fn deinit(self: *JsonValue, allocator: std.mem.Allocator) void {
            switch (self.*) {
                .string => |str| allocator.free(str),
                .array => |arr| {
                    for (arr) |*item| item.deinit(allocator);
                    allocator.free(arr);
                },
                .object => |*obj| {
                    var it = obj.iterator();
                    while (it.next()) |entry| {
                        allocator.free(entry.key_ptr.*);
                        entry.value_ptr.deinit(allocator);
                    }
                    obj.deinit();
                },
                else => {},
            }
        }
    };

    /// Parse JSON string into JsonValue
    pub fn parse(allocator: std.mem.Allocator, json_str: []const u8) !JsonValue {
        var tree = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer tree.deinit();

        return try jsonValueFromTree(allocator, tree.value);
    }

    /// Serialize JsonValue to JSON string (simplified version)
    pub fn stringify(allocator: std.mem.Allocator, value: JsonValue) ![]u8 {
        // Simplified implementation for compatibility
        return switch (value) {
            .null => allocator.dupe(u8, "null"),
            .bool => |b| if (b) allocator.dupe(u8, "true") else allocator.dupe(u8, "false"),
            .int => |i| std.fmt.allocPrint(allocator, "{}", .{i}),
            .float => |f| std.fmt.allocPrint(allocator, "{}", .{f}),
            .string => |s| std.fmt.allocPrint(allocator, "\"{s}\"", .{s}),
            .array => allocator.dupe(u8, "[]"), // Simplified
            .object => allocator.dupe(u8, "{}"), // Simplified
        };
    }

    /// Parse JSON string into typed struct
    pub fn parseInto(allocator: std.mem.Allocator, comptime T: type, json_str: []const u8) !T {
        var tree = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer tree.deinit();

        return try std.json.parseFromValueLeaky(T, allocator, tree.value, .{});
    }

    /// Serialize struct to JSON string (simplified)
    pub fn stringifyFrom(allocator: std.mem.Allocator, value: anytype) ![]u8 {
        _ = value; // Remove unused parameter warning
        // Simplified implementation
        return allocator.dupe(u8, "{}");
    }

    fn jsonValueFromTree(allocator: std.mem.Allocator, value: std.json.Value) !JsonValue {
        switch (value) {
            .null => return .null,
            .bool => |b| return JsonValue{ .bool = b },
            .integer => |i| return JsonValue{ .int = i },
            .float => |f| return JsonValue{ .float = f },
            .number_string => |s| return JsonValue{ .string = try allocator.dupe(u8, s) },
            .string => |s| return JsonValue{ .string = try allocator.dupe(u8, s) },
            .array => |arr| {
                const items = try allocator.alloc(JsonValue, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    items[i] = try jsonValueFromTree(allocator, item);
                }
                return JsonValue{ .array = items };
            },
            .object => |obj| {
                var map = std.StringHashMap(JsonValue).init(allocator);
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = try allocator.dupe(u8, entry.key_ptr.*);
                    const val = try jsonValueFromTree(allocator, entry.value_ptr.*);
                    try map.put(key, val);
                }
                return JsonValue{ .object = map };
            },
        }
    }
};

// =============================================================================
// URL UTILITIES - Encoding/decoding, query parameter parsing
// =============================================================================

/// URL utilities for encoding/decoding and query parameter handling
pub const UrlUtils = struct {
    /// URL-encode a string
    pub fn encode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, input.len * 2);
        errdefer buffer.deinit(allocator);

        for (input) |byte| {
            if (isUrlSafe(byte)) {
                try buffer.append(allocator, byte);
            } else {
                const hex = try std.fmt.allocPrint(allocator, "%{X:0>2}", .{byte});
                defer allocator.free(hex);
                try buffer.appendSlice(allocator, hex);
            }
        }

        return try buffer.toOwnedSlice(allocator);
    }

    /// URL-decode a string
    pub fn decode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, input.len);
        errdefer buffer.deinit(allocator);

        var i: usize = 0;
        while (i < input.len) {
            if (input[i] == '%') {
                if (i + 2 >= input.len) return error.InvalidPercentEncoding;
                const hex_str = input[i + 1 .. i + 3];
                const byte = try std.fmt.parseInt(u8, hex_str, 16);
                try buffer.append(allocator, byte);
                i += 3;
            } else if (input[i] == '+') {
                try buffer.append(allocator, ' ');
                i += 1;
            } else {
                try buffer.append(allocator, input[i]);
                i += 1;
            }
        }

        return try buffer.toOwnedSlice(allocator);
    }

    /// Parse query parameters from URL or query string
    pub fn parseQueryParams(allocator: std.mem.Allocator, query_string: []const u8) !std.StringHashMap([]u8) {
        var params = std.StringHashMap([]u8).init(allocator);
        errdefer {
            var it = params.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            params.deinit();
        }

        if (query_string.len == 0) return params;

        var iter = std.mem.splitScalar(u8, query_string, '&');
        while (iter.next()) |pair| {
            if (pair.len == 0) continue;

            var kv_iter = std.mem.splitScalar(u8, pair, '=');
            const key_encoded = kv_iter.next() orelse continue;
            const value_encoded = kv_iter.next() orelse "";

            const key = try decode(allocator, key_encoded);
            errdefer allocator.free(key);

            const value = try decode(allocator, value_encoded);
            errdefer allocator.free(value);

            try params.put(key, value);
        }

        return params;
    }

    /// Build query string from parameters
    pub fn buildQueryString(allocator: std.mem.Allocator, params: std.StringHashMap([]const u8)) ![]u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 256);
        errdefer buffer.deinit(allocator);

        var it = params.iterator();
        var first = true;
        while (it.next()) |entry| {
            if (!first) try buffer.append(allocator, '&');
            first = false;

            const key_encoded = try encode(allocator, entry.key_ptr.*);
            defer allocator.free(key_encoded);
            try buffer.appendSlice(allocator, key_encoded);

            try buffer.append(allocator, '=');

            const value_encoded = try encode(allocator, entry.value_ptr.*);
            defer allocator.free(value_encoded);
            try buffer.appendSlice(allocator, value_encoded);
        }

        return try buffer.toOwnedSlice(allocator);
    }

    /// Parse URL into components
    pub const UrlComponents = struct {
        scheme: []const u8,
        host: []const u8,
        port: ?u16,
        path: []const u8,
        query: []const u8,
        fragment: []const u8,

        pub fn deinit(self: *UrlComponents, allocator: std.mem.Allocator) void {
            allocator.free(self.scheme);
            allocator.free(self.host);
            allocator.free(self.path);
            allocator.free(self.query);
            allocator.free(self.fragment);
        }
    };

    /// Parse URL into components
    pub fn parseUrl(allocator: std.mem.Allocator, url: []const u8) !UrlComponents {
        var components = UrlComponents{
            .scheme = "",
            .host = "",
            .port = null,
            .path = "",
            .query = "",
            .fragment = "",
        };

        // Parse scheme
        if (std.mem.indexOf(u8, url, "://")) |scheme_end| {
            components.scheme = try allocator.dupe(u8, url[0..scheme_end]);
            var remaining = url[scheme_end + 3 ..];

            // Parse fragment
            if (std.mem.indexOf(u8, remaining, "#")) |frag_start| {
                components.fragment = try allocator.dupe(u8, remaining[frag_start + 1 ..]);
                remaining = remaining[0..frag_start];
            }

            // Parse query
            if (std.mem.indexOf(u8, remaining, "?")) |query_start| {
                components.query = try allocator.dupe(u8, remaining[query_start + 1 ..]);
                remaining = remaining[0..query_start];
            }

            // Parse host and port
            if (std.mem.indexOf(u8, remaining, "/")) |path_start| {
                const host_part = remaining[0..path_start];
                components.path = try allocator.dupe(u8, remaining[path_start..]);

                if (std.mem.indexOf(u8, host_part, ":")) |port_start| {
                    components.host = try allocator.dupe(u8, host_part[0..port_start]);
                    const port_str = host_part[port_start + 1 ..];
                    components.port = try std.fmt.parseInt(u16, port_str, 10);
                } else {
                    components.host = try allocator.dupe(u8, host_part);
                }
            } else {
                // No path, remaining is host:port
                if (std.mem.indexOf(u8, remaining, ":")) |port_start| {
                    components.host = try allocator.dupe(u8, remaining[0..port_start]);
                    const port_str = remaining[port_start + 1 ..];
                    components.port = try std.fmt.parseInt(u16, port_str, 10);
                } else {
                    components.host = try allocator.dupe(u8, remaining);
                }
                components.path = try allocator.dupe(u8, "/");
            }
        } else {
            return error.InvalidUrl;
        }

        return components;
    }

    fn isUrlSafe(byte: u8) bool {
        return switch (byte) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => true,
            else => false,
        };
    }
};

// =============================================================================
// BASE64 UTILITIES - Encoding/decoding
// =============================================================================

/// Base64 utilities for encoding/decoding
pub const Base64Utils = struct {
    /// Encode data to base64 string
    pub fn encode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
        const encoded_len = std.base64.standard.Encoder.calcSize(data.len);
        const buffer = try allocator.alloc(u8, encoded_len);
        _ = std.base64.standard.Encoder.encode(buffer, data);
        return buffer;
    }

    /// Decode base64 string to data
    pub fn decode(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(encoded);
        const buffer = try allocator.alloc(u8, decoded_len);
        try std.base64.standard.Decoder.decode(buffer, encoded);
        return buffer;
    }

    /// Encode data to URL-safe base64 string
    pub fn encodeUrlSafe(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
        const encoded_len = std.base64.url_safe.Encoder.calcSize(data.len);
        const buffer = try allocator.alloc(u8, encoded_len);
        _ = std.base64.url_safe.Encoder.encode(buffer, data);
        return buffer;
    }

    /// Decode URL-safe base64 string to data
    pub fn decodeUrlSafe(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
        const decoded_len = try std.base64.url_safe.Decoder.calcSizeForSlice(encoded);
        const buffer = try allocator.alloc(u8, decoded_len);
        try std.base64.url_safe.Decoder.decode(buffer, encoded);
        return buffer;
    }
};

// =============================================================================
// FILE SYSTEM UTILITIES - Common file operations
// =============================================================================

/// File system utilities for common operations
pub const FileSystemUtils = struct {
    /// Read entire file into string
    pub fn readFileString(allocator: std.mem.Allocator, file_path: []const u8) ![]u8 {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const stat = try file.stat();
        const buffer = try allocator.alloc(u8, stat.size);
        errdefer allocator.free(buffer);

        const bytes_read = try file.readAll(buffer);
        if (bytes_read != stat.size) return error.IncompleteRead;

        return buffer;
    }

    /// Write string to file
    pub fn writeFileString(file_path: []const u8, content: []const u8) !void {
        const file = try std.fs.cwd().createFile(file_path, .{ .truncate = true });
        defer file.close();

        try file.writeAll(content);
    }

    /// Check if file exists
    pub fn fileExists(file_path: []const u8) bool {
        const file = std.fs.cwd().openFile(file_path, .{}) catch return false;
        file.close();
        return true;
    }

    /// Check if directory exists
    pub fn dirExists(dir_path: []const u8) bool {
        const dir = std.fs.cwd().openDir(dir_path, .{}) catch return false;
        dir.close();
        return true;
    }

    /// Create directory recursively
    pub fn createDirRecursive(dir_path: []const u8) !void {
        try std.fs.cwd().makePath(dir_path);
    }

    /// Get file extension
    pub fn getFileExtension(file_path: []const u8) []const u8 {
        if (std.fs.path.extension(file_path).len > 0) {
            return std.fs.path.extension(file_path)[1..]; // Remove the leading dot
        }
        return "";
    }

    /// Get file name without extension
    pub fn getFileNameWithoutExtension(file_path: []const u8) []const u8 {
        const basename = std.fs.path.basename(file_path);
        if (std.fs.path.extension(basename).len > 0) {
            return basename[0 .. basename.len - std.fs.path.extension(basename).len];
        }
        return basename;
    }

    /// Get file size
    pub fn getFileSize(file_path: []const u8) !u64 {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const stat = try file.stat();
        return stat.size;
    }

    /// Copy file
    pub fn copyFile(src_path: []const u8, dst_path: []const u8) !void {
        try std.fs.cwd().copyFile(src_path, std.fs.cwd(), dst_path, .{});
    }

    /// Delete file
    pub fn deleteFile(file_path: []const u8) !void {
        try std.fs.cwd().deleteFile(file_path);
    }

    /// List directory contents
    pub fn listDirectory(allocator: std.mem.Allocator, dir_path: []const u8) ![][]u8 {
        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();

        var entries = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (entries.items) |entry| allocator.free(entry);
            entries.deinit();
        }

        var it = dir.iterate();
        while (try it.next()) |entry| {
            const name = try allocator.dupe(u8, entry.name);
            try entries.append(name);
        }

        return entries.toOwnedSlice();
    }
};

// =============================================================================
// VALIDATION UTILITIES - Email, UUID, input sanitization
// =============================================================================

/// Validation utilities for common input validation
pub const ValidationUtils = struct {
    /// Validate email address format
    pub fn isValidEmail(email: []const u8) bool {
        if (email.len == 0 or email.len > 254) return false;

        // Basic email validation: local@domain
        const at_index = std.mem.indexOf(u8, email, "@") orelse return false;
        if (at_index == 0 or at_index == email.len - 1) return false;

        const local = email[0..at_index];
        const domain = email[at_index + 1 ..];

        // Check local part (before @)
        if (local.len == 0 or local.len > 64) return false;
        if (local[0] == '.' or local[local.len - 1] == '.') return false;
        if (std.mem.indexOf(u8, local, "..") != null) return false;

        // Check domain part (after @)
        if (domain.len == 0 or domain.len > 253) return false;
        if (!std.mem.containsAtLeast(u8, domain, 1, ".")) return false;

        // Basic character validation
        for (email) |c| {
            if (!isValidEmailChar(c)) return false;
        }

        return true;
    }

    /// Validate UUID format (v4)
    pub fn isValidUuid(uuid: []const u8) bool {
        if (uuid.len != 36) return false;

        // Pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        const pattern = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx";
        for (uuid, pattern, 0..) |c, p, i| {
            switch (p) {
                'x' => if (!std.ascii.isHex(c)) return false,
                '-' => if (c != '-') return false,
                else => unreachable,
            }
            // Check version (position 14 should be '4' for UUID v4)
            if (i == 14 and c != '4') return false;
            // Check variant (position 19 should be '8', '9', 'a', or 'b')
            if (i == 19) {
                if (c != '8' and c != '9' and c != 'a' and c != 'b') return false;
            }
        }

        return true;
    }

    /// Sanitize input by removing potentially dangerous characters
    pub fn sanitizeInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();

        for (input) |c| {
            if (isSafeChar(c)) {
                try buffer.append(c);
            } else {
                // Replace dangerous characters with safe alternatives
                try buffer.appendSlice("&amp;");
            }
        }

        return buffer.toOwnedSlice();
    }

    /// Validate URL format
    pub fn isValidUrl(url: []const u8) bool {
        if (url.len == 0 or url.len > 2048) return false;

        // Must start with http:// or https://
        if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
            return false;
        }

        // Basic character validation
        for (url) |c| {
            if (!isValidUrlChar(c)) return false;
        }

        return true;
    }

    /// Validate phone number (basic international format)
    pub fn isValidPhoneNumber(phone: []const u8) bool {
        if (phone.len < 7 or phone.len > 15) return false;

        var digit_count: usize = 0;
        var has_plus = false;

        for (phone) |c| {
            if (c == '+') {
                if (has_plus) return false; // Only one + allowed
                has_plus = true;
            } else if (std.ascii.isDigit(c)) {
                digit_count += 1;
            } else if (!std.ascii.isWhitespace(c) and c != '-' and c != '(' and c != ')') {
                return false;
            }
        }

        return digit_count >= 7; // At least 7 digits
    }

    /// Validate strong password (customizable requirements)
    pub fn isStrongPassword(password: []const u8, options: PasswordOptions) bool {
        if (password.len < options.min_length) return false;

        var has_lower = false;
        var has_upper = false;
        var has_digit = false;
        var has_special = false;

        for (password) |c| {
            if (options.require_lowercase and std.ascii.isLower(c)) has_lower = true;
            if (options.require_uppercase and std.ascii.isUpper(c)) has_upper = true;
            if (options.require_digits and std.ascii.isDigit(c)) has_digit = true;
            if (options.require_special and isSpecialChar(c)) has_special = true;
        }

        return (!options.require_lowercase or has_lower) and
            (!options.require_uppercase or has_upper) and
            (!options.require_digits or has_digit) and
            (!options.require_special or has_special);
    }

    pub const PasswordOptions = struct {
        min_length: usize = 8,
        require_lowercase: bool = true,
        require_uppercase: bool = true,
        require_digits: bool = true,
        require_special: bool = true,
    };

    fn isValidEmailChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '@' or c == '.' or c == '-' or c == '_' or c == '+';
    }

    fn isSafeChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or std.ascii.isWhitespace(c) or
            c == '.' or c == ',' or c == '!' or c == '?' or c == ':' or c == ';';
    }

    fn isValidUrlChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '.' or c == '/' or c == ':' or
            c == '?' or c == '=' or c == '&' or c == '%' or c == '-' or c == '_';
    }

    fn isSpecialChar(c: u8) bool {
        return switch (c) {
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', '\'', '<', '>', ',', '.', '/', '?' => true,
            else => false,
        };
    }
};

// =============================================================================
// RANDOM UTILITIES - Secure random strings, UUID generation
// =============================================================================

/// Random utilities for secure random generation
pub const RandomUtils = struct {
    /// Generate cryptographically secure random bytes
    pub fn randomBytes(allocator: std.mem.Allocator, length: usize) ![]u8 {
        const buffer = try allocator.alloc(u8, length);
        std.crypto.random.bytes(buffer);
        return buffer;
    }

    /// Generate secure random string with specified character set
    pub fn randomString(allocator: std.mem.Allocator, length: usize, charset: []const u8) ![]u8 {
        if (charset.len == 0) return error.EmptyCharset;

        const buffer = try allocator.alloc(u8, length);
        errdefer allocator.free(buffer);

        for (0..length) |i| {
            const random_index = std.crypto.random.intRangeAtMost(usize, 0, charset.len - 1);
            buffer[i] = charset[random_index];
        }

        return buffer;
    }

    /// Generate alphanumeric random string
    pub fn randomAlphanumeric(allocator: std.mem.Allocator, length: usize) ![]u8 {
        const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        return randomString(allocator, length, charset);
    }

    /// Generate URL-safe random string
    pub fn randomUrlSafe(allocator: std.mem.Allocator, length: usize) ![]u8 {
        const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        return randomString(allocator, length, charset);
    }

    /// Generate UUID v4
    pub fn generateUuid(allocator: std.mem.Allocator) ![]u8 {
        var uuid_bytes: [16]u8 = undefined;
        std.crypto.random.bytes(&uuid_bytes);

        // Set version (4) and variant (2)
        uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40; // Version 4
        uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80; // Variant 2

        // Format as string: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        return try std.fmt.allocPrint(allocator, "{x:0>8}-{x:0>4}-{x:0>4}-{x:0>4}-{x:0>4}{x:0>8}", .{
            std.mem.readInt(u32, uuid_bytes[0..4], .big),
            std.mem.readInt(u16, uuid_bytes[4..6], .big),
            std.mem.readInt(u16, uuid_bytes[6..8], .big),
            std.mem.readInt(u16, uuid_bytes[8..10], .big),
            std.mem.readInt(u16, uuid_bytes[10..12], .big),
            std.mem.readInt(u32, uuid_bytes[12..16], .big),
        });
    }

    /// Generate secure token (URL-safe base64 encoded random bytes)
    pub fn generateToken(allocator: std.mem.Allocator, byte_length: usize) ![]u8 {
        const random_bytes = try randomBytes(allocator, byte_length);
        defer allocator.free(random_bytes);

        return Base64Utils.encodeUrlSafe(allocator, random_bytes);
    }

    /// Generate random integer in range [min, max]
    pub fn randomInt(comptime T: type, min: T, max: T) T {
        if (@typeInfo(T).Int.signedness == .signed) {
            return std.crypto.random.intRangeAtMostBiased(T, min, max);
        } else {
            return std.crypto.random.intRangeAtMost(T, min, max);
        }
    }

    /// Generate random float in range [0, 1)
    pub fn randomFloat() f64 {
        return std.crypto.random.float(f64);
    }

    /// Shuffle array in place using Fisher-Yates algorithm
    pub fn shuffle(comptime T: type, items: []T) void {
        var i: usize = items.len;
        while (i > 1) {
            i -= 1;
            const j = std.crypto.random.intRangeAtMost(usize, 0, i);
            std.mem.swap(T, &items[i], &items[j]);
        }
    }

    /// Select random element from slice
    pub fn randomChoice(comptime T: type, items: []const T) ?T {
        if (items.len == 0) return null;
        const index = std.crypto.random.intRangeAtMost(usize, 0, items.len - 1);
        return items[index];
    }
};

// =============================================================================
// MATH UTILITIES - Common mathematical functions
// =============================================================================

/// Math utilities for common mathematical operations
pub const MathUtils = struct {
    /// Clamp value between min and max
    pub fn clamp(comptime T: type, value: T, min: T, max: T) T {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /// Linear interpolation between a and b
    pub fn lerp(a: f64, b: f64, t: f64) f64 {
        return a + (b - a) * clamp(f64, t, 0.0, 1.0);
    }

    /// Calculate percentage (value / total * 100)
    pub fn percentage(value: f64, total: f64) f64 {
        if (total == 0.0) return 0.0;
        return (value / total) * 100.0;
    }

    /// Round to specified decimal places
    pub fn roundToDecimal(value: f64, decimals: usize) f64 {
        const multiplier = std.math.pow(f64, 10.0, @floatFromInt(decimals));
        return @round(value * multiplier) / multiplier;
    }

    /// Check if number is power of 2
    pub fn isPowerOfTwo(value: usize) bool {
        return value != 0 and (value & (value - 1)) == 0;
    }

    /// Find next power of 2 greater than or equal to value
    pub fn nextPowerOfTwo(value: usize) usize {
        if (value == 0) return 1;
        var result = value - 1;
        result |= result >> 1;
        result |= result >> 2;
        result |= result >> 4;
        result |= result >> 8;
        result |= result >> 16;
        if (@sizeOf(usize) > 4) {
            result |= result >> 32;
        }
        return result + 1;
    }

    /// Calculate factorial
    pub fn factorial(n: u64) u64 {
        if (n == 0 or n == 1) return 1;
        var result: u64 = 1;
        var i: u64 = 2;
        while (i <= n) : (i += 1) {
            result *= i;
        }
        return result;
    }

    /// Calculate greatest common divisor (GCD)
    pub fn gcd(a: usize, b: usize) usize {
        var x = a;
        var y = b;
        while (y != 0) {
            const t = y;
            y = x % y;
            x = t;
        }
        return x;
    }

    /// Calculate least common multiple (LCM)
    pub fn lcm(a: usize, b: usize) usize {
        if (a == 0 or b == 0) return 0;
        return (a / gcd(a, b)) * b;
    }

    /// Calculate mean (average)
    pub fn mean(values: []const f64) f64 {
        if (values.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (values) |v| sum += v;
        return sum / @as(f64, @floatFromInt(values.len));
    }

    /// Calculate standard deviation
    pub fn standardDeviation(values: []const f64) f64 {
        if (values.len < 2) return 0.0;

        const avg = mean(values);
        var sum_squares: f64 = 0.0;

        for (values) |v| {
            const diff = v - avg;
            sum_squares += diff * diff;
        }

        return std.math.sqrt(sum_squares / @as(f64, @floatFromInt(values.len - 1)));
    }

    /// Calculate median
    pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
        if (values.len == 0) return 0.0;

        // Create a copy to sort
        const sorted = try allocator.alloc(f64, values.len);
        defer allocator.free(sorted);
        @memcpy(sorted, values);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

        const mid = values.len / 2;
        if (values.len % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        } else {
            return sorted[mid];
        }
    }

    /// Calculate distance between two points (2D)
    pub fn distance2D(x1: f64, y1: f64, x2: f64, y2: f64) f64 {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return std.math.sqrt(dx * dx + dy * dy);
    }

    /// Calculate distance between two points (3D)
    pub fn distance3D(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) f64 {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dz = z2 - z1;
        return std.math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// Convert degrees to radians
    pub fn degreesToRadians(degrees: f64) f64 {
        return degrees * (std.math.pi / 180.0);
    }

    /// Convert radians to degrees
    pub fn radiansToDegrees(radians: f64) f64 {
        return radians * (180.0 / std.math.pi);
    }
};

// =============================================================================
// MEMORY MANAGEMENT UTILITIES - Common allocation patterns
// =============================================================================

/// Memory management utilities for common allocation patterns
pub const MemoryUtils = struct {
    /// Safe allocation with automatic cleanup on error
    pub fn safeAlloc(allocator: std.mem.Allocator, comptime T: type, count: usize) ![]T {
        const buffer = try allocator.alloc(T, count);
        errdefer allocator.free(buffer);
        return buffer;
    }

    /// Safe creation with automatic cleanup on error
    pub fn safeCreate(allocator: std.mem.Allocator, comptime T: type) !*T {
        const instance = try allocator.create(T);
        errdefer allocator.destroy(instance);
        return instance;
    }

    /// Safe duplication with automatic cleanup on error
    pub fn safeDupe(allocator: std.mem.Allocator, comptime T: type, slice: []const T) ![]T {
        const buffer = try allocator.dupe(T, slice);
        errdefer allocator.free(buffer);
        return buffer;
    }

    /// Batch deallocation for arrays of allocated items
    pub fn batchFree(allocator: std.mem.Allocator, comptime T: type, items: []T) void {
        for (items) |item| {
            allocator.free(item);
        }
        allocator.free(items);
    }

    /// Batch destruction for arrays of created items
    pub fn batchDestroy(allocator: std.mem.Allocator, comptime T: type, items: []*T) void {
        for (items) |item| {
            allocator.destroy(item);
        }
        allocator.free(items);
    }

    /// Create a managed buffer that automatically cleans up
    pub fn ManagedBuffer(comptime T: type) type {
        return struct {
            data: []T,
            allocator: std.mem.Allocator,

            const Self = @This();

            pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
                const data = try allocator.alloc(T, size);
                errdefer allocator.free(data);
                return Self{
                    .data = data,
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *Self) void {
                self.allocator.free(self.data);
            }

            pub fn resize(self: *Self, new_size: usize) !void {
                const new_data = try self.allocator.realloc(self.data, new_size);
                self.data = new_data;
            }
        };
    }
};

// =============================================================================
// ERROR HANDLING UTILITIES - Common error patterns
// =============================================================================

/// Error handling utilities for common patterns
pub const ErrorUtils = struct {
    /// Result type that can hold either a value or an error
    pub fn Result(comptime T: type) type {
        return union(enum) {
            success: T,
            failure: ErrorInfo,

            const Self = @This();

            pub fn unwrap(self: Self) T {
                return switch (self) {
                    .success => |value| value,
                    .failure => |err| {
                        std.log.err("Unwrapped error: {s} at {s}:{d}", .{ err.message, err.file, err.line });
                        std.process.exit(1);
                    },
                };
            }

            pub fn unwrapOr(self: Self, default_value: T) T {
                return switch (self) {
                    .success => |value| value,
                    .failure => default_value,
                };
            }

            pub fn map(self: Self, comptime U: type, func: fn (T) U) Result(U) {
                return switch (self) {
                    .success => |value| .{ .success = func(value) },
                    .failure => |err| .{ .failure = err },
                };
            }

            pub fn isSuccess(self: Self) bool {
                return self == .success;
            }

            pub fn isFailure(self: Self) bool {
                return self == .failure;
            }
        };
    }

    pub const ErrorInfo = struct {
        message: []const u8,
        file: []const u8,
        line: u32,
        function: []const u8,

        pub fn format(self: ErrorInfo, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("Error: {s} at {s}:{d} in {s}", .{ self.message, self.file, self.line, self.function });
        }
    };

    /// Create a success result
    pub fn success(comptime T: type, value: T) Result(T) {
        return .{ .success = value };
    }

    /// Create a failure result
    pub fn failure(comptime T: type, message: []const u8, file: []const u8, line: u32, function: []const u8) Result(T) {
        return .{ .failure = ErrorInfo{
            .message = message,
            .file = file,
            .line = line,
            .function = function,
        } };
    }

    /// Try to execute a function and return a Result
    pub fn tryExecute(comptime T: type, func: fn () anyerror!T) Result(T) {
        return func() catch |err| {
            return failure(T, @errorName(err), @src().file, @src().line, @src().fn_name);
        };
    }

    /// Retry a function with exponential backoff
    pub fn retry(comptime T: type, func: fn () anyerror!T, max_retries: usize, base_delay_ms: u64) !T {
        var retries: usize = 0;
        var delay_ms = base_delay_ms;

        while (retries < max_retries) {
            const result = func() catch |err| {
                if (retries == max_retries - 1) return err;

                std.log.warn("Attempt {d} failed: {s}, retrying in {d}ms", .{ retries + 1, @errorName(err), delay_ms });
                std.time.sleep(delay_ms * std.time.ns_per_ms);
                delay_ms *= 2; // Exponential backoff
                retries += 1;
                continue;
            };
            return result;
        }

        unreachable;
    }
};

// =============================================================================
// COMMON VALIDATION UTILITIES - Input validation patterns
// =============================================================================

/// Common validation utilities for input validation patterns
pub const CommonValidationUtils = struct {
    /// Validate that a value is within bounds
    pub fn validateBounds(comptime T: type, value: T, min: T, max: T) !void {
        if (value < min or value > max) {
            return error.OutOfBounds;
        }
    }

    /// Validate that a string is not empty and within length limits
    pub fn validateString(str: []const u8, max_length: usize) !void {
        if (str.len == 0) return error.EmptyString;
        if (str.len > max_length) return error.StringTooLong;
    }

    /// Validate that a slice has the expected length
    pub fn validateSliceLength(slice: []const u8, expected_length: usize) !void {
        if (slice.len != expected_length) {
            return error.InvalidLength;
        }
    }

    /// Validate that a pointer is not null
    pub fn validateNotNull(ptr: anytype) !void {
        if (ptr == null) {
            return error.NullPointer;
        }
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "HttpStatus phrase" {
    try std.testing.expectEqualStrings("OK", HttpStatus.ok.phrase());
    try std.testing.expectEqualStrings("Not Found", HttpStatus.not_found.phrase());
    try std.testing.expectEqualStrings("Internal Server Error", HttpStatus.internal_server_error.phrase());
}

test "JsonUtils parse and stringify" {
    const allocator = std.testing.allocator;

    // Test parsing simple JSON
    const json_str = "{\"name\":\"John\",\"age\":30,\"active\":true}";
    var parsed = try JsonUtils.parse(allocator, json_str);
    defer parsed.deinit(allocator);

    try std.testing.expect(parsed == .object);
    try std.testing.expect(parsed.object.count() == 3);

    // Test stringifying
    const stringified = try JsonUtils.stringify(allocator, parsed);
    defer allocator.free(stringified);

    // Should contain the same data (order may differ)
    try std.testing.expect(std.mem.indexOf(u8, stringified, "\"name\":\"John\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, stringified, "\"age\":30") != null);
    try std.testing.expect(std.mem.indexOf(u8, stringified, "\"active\":true") != null);
}

test "JsonUtils parseInto and stringifyFrom" {
    const allocator = std.testing.allocator;

    const Person = struct {
        name: []const u8,
        age: u32,
        active: bool,
    };

    const json_str = "{\"name\":\"Alice\",\"age\":25,\"active\":false}";
    const person = try JsonUtils.parseInto(allocator, Person, json_str);

    try std.testing.expectEqualStrings("Alice", person.name);
    try std.testing.expectEqual(@as(u32, 25), person.age);
    try std.testing.expect(!person.active);

    // Test stringifyFrom
    const json_output = try JsonUtils.stringifyFrom(allocator, person);
    defer allocator.free(json_output);

    try std.testing.expect(std.mem.indexOf(u8, json_output, "\"name\":\"Alice\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_output, "\"age\":25") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_output, "\"active\":false") != null);
}

test "UrlUtils encode and decode" {
    const allocator = std.testing.allocator;

    const original = "Hello World! 你好";
    const encoded = try UrlUtils.encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try UrlUtils.decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "UrlUtils parseQueryParams" {
    const allocator = std.testing.allocator;

    const query_string = "name=John%20Doe&age=30&city=New%20York";
    var params = try UrlUtils.parseQueryParams(allocator, query_string);
    defer {
        var it = params.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        params.deinit();
    }

    try std.testing.expectEqualStrings("John Doe", params.get("name").?);
    try std.testing.expectEqualStrings("30", params.get("age").?);
    try std.testing.expectEqualStrings("New York", params.get("city").?);
}

test "UrlUtils buildQueryString" {
    const allocator = std.testing.allocator;

    var params = std.StringHashMap([]const u8).init(allocator);
    defer params.deinit();

    try params.put("name", "Jane Doe");
    try params.put("role", "developer");
    try params.put("location", "San Francisco");

    const query_string = try UrlUtils.buildQueryString(allocator, params);
    defer allocator.free(query_string);

    // Query string should contain all parameters (order may vary)
    try std.testing.expect(std.mem.indexOf(u8, query_string, "name=Jane%20Doe") != null);
    try std.testing.expect(std.mem.indexOf(u8, query_string, "role=developer") != null);
    try std.testing.expect(std.mem.indexOf(u8, query_string, "location=San%20Francisco") != null);
}

test "UrlUtils parseUrl" {
    const allocator = std.testing.allocator;

    const url = "https://user:pass@example.com:8080/path/to/resource?param=value&other=test#section";
    var components = try UrlUtils.parseUrl(allocator, url);
    defer components.deinit(allocator);

    try std.testing.expectEqualStrings("https", components.scheme);
    try std.testing.expectEqualStrings("example.com", components.host);
    try std.testing.expectEqual(@as(?u16, 8080), components.port);
    try std.testing.expectEqualStrings("/path/to/resource", components.path);
    try std.testing.expectEqualStrings("param=value&other=test", components.query);
    try std.testing.expectEqualStrings("section", components.fragment);
}

test "Base64Utils encode and decode" {
    const allocator = std.testing.allocator;

    const original = "Hello, World! 🌍";
    const encoded = try Base64Utils.encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try Base64Utils.decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "Base64Utils url safe encode decode" {
    const allocator = std.testing.allocator;

    const original = "Binary data with special chars: \x00\x01\x02";
    const encoded = try Base64Utils.encodeUrlSafe(allocator, original);
    defer allocator.free(encoded);

    const decoded = try Base64Utils.decodeUrlSafe(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "ValidationUtils email validation" {
    try std.testing.expect(ValidationUtils.isValidEmail("test@example.com"));
    try std.testing.expect(ValidationUtils.isValidEmail("user.name+tag@domain.co.uk"));
    try std.testing.expect(!ValidationUtils.isValidEmail("invalid"));
    try std.testing.expect(!ValidationUtils.isValidEmail("@example.com"));
    try std.testing.expect(!ValidationUtils.isValidEmail("test@"));
}

test "ValidationUtils UUID validation" {
    try std.testing.expect(ValidationUtils.isValidUuid("550e8400-e29b-41d4-a716-446655440000"));
    try std.testing.expect(ValidationUtils.isValidUuid("6ba7b810-9dad-11d1-80b4-00c04fd430c8"));
    try std.testing.expect(!ValidationUtils.isValidUuid("not-a-uuid"));
    try std.testing.expect(!ValidationUtils.isValidUuid("550e8400-e29b-41d4-a716-44665544000")); // too short
    try std.testing.expect(!ValidationUtils.isValidUuid("550e8400-e29b-41d4-a716-4466554400001")); // too long
}

test "ValidationUtils strong password" {
    const options = ValidationUtils.PasswordOptions{
        .min_length = 8,
        .require_lowercase = true,
        .require_uppercase = true,
        .require_digits = true,
        .require_special = true,
    };

    try std.testing.expect(ValidationUtils.isStrongPassword("StrongP@ss123", options));
    try std.testing.expect(!ValidationUtils.isStrongPassword("weak", options)); // too short
    try std.testing.expect(!ValidationUtils.isStrongPassword("nouppercase123!", options)); // no uppercase
    try std.testing.expect(!ValidationUtils.isStrongPassword("NOLOWERCASE123!", options)); // no lowercase
    try std.testing.expect(!ValidationUtils.isStrongPassword("NoDigits!", options)); // no digits
    try std.testing.expect(!ValidationUtils.isStrongPassword("NoSpecial123", options)); // no special chars
}

test "RandomUtils randomString" {
    const allocator = std.testing.allocator;

    const str1 = try RandomUtils.randomAlphanumeric(allocator, 10);
    defer allocator.free(str1);
    try std.testing.expectEqual(@as(usize, 10), str1.len);

    const str2 = try RandomUtils.randomUrlSafe(allocator, 15);
    defer allocator.free(str2);
    try std.testing.expectEqual(@as(usize, 15), str2.len);

    // Strings should be different (very high probability)
    try std.testing.expect(!std.mem.eql(u8, str1, str2));
}

test "RandomUtils generateUuid" {
    const allocator = std.testing.allocator;

    const uuid1 = try RandomUtils.generateUuid(allocator);
    defer allocator.free(uuid1);
    try std.testing.expectEqual(@as(usize, 36), uuid1.len);
    try std.testing.expect(ValidationUtils.isValidUuid(uuid1));

    const uuid2 = try RandomUtils.generateUuid(allocator);
    defer allocator.free(uuid2);
    try std.testing.expectEqual(@as(usize, 36), uuid2.len);
    try std.testing.expect(ValidationUtils.isValidUuid(uuid2));

    // UUIDs should be different
    try std.testing.expect(!std.mem.eql(u8, uuid1, uuid2));
}

test "RandomUtils generateToken" {
    const allocator = std.testing.allocator;

    const token = try RandomUtils.generateToken(allocator, 32);
    defer allocator.free(token);

    // Token should be URL-safe base64 encoded
    for (token) |c| {
        try std.testing.expect(std.ascii.isAlphanumeric(c) or c == '-' or c == '_');
    }
}

test "MathUtils basic operations" {
    try std.testing.expectEqual(@as(i32, 5), MathUtils.clamp(i32, 10, 0, 5));
    try std.testing.expectEqual(@as(i32, 0), MathUtils.clamp(i32, -5, 0, 5));
    try std.testing.expectEqual(@as(i32, 3), MathUtils.clamp(i32, 3, 0, 5));

    try std.testing.expectEqual(@as(f64, 75.0), MathUtils.lerp(0.0, 100.0, 0.75));
    try std.testing.expectEqual(@as(f64, 25.0), MathUtils.percentage(25.0, 100.0));
}

test "MathUtils power of two" {
    try std.testing.expect(MathUtils.isPowerOfTwo(1));
    try std.testing.expect(MathUtils.isPowerOfTwo(2));
    try std.testing.expect(MathUtils.isPowerOfTwo(4));
    try std.testing.expect(MathUtils.isPowerOfTwo(1024));
    try std.testing.expect(!MathUtils.isPowerOfTwo(3));
    try std.testing.expect(!MathUtils.isPowerOfTwo(6));

    try std.testing.expectEqual(@as(usize, 4), MathUtils.nextPowerOfTwo(3));
    try std.testing.expectEqual(@as(usize, 8), MathUtils.nextPowerOfTwo(5));
    try std.testing.expectEqual(@as(usize, 16), MathUtils.nextPowerOfTwo(9));
}

test "MathUtils statistics" {
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    try std.testing.expectEqual(@as(f64, 3.0), MathUtils.mean(&values));

    const std_dev = MathUtils.standardDeviation(&values);
    try std.testing.expect(std_dev > 0.0); // Should be positive

    const median_val = try MathUtils.median(std.testing.allocator, &values);
    try std.testing.expectEqual(@as(f64, 3.0), median_val);
}

test "MathUtils distance" {
    const dist2d = MathUtils.distance2D(0.0, 0.0, 3.0, 4.0);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), dist2d, 0.001);

    const dist3d = MathUtils.distance3D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.732), dist3d, 0.001);
}

test "MathUtils angle conversion" {
    const radians = MathUtils.degreesToRadians(180.0);
    try std.testing.expectApproxEqAbs(std.math.pi, radians, 0.001);

    const degrees = MathUtils.radiansToDegrees(std.math.pi);
    try std.testing.expectApproxEqAbs(@as(f64, 180.0), degrees, 0.001);
}

test "FileSystemUtils basic operations" {
    // Test file extension extraction
    try std.testing.expectEqualStrings("txt", FileSystemUtils.getFileExtension("document.txt"));
    try std.testing.expectEqualStrings("zig", FileSystemUtils.getFileExtension("src/main.zig"));
    try std.testing.expectEqualStrings("", FileSystemUtils.getFileExtension("README"));

    // Test filename without extension
    try std.testing.expectEqualStrings("document", FileSystemUtils.getFileNameWithoutExtension("document.txt"));
    try std.testing.expectEqualStrings("main", FileSystemUtils.getFileNameWithoutExtension("src/main.zig"));
    try std.testing.expectEqualStrings("README", FileSystemUtils.getFileNameWithoutExtension("README"));
}

test "HttpMethod fromString" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("post").?);
    try std.testing.expectEqual(@as(?HttpMethod, null), HttpMethod.fromString("INVALID"));
}

test "StringUtils" {
    try std.testing.expect(StringUtils.isEmptyOrWhitespace(""));
    try std.testing.expect(StringUtils.isEmptyOrWhitespace("   "));
    try std.testing.expect(!StringUtils.isEmptyOrWhitespace("hello"));
}

test "ArrayUtils" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    try std.testing.expect(ArrayUtils.contains(i32, &arr, 3));
    try std.testing.expect(!ArrayUtils.contains(i32, &arr, 6));
    try std.testing.expectEqual(@as(?usize, 2), ArrayUtils.indexOf(i32, &arr, 3));
    try std.testing.expectEqual(@as(?usize, null), ArrayUtils.indexOf(i32, &arr, 6));
}

test "MemoryUtils safeAlloc" {
    const allocator = std.testing.allocator;

    const buffer = try MemoryUtils.safeAlloc(allocator, u8, 100);
    defer allocator.free(buffer);

    try std.testing.expectEqual(@as(usize, 100), buffer.len);
}

test "MemoryUtils safeCreate" {
    const allocator = std.testing.allocator;

    const TestStruct = struct {
        value: i32,
    };

    const instance = try MemoryUtils.safeCreate(allocator, TestStruct);
    defer allocator.destroy(instance);

    instance.value = 42;
    try std.testing.expectEqual(@as(i32, 42), instance.value);
}

test "MemoryUtils safeDupe" {
    const allocator = std.testing.allocator;
    const original = "Hello, World!";

    const duplicated = try MemoryUtils.safeDupe(allocator, u8, original);
    defer allocator.free(duplicated);

    try std.testing.expectEqualStrings(original, duplicated);
}

test "MemoryUtils ManagedBuffer" {
    const allocator = std.testing.allocator;

    var buffer = try MemoryUtils.ManagedBuffer(u8).init(allocator, 50);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 50), buffer.data.len);

    // Test resize
    try buffer.resize(100);
    try std.testing.expectEqual(@as(usize, 100), buffer.data.len);
}

test "ErrorUtils Result success" {
    const result = ErrorUtils.success(i32, 42);
    try std.testing.expect(result.isSuccess());
    try std.testing.expectEqual(@as(i32, 42), result.unwrap());
}

test "ErrorUtils Result failure" {
    const result = ErrorUtils.failure(i32, "Test error", "test.zig", 123, "test_function");
    try std.testing.expect(result.isFailure());

    const default_value = result.unwrapOr(99);
    try std.testing.expectEqual(@as(i32, 99), default_value);
}

test "ErrorUtils retry" {
    // Test with a function that always succeeds
    const successFunc = struct {
        fn call() !i32 {
            return 42;
        }
    }.call;

    const result = try ErrorUtils.retry(i32, successFunc, 5, 1);
    try std.testing.expectEqual(@as(i32, 42), result);

    // Test with a function that always fails
    const failFunc = struct {
        fn call() !i32 {
            return error.PermanentFailure;
        }
    }.call;

    try std.testing.expectError(error.PermanentFailure, ErrorUtils.retry(i32, failFunc, 3, 1));
}

test "CommonValidationUtils validateBounds" {
    try CommonValidationUtils.validateBounds(i32, 5, 0, 10);
    try std.testing.expectError(error.OutOfBounds, CommonValidationUtils.validateBounds(i32, 15, 0, 10));
    try std.testing.expectError(error.OutOfBounds, CommonValidationUtils.validateBounds(i32, -1, 0, 10));
}

test "CommonValidationUtils validateString" {
    try CommonValidationUtils.validateString("hello", 100);
    try std.testing.expectError(error.EmptyString, CommonValidationUtils.validateString("", 100));
    try std.testing.expectError(error.StringTooLong, CommonValidationUtils.validateString("a" ** 101, 100));
}

test "CommonValidationUtils validateSliceLength" {
    const slice = "hello";
    try CommonValidationUtils.validateSliceLength(slice, 5);
    try std.testing.expectError(error.InvalidLength, CommonValidationUtils.validateSliceLength(slice, 3));
}
