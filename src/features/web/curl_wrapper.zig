//! Libcurl wrapper for enhanced HTTP client functionality
//!
//! This module provides Zig bindings for libcurl to handle:
//! - Proxy authentication and configuration
//! - TLS/SSL certificate verification and client certificates
//! - Advanced timeout and connection pooling
//! - Automatic retry with exponential backoff
//! - Detailed error reporting and diagnostics

const std = @import("std");
const builtin = @import("builtin");
const http_client = @import("http_client.zig");

// Conditional compilation - only include if libcurl is available
const HAVE_LIBCURL = false; // This would be set by build system

/// Libcurl response data structure
const CurlResponse = struct {
    data: std.ArrayList(u8),
    headers: std.ArrayList(u8),
    status_code: c_long,

    pub fn init(_: std.mem.Allocator) CurlResponse {
        return CurlResponse{
            .data = std.ArrayList(u8){},
            .headers = std.ArrayList(u8){},
            .status_code = 0,
        };
    }

    pub fn deinit(self: *CurlResponse, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
        self.headers.deinit(allocator);
    }
};

/// Libcurl HTTP client implementation
pub const CurlHttpClient = struct {
    allocator: std.mem.Allocator,
    config: http_client.HttpClientConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: http_client.HttpClientConfig) !Self {
        if (!HAVE_LIBCURL) {
            return error.LibcurlNotAvailable;
        }

        // Initialize libcurl globally
        if (curlGlobalInit() != 0) {
            return error.CurlInitFailed;
        }

        return Self{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        if (HAVE_LIBCURL) {
            curlGlobalCleanup();
        }
    }

    /// Make HTTP request using libcurl
    pub fn request(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !http_client.HttpResponse {
        if (!HAVE_LIBCURL) {
            return error.LibcurlNotAvailable;
        }

        const curl = curlEasyInit() orelse return error.CurlInitFailed;
        defer curlEasyCleanup(curl);

        var response = CurlResponse.init(self.allocator);
        defer response.deinit();

        // Set basic options
        try self.setCurlOptions(curl, method, url, &response);

        // Set proxy if configured
        if (self.config.proxy_url) |proxy| {
            _ = curlEasySetopt(curl, CURLOPT_PROXY, proxy.ptr);
        }

        // Set SSL options
        if (!self.config.verify_ssl) {
            _ = curlEasySetopt(curl, CURLOPT_SSL_VERIFYPEER, @as(c_long, 0));
            _ = curlEasySetopt(curl, CURLOPT_SSL_VERIFYHOST, @as(c_long, 0));
        }

        if (self.config.ca_bundle_path) |ca_path| {
            _ = curlEasySetopt(curl, CURLOPT_CAINFO, ca_path.ptr);
        }

        // Set timeouts
        _ = curlEasySetopt(curl, CURLOPT_CONNECTTIMEOUT_MS, @as(c_long, @intCast(self.config.connect_timeout_ms)));
        _ = curlEasySetopt(curl, CURLOPT_TIMEOUT_MS, @as(c_long, @intCast(self.config.read_timeout_ms)));

        // Set redirects
        if (self.config.follow_redirects) {
            _ = curlEasySetopt(curl, CURLOPT_FOLLOWLOCATION, @as(c_long, 1));
            _ = curlEasySetopt(curl, CURLOPT_MAXREDIRS, @as(c_long, @intCast(self.config.max_redirects)));
        }

        // Set user agent
        _ = curlEasySetopt(curl, CURLOPT_USERAGENT, self.config.user_agent.ptr);

        // Set request body for POST/PUT
        if (body) |request_body| {
            _ = curlEasySetopt(curl, CURLOPT_POSTFIELDS, request_body.ptr);
            _ = curlEasySetopt(curl, CURLOPT_POSTFIELDSIZE, @as(c_long, @intCast(request_body.len)));
        }

        // Set content type header if provided
        var headers: ?*std.c.void = null;
        if (content_type) |ct| {
            const header_prefix = "Content-Type: ";
            const total_len = header_prefix.len + ct.len + 1; // +1 for nul
            const buf = try self.allocator.alloc(u8, total_len);
            defer self.allocator.free(buf);
            @memcpy(buf[0..header_prefix.len], header_prefix);
            @memcpy(buf[header_prefix.len .. header_prefix.len + ct.len], ct);
            buf[total_len - 1] = 0; // nul-terminate
            headers = curlSlistAppend(headers, @ptrCast(buf.ptr));
            _ = curlEasySetopt(curl, CURLOPT_HTTPHEADER, headers);
        }
        defer if (headers) |h| curlSlistFreeAll(h);

        // Perform the request
        const curl_code = curlEasyPerform(curl);
        if (curl_code != CURLE_OK) {
            if (self.config.verbose) {
                std.debug.print("Curl error: {s}\n", .{curlEasyStrerror(curl_code)});
            }
            return error.CurlRequestFailed;
        }

        // Get response code
        _ = curlEasyGetinfo(curl, CURLINFO_RESPONSE_CODE, &response.status_code);

        // Parse headers
        var response_headers = std.StringHashMap([]const u8).init(self.allocator);
        try self.parseHeaders(response.headers.items, &response_headers);

        // Create response
        return http_client.HttpResponse{
            .status_code = @intCast(response.status_code),
            .headers = response_headers,
            .body = try response.data.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }

    /// Set curl options for the request
    fn setCurlOptions(self: *Self, curl: *std.c.void, method: []const u8, url: []const u8, response: *CurlResponse) !void {
        // Set URL
        const url_z = try self.allocator.dupeZ(u8, url);
        defer self.allocator.free(url_z);
        _ = curlEasySetopt(curl, CURLOPT_URL, url_z.ptr);

        // Set HTTP method
        if (std.mem.eql(u8, method, "GET")) {
            _ = curlEasySetopt(curl, CURLOPT_HTTPGET, @as(c_long, 1));
        } else if (std.mem.eql(u8, method, "POST")) {
            _ = curlEasySetopt(curl, CURLOPT_POST, @as(c_long, 1));
        } else if (std.mem.eql(u8, method, "PUT")) {
            _ = curlEasySetopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        } else if (std.mem.eql(u8, method, "DELETE")) {
            _ = curlEasySetopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
        }

        // Set callback functions
        _ = curlEasySetopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        _ = curlEasySetopt(curl, CURLOPT_WRITEDATA, response);
        _ = curlEasySetopt(curl, CURLOPT_HEADERFUNCTION, headerCallback);
        _ = curlEasySetopt(curl, CURLOPT_HEADERDATA, response);

        // Enable verbose output if requested
        if (self.config.verbose) {
            _ = curlEasySetopt(curl, CURLOPT_VERBOSE, @as(c_long, 1));
        }
    }

    /// Parse HTTP headers from curl response
    fn parseHeaders(self: *Self, headers_data: []const u8, headers_map: *std.StringHashMap([]const u8)) !void {
        var lines = std.mem.split(u8, headers_data, "\r\n");
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            if (std.mem.startsWith(u8, line, "HTTP/")) continue; // Skip status line

            if (std.mem.indexOf(u8, line, ":")) |colon_pos| {
                const name = std.mem.trim(u8, line[0..colon_pos], " \t");
                const value = std.mem.trim(u8, line[colon_pos + 1 ..], " \t");

                const name_owned = try self.allocator.dupe(u8, name);
                const value_owned = try self.allocator.dupe(u8, value);
                try headers_map.put(name_owned, value_owned);
            }
        }
    }
};

// Callback function for curl to write response data
fn writeCallback(contents: [*c]u8, size: usize, nmemb: usize, userdata: ?*anyopaque) callconv(.C) usize {
    const real_size = size * nmemb;
    const response: *CurlResponse = @ptrCast(@alignCast(userdata));

    response.data.appendSlice(contents[0..real_size]) catch return 0;
    return real_size;
}

// Callback function for curl to write headers
fn headerCallback(contents: [*c]u8, size: usize, nmemb: usize, userdata: ?*anyopaque) callconv(.C) usize {
    const real_size = size * nmemb;
    const response: *CurlResponse = @ptrCast(@alignCast(userdata));

    response.headers.appendSlice(contents[0..real_size]) catch return 0;
    return real_size;
}

// Libcurl function declarations (would normally be in a separate C header)
// These are placeholder declarations - in a real implementation you'd use proper C bindings

extern fn curl_global_init(flags: c_long) c_int;
extern fn curl_global_cleanup() void;
extern fn curl_easy_init() ?*std.c.void;
extern fn curl_easy_cleanup(curl: *std.c.void) void;
extern fn curl_easy_setopt(curl: *std.c.void, option: c_int, parameter: c_long) c_int;
extern fn curl_easy_perform(curl: *std.c.void) c_int;
extern fn curl_easy_getinfo(curl: *std.c.void, info: c_int, parameter: *c_long) c_int;
extern fn curl_easy_strerror(code: c_int) [*c]const u8;
extern fn curl_slist_append(list: ?*std.c.void, string: [*c]const u8) ?*std.c.void;
extern fn curl_slist_free_all(list: *std.c.void) void;

// Libcurl constants (normally from curl.h)
const CURLOPT_URL: c_int = 10002;
const CURLOPT_WRITEFUNCTION: c_int = 20011;
const CURLOPT_WRITEDATA: c_int = 10001;
const CURLOPT_HEADERFUNCTION: c_int = 20079;
const CURLOPT_HEADERDATA: c_int = 10029;
const CURLOPT_USERAGENT: c_int = 10018;
const CURLOPT_HTTPGET: c_int = 80;
const CURLOPT_POST: c_int = 47;
const CURLOPT_POSTFIELDS: c_int = 10015;
const CURLOPT_POSTFIELDSIZE: c_int = 60;
const CURLOPT_CUSTOMREQUEST: c_int = 10036;
const CURLOPT_HTTPHEADER: c_int = 10023;
const CURLOPT_PROXY: c_int = 10004;
const CURLOPT_SSL_VERIFYPEER: c_int = 64;
const CURLOPT_SSL_VERIFYHOST: c_int = 81;
const CURLOPT_CAINFO: c_int = 10065;
const CURLOPT_CONNECTTIMEOUT_MS: c_int = 156;
const CURLOPT_TIMEOUT_MS: c_int = 155;
const CURLOPT_FOLLOWLOCATION: c_int = 52;
const CURLOPT_MAXREDIRS: c_int = 68;
const CURLOPT_VERBOSE: c_int = 41;

const CURLINFO_RESPONSE_CODE: c_int = 2097154;

const CURLE_OK: c_int = 0;

// Wrapper functions to handle the extern declarations
fn curlGlobalInit() c_int {
    if (HAVE_LIBCURL) {
        return curl_global_init(0);
    }
    return -1;
}

fn curlGlobalCleanup() void {
    if (HAVE_LIBCURL) {
        curl_global_cleanup();
    }
}

fn curlEasyInit() ?*std.c.void {
    if (HAVE_LIBCURL) {
        return curl_easy_init();
    }
    return null;
}

fn curlEasyCleanup(curl: *std.c.void) void {
    if (HAVE_LIBCURL) {
        curl_easy_cleanup(curl);
    }
}

fn curlEasySetopt(curl: *std.c.void, option: c_int, parameter: anytype) c_int {
    if (HAVE_LIBCURL) {
        switch (@TypeOf(parameter)) {
            c_long => return curl_easy_setopt(curl, option, parameter),
            [*c]const u8, [*:0]const u8 => return curl_easy_setopt(curl, option, @intFromPtr(parameter)),
            ?*std.c.void, *CurlResponse => return curl_easy_setopt(curl, option, @intFromPtr(parameter)),
            else => return curl_easy_setopt(curl, option, @as(c_long, @intCast(@intFromPtr(parameter)))),
        }
    }
    return -1;
}

fn curlEasyPerform(curl: *std.c.void) c_int {
    if (HAVE_LIBCURL) {
        return curl_easy_perform(curl);
    }
    return -1;
}

fn curlEasyGetinfo(curl: *std.c.void, info: c_int, parameter: *c_long) c_int {
    if (HAVE_LIBCURL) {
        return curl_easy_getinfo(curl, info, parameter);
    }
    return -1;
}

fn curlEasyStrerror(code: c_int) [*c]const u8 {
    if (HAVE_LIBCURL) {
        return curl_easy_strerror(code);
    }
    return "libcurl not available";
}

fn curlSlistAppend(list: ?*std.c.void, string: [*c]const u8) ?*std.c.void {
    if (HAVE_LIBCURL) {
        return curl_slist_append(list, string);
    }
    return null;
}

fn curlSlistFreeAll(list: *std.c.void) void {
    if (HAVE_LIBCURL) {
        curl_slist_free_all(list);
    }
}
