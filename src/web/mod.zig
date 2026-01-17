//! Web Module
//!
//! HTTP utilities and web service support.
//!
//! ## Features
//! - HTTP client
//! - Request/response handling
//! - JSON parsing utilities

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/web
const features_web = @import("../features/web/mod.zig");

pub const HttpClient = features_web.HttpClient;
pub const Response = features_web.Response;
pub const RequestOptions = features_web.RequestOptions;
pub const WeatherClient = features_web.WeatherClient;
pub const WeatherConfig = features_web.WeatherConfig;

pub const Error = error{
    WebDisabled,
    ConnectionFailed,
    RequestFailed,
    Timeout,
    InvalidUrl,
};

/// Web context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    client: ?*HttpClient = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.WebConfig) !*Context {
        if (!isEnabled()) return error.WebDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.client) |c| {
            c.deinit();
            self.allocator.destroy(c);
        }
        self.allocator.destroy(self);
    }

    /// Get or create the HTTP client.
    pub fn getClient(self: *Context) !*HttpClient {
        if (self.client) |c| return c;

        const client_ptr = try self.allocator.create(HttpClient);
        client_ptr.* = try HttpClient.init(self.allocator);
        self.client = client_ptr;
        return client_ptr;
    }

    /// Make an HTTP GET request.
    pub fn get(self: *Context, url: []const u8) !Response {
        const client_inst = try self.getClient();
        return client_inst.get(url);
    }

    /// Make an HTTP GET request with options.
    pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response {
        const client_inst = try self.getClient();
        return client_inst.getWithOptions(url, options);
    }

    /// Make an HTTP POST request with JSON body.
    pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response {
        const client_inst = try self.getClient();
        return client_inst.postJson(url, body);
    }

    /// Free a response body.
    pub fn freeResponse(self: *Context, response: Response) void {
        self.allocator.free(response.body);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_web;
}

pub fn isInitialized() bool {
    return features_web.isInitialized();
}

pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.WebDisabled;
    features_web.init(allocator) catch return error.WebDisabled;
}

pub fn deinit() void {
    features_web.deinit();
}
