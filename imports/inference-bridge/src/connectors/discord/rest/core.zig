//! Discord REST API — Core Client Infrastructure
//!
//! Shared HTTP client, configuration, and request helpers used by all
//! endpoint submodules.

const std = @import("std");
const types = @import("../types.zig");
const shared = @import("../../shared.zig");
const async_http = @import("../../../foundation/mod.zig").utils.async_http;

pub const DiscordError = types.DiscordError;
pub const GatewayIntent = types.GatewayIntent;

// ============================================================================
// Configuration
// ============================================================================

pub const Config = struct {
    bot_token: []u8,
    client_id: ?[]u8 = null,
    client_secret: ?[]u8 = null,
    public_key: ?[]u8 = null,
    api_version: u8 = 10,
    timeout_ms: u32 = 30_000,
    intents: u32 = GatewayIntent.ALL_UNPRIVILEGED,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        // Use shared secure cleanup helpers for sensitive credentials
        shared.secureFree(allocator, self.bot_token);
        if (self.client_id) |id| allocator.free(id);
        shared.secureFreeOptional(allocator, self.client_secret);
        shared.secureFreeOptional(allocator, self.public_key);
        self.* = undefined;
    }

    pub fn getBaseUrl(self: *const Config) []const u8 {
        _ = self;
        return "https://discord.com/api/v10";
    }
};

// ============================================================================
// Core Client (shared state for all endpoint submodules)
// ============================================================================

pub const ClientCore = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !ClientCore {
        const http = try async_http.AsyncHttpClient.init(allocator);
        errdefer http.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .http = http,
        };
    }

    pub fn deinit(self: *ClientCore) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    // ========================================================================
    // HTTP Helpers
    // ========================================================================

    pub fn makeRequest(
        self: *ClientCore,
        method: async_http.Method,
        endpoint: []const u8,
    ) !async_http.HttpRequest {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}{s}",
            .{ self.config.getBaseUrl(), endpoint },
        );
        errdefer self.allocator.free(url);

        var request = try async_http.HttpRequest.init(self.allocator, method, url);
        errdefer request.deinit();

        const auth = try std.fmt.allocPrint(
            self.allocator,
            "Bot {s}",
            .{self.config.bot_token},
        );
        defer self.allocator.free(auth);
        try request.setHeader("Authorization", auth);
        try request.setHeader("User-Agent", "DiscordBot (https://github.com/abi, 1.0)");

        return request;
    }

    pub fn doRequest(
        self: *ClientCore,
        request: *async_http.HttpRequest,
    ) !async_http.HttpResponse {
        const response = try self.http.fetch(request);

        if (response.status_code == 401) {
            return DiscordError.Unauthorized;
        } else if (response.status_code == 403) {
            return DiscordError.Forbidden;
        } else if (response.status_code == 404) {
            return DiscordError.NotFound;
        } else if (response.status_code == 429) {
            return DiscordError.RateLimitExceeded;
        } else if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return response;
    }
};

test {
    std.testing.refAllDecls(@This());
}
