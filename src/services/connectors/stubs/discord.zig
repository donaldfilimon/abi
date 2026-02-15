//! Discord connector stub — returns ConnectorsDisabled for all operations.

const std = @import("std");
const shared = @import("../shared.zig");

pub const DiscordError = error{
    MissingToken,
    ApiRequestFailed,
    InvalidResponse,
    RateLimited,
    Unauthorized,
    Forbidden,
    NotFound,
};

pub const Snowflake = u64;

pub const Config = struct {
    token: []u8,
    application_id: ?Snowflake = null,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.secureFree(allocator, self.token);
        self.* = undefined;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}
};

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return error.ConnectorsDisabled;
}

pub fn isAvailable() bool {
    return false;
}
