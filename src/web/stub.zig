//! Web Stub Module

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    WebDisabled,
    ConnectionFailed,
    RequestFailed,
    Timeout,
    InvalidUrl,
};

pub const HttpClient = struct {};
pub const Response = struct {
    status: u16 = 0,
    body: []const u8 = "",
};
pub const RequestOptions = struct {};
pub const WeatherClient = struct {};
pub const WeatherConfig = struct {};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.WebConfig) Error!*Context {
        return error.WebDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getClient(_: *Context) Error!*HttpClient {
        return error.WebDisabled;
    }

    pub fn get(_: *Context, _: []const u8) Error!Response {
        return error.WebDisabled;
    }

    pub fn getWithOptions(_: *Context, _: []const u8, _: RequestOptions) Error!Response {
        return error.WebDisabled;
    }

    pub fn postJson(_: *Context, _: []const u8, _: []const u8) Error!Response {
        return error.WebDisabled;
    }

    pub fn freeResponse(_: *Context, _: Response) void {}
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.WebDisabled;
}

pub fn deinit() void {}
