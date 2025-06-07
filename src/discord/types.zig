const std = @import("std");

pub const OpCode = enum(u8) {
    dispatch = 0,
    heartbeat = 1,
    identify = 2,
    presence_update = 3,
    voice_state_update = 4,
    resume = 6,
    reconnect = 7,
    request_guild_members = 8,
    invalid_session = 9,
    hello = 10,
    heartbeat_ack = 11,
};

pub const GatewayHello = struct {
    heartbeat_interval: u32,
};

pub const Identify = struct {
    token: []const u8,
    intents: u32,
    properties: struct {
        os: []const u8 = "linux",
        browser: []const u8 = "zig",
        device: []const u8 = "zig",
    } = .{},
};
