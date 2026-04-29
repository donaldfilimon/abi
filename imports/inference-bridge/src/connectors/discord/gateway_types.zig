//! Discord Gateway Types
//!
//! Types specific to the Discord Gateway WebSocket connection,
//! including connection state, event handlers, and opcodes.

const std = @import("std");
const types = @import("types.zig");

/// Re-export the canonical GatewayOpcode from types.zig
pub const GatewayOpcode = types.GatewayOpcode;

/// Connection state of the gateway client.
pub const GatewayState = enum {
    disconnected,
    connecting,
    identifying,
    connected,
    resuming,
};

/// Callbacks for gateway events. Each function pointer takes an opaque context.
pub const GatewayEventHandler = struct {
    ctx: ?*anyopaque = null,
    on_message_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_interaction_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_ready: ?*const fn (?*anyopaque, []const u8) void = null,
    on_guild_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_resumed: ?*const fn (?*anyopaque) void = null,
};

/// Payload structure for the HELLO opcode (op 10).
pub const HelloPayload = struct {
    heartbeat_interval: u64 = 0,
};

test "gateway state transitions" {
    const state: GatewayState = .disconnected;
    try std.testing.expectEqual(GatewayState.disconnected, state);
}

test "gateway event handler defaults" {
    const handler = GatewayEventHandler{};
    try std.testing.expect(handler.ctx == null);
    try std.testing.expect(handler.on_message_create == null);
    try std.testing.expect(handler.on_interaction_create == null);
    try std.testing.expect(handler.on_ready == null);
    try std.testing.expect(handler.on_guild_create == null);
    try std.testing.expect(handler.on_resumed == null);
}

test "gateway opcode values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(GatewayOpcode.DISPATCH));
    try std.testing.expectEqual(@as(u8, 10), @intFromEnum(GatewayOpcode.HELLO));
    try std.testing.expectEqual(@as(u8, 11), @intFromEnum(GatewayOpcode.HEARTBEAT_ACK));
}

test {
    std.testing.refAllDecls(@This());
}
