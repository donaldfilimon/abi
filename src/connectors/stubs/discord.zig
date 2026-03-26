//! Discord connector stub — returns ConnectorsDisabled for all operations.

const std = @import("std");
const shared = @import("../shared.zig");

pub const DiscordError = error{
    MissingBotToken,
    ApiRequestFailed,
    InvalidResponse,
    RateLimited,
    Unauthorized,
    Forbidden,
    NotFound,
};

pub const Snowflake = []const u8;

// Minimal stub types for code that references them through the connector.
// Fields must match the real types for any field accessed by route handlers
// (discord_routes.zig serializers use anytype and access fields directly).

pub const Message = struct {
    id: Snowflake = "0",
    channel_id: Snowflake = "0",
    content: []const u8 = "",
    timestamp: []const u8 = "",
    author: User = .{},
};

pub const Channel = struct {
    id: Snowflake = "0",
    channel_type: u8 = 0,
    guild_id: ?Snowflake = null,
    name: ?[]const u8 = null,
    topic: ?[]const u8 = null,
};

pub const Guild = struct {
    id: Snowflake = "0",
    name: []const u8 = "",
    icon: ?[]const u8 = null,
    owner_id: Snowflake = "0",
    approximate_member_count: u32 = 0,
};

pub const User = struct {
    id: Snowflake = "0",
    username: []const u8 = "",
    discriminator: []const u8 = "0000",
    bot: bool = false,
    avatar: ?[]const u8 = null,
};

pub const Interaction = struct {
    id: Snowflake = "0",
    application_id: Snowflake = "0",
    data: ?InteractionData = null,
};

pub const InteractionData = struct {
    name: []const u8 = "",
};

pub const InteractionResponse = struct {
    callback_type: u8 = 4,
    content: ?[]const u8 = null,
};

pub const ApplicationCommand = struct {
    id: Snowflake = "0",
    command_type: u8 = 1,
    application_id: Snowflake = "0",
    name: []const u8 = "",
    description: []const u8 = "",
    options: []const ApplicationCommandOption = &.{},
    version: Snowflake = "0",
};

pub const ApplicationCommandOption = struct {
    option_type: u8 = 3,
    name: []const u8 = "",
    description: []const u8 = "",
    required: bool = false,
};

pub const Config = struct {
    bot_token: []u8,
    client_id: ?[]u8 = null,
    client_secret: ?[]u8 = null,
    public_key: ?[]u8 = null,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.secureFree(allocator, self.bot_token);
        if (self.client_id) |id| allocator.free(id);
        shared.secureFreeOptional(allocator, self.client_secret);
        shared.secureFreeOptional(allocator, self.public_key);
        self.* = undefined;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn createMessage(_: *Client, _: Snowflake, _: []const u8) !Message {
        return error.ConnectorsDisabled;
    }

    pub fn getChannel(_: *Client, _: Snowflake) !Channel {
        return error.ConnectorsDisabled;
    }

    pub fn getChannelMessages(_: *Client, _: Snowflake, _: ?u32) ![]Message {
        return error.ConnectorsDisabled;
    }

    pub fn createReaction(_: *Client, _: Snowflake, _: Snowflake, _: []const u8) !void {
        return error.ConnectorsDisabled;
    }

    pub fn triggerTypingIndicator(_: *Client, _: Snowflake) !void {
        return error.ConnectorsDisabled;
    }

    pub fn executeWebhook(_: *Client, _: Snowflake, _: []const u8, _: []const u8) !void {
        return error.ConnectorsDisabled;
    }

    pub fn getCurrentUser(_: *Client) !User {
        return error.ConnectorsDisabled;
    }

    pub fn getCurrentUserGuilds(_: *Client) ![]Guild {
        return error.ConnectorsDisabled;
    }
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

// ============================================================================
// Gateway Stubs
// ============================================================================

pub const GatewayState = enum {
    disconnected,
    connecting,
    identifying,
    connected,
    resuming,
};

pub const GatewayEventHandler = struct {
    ctx: ?*anyopaque = null,
    on_message_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_interaction_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_ready: ?*const fn (?*anyopaque, []const u8) void = null,
    on_guild_create: ?*const fn (?*anyopaque, []const u8) void = null,
    on_resumed: ?*const fn (?*anyopaque) void = null,
};

pub const GatewayClient = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: u32, _: GatewayEventHandler) !GatewayClient {
        return error.ConnectorsDisabled;
    }
    pub fn connect(_: *GatewayClient) !void {
        return error.ConnectorsDisabled;
    }
    pub fn disconnect(_: *GatewayClient) void {}
    pub fn processPayload(_: *GatewayClient, _: []const u8) !void {
        return error.ConnectorsDisabled;
    }
    pub fn deinit(_: *GatewayClient) void {}
    pub fn getState(_: *const GatewayClient) GatewayState {
        return .disconnected;
    }
    pub fn getSequenceNumber(_: *const GatewayClient) ?u64 {
        return null;
    }
};

test {
    std.testing.refAllDecls(@This());
}
