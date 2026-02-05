const std = @import("std");

const T = @This(); // Alias for this struct to avoid ambiguity
pub const ParameterType = enum { string, integer, boolean, array, object, number };
pub const Parameter = struct {
    name: []const u8,
    type: ParameterType,
    required: bool = false,
    description: []const u8 = "",
    enum_values: ?[]const []const u8 = null,
};
pub const Tool = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    parameters: []const T.Parameter = &.{},
    execute: ?*const anyopaque = null,
};
pub const ToolResult = struct {
    success: bool = false,
    output: []const u8 = "",
    error_message: ?[]const u8 = null,
};
pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *@This()) void {}
    pub fn register(_: *@This(), _: *const T.Tool) !void {}
    pub fn get(_: *@This(), _: []const u8) ?*const T.Tool {
        return null;
    }
};
pub const TaskTool = struct {};
pub const Subagent = struct {};
pub const DiscordTools = struct {
    pub const send_message_tool = T.Tool{ .name = "discord_send_message" };
    pub const get_channel_tool = T.Tool{ .name = "discord_get_channel" };
    pub const list_guilds_tool = T.Tool{ .name = "discord_list_guilds" };
    pub const get_bot_info_tool = T.Tool{ .name = "discord_get_bot_info" };
    pub const execute_webhook_tool = T.Tool{ .name = "discord_execute_webhook" };
    pub const add_reaction_tool = T.Tool{ .name = "discord_add_reaction" };
    pub const get_messages_tool = T.Tool{ .name = "discord_get_messages" };
    pub fn registerAll(_: *T.ToolRegistry) void {}
};
pub const OsTools = struct {
    pub fn registerAll(_: *T.ToolRegistry) void {}
};
pub fn registerDiscordTools(registry: *T.ToolRegistry) !void {
    T.DiscordTools.registerAll(registry);
}
pub fn registerOsTools(registry: *T.ToolRegistry) !void {
    T.OsTools.registerAll(registry);
}
