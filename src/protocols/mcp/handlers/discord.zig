//! MCP Discord Tool Handlers
//!
//! Exposes Discord REST API operations as MCP tools for Claude Code.
//! Requires `DISCORD_BOT_TOKEN` env var. Comptime-gated by `feat_connectors`.

const std = @import("std");
const build_options = @import("build_options");
const json_utils = @import("../../../foundation/mod.zig").utils.json;

const registry = @import("../registry.zig");
const discord = if (build_options.feat_connectors)
    @import("../../../connectors/discord/mod.zig")
else
    @import("../../../connectors/stubs/discord.zig");

/// MCP Tool Definitions for Discord REST API operations
pub const tools = [_]registry.ToolDef{
    .{ .name = "discord_send_message", .description = "Send a message to a Discord channel", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"content":{"type":"string","description":"Message content"}},"required":["channel_id","content"]}
    , .handler = handleDiscordSendMessage },
    .{ .name = "discord_send_embed", .description = "Send a rich embed message to a Discord channel", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"title":{"type":"string","description":"Embed title"},"description":{"type":"string","description":"Embed description"},"content":{"type":"string","description":"Optional text content"},"color":{"type":"integer","description":"Embed color (decimal)"}},"required":["channel_id","title"]}
    , .handler = handleDiscordSendEmbed },
    .{ .name = "discord_edit_message", .description = "Edit an existing Discord message", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"},"content":{"type":"string","description":"New content"}},"required":["channel_id","message_id","content"]}
    , .handler = handleDiscordEditMessage },
    .{ .name = "discord_delete_message", .description = "Delete a Discord message", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"}},"required":["channel_id","message_id"]}
    , .handler = handleDiscordDeleteMessage },
    .{ .name = "discord_get_messages", .description = "Get recent messages from a Discord channel", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"limit":{"type":"integer","description":"Max messages (1-100, default 50)","default":50}},"required":["channel_id"]}
    , .handler = handleDiscordGetMessages },
    .{ .name = "discord_get_channel", .description = "Get Discord channel details", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"}},"required":["channel_id"]}
    , .handler = handleDiscordGetChannel },
    .{ .name = "discord_react", .description = "Add a reaction to a Discord message", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"},"emoji":{"type":"string","description":"Emoji (e.g. %F0%9F%91%8D or custom:name:id)"}},"required":["channel_id","message_id","emoji"]}
    , .handler = handleDiscordReact },
    .{ .name = "discord_typing", .description = "Show typing indicator in a Discord channel", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string"}},"required":["channel_id"]}
    , .handler = handleDiscordTyping },
    .{ .name = "discord_get_guild", .description = "Get Discord server (guild) details", .input_schema =
    \\{"type":"object","properties":{"guild_id":{"type":"string","description":"Discord guild/server ID"}},"required":["guild_id"]}
    , .handler = handleDiscordGetGuild },
    .{ .name = "discord_get_guild_channels", .description = "List all channels in a Discord server", .input_schema =
    \\{"type":"object","properties":{"guild_id":{"type":"string","description":"Discord guild/server ID"}},"required":["guild_id"]}
    , .handler = handleDiscordGetGuildChannels },
    .{ .name = "discord_list_guilds", .description = "List all Discord servers the bot is in", .input_schema =
    \\{"type":"object","properties":{},"required":[]}
    , .handler = handleDiscordListGuilds },
    .{ .name = "discord_get_bot", .description = "Get the bot's own Discord user info", .input_schema =
    \\{"type":"object","properties":{},"required":[]}
    , .handler = handleDiscordGetBot },
    .{ .name = "discord_create_dm", .description = "Open a DM channel with a Discord user", .input_schema =
    \\{"type":"object","properties":{"user_id":{"type":"string","description":"Discord user ID"}},"required":["user_id"]}
    , .handler = handleDiscordCreateDM },
    .{ .name = "discord_execute_webhook", .description = "Execute a Discord webhook", .input_schema =
    \\{"type":"object","properties":{"webhook_id":{"type":"string"},"token":{"type":"string"},"content":{"type":"string"}},"required":["webhook_id","token","content"]}
    , .handler = handleDiscordExecuteWebhook },
    .{ .name = "discord_get_member", .description = "Get a member's details in a Discord server", .input_schema =
    \\{"type":"object","properties":{"guild_id":{"type":"string"},"user_id":{"type":"string"}},"required":["guild_id","user_id"]}
    , .handler = handleDiscordGetMember },
    .{ .name = "discord_register_command", .description = "Register a global slash command for the bot", .input_schema =
    \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"},"name":{"type":"string","description":"Command name (lowercase, 1-32 chars)"},"description":{"type":"string","description":"Command description (1-100 chars)"}},"required":["application_id","name","description"]}
    , .handler = handleDiscordRegisterCommand },
    .{ .name = "discord_list_commands", .description = "List all registered global slash commands", .input_schema =
    \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"}},"required":["application_id"]}
    , .handler = handleDiscordListCommands },
    .{ .name = "discord_delete_command", .description = "Delete a global slash command by ID", .input_schema =
    \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"},"command_id":{"type":"string","description":"Command ID to delete"}},"required":["application_id","command_id"]}
    , .handler = handleDiscordDeleteCommand },
    .{ .name = "discord_get_message", .description = "Get a specific Discord message by ID", .input_schema =
    \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"}},"required":["channel_id","message_id"]}
    , .handler = handleDiscordGetMessage },
};

const DiscordError = if (build_options.feat_connectors)
    @import("../../../connectors/discord/types.zig").DiscordError
else
    error{};

fn getStringParam(p: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const val = p.get(key) orelse return null;
    return if (val == .string) val.string else null;
}

fn getIntParam(p: std.json.ObjectMap, key: []const u8) ?i64 {
    const val = p.get(key) orelse return null;
    return if (val == .integer) val.integer else null;
}

fn requireString(p: ?std.json.ObjectMap, key: []const u8) ![]const u8 {
    const map = p orelse return error.InvalidParams;
    return getStringParam(map, key) orelse error.InvalidParams;
}

fn appendError(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), msg: []const u8) !void {
    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":\"");
    try json_utils.appendJsonEscaped(allocator, out, msg);
    try out.appendSlice(allocator, "\"}],\"isError\":true}");
}

fn withClient(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8)) !?discord.Client {
    if (!build_options.feat_connectors) {
        try appendError(allocator, out, "connectors disabled");
        return null;
    }
    return discord.createClient(allocator) catch |err| {
        const msg = switch (err) {
            DiscordError.MissingBotToken => "DISCORD_BOT_TOKEN env var not set",
            DiscordError.MissingClientId => "DISCORD_CLIENT_ID env var not set",
            DiscordError.MissingClientSecret => "DISCORD_CLIENT_SECRET env var not set",
            else => "failed to create Discord client",
        };
        try appendError(allocator, out, msg);
        return null;
    };
}

// ── JSON Serialization Helpers ──

fn serializeUser(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), user: anytype) !void {
    try out.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, user.id);
    try out.appendSlice(allocator, "\",\"username\":\"");
    try json_utils.appendJsonEscaped(allocator, out, user.username);
    try out.appendSlice(allocator, "\",\"discriminator\":\"");
    try json_utils.appendJsonEscaped(allocator, out, user.discriminator);
    try out.appendSlice(allocator, "\",\"bot\":");
    try out.appendSlice(allocator, if (user.bot) "true" else "false");
    if (user.avatar) |av| {
        try out.appendSlice(allocator, ",\"avatar\":\"");
        try json_utils.appendJsonEscaped(allocator, out, av);
        try out.appendSlice(allocator, "\"");
    }
    try out.appendSlice(allocator, "}");
}

fn serializeMessage(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), msg: anytype) !void {
    try out.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, msg.id);
    try out.appendSlice(allocator, "\",\"channel_id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, msg.channel_id);
    try out.appendSlice(allocator, "\",\"content\":\"");
    try json_utils.appendJsonEscaped(allocator, out, msg.content);
    try out.appendSlice(allocator, "\",\"timestamp\":\"");
    try json_utils.appendJsonEscaped(allocator, out, msg.timestamp);
    try out.appendSlice(allocator, "\",\"author\":");
    try serializeUser(allocator, out, &msg.author);
    try out.appendSlice(allocator, "}");
}

fn serializeChannel(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), ch: anytype) !void {
    try out.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, ch.id);
    try out.appendSlice(allocator, "\",\"type\":");
    var type_buf: [16]u8 = undefined;
    const type_str = std.fmt.bufPrint(&type_buf, "{d}", .{ch.channel_type}) catch "0";
    try out.appendSlice(allocator, type_str);
    if (ch.guild_id) |gid| {
        try out.appendSlice(allocator, ",\"guild_id\":\"");
        try json_utils.appendJsonEscaped(allocator, out, gid);
        try out.appendSlice(allocator, "\"");
    }
    if (ch.name) |name| {
        try out.appendSlice(allocator, ",\"name\":\"");
        try json_utils.appendJsonEscaped(allocator, out, name);
        try out.appendSlice(allocator, "\"");
    }
    if (ch.topic) |topic| {
        try out.appendSlice(allocator, ",\"topic\":\"");
        try json_utils.appendJsonEscaped(allocator, out, topic);
        try out.appendSlice(allocator, "\"");
    }
    try out.appendSlice(allocator, "}");
}

fn serializeGuild(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), guild: anytype) !void {
    try out.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, guild.id);
    try out.appendSlice(allocator, "\",\"name\":\"");
    try json_utils.appendJsonEscaped(allocator, out, guild.name);
    try out.appendSlice(allocator, "\"");
    if (guild.icon) |ic| {
        try out.appendSlice(allocator, ",\"icon\":\"");
        try json_utils.appendJsonEscaped(allocator, out, ic);
        try out.appendSlice(allocator, "\"");
    }
    try out.appendSlice(allocator, ",\"owner_id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, guild.owner_id);
    try out.appendSlice(allocator, "\",\"member_count\":");
    var count_buf: [16]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, "{d}", .{guild.approximate_member_count}) catch "0";
    try out.appendSlice(allocator, count_str);
    try out.appendSlice(allocator, "}");
}

fn serializeMember(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), member: anytype) !void {
    try out.appendSlice(allocator, "{\"nick\":");
    if (member.nick) |n| {
        try out.appendSlice(allocator, "\"");
        try json_utils.appendJsonEscaped(allocator, out, n);
        try out.appendSlice(allocator, "\"");
    } else {
        try out.appendSlice(allocator, "null");
    }
    if (member.user) |*user| {
        try out.appendSlice(allocator, ",\"user\":");
        try serializeUser(allocator, out, user);
    }
    try out.appendSlice(allocator, ",\"joined_at\":\"");
    try json_utils.appendJsonEscaped(allocator, out, member.joined_at);
    try out.appendSlice(allocator, "\"}");
}

// ── Tool Handlers ──

pub fn handleDiscordSendMessage(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const content = try requireString(params, "content");

    const msg = client.createMessage(channel_id, content) catch {
        try appendError(allocator, out, "failed to send message");
        return;
    };

    try serializeMessage(allocator, out, &msg);
}

pub fn handleDiscordSendEmbed(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const p = params orelse return error.InvalidParams;
    const channel_id = getStringParam(p, "channel_id") orelse return error.InvalidParams;
    const title = getStringParam(p, "title") orelse "Untitled";
    const description = getStringParam(p, "description") orelse "";
    const content = getStringParam(p, "content");
    const color: ?u32 = if (getIntParam(p, "color")) |c| @intCast(c) else null;

    const embed = discord.Embed{
        .title = title,
        .description = description,
        .color = color,
    };

    const msg = client.createMessageWithEmbed(channel_id, content, embed) catch {
        try appendError(allocator, out, "failed to send embed");
        return;
    };

    try serializeMessage(allocator, out, &msg);
}

pub fn handleDiscordEditMessage(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const message_id = try requireString(params, "message_id");
    const content = try requireString(params, "content");

    const msg = client.editMessage(channel_id, message_id, content) catch {
        try appendError(allocator, out, "failed to edit message");
        return;
    };

    try serializeMessage(allocator, out, &msg);
}

pub fn handleDiscordDeleteMessage(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const message_id = try requireString(params, "message_id");

    client.deleteMessage(channel_id, message_id) catch {
        try appendError(allocator, out, "failed to delete message");
        return;
    };

    try out.appendSlice(allocator, "{\"ok\":true}");
}

pub fn handleDiscordGetMessages(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const limit: ?u8 = if (params) |p| blk: {
        break :blk if (getIntParam(p, "limit")) |l| @intCast(@min(100, @max(1, l))) else null;
    } else null;

    const msgs = client.getChannelMessages(channel_id, limit) catch {
        try appendError(allocator, out, "failed to get messages");
        return;
    };

    try out.appendSlice(allocator, "[");
    for (msgs, 0..) |*msg, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        try serializeMessage(allocator, out, msg);
    }
    try out.appendSlice(allocator, "]");
}

pub fn handleDiscordGetChannel(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");

    const ch = client.getChannel(channel_id) catch {
        try appendError(allocator, out, "failed to get channel");
        return;
    };

    try serializeChannel(allocator, out, &ch);
}

pub fn handleDiscordReact(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const message_id = try requireString(params, "message_id");
    const emoji = try requireString(params, "emoji");

    client.createReaction(channel_id, message_id, emoji) catch {
        try appendError(allocator, out, "failed to add reaction");
        return;
    };

    try out.appendSlice(allocator, "{\"ok\":true}");
}

pub fn handleDiscordTyping(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");

    client.triggerTypingIndicator(channel_id) catch {
        try appendError(allocator, out, "failed to trigger typing");
        return;
    };

    try out.appendSlice(allocator, "{\"ok\":true}");
}

pub fn handleDiscordGetGuild(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const guild_id = try requireString(params, "guild_id");

    const guild = client.getGuild(guild_id) catch {
        try appendError(allocator, out, "failed to get guild");
        return;
    };

    try serializeGuild(allocator, out, &guild);
}

pub fn handleDiscordGetGuildChannels(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const guild_id = try requireString(params, "guild_id");

    const channels = client.getGuildChannels(guild_id) catch {
        try appendError(allocator, out, "failed to get guild channels");
        return;
    };

    try out.appendSlice(allocator, "[");
    for (channels, 0..) |*ch, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        try serializeChannel(allocator, out, ch);
    }
    try out.appendSlice(allocator, "]");
}

pub fn handleDiscordListGuilds(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const guilds = client.getCurrentUserGuilds() catch {
        try appendError(allocator, out, "failed to list guilds");
        return;
    };

    try out.appendSlice(allocator, "[");
    for (guilds, 0..) |*guild, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        try serializeGuild(allocator, out, guild);
    }
    try out.appendSlice(allocator, "]");
}

pub fn handleDiscordGetBot(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const user = client.getCurrentUser() catch {
        try appendError(allocator, out, "failed to get bot user");
        return;
    };

    try serializeUser(allocator, out, &user);
}

pub fn handleDiscordCreateDM(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const user_id = try requireString(params, "user_id");

    const dm = client.createDM(user_id) catch {
        try appendError(allocator, out, "failed to create DM");
        return;
    };

    try serializeChannel(allocator, out, &dm);
}

pub fn handleDiscordExecuteWebhook(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const webhook_id = try requireString(params, "webhook_id");
    const token = try requireString(params, "token");
    const content = try requireString(params, "content");

    client.executeWebhook(webhook_id, token, content) catch {
        try appendError(allocator, out, "failed to execute webhook");
        return;
    };

    try out.appendSlice(allocator, "{\"ok\":true}");
}

pub fn handleDiscordGetMember(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const guild_id = try requireString(params, "guild_id");
    const user_id = try requireString(params, "user_id");

    const member = client.getGuildMember(guild_id, user_id) catch {
        try appendError(allocator, out, "failed to get guild member");
        return;
    };

    try serializeMember(allocator, out, &member);
}

pub fn handleDiscordRegisterCommand(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const p = params orelse return error.InvalidParams;
    const name = getStringParam(p, "name") orelse return error.InvalidParams;
    const description = getStringParam(p, "description") orelse return error.InvalidParams;
    const app_id = getStringParam(p, "application_id") orelse {
        try appendError(allocator, out, "application_id required");
        return;
    };

    const no_options: []const discord.ApplicationCommandOption = &.{};
    const cmd = client.createGlobalApplicationCommand(app_id, name, description, no_options) catch {
        try appendError(allocator, out, "failed to register command");
        return;
    };

    try out.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendJsonEscaped(allocator, out, cmd.id);
    try out.appendSlice(allocator, "\",\"name\":\"");
    try json_utils.appendJsonEscaped(allocator, out, cmd.name);
    try out.appendSlice(allocator, "\",\"description\":\"");
    try json_utils.appendJsonEscaped(allocator, out, cmd.description);
    try out.appendSlice(allocator, "\"}");
}

// ── List Commands ──

pub fn handleDiscordListCommands(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const app_id = if (params) |p| getStringParam(p, "application_id") else null;
    const id = app_id orelse {
        try appendError(allocator, out, "application_id required");
        return;
    };

    const cmds = client.getGlobalApplicationCommands(id) catch {
        try appendError(allocator, out, "failed to list commands");
        return;
    };

    try out.appendSlice(allocator, "[");
    for (cmds, 0..) |*cmd, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        try out.appendSlice(allocator, "{\"id\":\"");
        try json_utils.appendJsonEscaped(allocator, out, cmd.id);
        try out.appendSlice(allocator, "\",\"name\":\"");
        try json_utils.appendJsonEscaped(allocator, out, cmd.name);
        try out.appendSlice(allocator, "\",\"description\":\"");
        try json_utils.appendJsonEscaped(allocator, out, cmd.description);
        try out.appendSlice(allocator, "\"}");
    }
    try out.appendSlice(allocator, "]");
}

// ── Delete Command ──

pub fn handleDiscordDeleteCommand(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const app_id = try requireString(params, "application_id");
    const cmd_id = try requireString(params, "command_id");

    client.deleteGlobalApplicationCommand(app_id, cmd_id) catch {
        try appendError(allocator, out, "failed to delete command");
        return;
    };

    try out.appendSlice(allocator, "{\"ok\":true}");
}

// ── Get Message ──

pub fn handleDiscordGetMessage(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    var client = try withClient(allocator, out) orelse return;
    defer client.deinit();

    const channel_id = try requireString(params, "channel_id");
    const message_id = try requireString(params, "message_id");

    const msg = client.getMessage(channel_id, message_id) catch {
        try appendError(allocator, out, "failed to get message");
        return;
    };

    try serializeMessage(allocator, out, &msg);
}

test {
    std.testing.refAllDecls(@This());
}
