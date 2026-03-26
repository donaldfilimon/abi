//! Discord REST operations exposed via ACP HTTP.
//!
//! Routes:
//!   POST /discord/send          — Send a message to a channel
//!   GET  /discord/channels/{id} — Get a channel
//!   POST /discord/webhook       — Execute a webhook
//!   GET  /discord/guilds        — List current user's guilds
//!   GET  /discord/bot           — Get current bot user

const std = @import("std");
const build_options = @import("build_options");
const routing = @import("routing.zig");
const json_utils = @import("json_utils.zig");
const respondJson = routing.respondJson;

const discord = if (build_options.feat_connectors)
    @import("../../../connectors/discord/mod.zig")
else
    @import("../../../connectors/stubs/discord.zig");

pub fn handleDiscordHttpRoute(
    allocator: std.mem.Allocator,
    request: *std.http.Server.Request,
    path: []const u8,
) !void {
    // POST /discord/send
    if (std.mem.eql(u8, path, "/discord/send")) {
        if (request.head.method != .POST) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        return handleSendMessage(allocator, request);
    }

    // GET /discord/channels/{id}
    if (std.mem.startsWith(u8, path, "/discord/channels/")) {
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        const channel_id = path["/discord/channels/".len..];
        if (channel_id.len == 0) {
            return respondJson(request, "{\"error\":\"missing channel_id\"}", .bad_request);
        }
        return handleGetChannel(allocator, request, channel_id);
    }

    // POST /discord/webhook
    if (std.mem.eql(u8, path, "/discord/webhook")) {
        if (request.head.method != .POST) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        return handleExecuteWebhook(allocator, request);
    }

    // GET /discord/guilds
    if (std.mem.eql(u8, path, "/discord/guilds")) {
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        return handleGetGuilds(allocator, request);
    }

    // GET /discord/bot
    if (std.mem.eql(u8, path, "/discord/bot")) {
        if (request.head.method != .GET) {
            return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
        }
        return handleGetBot(allocator, request);
    }

    return respondJson(request, "{\"error\":\"not found\"}", .not_found);
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

fn handleSendMessage(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    const body = readBody(allocator, request) orelse return;
    defer allocator.free(body);

    // Parse { "channel_id": "...", "content": "..." }
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return respondJson(request, "{\"error\":\"invalid json\"}", .bad_request);
    };
    defer parsed.deinit();

    const obj = if (parsed.value == .object) &parsed.value.object else {
        return respondJson(request, "{\"error\":\"expected object\"}", .bad_request);
    };

    const channel_id = if (obj.get("channel_id")) |v| (if (v == .string) v.string else null) else null;
    const content = if (obj.get("content")) |v| (if (v == .string) v.string else null) else null;

    if (channel_id == null or content == null) {
        return respondJson(request, "{\"error\":\"missing channel_id or content\"}", .bad_request);
    }

    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    const msg = client.createMessage(channel_id.?, content.?) catch |err| {
        return respondDiscordError(request, err);
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try serializeMessage(allocator, &buf, &msg);
    return respondJson(request, buf.items, .ok);
}

fn handleGetChannel(allocator: std.mem.Allocator, request: *std.http.Server.Request, channel_id: []const u8) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    const channel = client.getChannel(channel_id) catch |err| {
        return respondDiscordError(request, err);
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try serializeChannel(allocator, &buf, &channel);
    return respondJson(request, buf.items, .ok);
}

fn handleExecuteWebhook(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    const body = readBody(allocator, request) orelse return;
    defer allocator.free(body);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return respondJson(request, "{\"error\":\"invalid json\"}", .bad_request);
    };
    defer parsed.deinit();

    const obj = if (parsed.value == .object) &parsed.value.object else {
        return respondJson(request, "{\"error\":\"expected object\"}", .bad_request);
    };

    const webhook_id = if (obj.get("webhook_id")) |v| (if (v == .string) v.string else null) else null;
    const webhook_token = if (obj.get("webhook_token")) |v| (if (v == .string) v.string else null) else null;
    const content = if (obj.get("content")) |v| (if (v == .string) v.string else null) else null;

    if (webhook_id == null or webhook_token == null or content == null) {
        return respondJson(request, "{\"error\":\"missing webhook_id, webhook_token, or content\"}", .bad_request);
    }

    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    // executeWebhook returns void on success — echo request params as confirmation.
    client.executeWebhook(webhook_id.?, webhook_token.?, content.?) catch |err| {
        return respondDiscordError(request, err);
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"ok\":true,\"webhook_id\":\"");
    try json_utils.appendEscaped(allocator, &buf, webhook_id.?);
    try buf.appendSlice(allocator, "\",\"content\":\"");
    try json_utils.appendEscaped(allocator, &buf, content.?);
    try buf.appendSlice(allocator, "\"}");
    return respondJson(request, buf.items, .ok);
}

fn handleGetGuilds(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    const guilds = client.getCurrentUserGuilds() catch |err| {
        return respondDiscordError(request, err);
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"guilds\":[");
    for (guilds, 0..) |*guild, i| {
        if (i > 0) try buf.appendSlice(allocator, ",");
        try serializeGuild(allocator, &buf, guild);
    }
    try buf.appendSlice(allocator, "]}");
    return respondJson(request, buf.items, .ok);
}

fn handleGetBot(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    const user = client.getCurrentUser() catch |err| {
        return respondDiscordError(request, err);
    };

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try serializeUser(allocator, &buf, &user);
    return respondJson(request, buf.items, .ok);
}

// ---------------------------------------------------------------------------
// JSON Serialization Helpers
// ---------------------------------------------------------------------------

/// Serialize a User object: {id, username, discriminator, bot, avatar?}
fn serializeUser(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), user: anytype) !void {
    try buf.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendEscaped(allocator, buf, user.id);
    try buf.appendSlice(allocator, "\",\"username\":\"");
    try json_utils.appendEscaped(allocator, buf, user.username);
    try buf.appendSlice(allocator, "\",\"discriminator\":\"");
    try json_utils.appendEscaped(allocator, buf, user.discriminator);
    try buf.appendSlice(allocator, "\",\"bot\":");
    try buf.appendSlice(allocator, if (user.bot) "true" else "false");
    if (user.avatar) |av| {
        try buf.appendSlice(allocator, ",\"avatar\":\"");
        try json_utils.appendEscaped(allocator, buf, av);
        try buf.appendSlice(allocator, "\"");
    }
    try buf.appendSlice(allocator, "}");
}

/// Serialize a Message object: {id, channel_id, content, timestamp, author}
fn serializeMessage(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), msg: anytype) !void {
    try buf.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendEscaped(allocator, buf, msg.id);
    try buf.appendSlice(allocator, "\",\"channel_id\":\"");
    try json_utils.appendEscaped(allocator, buf, msg.channel_id);
    try buf.appendSlice(allocator, "\",\"content\":\"");
    try json_utils.appendEscaped(allocator, buf, msg.content);
    try buf.appendSlice(allocator, "\",\"timestamp\":\"");
    try json_utils.appendEscaped(allocator, buf, msg.timestamp);
    try buf.appendSlice(allocator, "\",\"author\":");
    try serializeUser(allocator, buf, &msg.author);
    try buf.appendSlice(allocator, "}");
}

/// Serialize a Channel object: {id, type, guild_id?, name?, topic?}
fn serializeChannel(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), channel: anytype) !void {
    try buf.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendEscaped(allocator, buf, channel.id);
    try buf.appendSlice(allocator, "\",\"type\":");
    var type_buf: [16]u8 = undefined;
    const type_str = std.fmt.bufPrint(&type_buf, "{d}", .{channel.channel_type}) catch "0";
    try buf.appendSlice(allocator, type_str);
    if (channel.guild_id) |gid| {
        try buf.appendSlice(allocator, ",\"guild_id\":\"");
        try json_utils.appendEscaped(allocator, buf, gid);
        try buf.appendSlice(allocator, "\"");
    }
    if (channel.name) |name| {
        try buf.appendSlice(allocator, ",\"name\":\"");
        try json_utils.appendEscaped(allocator, buf, name);
        try buf.appendSlice(allocator, "\"");
    }
    if (channel.topic) |topic| {
        try buf.appendSlice(allocator, ",\"topic\":\"");
        try json_utils.appendEscaped(allocator, buf, topic);
        try buf.appendSlice(allocator, "\"");
    }
    try buf.appendSlice(allocator, "}");
}

/// Serialize a Guild object: {id, name, icon?, owner_id, approximate_member_count}
fn serializeGuild(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), guild: anytype) !void {
    try buf.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendEscaped(allocator, buf, guild.id);
    try buf.appendSlice(allocator, "\",\"name\":\"");
    try json_utils.appendEscaped(allocator, buf, guild.name);
    try buf.appendSlice(allocator, "\"");
    if (guild.icon) |ic| {
        try buf.appendSlice(allocator, ",\"icon\":\"");
        try json_utils.appendEscaped(allocator, buf, ic);
        try buf.appendSlice(allocator, "\"");
    }
    try buf.appendSlice(allocator, ",\"owner_id\":\"");
    try json_utils.appendEscaped(allocator, buf, guild.owner_id);
    try buf.appendSlice(allocator, "\",\"member_count\":");
    var count_buf: [16]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, "{d}", .{guild.approximate_member_count}) catch "0";
    try buf.appendSlice(allocator, count_str);
    try buf.appendSlice(allocator, "}");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn readBody(allocator: std.mem.Allocator, request: *std.http.Server.Request) ?[]u8 {
    return routing.readRequestBody(allocator, request) catch |err| {
        switch (err) {
            routing.HttpError.RequestTooLarge => {
                respondJson(request, "{\"error\":\"payload too large\"}", .payload_too_large) catch {};
            },
            routing.HttpError.ReadFailed => {
                respondJson(request, "{\"error\":\"invalid body\"}", .bad_request) catch {};
            },
            else => {
                respondJson(request, "{\"error\":\"internal error\"}", .internal_server_error) catch {};
            },
        }
        return null;
    };
}

fn respondDiscordError(request: *std.http.Server.Request, err: anyerror) !void {
    const status: std.http.Status = switch (err) {
        error.Unauthorized => .unauthorized,
        error.Forbidden => .forbidden,
        error.NotFound => .not_found,
        error.RateLimited => .too_many_requests,
        error.MissingBotToken => .service_unavailable,
        error.ConnectorsDisabled => .service_unavailable,
        else => .bad_gateway,
    };
    const body = switch (err) {
        error.Unauthorized => "{\"error\":\"unauthorized\"}",
        error.Forbidden => "{\"error\":\"forbidden\"}",
        error.NotFound => "{\"error\":\"not found\"}",
        error.RateLimited => "{\"error\":\"rate limited\"}",
        error.MissingBotToken => "{\"error\":\"discord unavailable\"}",
        error.ConnectorsDisabled => "{\"error\":\"connectors disabled\"}",
        else => "{\"error\":\"discord api error\"}",
    };
    return respondJson(request, body, status);
}

test "discord_routes: module compiles" {
    // Ensure all declarations are reachable
    std.testing.refAllDecls(@This());
}
