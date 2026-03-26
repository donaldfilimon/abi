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

    _ = client.createMessage(channel_id.?, content.?) catch |err| {
        return respondDiscordError(request, err);
    };

    return respondJson(request, "{\"ok\":true}", .ok);
}

fn handleGetChannel(allocator: std.mem.Allocator, request: *std.http.Server.Request, channel_id: []const u8) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    _ = client.getChannel(channel_id) catch |err| {
        return respondDiscordError(request, err);
    };

    // For now, return a simple acknowledgment — full serialization would
    // require the Channel struct to implement toJson.
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"id\":\"");
    try json_utils.appendEscaped(allocator, &buf, channel_id);
    try buf.appendSlice(allocator, "\"}");
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

    _ = client.executeWebhook(webhook_id.?, webhook_token.?, content.?) catch |err| {
        return respondDiscordError(request, err);
    };

    return respondJson(request, "{\"ok\":true}", .ok);
}

fn handleGetGuilds(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    _ = client.getCurrentUserGuilds() catch |err| {
        return respondDiscordError(request, err);
    };

    return respondJson(request, "{\"guilds\":[]}", .ok);
}

fn handleGetBot(allocator: std.mem.Allocator, request: *std.http.Server.Request) !void {
    var client = discord.createClient(allocator) catch {
        return respondJson(request, "{\"error\":\"discord unavailable\"}", .service_unavailable);
    };
    defer client.deinit();

    _ = client.getCurrentUser() catch |err| {
        return respondDiscordError(request, err);
    };

    return respondJson(request, "{\"bot\":true}", .ok);
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
