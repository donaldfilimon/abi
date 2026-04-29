//! Discord REST API — User Endpoints
//!
//! User info, current user management, DMs, and guild membership.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const ClientCore = @import("core.zig").ClientCore;

const DiscordError = types.DiscordError;
const Snowflake = types.Snowflake;
const User = types.User;
const Guild = types.Guild;
const Channel = types.Channel;

/// Get the current user
pub fn getCurrentUser(core: *ClientCore) !User {
    var request = try core.makeRequest(.get, "/users/@me");
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseUser(core.allocator, response.body);
}

/// Get a user by ID
pub fn getUser(core: *ClientCore, user_id: Snowflake) !User {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/users/{s}",
        .{user_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseUser(core.allocator, response.body);
}

/// Modify the current user
pub fn modifyCurrentUser(
    core: *ClientCore,
    username: ?[]const u8,
    avatar: ?[]const u8,
) !User {
    var request = try core.makeRequest(.patch, "/users/@me");
    defer request.deinit();

    var body = std.ArrayListUnmanaged(u8).empty;
    defer body.deinit(core.allocator);

    try body.appendSlice(core.allocator, "{");
    var first = true;

    if (username) |name| {
        try body.print(core.allocator, "\"username\":\"{s}\"", .{name});
        first = false;
    }

    if (avatar) |av| {
        if (!first) try body.appendSlice(core.allocator, ",");
        try body.print(core.allocator, "\"avatar\":\"{s}\"", .{av});
    }

    try body.appendSlice(core.allocator, "}");
    try request.setJsonBody(body.items);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseUser(core.allocator, response.body);
}

/// Get current user's guilds
pub fn getCurrentUserGuilds(core: *ClientCore) ![]Guild {
    var request = try core.makeRequest(.get, "/users/@me/guilds");
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseGuildArray(core.allocator, response.body);
}

/// Leave a guild
pub fn leaveGuild(core: *ClientCore, guild_id: Snowflake) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/users/@me/guilds/{s}",
        .{guild_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Create a DM channel
pub fn createDM(core: *ClientCore, recipient_id: Snowflake) !Channel {
    var request = try core.makeRequest(.post, "/users/@me/channels");
    defer request.deinit();

    const body = try std.fmt.allocPrint(
        core.allocator,
        "{{\"recipient_id\":\"{s}\"}}",
        .{recipient_id},
    );
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseChannel(core.allocator, response.body);
}

test {
    std.testing.refAllDecls(@This());
}
