//! Discord REST API — Channel and Message Endpoints
//!
//! Channel CRUD, message send/edit/delete, reactions, and typing indicators.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const encoders = @import("../rest_encoders.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;
const ClientCore = @import("core.zig").ClientCore;

const Snowflake = types.Snowflake;
const Channel = types.Channel;
const Message = types.Message;
const Embed = types.Embed;

/// Get a channel
pub fn getChannel(core: *ClientCore, channel_id: Snowflake) !Channel {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}",
        .{channel_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseChannel(core.allocator, response.body);
}

/// Delete a channel
pub fn deleteChannel(core: *ClientCore, channel_id: Snowflake) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}",
        .{channel_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Get channel messages
pub fn getChannelMessages(
    core: *ClientCore,
    channel_id: Snowflake,
    limit: ?u8,
) ![]Message {
    const lim = limit orelse 50;
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages?limit={d}",
        .{ channel_id, lim },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseMessageArray(core.allocator, response.body);
}

/// Get a specific message
pub fn getMessage(
    core: *ClientCore,
    channel_id: Snowflake,
    message_id: Snowflake,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages/{s}",
        .{ channel_id, message_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseMessage(core.allocator, response.body);
}

/// Create a message
pub fn createMessage(
    core: *ClientCore,
    channel_id: Snowflake,
    content: []const u8,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages",
        .{channel_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    const body = try std.fmt.allocPrint(
        core.allocator,
        "{{\"content\":\"{}\"}}",
        .{json_utils.jsonEscape(content)},
    );
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseMessage(core.allocator, response.body);
}

/// Create a message with embed
pub fn createMessageWithEmbed(
    core: *ClientCore,
    channel_id: Snowflake,
    content: ?[]const u8,
    embed: Embed,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages",
        .{channel_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    const body = try encoders.encodeMessageWithEmbed(core.allocator, content, embed);
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseMessage(core.allocator, response.body);
}

/// Edit a message
pub fn editMessage(
    core: *ClientCore,
    channel_id: Snowflake,
    message_id: Snowflake,
    content: []const u8,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages/{s}",
        .{ channel_id, message_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.patch, endpoint);
    defer request.deinit();

    const body = try std.fmt.allocPrint(
        core.allocator,
        "{{\"content\":\"{}\"}}",
        .{json_utils.jsonEscape(content)},
    );
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseMessage(core.allocator, response.body);
}

/// Delete a message
pub fn deleteMessage(
    core: *ClientCore,
    channel_id: Snowflake,
    message_id: Snowflake,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages/{s}",
        .{ channel_id, message_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Add a reaction to a message
pub fn createReaction(
    core: *ClientCore,
    channel_id: Snowflake,
    message_id: Snowflake,
    emoji: []const u8,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages/{s}/reactions/{s}/@me",
        .{ channel_id, message_id, emoji },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.put, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Delete own reaction
pub fn deleteOwnReaction(
    core: *ClientCore,
    channel_id: Snowflake,
    message_id: Snowflake,
    emoji: []const u8,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/messages/{s}/reactions/{s}/@me",
        .{ channel_id, message_id, emoji },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Trigger a typing indicator in a channel.
/// The indicator lasts ~10 seconds or until a message is sent.
pub fn triggerTypingIndicator(
    core: *ClientCore,
    channel_id: Snowflake,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/channels/{s}/typing",
        .{channel_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

test {
    std.testing.refAllDecls(@This());
}
