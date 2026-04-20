//! Discord REST API — Interaction and Application Command Endpoints
//!
//! Slash commands, components, modals, interaction responses, and followups.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const encoders = @import("../rest_encoders.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;
const ClientCore = @import("core.zig").ClientCore;

const DiscordError = types.DiscordError;
const Snowflake = types.Snowflake;
const Message = types.Message;
const ApplicationCommand = types.ApplicationCommand;
const ApplicationCommandOption = types.ApplicationCommandOption;
const InteractionCallbackType = types.InteractionCallbackType;

/// Get global application commands
pub fn getGlobalApplicationCommands(
    core: *ClientCore,
    application_id: Snowflake,
) ![]ApplicationCommand {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/applications/{s}/commands",
        .{application_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseApplicationCommandArray(core.allocator, response.body);
}

/// Create a global application command
pub fn createGlobalApplicationCommand(
    core: *ClientCore,
    application_id: Snowflake,
    name: []const u8,
    description: []const u8,
    options: []const ApplicationCommandOption,
) !ApplicationCommand {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/applications/{s}/commands",
        .{application_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    const body = try encoders.encodeApplicationCommand(core.allocator, name, description, options);
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseApplicationCommand(core.allocator, response.body);
}

/// Delete a global application command
pub fn deleteGlobalApplicationCommand(
    core: *ClientCore,
    application_id: Snowflake,
    command_id: Snowflake,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/applications/{s}/commands/{s}",
        .{ application_id, command_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Get guild application commands
pub fn getGuildApplicationCommands(
    core: *ClientCore,
    application_id: Snowflake,
    guild_id: Snowflake,
) ![]ApplicationCommand {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/applications/{s}/guilds/{s}/commands",
        .{ application_id, guild_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseApplicationCommandArray(core.allocator, response.body);
}

/// Create a guild application command
pub fn createGuildApplicationCommand(
    core: *ClientCore,
    application_id: Snowflake,
    guild_id: Snowflake,
    name: []const u8,
    description: []const u8,
    options: []const ApplicationCommandOption,
) !ApplicationCommand {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/applications/{s}/guilds/{s}/commands",
        .{ application_id, guild_id },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    const body = try encoders.encodeApplicationCommand(core.allocator, name, description, options);
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseApplicationCommand(core.allocator, response.body);
}

/// Create an interaction response
pub fn createInteractionResponse(
    core: *ClientCore,
    interaction_id: Snowflake,
    interaction_token: []const u8,
    response_type: InteractionCallbackType,
    content: ?[]const u8,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/interactions/{s}/{s}/callback",
        .{ interaction_id, interaction_token },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    var body = std.ArrayListUnmanaged(u8).empty;
    defer body.deinit(core.allocator);

    try body.print(core.allocator, "{{\"type\":{d}", .{@intFromEnum(response_type)});

    if (content) |c| {
        try body.print(
            core.allocator,
            ",\"data\":{{\"content\":\"{}\"}}",
            .{json_utils.jsonEscape(c)},
        );
    }

    try body.appendSlice(core.allocator, "}");
    try request.setJsonBody(body.items);

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Edit the original interaction response
pub fn editOriginalInteractionResponse(
    core: *ClientCore,
    application_id: Snowflake,
    interaction_token: []const u8,
    content: []const u8,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}/{s}/messages/@original",
        .{ application_id, interaction_token },
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

/// Delete the original interaction response
pub fn deleteOriginalInteractionResponse(
    core: *ClientCore,
    application_id: Snowflake,
    interaction_token: []const u8,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}/{s}/messages/@original",
        .{ application_id, interaction_token },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Create a followup message
pub fn createFollowupMessage(
    core: *ClientCore,
    application_id: Snowflake,
    interaction_token: []const u8,
    content: []const u8,
) !Message {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}/{s}",
        .{ application_id, interaction_token },
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

test {
    std.testing.refAllDecls(@This());
}
