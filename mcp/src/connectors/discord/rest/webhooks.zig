//! Discord REST API — Webhook Endpoints
//!
//! Webhook CRUD and execution with text and embed payloads.

const std = @import("std");
const types = @import("../types.zig");
const parsers = @import("../rest_parsers.zig");
const encoders = @import("../rest_encoders.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;
const ClientCore = @import("core.zig").ClientCore;

const Snowflake = types.Snowflake;
const Webhook = types.Webhook;
const Embed = types.Embed;

/// Get a webhook by ID
pub fn getWebhook(core: *ClientCore, webhook_id: Snowflake) !Webhook {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}",
        .{webhook_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.get, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();

    return try parsers.parseWebhook(core.allocator, response.body);
}

/// Execute a webhook
pub fn executeWebhook(
    core: *ClientCore,
    webhook_id: Snowflake,
    webhook_token: []const u8,
    content: []const u8,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}/{s}",
        .{ webhook_id, webhook_token },
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
}

/// Execute a webhook with embeds
pub fn executeWebhookWithEmbed(
    core: *ClientCore,
    webhook_id: Snowflake,
    webhook_token: []const u8,
    content: ?[]const u8,
    embed: Embed,
) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}/{s}",
        .{ webhook_id, webhook_token },
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.post, endpoint);
    defer request.deinit();

    const body = try encoders.encodeMessageWithEmbed(core.allocator, content, embed);
    defer core.allocator.free(body);
    try request.setJsonBody(body);

    var response = try core.doRequest(&request);
    defer response.deinit();
}

/// Delete a webhook
pub fn deleteWebhook(core: *ClientCore, webhook_id: Snowflake) !void {
    const endpoint = try std.fmt.allocPrint(
        core.allocator,
        "/webhooks/{s}",
        .{webhook_id},
    );
    defer core.allocator.free(endpoint);

    var request = try core.makeRequest(.delete, endpoint);
    defer request.deinit();

    var response = try core.doRequest(&request);
    defer response.deinit();
}

test {
    std.testing.refAllDecls(@This());
}
