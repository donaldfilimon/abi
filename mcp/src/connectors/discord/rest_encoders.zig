//! Discord REST API JSON Encoders
//!
//! Standalone encoding functions for Discord API request bodies.

const std = @import("std");
const types = @import("types.zig");
const json_utils = @import("../../foundation/mod.zig").utils.json;

const Embed = types.Embed;
const ApplicationCommandOption = types.ApplicationCommandOption;

fn appendJsonString(allocator: std.mem.Allocator, json: *std.ArrayListUnmanaged(u8), value: []const u8) !void {
    try json.append(allocator, '"');
    try json_utils.appendJsonEscaped(allocator, json, value);
    try json.append(allocator, '"');
}

pub fn encodeMessage(allocator: std.mem.Allocator, content: []const u8) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"content\":");
    try appendJsonString(allocator, &json, content);
    try json.append(allocator, '}');

    return try json.toOwnedSlice(allocator);
}

pub fn encodeMessageWithEmbed(
    allocator: std.mem.Allocator,
    content: ?[]const u8,
    embed: Embed,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{");

    if (content) |c| {
        try json.appendSlice(allocator, "\"content\":");
        try appendJsonString(allocator, &json, c);
        try json.append(allocator, ',');
    }

    try json.appendSlice(allocator, "\"embeds\":[{");

    var first = true;
    if (embed.title) |title| {
        try json.appendSlice(allocator, "\"title\":");
        try appendJsonString(allocator, &json, title);
        first = false;
    }

    if (embed.description) |desc| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.appendSlice(allocator, "\"description\":");
        try appendJsonString(allocator, &json, desc);
        first = false;
    }

    if (embed.color) |color| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.print(allocator, "\"color\":{d}", .{color});
        first = false;
    }

    if (embed.url) |url_val| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.print(allocator, "\"url\":\"{s}\"", .{url_val});
        first = false;
    }

    if (embed.timestamp) |ts| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.print(allocator, "\"timestamp\":\"{s}\"", .{ts});
        first = false;
    }

    if (embed.footer) |footer| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.appendSlice(allocator, "\"footer\":{\"text\":");
        try appendJsonString(allocator, &json, footer.text);
        if (footer.icon_url) |icon| {
            try json.print(allocator, ",\"icon_url\":\"{s}\"", .{icon});
        }
        try json.appendSlice(allocator, "}");
        first = false;
    }

    if (embed.author) |author| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.appendSlice(allocator, "\"author\":{\"name\":");
        try appendJsonString(allocator, &json, author.name);
        if (author.url) |url_val| {
            try json.print(allocator, ",\"url\":\"{s}\"", .{url_val});
        }
        if (author.icon_url) |icon| {
            try json.print(allocator, ",\"icon_url\":\"{s}\"", .{icon});
        }
        try json.appendSlice(allocator, "}");
        first = false;
    }

    if (embed.fields.len > 0) {
        if (!first) try json.appendSlice(allocator, ",");
        try json.appendSlice(allocator, "\"fields\":[");
        for (embed.fields, 0..) |field, i| {
            if (i > 0) try json.appendSlice(allocator, ",");
            try json.appendSlice(allocator, "{\"name\":");
            try appendJsonString(allocator, &json, field.name);
            try json.appendSlice(allocator, ",\"value\":");
            try appendJsonString(allocator, &json, field.value);
            try json.print(
                allocator,
                ",\"inline\":{s}}}",
                .{if (field.inline_field) "true" else "false"},
            );
        }
        try json.appendSlice(allocator, "]");
    }

    try json.appendSlice(allocator, "}]}");

    return try json.toOwnedSlice(allocator);
}

pub fn encodeApplicationCommand(
    allocator: std.mem.Allocator,
    name: []const u8,
    description: []const u8,
    options: []const ApplicationCommandOption,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"name\":");
    try appendJsonString(allocator, &json, name);
    try json.appendSlice(allocator, ",\"description\":");
    try appendJsonString(allocator, &json, description);

    if (options.len > 0) {
        try json.appendSlice(allocator, ",\"options\":[");
        for (options, 0..) |opt, i| {
            if (i > 0) try json.appendSlice(allocator, ",");
            try json.print(allocator, "{{\"type\":{d},\"name\":", .{opt.option_type});
            try appendJsonString(allocator, &json, opt.name);
            try json.appendSlice(allocator, ",\"description\":");
            try appendJsonString(allocator, &json, opt.description);
            try json.print(
                allocator,
                ",\"required\":{s}}}",
                .{if (opt.required) "true" else "false"},
            );
        }
        try json.appendSlice(allocator, "]");
    }

    try json.appendSlice(allocator, "}");

    return try json.toOwnedSlice(allocator);
}

pub fn encodeWebhookExecute(
    allocator: std.mem.Allocator,
    content: []const u8,
    username: ?[]const u8,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"content\":");
    try appendJsonString(allocator, &json, content);
    if (username) |value| {
        try json.appendSlice(allocator, ",\"username\":");
        try appendJsonString(allocator, &json, value);
    }
    try json.appendSlice(allocator, "}");

    return try json.toOwnedSlice(allocator);
}

pub fn encodeInteractionResponse(
    allocator: std.mem.Allocator,
    response_type: types.InteractionCallbackType,
    content: ?[]const u8,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.print(allocator, "{{\"type\":{d}", .{@intFromEnum(response_type)});
    if (content) |value| {
        try json.appendSlice(allocator, ",\"data\":{\"content\":");
        try appendJsonString(allocator, &json, value);
        try json.append(allocator, '}');
    }
    try json.appendSlice(allocator, "}");

    return try json.toOwnedSlice(allocator);
}

fn isFormUnreserved(byte: u8) bool {
    return (byte >= 'A' and byte <= 'Z') or
        (byte >= 'a' and byte <= 'z') or
        (byte >= '0' and byte <= '9') or
        byte == '-' or byte == '_' or byte == '.' or byte == '~';
}

fn appendFormEncoded(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(u8), value: []const u8) !void {
    const hex = "0123456789ABCDEF";
    for (value) |byte| {
        if (byte == ' ') {
            try list.append(allocator, '+');
        } else if (isFormUnreserved(byte)) {
            try list.append(allocator, byte);
        } else {
            try list.appendSlice(allocator, &[_]u8{ '%', hex[byte >> 4], hex[byte & 0x0f] });
        }
    }
}

pub fn encodeOAuthClientCredentials(
    allocator: std.mem.Allocator,
    client_id: []const u8,
    client_secret: []const u8,
    scope: []const u8,
) ![]u8 {
    var body = std.ArrayListUnmanaged(u8).empty;
    errdefer body.deinit(allocator);

    try body.appendSlice(allocator, "grant_type=client_credentials&client_id=");
    try appendFormEncoded(allocator, &body, client_id);
    try body.appendSlice(allocator, "&client_secret=");
    try appendFormEncoded(allocator, &body, client_secret);
    try body.appendSlice(allocator, "&scope=");
    try appendFormEncoded(allocator, &body, scope);

    return try body.toOwnedSlice(allocator);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

test "encodeMessageWithEmbed basic" {
    const allocator = std.testing.allocator;
    const embed = Embed{
        .title = "Test",
        .description = "Body",
    };
    const json_str = try encodeMessageWithEmbed(allocator, "Hello", embed);
    defer allocator.free(json_str);

    // Should contain content and embed structure
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"content\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"embeds\":[{") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"title\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"description\":") != null);
}

test "encodeMessage escapes content" {
    const allocator = std.testing.allocator;
    const json_str = try encodeMessage(allocator, "hello \"discord\"");
    defer allocator.free(json_str);

    try std.testing.expectEqualStrings("{\"content\":\"hello \\\"discord\\\"\"}", json_str);
}

test "encodeMessageWithEmbed no content" {
    const allocator = std.testing.allocator;
    const embed = Embed{
        .title = "Embed",
    };
    const json_str = try encodeMessageWithEmbed(allocator, null, embed);
    defer allocator.free(json_str);

    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"content\":") == null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"title\":") != null);
}

test "encodeMessageWithEmbed with color" {
    const allocator = std.testing.allocator;
    const embed = Embed{
        .title = "Colored",
        .color = 0xFF0000,
    };
    const json_str = try encodeMessageWithEmbed(allocator, null, embed);
    defer allocator.free(json_str);

    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"color\":") != null);
}

test "encodeApplicationCommand basic" {
    const allocator = std.testing.allocator;
    const json_str = try encodeApplicationCommand(allocator, "ping", "Ping the bot", &.{});
    defer allocator.free(json_str);

    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"name\":\"ping\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"description\":") != null);
}

test "encodeApplicationCommand with options" {
    const allocator = std.testing.allocator;
    const options = [_]ApplicationCommandOption{
        .{
            .option_type = 3, // STRING
            .name = "query",
            .description = "Search query",
            .required = true,
        },
    };
    const json_str = try encodeApplicationCommand(allocator, "search", "Search for something", &options);
    defer allocator.free(json_str);

    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"options\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"name\":\"query\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"required\":true") != null);
}

test "encodeWebhookExecute includes optional username" {
    const allocator = std.testing.allocator;
    const json_str = try encodeWebhookExecute(allocator, "deploy done", "ABI Bot");
    defer allocator.free(json_str);

    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"content\":\"deploy done\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_str, "\"username\":\"ABI Bot\"") != null);
}

test "encodeInteractionResponse wraps message data" {
    const allocator = std.testing.allocator;
    const json_str = try encodeInteractionResponse(
        allocator,
        .CHANNEL_MESSAGE_WITH_SOURCE,
        "ack",
    );
    defer allocator.free(json_str);

    try std.testing.expectEqualStrings("{\"type\":4,\"data\":{\"content\":\"ack\"}}", json_str);
}

test "encodeOAuthClientCredentials form-encodes secrets and scopes" {
    const allocator = std.testing.allocator;
    const body = try encodeOAuthClientCredentials(
        allocator,
        "client id",
        "secret&value",
        "identify guilds.members.read",
    );
    defer allocator.free(body);

    try std.testing.expectEqualStrings(
        "grant_type=client_credentials&client_id=client+id&client_secret=secret%26value&scope=identify+guilds.members.read",
        body,
    );
}

test {
    std.testing.refAllDecls(@This());
}
