//! Discord REST API JSON Encoders
//!
//! Standalone encoding functions for Discord API request bodies.

const std = @import("std");
const types = @import("types.zig");
const json_utils = @import("../../shared/utils.zig").json;

const Embed = types.Embed;
const ApplicationCommandOption = types.ApplicationCommandOption;

pub fn encodeMessageWithEmbed(
    allocator: std.mem.Allocator,
    content: ?[]const u8,
    embed: Embed,
) ![]u8 {
    var json = std.ArrayListUnmanaged(u8).empty;
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{");

    if (content) |c| {
        try json.print(
            allocator,
            "\"content\":\"{}\",",
            .{json_utils.jsonEscape(c)},
        );
    }

    try json.appendSlice(allocator, "\"embeds\":[{");

    var first = true;
    if (embed.title) |title| {
        try json.print(
            allocator,
            "\"title\":\"{}\"",
            .{json_utils.jsonEscape(title)},
        );
        first = false;
    }

    if (embed.description) |desc| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.print(
            allocator,
            "\"description\":\"{}\"",
            .{json_utils.jsonEscape(desc)},
        );
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
        try json.print(
            allocator,
            "\"footer\":{{\"text\":\"{}\"",
            .{json_utils.jsonEscape(footer.text)},
        );
        if (footer.icon_url) |icon| {
            try json.print(allocator, ",\"icon_url\":\"{s}\"", .{icon});
        }
        try json.appendSlice(allocator, "}");
        first = false;
    }

    if (embed.author) |author| {
        if (!first) try json.appendSlice(allocator, ",");
        try json.print(
            allocator,
            "\"author\":{{\"name\":\"{}\"",
            .{json_utils.jsonEscape(author.name)},
        );
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
            try json.print(
                allocator,
                "{{\"name\":\"{}\",\"value\":\"{}\",\"inline\":{s}}}",
                .{
                    json_utils.jsonEscape(field.name),
                    json_utils.jsonEscape(field.value),
                    if (field.inline_field) "true" else "false",
                },
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

    try json.print(
        allocator,
        "{{\"name\":\"{s}\",\"description\":\"{}\"",
        .{ name, json_utils.jsonEscape(description) },
    );

    if (options.len > 0) {
        try json.appendSlice(allocator, ",\"options\":[");
        for (options, 0..) |opt, i| {
            if (i > 0) try json.appendSlice(allocator, ",");
            try json.print(
                allocator,
                "{{\"type\":{d},\"name\":\"{s}\",\"description\":\"{}\",\"required\":{s}}}",
                .{
                    opt.option_type,
                    opt.name,
                    json_utils.jsonEscape(opt.description),
                    if (opt.required) "true" else "false",
                },
            );
        }
        try json.appendSlice(allocator, "]");
    }

    try json.appendSlice(allocator, "}");

    return try json.toOwnedSlice(allocator);
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

test {
    std.testing.refAllDecls(@This());
}
