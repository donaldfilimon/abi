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
    var json = std.ArrayListUnmanaged(u8){};
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
    var json = std.ArrayListUnmanaged(u8){};
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
