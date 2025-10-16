const std = @import("std");

pub const CLI_NAME = "ABI Framework CLI";
pub const CLI_VERSION = "0.1.0a";

pub const CommandId = enum {
    gpu,
    db,
    config,
    neural,
    simd,
    plugin,
    server,
    weather,
    llm,
    chat,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
};

pub const Command = struct {
    id: CommandId,
    name: []const u8,
    aliases: []const []const u8 = &.{},
    summary: []const u8,
    usage: []const u8,
    details: ?[]const u8 = null,
    run: *const fn (ctx: *Context, args: [][:0]u8) anyerror!void,
};

pub fn isHelpToken(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--help") or
        std.mem.eql(u8, arg, "-h") or
        std.mem.eql(u8, arg, "help");
}

pub fn parseCsvFloats(allocator: std.mem.Allocator, csv: []const u8) ![]f32 {
    var count: usize = 1;
    for (csv) |ch| {
        if (ch == ',') count += 1;
    }

    var vals = try allocator.alloc(f32, count);
    var idx: usize = 0;
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t\r\n");
        if (trimmed.len == 0) continue;
        vals[idx] = try std.fmt.parseFloat(f32, trimmed);
        idx += 1;
    }

    if (idx != count) {
        vals = try allocator.realloc(vals, idx);
    }

    return vals;
}
