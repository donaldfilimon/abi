const std = @import("std");
const common = @import("common.zig");

const gpu = @import("gpu.zig");
const db = @import("db.zig");
const config = @import("config.zig");
const neural = @import("neural.zig");
const simd = @import("simd.zig");
const plugin = @import("plugin.zig");
const server = @import("server.zig");
const weather = @import("weather.zig");
const llm = @import("llm.zig");
const chat = @import("chat.zig");

pub const commands = [_]common.Command{
    gpu.command,
    db.command,
    config.command,
    neural.command,
    simd.command,
    plugin.command,
    server.command,
    weather.command,
    llm.command,
    chat.command,
};

pub fn all() []const common.Command {
    return commands[0..];
}

pub fn find(name: []const u8) ?*const common.Command {
    for (&commands) |*cmd| {
        if (std.mem.eql(u8, name, cmd.name)) return cmd;
        for (cmd.aliases) |alias| {
            if (std.mem.eql(u8, name, alias)) return cmd;
        }
    }
    return null;
}

pub fn formatSummary(writer: anytype) !void {
    const entries = commands[0..];
    var longest: usize = 0;
    for (entries) |cmd| {
        longest = @max(longest, cmd.name.len);
        for (cmd.aliases) |alias| {
            longest = @max(longest, alias.len);
        }
    }

    for (entries) |cmd| {
        try writer.print("  {s}  {s}\n", .{ cmd.name, cmd.summary });
    }
}
