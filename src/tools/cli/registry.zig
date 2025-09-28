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

pub const CommandRegistry = struct {
    pub const Entry = struct {
        id: common.CommandId,
        command: *const common.Command,
    };

    entries: []const Entry,

    pub inline fn init(comptime entries: []const Entry) CommandRegistry {
        comptime {
            const all_ids = std.meta.fields(common.CommandId);
            var seen = [_]bool{false} ** all_ids.len;
            inline for (entries) |entry| {
                const idx = @intFromEnum(entry.id);
                if (idx >= seen.len) {
                    @compileError("Command id out of range");
                }
                if (seen[idx]) {
                    @compileError("Duplicate command id registered: " ++ all_ids[idx].name);
                }
                if (entry.command.id != entry.id) {
                    @compileError("Command id mismatch for " ++ entry.command.name);
                }
                seen[idx] = true;
            }

            inline for (seen, 0..) |was_seen, idx| {
                if (!was_seen) {
                    const field = all_ids[idx];
                    @compileError("Command registry missing entry for " ++ field.name);
                }
            }
        }

        return .{ .entries = entries };
    }

    pub inline fn iter(self: CommandRegistry) []const Entry {
        return self.entries;
    }

    pub inline fn find(self: CommandRegistry, name: []const u8) ?*const common.Command {
        for (self.entries) |entry| {
            const cmd = entry.command;
            if (std.mem.eql(u8, name, cmd.name)) return cmd;
            for (cmd.aliases) |alias| {
                if (std.mem.eql(u8, name, alias)) return cmd;
            }
        }
        return null;
    }

    pub inline fn byId(self: CommandRegistry, id: common.CommandId) *const common.Command {
        for (self.entries) |entry| {
            if (entry.id == id) return entry.command;
        }
        @panic("Command id not registered");
    }

    pub inline fn formatSummary(self: CommandRegistry, writer: anytype) !void {
        for (self.entries) |entry| {
            const cmd = entry.command;
            try writer.print("  {s}  {s}\n", .{ cmd.name, cmd.summary });
        }
    }
};

const entries = [_]CommandRegistry.Entry{
    .{ .id = .gpu, .command = &gpu.command },
    .{ .id = .db, .command = &db.command },
    .{ .id = .config, .command = &config.command },
    .{ .id = .neural, .command = &neural.command },
    .{ .id = .simd, .command = &simd.command },
    .{ .id = .plugin, .command = &plugin.command },
    .{ .id = .server, .command = &server.command },
    .{ .id = .weather, .command = &weather.command },
    .{ .id = .llm, .command = &llm.command },
    .{ .id = .chat, .command = &chat.command },
};

pub const global = CommandRegistry.init(entries);

pub fn find(name: []const u8) ?*const common.Command {
    return global.find(name);
}

pub fn iter() []const CommandRegistry.Entry {
    return global.iter();
}

pub fn formatSummary(writer: anytype) !void {
    try global.formatSummary(writer);
}

pub fn byId(id: common.CommandId) *const common.Command {
    return global.byId(id);
}
