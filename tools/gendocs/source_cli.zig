const std = @import("std");
const cli_root = @import("cli_root");
const model = @import("model.zig");

const commands_mod = cli_root.commands;
const CommandDescriptor = std.meta.Elem(@TypeOf(commands_mod.descriptors));

/// Discover CLI commands directly from the canonical descriptor registry so
/// docs, help, completions, and smoke coverage all resolve the same metadata.
pub fn discoverCommands(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.CliCommand {
    _ = io;
    _ = cwd;

    var commands = std.ArrayListUnmanaged(model.CliCommand).empty;
    errdefer {
        for (commands.items) |command| command.deinit(allocator);
        commands.deinit(allocator);
    }

    for (commands_mod.descriptors) |descriptor| {
        if (descriptor.visibility == .hidden) continue;
        try commands.append(allocator, try cloneDescriptor(allocator, descriptor));
    }

    std.mem.sort(model.CliCommand, commands.items, {}, model.compareCommands);
    return commands.toOwnedSlice(allocator);
}

fn cloneDescriptor(allocator: std.mem.Allocator, descriptor: CommandDescriptor) !model.CliCommand {
    return .{
        .name = try allocator.dupe(u8, descriptor.name),
        .description = try allocator.dupe(u8, descriptor.description),
        .aliases = try cloneStringSlice(allocator, descriptor.aliases),
        .subcommands = try cloneStringSlice(allocator, descriptor.subcommands),
    };
}

fn cloneStringSlice(allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    const cloned = try allocator.alloc([]const u8, values.len);
    errdefer allocator.free(cloned);

    var initialized: usize = 0;
    errdefer {
        for (cloned[0..initialized]) |value| allocator.free(value);
    }

    for (values, 0..) |value, i| {
        cloned[i] = try allocator.dupe(u8, value);
        initialized += 1;
    }

    return cloned;
}

test "discoverCommands includes canonical top-level editor descriptor" {
    const commands = try discoverCommands(std.testing.allocator, std.testing.io, std.Io.Dir.cwd());
    defer model.deinitCommandSlice(std.testing.allocator, commands);

    var found_editor = false;
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, "editor")) {
            found_editor = true;
            try std.testing.expectEqualStrings("Open the shared inline terminal text editor", command.description);
            try std.testing.expectEqual(@as(usize, 1), command.aliases.len);
            try std.testing.expectEqualStrings("edit", command.aliases[0]);
        }
    }

    try std.testing.expect(found_editor);
}
