const std = @import("std");
const usage_mod = @import("usage.zig");
const registry = @import("registry.zig");
const completion = @import("completion.zig");

fn dispatchKind(command: registry.Command) []const u8 {
    if (command.handler != null) return "typed";
    if (command.raw_handler != null) return "raw";
    return "metadata";
}

fn writeExamplesJson(json: anytype, examples: []const []const u8) !void {
    try json.objectField("examples");
    try json.beginArray();
    for (examples) |example| try json.write(example);
    try json.endArray();
}

fn writeChoicesJson(json: anytype, choices: []const []const u8) !void {
    try json.objectField("choices");
    try json.beginArray();
    for (choices) |choice| try json.write(choice);
    try json.endArray();
}

fn writeArgJson(json: anytype, a: registry.Arg) !void {
    try json.beginObject();
    try json.objectField("name");
    try json.write(a.name);
    try json.objectField("kind");
    try json.write(@tagName(a.kind));
    try json.objectField("required");
    try json.write(a.required);
    try json.objectField("value_kind");
    try json.write(@tagName(a.value_kind));
    try json.objectField("greedy");
    try json.write(a.greedy);
    try json.objectField("help");
    try json.write(a.help);
    try writeChoicesJson(json, a.choices);
    try json.endObject();
}

fn writeArgsJson(json: anytype, args: []const registry.Arg) !void {
    try json.objectField("args");
    try json.beginArray();
    for (args) |a| try writeArgJson(json, a);
    try json.endArray();
}

fn writeSubcommandJson(json: anytype, subcommand: registry.Command) !void {
    try json.beginObject();
    try json.objectField("name");
    try json.write(subcommand.name);
    try json.objectField("usage");
    try json.write(subcommand.usage);
    try json.objectField("summary");
    try json.write(subcommand.summary);
    try json.objectField("dispatch");
    try json.write(dispatchKind(subcommand));
    try writeArgsJson(json, subcommand.args);
    try json.endObject();
}

fn writeSubcommandsJson(json: anytype, subcommands: []const registry.Command) !void {
    try json.objectField("subcommands");
    try json.beginArray();
    for (subcommands) |subcommand| try writeSubcommandJson(json, subcommand);
    try json.endArray();
}

fn writeShortcutsJson(json: anytype) !void {
    try json.objectField("shortcuts");
    try json.beginArray();
    for (registry.shortcuts) |shortcut| {
        try json.beginObject();
        try json.objectField("token");
        try json.write(shortcut.token);
        try json.objectField("command");
        try json.write(shortcut.command);
        try json.objectField("summary");
        try json.write(shortcut.summary);
        try json.endObject();
    }
    try json.endArray();
}

fn writeShortcutsJsonForCommand(json: anytype, command_name: []const u8) !void {
    try json.objectField("shortcuts");
    try json.beginArray();
    for (registry.shortcuts) |shortcut| {
        if (!std.mem.eql(u8, shortcut.command, command_name)) continue;
        try json.beginObject();
        try json.objectField("token");
        try json.write(shortcut.token);
        try json.objectField("command");
        try json.write(shortcut.command);
        try json.objectField("summary");
        try json.write(shortcut.summary);
        try json.endObject();
    }
    try json.endArray();
}

fn writeCompletionJson(json: anytype) !void {
    try json.objectField("completion");
    try json.beginObject();
    try json.objectField("usage");
    try json.write("abi help --completion <bash|zsh|fish>");
    try json.objectField("shells");
    try json.beginArray();
    try json.write("bash");
    try json.write("zsh");
    try json.write("fish");
    try json.endArray();
    try json.endObject();
}

fn writeCommandJson(json: anytype, command: registry.Command) !void {
    const usage_meta = usage_mod.findCommand(command.name);
    try json.beginObject();
    try json.objectField("name");
    try json.write(command.name);
    try json.objectField("usage");
    try json.write(command.usage);
    try json.objectField("summary");
    try json.write(command.summary);
    try json.objectField("category");
    try json.write(if (usage_meta) |meta_value| usage_mod.categoryName(meta_value.category) else "unknown");
    try json.objectField("details");
    try json.write(if (usage_meta) |meta_value| meta_value.details else "");
    try json.objectField("dispatch");
    try json.write(dispatchKind(command));
    try writeExamplesJson(json, if (usage_meta) |meta_value| meta_value.examples else &.{});
    try writeArgsJson(json, command.args);
    try writeSubcommandsJson(json, command.subcommands);
    try json.endObject();
}

fn findRegistrySubcommand(command: registry.Command, name: []const u8) ?registry.Command {
    for (command.subcommands) |subcommand| {
        if (std.mem.eql(u8, subcommand.name, name)) return subcommand;
    }
    return null;
}

pub fn writeHelpJson(writer: anytype, allocator: std.mem.Allocator, command_name: ?[]const u8, subcommand_name: ?[]const u8) !bool {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json = std.json.Stringify{
        .writer = &out.writer,
        .options = .{ .whitespace = .minified },
    };

    try json.beginObject();
    try json.objectField("type");
    try json.write("abi.cli.help");
    try json.objectField("version");
    try json.write(@as(u32, 1));
    try writeCompletionJson(&json);

    if (command_name) |name| {
        const resolved_name = registry.commandNameForShortcut(name) orelse name;
        const command = registry.findRegistryCommand(resolved_name) orelse return false;
        try json.objectField("command");
        try json.write(command.name);
        try writeShortcutsJsonForCommand(&json, command.name);
        if (subcommand_name) |sub_name| {
            const subcommand = findRegistrySubcommand(command, sub_name) orelse return false;
            try json.objectField("subcommand");
            try writeSubcommandJson(&json, subcommand);
        } else {
            try json.objectField("command_detail");
            try writeCommandJson(&json, command);
        }
    } else {
        if (subcommand_name != null) return false;
        try writeShortcutsJson(&json);
        try json.objectField("commands");
        try json.beginArray();
        for (registry.commands) |command| try writeCommandJson(&json, command);
        try json.endArray();
    }

    try json.endObject();
    try writer.writeAll(out.written());
    try writer.writeAll("\n");
    return true;
}

pub fn printHelpJson(allocator: std.mem.Allocator, command_name: ?[]const u8, subcommand_name: ?[]const u8) !?u8 {
    const DebugWriter = struct {
        pub fn writeAll(_: *@This(), bytes: []const u8) !void {
            std.debug.print("{s}", .{bytes});
        }
    };
    var writer = DebugWriter{};
    if (!try writeHelpJson(&writer, allocator, command_name, subcommand_name)) return null;
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
