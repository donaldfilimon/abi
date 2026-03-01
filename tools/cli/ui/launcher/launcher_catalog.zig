//! Descriptor-driven launcher catalog for the TUI command launcher.

const std = @import("std");
const framework = @import("../../framework/mod.zig");
const commands = @import("../../mod.zig");
const types = @import("types.zig");

const MenuItem = types.MenuItem;
const CommandRef = types.CommandRef;

const empty_args = &[_][:0]const u8{};

pub fn menuItems() []const MenuItem {
    return &catalog_items;
}

pub fn findCommandById(command_id: []const u8) ?CommandRef {
    for (catalog_items) |item| {
        switch (item.action) {
            .command => |cmd| {
                if (std.mem.eql(u8, cmd.id, command_id)) return cmd;
            },
            else => {},
        }
    }
    return null;
}

pub fn commandName(command_id: []const u8) []const u8 {
    if (findCommandById(command_id)) |cmd| return cmd.id;
    return command_id;
}

fn mapCategory(category: framework.types.UiCategory) types.Category {
    return switch (category) {
        .ai => .ai,
        .data => .data,
        .system => .system,
        .tools => .tools,
        .meta => .meta,
    };
}

fn heuristicCategory(comptime command_name: []const u8) types.Category {
    if (std.mem.eql(u8, command_name, "agent") or
        std.mem.eql(u8, command_name, "llm") or
        std.mem.eql(u8, command_name, "train") or
        std.mem.eql(u8, command_name, "embed") or
        std.mem.eql(u8, command_name, "model") or
        std.mem.eql(u8, command_name, "brain") or
        std.mem.eql(u8, command_name, "ralph") or
        std.mem.eql(u8, command_name, "ui"))
    {
        return .ai;
    }
    if (std.mem.eql(u8, command_name, "db") or
        std.mem.eql(u8, command_name, "explore") or
        std.mem.eql(u8, command_name, "task"))
    {
        return .data;
    }
    if (std.mem.eql(u8, command_name, "gpu") or
        std.mem.eql(u8, command_name, "network") or
        std.mem.eql(u8, command_name, "system-info") or
        std.mem.eql(u8, command_name, "toolchain") or
        std.mem.eql(u8, command_name, "lsp") or
        std.mem.eql(u8, command_name, "mcp") or
        std.mem.eql(u8, command_name, "acp") or
        std.mem.eql(u8, command_name, "env"))
    {
        return .system;
    }
    return .tools;
}

fn defaultArgs(comptime desc: framework.types.CommandDescriptor) []const [:0]const u8 {
    if (desc.default_subcommand) |default_subcommand| {
        const Holder = struct {
            const args = [_][:0]const u8{default_subcommand};
        };
        return &Holder.args;
    }
    return empty_args;
}

fn commandLabel(comptime desc: framework.types.CommandDescriptor) []const u8 {
    return if (desc.ui.label) |label| label else desc.name;
}

fn commandUsage(comptime desc: framework.types.CommandDescriptor) []const u8 {
    return if (desc.ui.usage) |usage|
        usage
    else
        std.fmt.comptimePrint("abi {s}", .{desc.name});
}

fn includeInLauncher(comptime desc: framework.types.CommandDescriptor) bool {
    return desc.visibility == .public and desc.ui.include_in_launcher;
}

const visible_command_count: usize = blk: {
    var count: usize = 0;
    for (commands.descriptors) |desc| {
        if (includeInLauncher(desc)) count += 1;
    }
    break :blk count;
};

const catalog_items = blk: {
    var out: [visible_command_count + 3]MenuItem = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        if (!includeInLauncher(desc)) continue;

        const category = if (desc.ui.category) |explicit_category|
            mapCategory(explicit_category)
        else
            heuristicCategory(desc.name);

        out[index] = .{
            .label = commandLabel(desc),
            .description = desc.description,
            .action = .{ .command = .{
                .id = desc.name,
                .command = desc.name,
                .args = defaultArgs(desc),
            } },
            .category = category,
            .shortcut = desc.ui.shortcut,
            .usage = commandUsage(desc),
            .examples = desc.ui.examples,
            .related = desc.ui.related,
        };
        index += 1;
    }

    out[index] = .{ .label = "Help", .description = "Show CLI usage", .action = .help, .category = .meta };
    index += 1;
    out[index] = .{ .label = "Version", .description = "Show version", .action = .version, .category = .meta };
    index += 1;
    out[index] = .{ .label = "Quit", .description = "Exit the launcher", .action = .quit, .category = .meta };

    break :blk out;
};

comptime {
    validateCatalog(&catalog_items);
}

fn validateCatalog(comptime items: []const MenuItem) void {
    inline for (items, 0..) |item, index| {
        if (item.shortcut) |shortcut| {
            if (shortcut < 1 or shortcut > 9) {
                @compileError(std.fmt.comptimePrint(
                    "launcher item '{s}' has shortcut {d}; expected 1-9",
                    .{ item.label, shortcut },
                ));
            }
            inline for (items[0..index]) |prev| {
                if (prev.shortcut) |prev_shortcut| {
                    if (prev_shortcut == shortcut) {
                        @compileError(std.fmt.comptimePrint(
                            "duplicate launcher shortcut {d} between '{s}' and '{s}'",
                            .{ shortcut, prev.label, item.label },
                        ));
                    }
                }
            }
        }

        switch (item.action) {
            .command => |cmd| {
                if (findTopLevelDescriptor(cmd.command) == null) {
                    @compileError(std.fmt.comptimePrint(
                        "launcher item '{s}' references unknown command '{s}'",
                        .{ item.label, cmd.command },
                    ));
                }
            },
            else => {},
        }
    }
}

fn findTopLevelDescriptor(comptime raw_command: []const u8) ?*const framework.types.CommandDescriptor {
    inline for (&commands.descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw_command, descriptor.name)) return descriptor;
        inline for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw_command, alias)) return descriptor;
        }
    }
    return null;
}

test {
    std.testing.refAllDecls(@This());
}
