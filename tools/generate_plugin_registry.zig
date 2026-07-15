const std = @import("std");

const PluginEntry = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
    entry_point: []const u8,
    commands: []const CommandEntry = &.{},
    context_providers: []const ContextProviderEntry = &.{},

    fn deinit(self: PluginEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.description);
        allocator.free(self.target_feature);
        allocator.free(self.entry_point);
        for (self.commands) |cmd| {
            allocator.free(cmd.name);
            allocator.free(cmd.summary);
            for (cmd.aliases) |a| allocator.free(a);
            allocator.free(cmd.aliases);
        }
        allocator.free(self.commands);
        for (self.context_providers) |cp| cp.deinit(allocator);
        allocator.free(self.context_providers);
    }
};

const CommandEntry = struct {
    name: []const u8,
    summary: []const u8,
    aliases: []const []const u8,
};

const ContextProviderEntry = struct {
    name: []const u8,
    summary: []const u8,

    fn deinit(self: ContextProviderEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.summary);
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 3) return error.InvalidArguments;

    const plugins_dir = args[1];
    const output_path = args[2];

    var entries = std.ArrayListUnmanaged(PluginEntry).empty;
    defer {
        for (entries.items) |entry| entry.deinit(allocator);
        entries.deinit(allocator);
    }

    var dir = try std.Io.Dir.cwd().openDir(io, plugins_dir, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.eql(u8, std.fs.path.basename(entry.path), "abi-plugin.json")) continue;
        // A plugin is a *directory* under `plugins_dir` that contains its
        // manifest and entry point; a manifest at the walk root (dirname == null)
        // has no owning plugin directory, so it is intentionally ignored rather
        // than registered. Do not remove this guard to "simplify" discovery — it
        // is what keeps a stray top-level `abi-plugin.json` from being ingested.
        if (std.fs.path.dirname(entry.path) == null) continue;

        const path = try std.fs.path.join(allocator, &.{ plugins_dir, entry.path });
        defer allocator.free(path);

        const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024));
        defer allocator.free(source);

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, source, .{}) catch return error.InvalidPluginManifest;
        defer parsed.deinit();

        const name = stringField(parsed.value, "name") orelse return error.InvalidPluginManifest;
        if (name.len == 0) return error.InvalidPluginManifest;

        const version = stringField(parsed.value, "version") orelse return error.InvalidPluginManifest;
        if (version.len == 0) return error.InvalidPluginManifest;

        const target_feature = stringField(parsed.value, "target_feature") orelse stringField(parsed.value, "targetFeature") orelse return error.InvalidPluginManifest;
        if (target_feature.len == 0) return error.InvalidPluginManifest;

        const entry_point = stringField(parsed.value, "entry_point") orelse stringField(parsed.value, "entryPoint") orelse return error.InvalidPluginManifest;
        if (!isSafeEntryPoint(entry_point)) return error.InvalidPluginManifest;

        const description = stringField(parsed.value, "description") orelse return error.InvalidPluginManifest;
        if (description.len == 0) return error.InvalidPluginManifest;

        // Parse optional commands array
        const commands = blk: {
            const cmds_val = parsed.value.object.get("commands") orelse break :blk &.{};
            if (cmds_val != .array) break :blk &.{};
            const arr = cmds_val.array.items;
            const cmds = try allocator.alloc(CommandEntry, arr.len);
            errdefer allocator.free(cmds);
            for (arr, 0..) |cmd_val, i| {
                if (cmd_val != .object) return error.InvalidPluginManifest;
                const cmd_obj = cmd_val.object;
                const cmd_name = stringField(cmd_val, "name") orelse return error.InvalidPluginManifest;
                const cmd_summary = stringField(cmd_val, "summary") orelse "";

                var aliases: []const []const u8 = &.{};
                if (cmd_obj.get("aliases")) |alias_val| {
                    if (alias_val == .array) {
                        const alias_arr = try allocator.alloc([]const u8, alias_val.array.items.len);
                        errdefer allocator.free(alias_arr);
                        for (alias_val.array.items, 0..) |a, j| {
                            if (a != .string) return error.InvalidPluginManifest;
                            alias_arr[j] = try allocator.dupe(u8, a.string);
                        }
                        aliases = alias_arr;
                    }
                }

                cmds[i] = .{
                    .name = try allocator.dupe(u8, cmd_name),
                    .summary = try allocator.dupe(u8, cmd_summary),
                    .aliases = aliases,
                };
            }
            break :blk cmds;
        };

        // Parse optional context_providers array
        const context_providers = blk: {
            const cps_val = parsed.value.object.get("context_providers") orelse break :blk &.{};
            if (cps_val != .array) break :blk &.{};
            const arr = cps_val.array.items;
            const cps = try allocator.alloc(ContextProviderEntry, arr.len);
            errdefer allocator.free(cps);
            for (arr, 0..) |cp_val, i| {
                if (cp_val != .object) return error.InvalidPluginManifest;
                const cp_name = stringField(cp_val, "name") orelse return error.InvalidPluginManifest;
                if (cp_name.len == 0) return error.InvalidPluginManifest;
                const cp_summary = stringField(cp_val, "summary") orelse "";
                cps[i] = .{
                    .name = try allocator.dupe(u8, cp_name),
                    .summary = try allocator.dupe(u8, cp_summary),
                };
            }
            break :blk cps;
        };

        const entry_dir = std.fs.path.dirname(entry.path) orelse return error.InvalidPluginManifest;
        const entry_disk_path = try std.fs.path.join(allocator, &.{ plugins_dir, entry_dir, entry_point });
        defer allocator.free(entry_disk_path);
        _ = std.Io.Dir.cwd().statFile(io, entry_disk_path, .{}) catch return error.InvalidPluginManifest;

        try entries.append(allocator, .{
            .name = try allocator.dupe(u8, name),
            .version = try allocator.dupe(u8, version),
            .description = try allocator.dupe(u8, description),
            .target_feature = try allocator.dupe(u8, target_feature),
            .entry_point = try allocator.dupe(u8, entry_point),
            .commands = commands,
            .context_providers = context_providers,
        });
    }

    std.mem.sort(PluginEntry, entries.items, {}, lessThanName);

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator,
        \\//! Generated plugin registry. DO NOT EDIT.
        \\const Registry = @import("core/registry.zig").Registry;
        \\
        \\pub fn registerPlugins(registry: *Registry) !void {
        \\
    );

    for (entries.items) |entry| {
        try out.appendSlice(allocator, "    try registry.registerPlugin(.{ .name = ");
        try appendZigString(&out, allocator, entry.name);
        try out.appendSlice(allocator, ", .version = ");
        try appendZigString(&out, allocator, entry.version);
        try out.appendSlice(allocator, ", .description = ");
        try appendZigString(&out, allocator, entry.description);
        try out.appendSlice(allocator, ", .target_feature = ");
        try appendZigString(&out, allocator, entry.target_feature);
        try out.appendSlice(allocator, ", .entry_point = ");
        try appendZigString(&out, allocator, entry.entry_point);
        if (entry.commands.len > 0) {
            try out.appendSlice(allocator, ", .commands = &.{\n");
            for (entry.commands) |cmd| {
                try out.appendSlice(allocator, "        .{ .name = ");
                try appendZigString(&out, allocator, cmd.name);
                try out.appendSlice(allocator, ", .summary = ");
                try appendZigString(&out, allocator, cmd.summary);
                if (cmd.aliases.len > 0) {
                    try out.appendSlice(allocator, ", .aliases = &.{");
                    for (cmd.aliases, 0..) |alias, ai| {
                        if (ai > 0) try out.appendSlice(allocator, ", ");
                        try appendZigString(&out, allocator, alias);
                    }
                    try out.appendSlice(allocator, "}");
                }
                try out.appendSlice(allocator, " },\n");
            }
            try out.appendSlice(allocator, "    }");
        }
        if (entry.context_providers.len > 0) {
            try out.appendSlice(allocator, ", .context_providers = &.{\n");
            for (entry.context_providers) |cp| {
                try out.appendSlice(allocator, "        .{ .name = ");
                try appendZigString(&out, allocator, cp.name);
                try out.appendSlice(allocator, ", .summary = ");
                try appendZigString(&out, allocator, cp.summary);
                try out.appendSlice(allocator, " },\n");
            }
            try out.appendSlice(allocator, "    }");
        }
        try out.appendSlice(allocator, " });\n");
    }

    try out.appendSlice(allocator, "}\n");
    const generated = try out.toOwnedSlice(allocator);

    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = output_path,
        .data = generated,
    });
}

fn stringField(value: std.json.Value, field: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const entry = value.object.get(field) orelse return null;
    return if (entry == .string) entry.string else null;
}

fn lessThanName(_: void, lhs: PluginEntry, rhs: PluginEntry) bool {
    return std.mem.lessThan(u8, lhs.name, rhs.name);
}

fn isSafeEntryPoint(entry_point: []const u8) bool {
    if (entry_point.len == 0) return false;
    if (!std.mem.endsWith(u8, entry_point, ".zig")) return false;
    if (entry_point[0] == '/' or entry_point[0] == '\\') return false;
    if (std.mem.indexOfScalar(u8, entry_point, ':') != null) return false;

    var parts = std.mem.splitAny(u8, entry_point, "/\\");
    while (parts.next()) |part| {
        if (part.len == 0) return false;
        if (std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return true;
}

fn appendZigString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.print(allocator, "\"{f}\"", .{std.zig.fmtString(value)});
}

test "plugin registry string literals use Zig escaping" {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(std.testing.allocator);

    try appendZigString(&out, std.testing.allocator, "quote=\" slash=\\ newline=\n control=\x0f");
    try std.testing.expectEqualStrings("\"quote=\\\" slash=\\\\ newline=\\n control=\\x0f\"", out.items);
}

test {
    std.testing.refAllDecls(@This());
}
