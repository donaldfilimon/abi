const std = @import("std");

const PluginEntry = struct {
    name: []const u8,
    description: []const u8,

    fn deinit(self: PluginEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
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

        const entry_point = stringField(parsed.value, "entry_point") orelse return error.InvalidPluginManifest;
        if (entry_point.len == 0) return error.InvalidPluginManifest;

        const description = stringField(parsed.value, "description") orelse version;

        try entries.append(allocator, .{
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, description),
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
        try out.appendSlice(allocator, "    try registry.register(");
        try appendZigString(&out, allocator, entry.name);
        try out.appendSlice(allocator, ", ");
        try appendZigString(&out, allocator, entry.description);
        try out.appendSlice(allocator, ");\n");
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

fn appendZigString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '"' => try out.appendSlice(allocator, "\\\""),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => {
                if (byte < 0x20 or byte > 0x7e) return error.InvalidPluginManifest;
                try out.append(allocator, byte);
            },
        }
    }
    try out.append(allocator, '"');
}
