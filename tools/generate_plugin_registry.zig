const std = @import("std");

const PluginEntry = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
    entry_point: []const u8,

    fn deinit(self: PluginEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.description);
        allocator.free(self.target_feature);
        allocator.free(self.entry_point);
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
    // Mirrors src/connectors/json.zig appendJsonString — full RFC 8259 JSON string
    // escaping with \uXXXX for control characters. This is intentionally identical
    // to avoid duplicate maintenance of two JSON string escaper implementations.
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}
