const std = @import("std");

pub fn validatePluginStructure(allocator: std.mem.Allocator, plugin_path: []const u8) !bool {
    const io = std.Options.debug_io;
    var dir = try std.Io.Dir.openDirAbsolute(io, plugin_path, .{});
    defer dir.close(io);

    _ = dir.statFile(io, "mod.zig", .{}) catch return false;
    _ = dir.statFile(io, "stub.zig", .{}) catch return false;

    const manifest_json = std.Io.Dir.readFileAlloc(dir, io, "abi-plugin.json", allocator, .limited(64 * 1024)) catch return false;
    defer allocator.free(manifest_json);

    const manifest = std.json.parseFromSlice(std.json.Value, allocator, manifest_json, .{}) catch return false;
    defer manifest.deinit();

    const obj = switch (manifest.value) {
        .object => |object| object,
        else => return false,
    };

    const name = jsonStringField(obj, "name") orelse return false;
    const version = jsonStringField(obj, "version") orelse return false;
    const description = jsonStringField(obj, "description") orelse return false;
    const entry_point = jsonStringField(obj, "entry_point") orelse jsonStringField(obj, "entryPoint") orelse return false;
    const target_feature = jsonStringField(obj, "target_feature") orelse jsonStringField(obj, "targetFeature") orelse return false;

    if (name.len == 0 or version.len == 0 or description.len == 0 or target_feature.len == 0) return false;
    if (!isSafeEntryPoint(entry_point)) return false;
    _ = dir.statFile(io, entry_point, .{}) catch return false;

    return true;
}

test {
    std.testing.refAllDecls(@This());
}

test "validatePluginStructure rejects missing directory" {
    try std.testing.expectError(error.FileNotFound, validatePluginStructure(std.testing.allocator, "/tmp/nonexistent_plugin_dir_abc123"));
}

test "validatePluginStructure accepts bundled plugin fixtures" {
    inline for (.{
        "src/plugins/example-plugin/mod.zig",
        "src/plugins/example-wdbx-plugin/mod.zig",
    }) |fixture_path| {
        try expectValidBundledPlugin(fixture_path);
    }
}

test "validatePluginStructure rejects unsafe manifests" {
    const fixture_dir = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_plugin_validator_unsafe_{d}", .{std.c.getpid()});
    defer std.testing.allocator.free(fixture_dir);
    try resetPluginFixture(fixture_dir);
    defer cleanupPluginFixture(fixture_dir);

    try writePluginFixtureFile(fixture_dir, "mod.zig", "pub fn main() void {}\n");
    try writePluginFixtureFile(fixture_dir, "stub.zig", "pub fn main() void {}\n");

    try writePluginFixtureFile(fixture_dir, "abi-plugin.json",
        \\{"name":"bad-plugin","version":"0.1.0","description":"bad","target_feature":"plugins","entry_point":"../mod.zig"}
    );
    try std.testing.expect(!try validatePluginStructure(std.testing.allocator, fixture_dir));

    try writePluginFixtureFile(fixture_dir, "abi-plugin.json",
        \\{"name":"bad-plugin","version":"0.1.0","description":"bad","target_feature":"plugins","entry_point":"missing.zig"}
    );
    try std.testing.expect(!try validatePluginStructure(std.testing.allocator, fixture_dir));

    try writePluginFixtureFile(fixture_dir, "abi-plugin.json",
        \\{"name":"bad-plugin","version":"0.1.0","description":"bad","target_feature":"","entry_point":"mod.zig"}
    );
    try std.testing.expect(!try validatePluginStructure(std.testing.allocator, fixture_dir));
}

test "validatePluginStructure accepts camelCase aliases and nested entry" {
    const fixture_dir = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_plugin_validator_nested_{d}", .{std.c.getpid()});
    defer std.testing.allocator.free(fixture_dir);
    try resetPluginFixture(fixture_dir);
    defer cleanupPluginFixture(fixture_dir);

    try writePluginFixtureFile(fixture_dir, "mod.zig", "pub fn main() void {}\n");
    try writePluginFixtureFile(fixture_dir, "stub.zig", "pub fn main() void {}\n");
    const nested_dir = try std.fs.path.join(std.testing.allocator, &.{ fixture_dir, "nested" });
    defer std.testing.allocator.free(nested_dir);
    try std.Io.Dir.createDirPath(.cwd(), std.Options.debug_io, nested_dir);
    try writePluginFixtureFile(fixture_dir, "nested/entry.zig", "pub fn main() void {}\n");
    try writePluginFixtureFile(fixture_dir, "abi-plugin.json",
        \\{"name":"nested-plugin","version":"0.1.0","description":"ok","targetFeature":"plugins","entryPoint":"nested/entry.zig"}
    );

    try std.testing.expect(try validatePluginStructure(std.testing.allocator, fixture_dir));
}

fn expectValidBundledPlugin(mod_path: []const u8) !void {
    var buf: [4096]u8 = undefined;
    const len = try std.Io.Dir.realPathFile(.cwd(), std.Options.debug_io, mod_path, &buf);
    const fixture_mod = buf[0..len];
    const fixture_dir = std.fs.path.dirname(fixture_mod) orelse return error.MissingFixtureDir;
    try std.testing.expect(try validatePluginStructure(std.testing.allocator, fixture_dir));
}

fn resetPluginFixture(path: []const u8) !void {
    std.Io.Dir.deleteTree(.cwd(), std.Options.debug_io, path) catch |err| {
        std.log.debug("plugin validator fixture reset cleanup skipped: {s}", .{@errorName(err)});
    };
    try std.Io.Dir.createDirPath(.cwd(), std.Options.debug_io, path);
}

fn cleanupPluginFixture(path: []const u8) void {
    std.Io.Dir.deleteTree(.cwd(), std.Options.debug_io, path) catch |err| std.log.warn("plugin validator fixture cleanup failed: {s}", .{@errorName(err)});
}

fn writePluginFixtureFile(base: []const u8, relative_path: []const u8, content: []const u8) !void {
    const full_path = try std.fs.path.join(std.testing.allocator, &.{ base, relative_path });
    defer std.testing.allocator.free(full_path);
    const file = try std.Io.Dir.createFileAbsolute(std.Options.debug_io, full_path, .{ .truncate = true });
    defer file.close(std.Options.debug_io);
    try file.writeStreamingAll(std.Options.debug_io, content);
}

fn jsonStringField(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
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
