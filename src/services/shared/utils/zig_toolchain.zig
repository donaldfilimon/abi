const std = @import("std");
const builtin = @import("builtin");

pub fn zigBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zig.exe" else "zig";
}

pub fn zlsBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zls.exe" else "zls";
}

pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator) !?[]u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
    const home = std.mem.span(home_ptr);
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "master", zigBinaryName() });
}

pub fn resolveExistingZvmMasterZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const path = try allocZvmMasterZigPath(allocator) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

pub fn resolveExistingRepoLocalCelToolPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
    tool_name: []const u8,
) !?[]u8 {
    const repo_root = try findAbiRepoRoot(allocator, io, start_path) orelse return null;
    defer allocator.free(repo_root);

    const bootstrap_path = try std.fs.path.join(allocator, &.{ repo_root, ".zig-bootstrap", "bin", tool_name });
    if (fileExistsAbsolute(io, bootstrap_path)) {
        return bootstrap_path;
    }
    allocator.free(bootstrap_path);

    const legacy_path = try std.fs.path.join(allocator, &.{ repo_root, ".cel", "bin", tool_name });
    if (!fileExistsAbsolute(io, legacy_path)) {
        allocator.free(legacy_path);
        return null;
    }
    return legacy_path;
}

pub fn resolveExistingPreferredZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
) !?[]u8 {
    if (try resolveExistingRepoLocalCelToolPath(allocator, io, start_path, zigBinaryName())) |path| {
        return path;
    }
    return resolveExistingZvmMasterZigPath(allocator, io);
}

pub fn findAbiRepoRoot(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
) !?[]u8 {
    var current = try std.fs.path.resolve(allocator, &.{start_path});
    errdefer allocator.free(current);

    while (true) {
        if (try looksLikeAbiRepoRoot(allocator, io, current)) {
            return current;
        }

        const parent = std.fs.path.dirname(current) orelse break;
        if (std.mem.eql(u8, parent, current)) break;

        const next = try allocator.dupe(u8, parent);
        allocator.free(current);
        current = next;
    }

    allocator.free(current);
    return null;
}

fn looksLikeAbiRepoRoot(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
) !bool {
    const build_zig = try std.fs.path.join(allocator, &.{ root_path, "build.zig" });
    defer allocator.free(build_zig);
    if (!fileExistsAbsolute(io, build_zig)) return false;

    const root_entry = try std.fs.path.join(allocator, &.{ root_path, "src", "root.zig" });
    defer allocator.free(root_entry);
    if (!fileExistsAbsolute(io, root_entry)) return false;

    const bootstrap_build = try std.fs.path.join(allocator, &.{ root_path, ".zig-bootstrap", "build.sh" });
    defer allocator.free(bootstrap_build);
    if (fileExistsAbsolute(io, bootstrap_build)) return true;

    const legacy_build = try std.fs.path.join(allocator, &.{ root_path, ".cel", "build.sh" });
    defer allocator.free(legacy_build);
    return fileExistsAbsolute(io, legacy_build);
}

fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
