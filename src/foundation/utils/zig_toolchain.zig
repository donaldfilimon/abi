const std = @import("std");
const builtin = @import("builtin");
const os = @import("../os.zig");

const PreferredSource = enum {
    zvm_bin,
    zigly_cache,
    none,
};

pub fn zigBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zig.exe" else "zig";
}

pub fn zlsBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zls.exe" else "zls";
}

pub fn allocZiglyZigPathFromHome(allocator: std.mem.Allocator, home: []const u8, version: []const u8) ![]u8 {
    if (comptime os.no_os) return error.NotSupported;
    return try std.fs.path.join(allocator, &.{ home, ".zigly", "versions", version, "bin", zigBinaryName() });
}

pub fn allocZiglyZigPath(allocator: std.mem.Allocator, version: []const u8) !?[]u8 {
    if (comptime os.no_os) return null;
    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    return try allocZiglyZigPathFromHome(allocator, home, version);
}

pub fn resolveExistingZiglyZigPath(allocator: std.mem.Allocator, io: std.Io, version: []const u8) !?[]u8 {
    const path = try allocZiglyZigPath(allocator, version) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

pub fn allocZvmBinZigPathFromHome(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    if (comptime os.no_os) return error.NotSupported;
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "bin", zigBinaryName() });
}

pub fn allocZvmBinZigPath(allocator: std.mem.Allocator) !?[]u8 {
    if (comptime os.no_os) return null;
    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    return try allocZvmBinZigPathFromHome(allocator, home);
}

pub fn resolveExistingZvmBinZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const path = try allocZvmBinZigPath(allocator) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

/// Resolve the preferred zig path: active ZVM binary when it matches .zigversion
/// → zigly cache for the pinned version.
pub fn resolveExistingPreferredZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
) !?[]u8 {
    _ = start_path;

    const version = try readProjectVersion(allocator, io) orelse return null;
    defer allocator.free(version);

    const zvm_path = try resolveExistingZvmBinZigPath(allocator, io);
    const zvm_version = if (zvm_path) |path| try probeBinaryVersion(allocator, io, path) else null;
    defer if (zvm_version) |v| allocator.free(v);

    const zigly_path = try resolveExistingZiglyZigPath(allocator, io, version);
    switch (selectPreferredSource(version, zvm_version, zigly_path != null)) {
        .zvm_bin => {
            if (zigly_path) |path| allocator.free(path);
            return zvm_path;
        },
        .zigly_cache => {
            if (zvm_path) |path| allocator.free(path);
            return zigly_path;
        },
        .none => {
            if (zvm_path) |path| allocator.free(path);
            if (zigly_path) |path| allocator.free(path);
            return null;
        },
    }
}

fn selectPreferredSource(
    requested_version: []const u8,
    zvm_version: ?[]const u8,
    has_zigly_cache: bool,
) PreferredSource {
    if (versionMatches(requested_version, zvm_version)) return .zvm_bin;
    if (has_zigly_cache) return .zigly_cache;
    return .none;
}

fn versionMatches(requested_version: []const u8, actual_version: ?[]const u8) bool {
    const actual = actual_version orelse return false;
    return std.mem.eql(u8, requested_version, actual);
}

fn readProjectVersion(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const version_bytes = std.Io.Dir.cwd().readFileAlloc(io, ".zigversion", allocator, .limited(4096)) catch return null;
    defer allocator.free(version_bytes);

    const version = std.mem.trim(u8, version_bytes, " \t\r\n");
    if (version.len == 0) return null;
    return try allocator.dupe(u8, version);
}

fn probeBinaryVersion(allocator: std.mem.Allocator, io: std.Io, binary_path: []const u8) !?[]u8 {
    if (!fileExistsAbsolute(io, binary_path)) return null;
    if (comptime os.no_os) return null;

    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    const tmp_dir = try std.fs.path.join(allocator, &.{ home, ".zigly", "tmp" });
    defer allocator.free(tmp_dir);
    try std.Io.Dir.cwd().createDirPath(io, tmp_dir);

    const output_path = try std.fs.path.join(allocator, &.{ tmp_dir, "toolchain-version.txt" });
    defer allocator.free(output_path);

    var child_args = std.ArrayListUnmanaged([]const u8).empty;
    defer child_args.deinit(allocator);
    try child_args.append(allocator, "sh");
    try child_args.append(allocator, "-c");
    try child_args.append(allocator, "\"$1\" version > \"$2\"");
    try child_args.append(allocator, "sh");
    try child_args.append(allocator, binary_path);
    try child_args.append(allocator, output_path);

    var child = try std.process.spawn(io, .{ .argv = child_args.items });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return null,
        else => return null,
    }

    const file = std.Io.Dir.openFileAbsolute(io, output_path, .{}) catch return null;
    defer file.close(io);
    const stat = file.stat(io) catch return null;
    if (stat.size == 0) return null;

    const content = try allocator.alloc(u8, @min(stat.size, 128));
    const bytes_read = try file.readPositionalAll(io, content, 0);
    const trimmed = std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
    if (trimmed.len == 0) {
        allocator.free(content);
        return null;
    }
    const version = try allocator.dupe(u8, trimmed);
    allocator.free(content);
    return version;
}

fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

test "preferred source uses zvm when active version matches" {
    try std.testing.expectEqual(
        PreferredSource.zvm_bin,
        selectPreferredSource("0.17.0-dev.135+9df02121d", "0.17.0-dev.135+9df02121d", true),
    );
}

test "preferred source falls back to zigly cache when zvm does not match" {
    try std.testing.expectEqual(
        PreferredSource.zigly_cache,
        selectPreferredSource("0.17.0-dev.135+9df02121d", "0.17.0-dev.27+0dd99c37c", true),
    );
}

test "preferred source returns none when no resolver matches" {
    try std.testing.expectEqual(
        PreferredSource.none,
        selectPreferredSource("0.17.0-dev.135+9df02121d", null, false),
    );
}

test {
    std.testing.refAllDecls(@This());
}
