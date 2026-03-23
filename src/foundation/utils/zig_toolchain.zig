const std = @import("std");
const builtin = @import("builtin");
const os = @import("../os.zig");

pub fn zigBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zig.exe" else "zig";
}

pub fn zlsBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zls.exe" else "zls";
}

/// Returns the path to ~/.cache/abi-zig/<version>/bin/zig for the given version.
pub fn allocAbiZigCachePath(allocator: std.mem.Allocator, version: []const u8) !?[]u8 {
    if (comptime os.no_os) return null;
    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    return try std.fs.path.join(allocator, &.{ home, ".cache", "abi-zig", version, "bin", zigBinaryName() });
}

/// Returns the path to ~/.cache/abi-zig/<version>/bin/zls for the given version.
pub fn allocAbiZlsCachePath(allocator: std.mem.Allocator, version: []const u8) !?[]u8 {
    if (comptime os.no_os) return null;
    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    return try std.fs.path.join(allocator, &.{ home, ".cache", "abi-zig", version, "bin", zlsBinaryName() });
}

/// Resolve zig from the abi-zig cache for a given version. Returns null if not found.
pub fn resolveExistingAbiZigPath(allocator: std.mem.Allocator, io: std.Io, version: []const u8) !?[]u8 {
    const path = try allocAbiZigCachePath(allocator, version) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

pub fn allocZvmMasterZigPathFromHome(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    if (comptime os.no_os) return error.NotSupported;
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "master", zigBinaryName() });
}

pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator) !?[]u8 {
    if (comptime os.no_os) return null;
    const home = os.Env.get("HOME") orelse os.Env.get("USERPROFILE") orelse return null;
    return try allocZvmMasterZigPathFromHome(allocator, home);
}

pub fn resolveExistingZvmMasterZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const path = try allocZvmMasterZigPath(allocator) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

/// Resolve the preferred zig path: abi-zig cache (if version available) → zvm master → PATH
pub fn resolveExistingPreferredZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
) !?[]u8 {
    _ = start_path;

    // Check abi-zig cache first (read version from .zigversion in cwd)
    if (resolveAbiZigFromVersionFile(allocator, io)) |abi_path| {
        return abi_path;
    } else |_| {}

    // Fall back to zvm master
    return resolveExistingZvmMasterZigPath(allocator, io);
}

/// Read .zigversion from cwd and resolve the abi-zig cached binary.
fn resolveAbiZigFromVersionFile(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const version_bytes = std.Io.Dir.cwd().readFileAlloc(io, ".zigversion", allocator, .limited(4096)) catch return null;
    defer allocator.free(version_bytes);
    // Trim whitespace
    const version = std.mem.trim(u8, version_bytes, " \t\r\n");
    if (version.len == 0) return null;
    return try resolveExistingAbiZigPath(allocator, io, version);
}

fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
