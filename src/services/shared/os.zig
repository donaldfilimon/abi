//! Cross-Platform OS Features Module
//!
//! Provides unified OS-level operations for UI and agent actions across all platforms:
//! - Windows, Linux, macOS, FreeBSD, NetBSD, OpenBSD, DragonFly BSD
//! - WASM/WASI (with graceful degradation)

const std = @import("std");
const builtin = @import("builtin");

// libc imports for cross-platform compatibility (Zig 0.16)
const libc = if (builtin.link_libc) std.c else struct {
    pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
};

// ============================================================================
// Platform Detection
// ============================================================================

pub const OsKind = enum {
    windows,
    linux,
    macos,
    freebsd,
    netbsd,
    openbsd,
    dragonfly,
    ios,
    wasi,
    other,

    pub fn isPosix(self: OsKind) bool {
        return switch (self) {
            .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly, .ios => true,
            else => false,
        };
    }
};

pub const current_os: OsKind = switch (builtin.os.tag) {
    .windows => .windows,
    .linux => .linux,
    .macos => .macos,
    .freebsd => .freebsd,
    .netbsd => .netbsd,
    .openbsd => .openbsd,
    .dragonfly => .dragonfly,
    .ios => .ios,
    .wasi => .wasi,
    else => .other,
};

pub const is_wasm = builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;

// ============================================================================
// System Information
// ============================================================================

/// Get the system temp directory path (platform-agnostic)
pub fn getTempPath(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) return allocator.dupe(u8, "/tmp");

    if (comptime builtin.os.tag == .windows) {
        if (Env.get("TEMP")) |val| return allocator.dupe(u8, val);
        if (Env.get("TMP")) |val| return allocator.dupe(u8, val);
        return allocator.dupe(u8, "C:\\Windows\\Temp");
    }

    if (Env.get("TMPDIR")) |val| return allocator.dupe(u8, val);
    return allocator.dupe(u8, "/tmp");
}

/// Helper to get environment variable safely
pub const Env = struct {
    pub fn get(name: []const u8) ?[]const u8 {
        var buf: [256]u8 = undefined;
        if (name.len >= buf.len) return null;
        @memcpy(buf[0..name.len], name);
        buf[name.len] = 0;
        const name_z: [:0]const u8 = buf[0..name.len :0];

        // Prefer std.c.getenv for Darwin 26+ and general Zig 0.16 stability
        const ptr = std.c.getenv(name_z.ptr);
        if (ptr) |p| {
            return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(p)), 0);
        }
        return null;
    }
};

/// Get CPU core count
pub fn getCpuCount() u32 {
    if (comptime is_wasm) return 1;
    const count = std.Thread.getCpuCount() catch 1;
    return @intCast(@max(1, count));
}

/// Simplified exec for internal tools.
/// Returns the exit code of the spawned process (0 = success).
/// Non-`.exited` terminations (signal, stop, unknown) return exit code 1.
pub fn exec(allocator: std.mem.Allocator, command: []const u8) !u8 {
    if (comptime is_wasm) return 0;
    const shell = if (comptime builtin.os.tag == .windows) "cmd.exe" else "/bin/sh";
    const flag = if (comptime builtin.os.tag == .windows) "/c" else "-c";

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var child = try std.process.spawn(io, .{
        .argv = &[_][]const u8{ shell, flag, command },
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    });
    const term = try child.wait(io);
    return switch (term) {
        .exited => |code| code,
        else => 1,
    };
}

test "temp path detection" {
    const allocator = std.testing.allocator;
    const tmp = try getTempPath(allocator);
    defer allocator.free(tmp);
    try std.testing.expect(tmp.len > 0);
}
