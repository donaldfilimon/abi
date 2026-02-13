//! Cross-Platform Test Utilities
//!
//! Provides utilities for writing tests that work across Linux, Windows, and macOS.
//! Handles platform-specific differences in file paths, environment variables,
//! terminal capabilities, and system APIs.

const std = @import("std");
const abi = @import("abi");
const time = abi.shared.time;
const sync = abi.shared.sync;
const builtin = @import("builtin");

const WindowsKernel32 = if (builtin.os.tag == .windows)
    struct {
        extern "kernel32" fn GetStdHandle(nStdHandle: std.os.windows.DWORD) callconv(.winapi) ?std.os.windows.HANDLE;
    }
else
    struct {};

/// Current platform information
pub const Platform = struct {
    os: std.Target.Os.Tag,
    arch: std.Target.Cpu.Arch,
    is_windows: bool,
    is_linux: bool,
    is_macos: bool,
    is_bsd: bool,
    is_posix: bool,
    is_64bit: bool,
    path_sep: []const u8,
    line_ending: []const u8,

    pub fn detect() Platform {
        const os = builtin.os.tag;
        return .{
            .os = os,
            .arch = builtin.cpu.arch,
            .is_windows = os == .windows,
            .is_linux = os == .linux,
            .is_macos = os == .macos,
            .is_bsd = os == .freebsd or os == .netbsd or os == .openbsd or os == .dragonfly,
            .is_posix = os != .windows,
            .is_64bit = @sizeOf(usize) == 8,
            .path_sep = std.fs.path.sep_str,
            .line_ending = if (os == .windows) "\r\n" else "\n",
        };
    }

    /// Get the current platform
    pub fn current() Platform {
        return comptime detect();
    }
};

/// Test environment setup and teardown
pub const TestEnv = struct {
    allocator: std.mem.Allocator,
    temp_dir: ?[]const u8,
    original_cwd: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) TestEnv {
        return .{
            .allocator = allocator,
            .temp_dir = null,
            .original_cwd = null,
        };
    }

    pub fn deinit(self: *TestEnv) void {
        if (self.temp_dir) |dir| {
            self.allocator.free(dir);
        }
        if (self.original_cwd) |cwd| {
            self.allocator.free(cwd);
        }
    }

    /// Create a temporary directory for test files
    pub fn createTempDir(self: *TestEnv, prefix: []const u8) ![]const u8 {
        const tmp_base = std.fs.tmpPath() orelse switch (builtin.os.tag) {
            .windows => "C:\\Windows\\Temp",
            else => "/tmp",
        };

        var timer = time.Timer.start() catch {
            // Fallback: use a simple counter
            const path = try std.fmt.allocPrint(
                self.allocator,
                "{s}{s}{s}_test_{d}",
                .{ tmp_base, std.fs.path.sep_str, prefix, @as(u64, 0) },
            );
            self.temp_dir = path;
            return path;
        };

        const unique = timer.read();
        const path = try std.fmt.allocPrint(
            self.allocator,
            "{s}{s}{s}_test_{d}",
            .{ tmp_base, std.fs.path.sep_str, prefix, unique },
        );

        self.temp_dir = path;
        return path;
    }
};

/// Skip test if platform condition is not met
pub fn skipUnless(condition: bool, reason: []const u8) error{SkipZigTest}!void {
    if (!condition) {
        std.debug.print("Test skipped: {s}\n", .{reason});
        return error.SkipZigTest;
    }
}

/// Skip test on specific platform
pub fn skipOnPlatform(os: std.Target.Os.Tag, reason: []const u8) error{SkipZigTest}!void {
    if (builtin.os.tag == os) {
        std.debug.print("Test skipped on {t}: {s}\n", .{ os, reason });
        return error.SkipZigTest;
    }
}

/// Skip test if not on specific platform
pub fn requirePlatform(os: std.Target.Os.Tag, reason: []const u8) error{SkipZigTest}!void {
    if (builtin.os.tag != os) {
        std.debug.print("Test requires {t}: {s}\n", .{ os, reason });
        return error.SkipZigTest;
    }
}

/// Check if running in CI environment
pub fn isCI() bool {
    // Common CI environment variables
    const ci_vars = [_][]const u8{
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "BUILDKITE",
    };

    for (ci_vars) |var_name| {
        if (std.c.getenv(var_name.ptr) != null) {
            return true;
        }
    }
    return false;
}

/// Check if running with terminal attached
pub fn hasTty() bool {
    if (builtin.os.tag == .windows) {
        const handle = WindowsKernel32.GetStdHandle(std.os.windows.STD_INPUT_HANDLE) orelse return false;
        return handle != std.os.windows.INVALID_HANDLE_VALUE;
    } else {
        return std.posix.isatty(std.posix.STDIN_FILENO);
    }
}

/// Normalize path separators for comparison
pub fn normalizePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var result = try allocator.alloc(u8, path.len);
    for (path, 0..) |c, i| {
        result[i] = if (c == '/' or c == '\\') std.fs.path.sep else c;
    }
    return result;
}

/// Join paths in a cross-platform way
pub fn joinPath(allocator: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    var total_len: usize = 0;
    for (parts, 0..) |part, i| {
        total_len += part.len;
        if (i < parts.len - 1) total_len += 1; // separator
    }

    var result = try allocator.alloc(u8, total_len);
    var pos: usize = 0;

    for (parts, 0..) |part, i| {
        @memcpy(result[pos..][0..part.len], part);
        pos += part.len;
        if (i < parts.len - 1) {
            result[pos] = std.fs.path.sep;
            pos += 1;
        }
    }

    return result;
}

/// Assert approximately equal for floating point (cross-platform consistent)
pub fn expectApproxEqFloat(comptime T: type, expected: T, actual: T, tolerance: T) !void {
    const diff = @abs(expected - actual);
    if (diff > tolerance) {
        std.debug.print(
            "Expected {d} to be approximately {d} (tolerance: {d}, diff: {d})\n",
            .{ actual, expected, tolerance, diff },
        );
        return error.TestExpectedApproxEq;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "Platform: detection" {
    const platform = Platform.current();

    // Basic sanity checks
    try std.testing.expect(platform.path_sep.len > 0);
    try std.testing.expect(platform.line_ending.len > 0);

    // Verify consistency
    if (platform.is_windows) {
        try std.testing.expectEqualStrings("\\", platform.path_sep);
        try std.testing.expectEqualStrings("\r\n", platform.line_ending);
        try std.testing.expect(!platform.is_posix);
    } else {
        try std.testing.expectEqualStrings("/", platform.path_sep);
        try std.testing.expectEqualStrings("\n", platform.line_ending);
        try std.testing.expect(platform.is_posix);
    }
}

test "Platform: mutually exclusive flags" {
    const platform = Platform.current();

    // Only one OS should be true
    var count: u32 = 0;
    if (platform.is_windows) count += 1;
    if (platform.is_linux) count += 1;
    if (platform.is_macos) count += 1;

    // Should be at most one (could be zero for other Unix systems)
    try std.testing.expect(count <= 1);
}

test "TestEnv: basic lifecycle" {
    const allocator = std.testing.allocator;

    var env = TestEnv.init(allocator);
    defer env.deinit();

    try std.testing.expect(env.temp_dir == null);
}

test "normalizePath: converts separators" {
    const allocator = std.testing.allocator;

    const unix_path = "foo/bar/baz";
    const normalized = try normalizePath(allocator, unix_path);
    defer allocator.free(normalized);

    // On Windows, should be foo\bar\baz; on Unix, foo/bar/baz
    const platform = Platform.current();
    if (platform.is_windows) {
        try std.testing.expectEqualStrings("foo\\bar\\baz", normalized);
    } else {
        try std.testing.expectEqualStrings("foo/bar/baz", normalized);
    }
}

test "joinPath: combines parts" {
    const allocator = std.testing.allocator;

    const parts = [_][]const u8{ "home", "user", "docs" };
    const joined = try joinPath(allocator, &parts);
    defer allocator.free(joined);

    const platform = Platform.current();
    if (platform.is_windows) {
        try std.testing.expectEqualStrings("home\\user\\docs", joined);
    } else {
        try std.testing.expectEqualStrings("home/user/docs", joined);
    }
}

test "expectApproxEqFloat: within tolerance" {
    try expectApproxEqFloat(f32, 1.0, 1.00001, 0.001);
    try expectApproxEqFloat(f64, 3.14159, 3.14160, 0.0001);
}

test "isCI: returns bool" {
    // Just verify it doesn't crash and returns a bool
    const ci = isCI();
    _ = ci;
}
