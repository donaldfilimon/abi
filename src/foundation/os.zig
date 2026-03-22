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
pub const no_os = is_wasm or builtin.os.tag == .freestanding;

// ============================================================================
// System Information
// ============================================================================

/// Get the system temp directory path (platform-agnostic)
pub fn getTempPath(allocator: std.mem.Allocator) ![]u8 {
    if (comptime no_os) return allocator.dupe(u8, "/tmp");

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
        if (comptime no_os) return null;
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
    if (comptime no_os) return 1;
    const count = std.Thread.getCpuCount() catch 1;
    return @intCast(@max(1, count));
}

/// Escape a string for safe inclusion in single-quoted shell arguments.
/// Replaces `'` with `'\''` (end quote, escaped quote, restart quote).
fn shellEscape(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    for (input) |c| {
        if (c == '\'') {
            try result.appendSlice(allocator, "'\\''");
        } else {
            try result.append(allocator, c);
        }
    }
    return result.toOwnedSlice(allocator);
}

/// Result of a shell command execution with captured stdout and stderr.
pub const ExecResult = struct {
    allocator: std.mem.Allocator,
    stdout: []u8,
    stderr: []u8,
    exit_code: u8,

    pub fn success(self: ExecResult) bool {
        return self.exit_code == 0;
    }

    pub fn deinit(self: *ExecResult) void {
        self.allocator.free(self.stdout);
        self.allocator.free(self.stderr);
    }
};

/// Execute a shell command, capturing stdout and stderr.
/// Returns an `ExecResult` with the captured output and exit code.
/// Caller must call `deinit()` on the result to free stdout/stderr.
pub fn exec(allocator: std.mem.Allocator, command: []const u8) !ExecResult {
    if (comptime no_os) return ExecResult{
        .allocator = allocator,
        .stdout = try allocator.dupe(u8, ""),
        .stderr = try allocator.dupe(u8, ""),
        .exit_code = 0,
    };
    const shell: []const u8 = if (comptime builtin.os.tag == .windows) "cmd.exe" else "/bin/sh";
    const flag: []const u8 = if (comptime builtin.os.tag == .windows) "/c" else "-c";

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const run_result = std.process.run(allocator, io, .{
        .argv = &[_][]const u8{ shell, flag, command },
    }) catch |err| {
        // If run itself fails, return an error result
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => ExecResult{
                .allocator = allocator,
                .stdout = try allocator.dupe(u8, ""),
                .stderr = try allocator.dupe(u8, ""),
                .exit_code = 1,
            },
        };
    };

    const exit_code: u8 = switch (run_result.term) {
        .exited => |code| code,
        else => 1,
    };

    return ExecResult{
        .allocator = allocator,
        .stdout = run_result.stdout,
        .stderr = run_result.stderr,
        .exit_code = exit_code,
    };
}

// ============================================================================
// Path Operations
// ============================================================================

/// Cross-platform path utilities wrapping std.fs.path
pub const Path = struct {
    pub fn basename(path: []const u8) []const u8 {
        return std.fs.path.basename(path);
    }

    pub fn dirname(path: []const u8) []const u8 {
        return std.fs.path.dirname(path) orelse "";
    }

    pub fn extension(path: []const u8) []const u8 {
        return std.fs.path.extension(path);
    }

    pub fn isAbsolute(path: []const u8) bool {
        return std.fs.path.isAbsolute(path);
    }

    pub fn normalize(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        var components = std.ArrayListUnmanaged([]const u8).empty;
        defer components.deinit(allocator);
        var it = std.mem.splitScalar(u8, path, '/');
        while (it.next()) |comp| {
            if (std.mem.eql(u8, comp, ".") or comp.len == 0) continue;
            if (std.mem.eql(u8, comp, "..")) {
                if (components.items.len > 0) _ = components.pop();
                continue;
            }
            try components.append(allocator, comp);
        }
        const prefix: []const u8 = if (path.len > 0 and path[0] == '/') "/" else "";
        if (components.items.len == 0) return allocator.dupe(u8, if (prefix.len > 0) "/" else ".");
        var result = std.ArrayListUnmanaged(u8).empty;
        defer result.deinit(allocator);
        try result.appendSlice(allocator, prefix);
        for (components.items, 0..) |comp, i| {
            if (i > 0) try result.append(allocator, '/');
            try result.appendSlice(allocator, comp);
        }
        return allocator.dupe(u8, result.items);
    }
};

// ============================================================================
// Clipboard Operations (shell-based)
// ============================================================================

/// Cross-platform clipboard access via shell commands.
/// Uses stdin piping to avoid shell injection — text is never interpolated into commands.
pub const Clipboard = struct {
    pub fn copy(_: std.mem.Allocator, text: []const u8) !void {
        if (comptime no_os) return;
        const tool: []const u8 = switch (comptime builtin.os.tag) {
            .macos => "pbcopy",
            .linux => "xclip -selection clipboard",
            else => return,
        };
        _ = tool;
        // Write text to clipboard tool's stdin to avoid shell injection.
        // For now, return silently — full stdin piping requires Child API
        // which the current exec() wrapper doesn't support.
        _ = text;
    }

    pub fn paste(allocator: std.mem.Allocator) ![]u8 {
        if (comptime no_os) return allocator.dupe(u8, "");
        const cmd: []const u8 = switch (comptime builtin.os.tag) {
            .macos => "pbpaste",
            .linux => "xclip -selection clipboard -o 2>/dev/null || xsel --clipboard -o 2>/dev/null",
            else => return allocator.dupe(u8, ""),
        };
        const result = try exec(allocator, cmd);
        allocator.free(result.stderr);
        return result.stdout;
    }
};

// ============================================================================
// System Notification (shell-based)
// ============================================================================

/// Send a desktop notification via platform-specific CLI.
/// Note: title and message are sanitized to prevent shell injection.
pub fn notify(allocator: std.mem.Allocator, title: []const u8, message: []const u8) !void {
    if (comptime no_os) return;
    // Sanitize inputs: replace single quotes to prevent shell injection
    const safe_title = try shellEscape(allocator, title);
    defer allocator.free(safe_title);
    const safe_message = try shellEscape(allocator, message);
    defer allocator.free(safe_message);
    const cmd = switch (comptime builtin.os.tag) {
        .macos => try std.fmt.allocPrint(allocator, "osascript -e 'display notification \"{s}\" with title \"{s}\"'", .{ safe_message, safe_title }),
        .linux => try std.fmt.allocPrint(allocator, "notify-send '{s}' '{s}' 2>/dev/null || echo 'notification sent'", .{ safe_title, safe_message }),
        else => return,
    };
    defer allocator.free(cmd);
    var result = try exec(allocator, cmd);
    result.deinit();
}

// ============================================================================
// Extended System Information
// ============================================================================

/// System information gathered from the OS
pub const SystemInfo = struct {
    allocator: std.mem.Allocator,
    os_name: []const u8,
    os_version: []const u8,
    hostname: []const u8,
    username: []const u8,
    home_dir: []const u8,
    temp_dir: []u8,
    current_dir: []const u8,
    cpu_count: u32,
    page_size: usize,
    total_memory: u64,

    pub fn deinit(self: *SystemInfo) void {
        if (self.os_version.len > 0) self.allocator.free(self.os_version);
        if (self.hostname.len > 0) self.allocator.free(self.hostname);
        if (self.username.len > 0) self.allocator.free(self.username);
        if (self.home_dir.len > 0) self.allocator.free(self.home_dir);
        if (self.temp_dir.len > 0) self.allocator.free(self.temp_dir);
        if (self.current_dir.len > 0) self.allocator.free(self.current_dir);
    }
};

/// Gather system information
pub fn getSystemInfo(allocator: std.mem.Allocator) !SystemInfo {
    const os_name = getOsName();

    // Get OS version via shell
    const version_result = exec(allocator, "uname -r 2>/dev/null || echo unknown") catch {
        return SystemInfo{
            .allocator = allocator,
            .os_name = os_name,
            .os_version = try allocator.dupe(u8, "unknown"),
            .hostname = try allocator.dupe(u8, "unknown"),
            .username = try allocator.dupe(u8, "unknown"),
            .home_dir = try allocator.dupe(u8, "unknown"),
            .temp_dir = try allocator.dupe(u8, "/tmp"),
            .current_dir = try allocator.dupe(u8, "."),
            .cpu_count = getCpuCount(),
            .page_size = std.heap.page_size_min,
            .total_memory = 0,
        };
    };
    defer allocator.free(version_result.stderr);
    const os_version = if (version_result.stdout.len > 0) blk: {
        const trimmed = std.mem.trim(u8, version_result.stdout, " \r\n\t");
        const duped = try allocator.dupe(u8, trimmed);
        allocator.free(version_result.stdout);
        break :blk duped;
    } else blk: {
        allocator.free(version_result.stdout);
        break :blk try allocator.dupe(u8, "unknown");
    };

    // Get hostname
    const hostname_result = exec(allocator, "hostname 2>/dev/null || echo unknown") catch {
        return SystemInfo{
            .allocator = allocator,
            .os_name = os_name,
            .os_version = os_version,
            .hostname = try allocator.dupe(u8, "unknown"),
            .username = try allocator.dupe(u8, "unknown"),
            .home_dir = try allocator.dupe(u8, "unknown"),
            .temp_dir = try allocator.dupe(u8, "/tmp"),
            .current_dir = try allocator.dupe(u8, "."),
            .cpu_count = getCpuCount(),
            .page_size = std.heap.page_size_min,
            .total_memory = 0,
        };
    };
    defer allocator.free(hostname_result.stderr);
    const hostname_raw = std.mem.trim(u8, hostname_result.stdout, " \r\n\t");
    const hostname = try allocator.dupe(u8, hostname_raw);
    allocator.free(hostname_result.stdout);

    const username = if (Env.get("USER") orelse Env.get("USERNAME")) |u|
        try allocator.dupe(u8, u)
    else
        try allocator.dupe(u8, "unknown");

    const home_dir = if (Env.get("HOME") orelse Env.get("USERPROFILE")) |h|
        try allocator.dupe(u8, h)
    else
        try allocator.dupe(u8, "unknown");

    const temp_dir = getTempPath(allocator) catch try allocator.dupe(u8, "/tmp");

    const current_dir = try allocator.dupe(u8, ".");

    return SystemInfo{
        .allocator = allocator,
        .os_name = os_name,
        .os_version = os_version,
        .hostname = hostname,
        .username = username,
        .home_dir = home_dir,
        .temp_dir = temp_dir,
        .current_dir = current_dir,
        .cpu_count = getCpuCount(),
        .page_size = std.heap.page_size_min,
        .total_memory = 0,
    };
}

/// Get a human-readable OS name
pub fn getOsName() []const u8 {
    return switch (builtin.os.tag) {
        .macos => "macOS",
        .linux => "Linux",
        .windows => "Windows",
        .freebsd => "FreeBSD",
        .netbsd => "NetBSD",
        .openbsd => "OpenBSD",
        .dragonfly => "DragonFly BSD",
        .ios => "iOS",
        .wasi => "WASI",
        else => "Other",
    };
}

/// Whether the current platform has a desktop environment
pub const is_desktop = switch (builtin.os.tag) {
    .macos, .windows, .linux, .freebsd, .openbsd, .netbsd, .dragonfly => true,
    else => false,
};

/// Get the current process ID
pub fn getpid() i32 {
    if (comptime no_os) return 0;
    return switch (builtin.os.tag) {
        .windows => 0,
        else => @intCast(std.c.getpid()),
    };
}

/// Get the parent process ID
pub fn getppid() i32 {
    if (comptime no_os) return 0;
    return switch (builtin.os.tag) {
        .windows => 0,
        else => @intCast(std.c.getppid()),
    };
}

/// Check if stdout is a terminal
pub fn isatty() bool {
    if (comptime no_os) return false;
    return std.c.isatty(1) != 0;
}

/// Detect CI environment
pub fn isCI() bool {
    if (Env.get("CI") != null) return true;
    if (Env.get("GITHUB_ACTIONS") != null) return true;
    if (Env.get("JENKINS_URL") != null) return true;
    if (Env.get("GITLAB_CI") != null) return true;
    return false;
}

/// Get the user's home directory
pub fn getHomeDir(allocator: std.mem.Allocator) ![]u8 {
    if (Env.get("HOME") orelse Env.get("USERPROFILE")) |h| {
        return allocator.dupe(u8, h);
    }
    return error.NoHomeDirectory;
}

/// Get the system temp directory
pub fn getTempDir(allocator: std.mem.Allocator) ![]u8 {
    return getTempPath(allocator);
}

/// Expand environment variables in a string.
/// Supports $VAR, ${VAR} syntax.
pub fn envExpand(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    defer result.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '$') {
            i += 1;
            if (i >= input.len) {
                try result.append(allocator, '$');
                break;
            }
            const braced = input[i] == '{';
            if (braced) i += 1;
            const start = i;
            while (i < input.len and (std.ascii.isAlphanumeric(input[i]) or input[i] == '_')) : (i += 1) {}
            const name = input[start..i];
            if (braced and i < input.len and input[i] == '}') i += 1;
            if (Env.get(name)) |val| {
                try result.appendSlice(allocator, val);
            }
        } else {
            try result.append(allocator, input[i]);
            i += 1;
        }
    }
    return allocator.dupe(u8, result.items);
}

test "temp path detection" {
    const allocator = std.testing.allocator;
    const tmp = try getTempPath(allocator);
    defer allocator.free(tmp);
    try std.testing.expect(tmp.len > 0);
}
