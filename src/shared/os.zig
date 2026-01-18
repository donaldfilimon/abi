//! Cross-Platform OS Features Module
//!
//! Provides unified OS-level operations for UI and agent actions across all platforms:
//! - Windows, Linux, macOS, FreeBSD, NetBSD, OpenBSD, DragonFly BSD
//! - WASM/WASI (with graceful degradation)
//!
//! Features:
//! - Process management (spawn, signals, environment)
//! - System information (hostname, username, directories)
//! - File system operations (paths, permissions, watching)
//! - Clipboard operations (for UI)
//! - Shell command execution
//! - System notifications

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

// Platform-specific imports
const windows = if (builtin.os.tag == .windows) std.os.windows else struct {};
const linux = if (builtin.os.tag == .linux) std.os.linux else struct {};

// ============================================================================
// Platform Detection
// ============================================================================

/// Operating system classification
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

    /// Check if this is a POSIX-compatible OS
    pub fn isPosix(self: OsKind) bool {
        return switch (self) {
            .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly, .ios => true,
            else => false,
        };
    }

    /// Check if this is a BSD variant
    pub fn isBsd(self: OsKind) bool {
        return switch (self) {
            .freebsd, .netbsd, .openbsd, .dragonfly => true,
            else => false,
        };
    }

    /// Check if this is a Unix-like OS
    pub fn isUnixLike(self: OsKind) bool {
        return self.isPosix() or self == .wasi;
    }
};

/// Current operating system at compile time
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

/// Check if we're running on WASM
pub const is_wasm = builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;

/// Check if we're running on a desktop OS (has GUI capabilities)
pub const is_desktop = current_os == .windows or current_os == .linux or current_os == .macos;

/// Check if standard I/O is available
pub const has_stdio = !is_wasm or current_os == .wasi;

// ============================================================================
// System Information
// ============================================================================

/// System information structure
pub const SystemInfo = struct {
    hostname: []const u8,
    username: []const u8,
    home_dir: []const u8,
    temp_dir: []const u8,
    current_dir: []const u8,
    os_name: []const u8,
    os_version: []const u8,
    cpu_count: u32,
    page_size: usize,
    total_memory: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SystemInfo) void {
        self.allocator.free(self.hostname);
        self.allocator.free(self.username);
        self.allocator.free(self.home_dir);
        self.allocator.free(self.temp_dir);
        self.allocator.free(self.current_dir);
        self.allocator.free(self.os_name);
        self.allocator.free(self.os_version);
    }
};

/// Get comprehensive system information
pub fn getSystemInfo(allocator: std.mem.Allocator) !SystemInfo {
    const hostname = try getHostname(allocator);
    errdefer allocator.free(hostname);

    const username = try getUsername(allocator);
    errdefer allocator.free(username);

    const home_dir = try getHomeDir(allocator);
    errdefer allocator.free(home_dir);

    const temp_dir = try getTempDir(allocator);
    errdefer allocator.free(temp_dir);

    const current_dir = try getCurrentDir(allocator);
    errdefer allocator.free(current_dir);

    const os_name = try allocator.dupe(u8, getOsName());
    errdefer allocator.free(os_name);

    const os_version = try getOsVersion(allocator);
    errdefer allocator.free(os_version);

    return .{
        .hostname = hostname,
        .username = username,
        .home_dir = home_dir,
        .temp_dir = temp_dir,
        .current_dir = current_dir,
        .os_name = os_name,
        .os_version = os_version,
        .cpu_count = getCpuCount(),
        .page_size = getPageSize(),
        .total_memory = getTotalMemory(),
        .allocator = allocator,
    };
}

/// Get the system hostname
pub fn getHostname(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "wasm-host");
    }

    if (comptime builtin.os.tag == .windows) {
        var buffer: [256]u8 = undefined;
        var size: u32 = @intCast(buffer.len);
        if (windows.kernel32.GetComputerNameA(&buffer, &size) != 0) {
            return allocator.dupe(u8, buffer[0..size]);
        }
        return allocator.dupe(u8, "localhost");
    }

    // POSIX systems use gethostname via uname
    var uname_buf: posix.utsname = undefined;
    if (posix.uname(&uname_buf)) |_| {
        const nodename = std.mem.sliceTo(&uname_buf.nodename, 0);
        return allocator.dupe(u8, nodename);
    } else |_| {
        return allocator.dupe(u8, "localhost");
    }
}

/// Get the current username
pub fn getUsername(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "wasm-user");
    }

    if (comptime builtin.os.tag == .windows) {
        var buffer: [256]u8 = undefined;
        var size: u32 = @intCast(buffer.len);
        if (windows.kernel32.GetUserNameA(&buffer, &size) != 0) {
            return allocator.dupe(u8, buffer[0 .. size - 1]); // -1 to exclude null terminator
        }
        return allocator.dupe(u8, "user");
    }

    // POSIX: try environment variables first
    if (std.process.getenv("USER")) |user| {
        return allocator.dupe(u8, user);
    }
    if (std.process.getenv("LOGNAME")) |user| {
        return allocator.dupe(u8, user);
    }

    // Fallback to getuid-based lookup would require libc
    return allocator.dupe(u8, "user");
}

/// Get the user's home directory
pub fn getHomeDir(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "/home");
    }

    if (comptime builtin.os.tag == .windows) {
        if (std.process.getenv("USERPROFILE")) |profile| {
            return allocator.dupe(u8, profile);
        }
        if (std.process.getenv("HOMEDRIVE")) |drive| {
            if (std.process.getenv("HOMEPATH")) |path| {
                return std.fmt.allocPrint(allocator, "{s}{s}", .{ drive, path });
            }
        }
        return allocator.dupe(u8, "C:\\Users\\Default");
    }

    // POSIX
    if (std.process.getenv("HOME")) |home| {
        return allocator.dupe(u8, home);
    }
    return allocator.dupe(u8, "/home");
}

/// Get the system temp directory
pub fn getTempDir(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "/tmp");
    }

    if (comptime builtin.os.tag == .windows) {
        if (std.process.getenv("TEMP")) |temp| {
            return allocator.dupe(u8, temp);
        }
        if (std.process.getenv("TMP")) |tmp| {
            return allocator.dupe(u8, tmp);
        }
        return allocator.dupe(u8, "C:\\Windows\\Temp");
    }

    // POSIX
    if (std.process.getenv("TMPDIR")) |tmpdir| {
        return allocator.dupe(u8, tmpdir);
    }
    return allocator.dupe(u8, "/tmp");
}

/// Get the current working directory
pub fn getCurrentDir(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "/");
    }

    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.posix.getcwd(&buffer) catch {
        return allocator.dupe(u8, ".");
    };
    return allocator.dupe(u8, cwd);
}

/// Get OS name string
pub fn getOsName() []const u8 {
    return switch (current_os) {
        .windows => "Windows",
        .linux => "Linux",
        .macos => "macOS",
        .freebsd => "FreeBSD",
        .netbsd => "NetBSD",
        .openbsd => "OpenBSD",
        .dragonfly => "DragonFly BSD",
        .ios => "iOS",
        .wasi => "WASI",
        .other => "Unknown",
    };
}

/// Get OS version string
pub fn getOsVersion(allocator: std.mem.Allocator) ![]u8 {
    if (comptime is_wasm) {
        return allocator.dupe(u8, "1.0");
    }

    if (comptime current_os.isPosix()) {
        var uname_buf: posix.utsname = undefined;
        if (posix.uname(&uname_buf)) |_| {
            const release = std.mem.sliceTo(&uname_buf.release, 0);
            return allocator.dupe(u8, release);
        } else |_| {
            return allocator.dupe(u8, "unknown");
        }
    }

    if (comptime builtin.os.tag == .windows) {
        // Windows version detection would require RtlGetVersion
        return allocator.dupe(u8, "10.0");
    }

    return allocator.dupe(u8, "unknown");
}

/// Get CPU core count
pub fn getCpuCount() u32 {
    const count = std.Thread.getCpuCount() catch 1;
    return @intCast(@max(1, @min(count, std.math.maxInt(u32))));
}

/// Get system page size
pub fn getPageSize() usize {
    if (comptime is_wasm) {
        return 65536; // WASM page size
    }
    return std.mem.page_size;
}

/// Get total system memory (best effort, returns 0 if unavailable)
pub fn getTotalMemory() u64 {
    if (comptime is_wasm) {
        return 0;
    }

    if (comptime builtin.os.tag == .linux) {
        // Read from /proc/meminfo using posix API
        const fd = std.posix.open("/proc/meminfo", .{}, 0) catch return 0;
        defer std.posix.close(fd);

        var buffer: [256]u8 = undefined;
        const bytes_read = std.posix.read(fd, &buffer) catch return 0;
        const content = buffer[0..bytes_read];

        // Parse "MemTotal: NNNN kB"
        if (std.mem.indexOf(u8, content, "MemTotal:")) |start| {
            const after_colon = content[start + 9 ..];
            const trimmed = std.mem.trimLeft(u8, after_colon, " ");
            if (std.mem.indexOf(u8, trimmed, " ")) |space_idx| {
                const num_str = trimmed[0..space_idx];
                const kb = std.fmt.parseInt(u64, num_str, 10) catch return 0;
                return kb * 1024;
            }
        }
        return 0;
    }

    // Other platforms - would require platform-specific syscalls
    return 0;
}

// ============================================================================
// Environment Variables
// ============================================================================

/// Environment variable operations
pub const Env = struct {
    /// Get an environment variable
    pub fn get(name: []const u8) ?[]const u8 {
        if (comptime is_wasm) return null;
        return std.process.getenv(name);
    }

    /// Get an environment variable or return a default
    pub fn getOr(name: []const u8, default: []const u8) []const u8 {
        return get(name) orelse default;
    }

    /// Check if an environment variable exists
    pub fn exists(name: []const u8) bool {
        return get(name) != null;
    }

    /// Get environment variable as boolean
    pub fn getBool(name: []const u8) ?bool {
        const val = get(name) orelse return null;
        if (std.ascii.eqlIgnoreCase(val, "true") or
            std.ascii.eqlIgnoreCase(val, "yes") or
            std.ascii.eqlIgnoreCase(val, "1"))
        {
            return true;
        }
        if (std.ascii.eqlIgnoreCase(val, "false") or
            std.ascii.eqlIgnoreCase(val, "no") or
            std.ascii.eqlIgnoreCase(val, "0"))
        {
            return false;
        }
        return null;
    }

    /// Get environment variable as integer
    pub fn getInt(comptime T: type, name: []const u8) ?T {
        const val = get(name) orelse return null;
        return std.fmt.parseInt(T, val, 10) catch null;
    }

    /// Expand environment variables in a string (e.g., $HOME or %USERPROFILE%)
    pub fn expand(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < input.len) {
            if (comptime builtin.os.tag == .windows) {
                // Windows style: %VAR%
                if (input[i] == '%') {
                    if (std.mem.indexOfPos(u8, input, i + 1, "%")) |end| {
                        const var_name = input[i + 1 .. end];
                        if (get(var_name)) |value| {
                            try result.appendSlice(value);
                        }
                        i = end + 1;
                        continue;
                    }
                }
            }

            // Unix style: $VAR or ${VAR}
            if (input[i] == '$') {
                if (i + 1 < input.len and input[i + 1] == '{') {
                    // ${VAR} format
                    if (std.mem.indexOfPos(u8, input, i + 2, "}")) |end| {
                        const var_name = input[i + 2 .. end];
                        if (get(var_name)) |value| {
                            try result.appendSlice(value);
                        }
                        i = end + 1;
                        continue;
                    }
                } else {
                    // $VAR format
                    const start = i + 1;
                    var end = start;
                    while (end < input.len and (std.ascii.isAlphanumeric(input[end]) or input[end] == '_')) {
                        end += 1;
                    }
                    if (end > start) {
                        const var_name = input[start..end];
                        if (get(var_name)) |value| {
                            try result.appendSlice(value);
                        }
                        i = end;
                        continue;
                    }
                }
            }

            try result.append(input[i]);
            i += 1;
        }

        return result.toOwnedSlice();
    }
};

// ============================================================================
// Path Operations
// ============================================================================

/// Cross-platform path separator
pub const path_separator: u8 = if (builtin.os.tag == .windows) '\\' else '/';

/// Cross-platform path list separator (PATH variable)
pub const path_list_separator: u8 = if (builtin.os.tag == .windows) ';' else ':';

/// Path operations
pub const Path = struct {
    /// Join path components with the platform separator
    pub fn join(allocator: std.mem.Allocator, components: []const []const u8) ![]u8 {
        if (components.len == 0) return allocator.dupe(u8, "");

        var total_len: usize = 0;
        for (components) |comp| {
            total_len += comp.len + 1; // +1 for separator
        }

        var result = try allocator.alloc(u8, total_len);
        var pos: usize = 0;

        for (components, 0..) |comp, i| {
            @memcpy(result[pos .. pos + comp.len], comp);
            pos += comp.len;
            if (i < components.len - 1) {
                result[pos] = path_separator;
                pos += 1;
            }
        }

        return allocator.realloc(result, pos);
    }

    /// Get the base name of a path
    pub fn basename(path: []const u8) []const u8 {
        if (path.len == 0) return "";

        // Find last separator
        var end = path.len;
        while (end > 0 and (path[end - 1] == '/' or path[end - 1] == '\\')) {
            end -= 1;
        }

        if (end == 0) return if (path.len > 0) path[0..1] else "";

        var start = end;
        while (start > 0 and path[start - 1] != '/' and path[start - 1] != '\\') {
            start -= 1;
        }

        return path[start..end];
    }

    /// Get the directory name of a path
    pub fn dirname(path: []const u8) []const u8 {
        if (path.len == 0) return ".";

        var end = path.len;
        while (end > 0 and (path[end - 1] == '/' or path[end - 1] == '\\')) {
            end -= 1;
        }

        while (end > 0 and path[end - 1] != '/' and path[end - 1] != '\\') {
            end -= 1;
        }

        while (end > 1 and (path[end - 1] == '/' or path[end - 1] == '\\')) {
            end -= 1;
        }

        if (end == 0) {
            if (path[0] == '/' or path[0] == '\\') return path[0..1];
            return ".";
        }

        return path[0..end];
    }

    /// Get the file extension (including the dot)
    pub fn extension(path: []const u8) []const u8 {
        const base = basename(path);
        if (base.len == 0) return "";

        var i = base.len;
        while (i > 0) {
            i -= 1;
            if (base[i] == '.') {
                if (i == 0) return ""; // Hidden file, no extension
                return base[i..];
            }
        }
        return "";
    }

    /// Check if path is absolute
    pub fn isAbsolute(path: []const u8) bool {
        if (path.len == 0) return false;

        if (comptime builtin.os.tag == .windows) {
            // Windows: C:\ or \\server\share
            if (path.len >= 3 and std.ascii.isAlphabetic(path[0]) and path[1] == ':' and (path[2] == '\\' or path[2] == '/')) {
                return true;
            }
            if (path.len >= 2 and path[0] == '\\' and path[1] == '\\') {
                return true; // UNC path
            }
            return false;
        }

        // POSIX
        return path[0] == '/';
    }

    /// Normalize path separators to the platform's native separator
    pub fn normalize(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        var result = try allocator.dupe(u8, path);
        for (result) |*c| {
            if (c.* == '/' or c.* == '\\') {
                c.* = path_separator;
            }
        }
        return result;
    }
};

// ============================================================================
// Process Management
// ============================================================================

/// Process ID type
pub const Pid = if (builtin.os.tag == .windows) u32 else i32;

/// Get current process ID
pub fn getpid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        return windows.kernel32.GetCurrentProcessId();
    }

    return @intCast(linux.getpid());
}

/// Get parent process ID
pub fn getppid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        // Windows doesn't have a direct getppid, would need NtQueryInformationProcess
        return 0;
    }

    return @intCast(linux.getppid());
}

/// Signal types (cross-platform subset)
pub const Signal = enum(u8) {
    interrupt = 2, // SIGINT
    terminate = 15, // SIGTERM
    kill = 9, // SIGKILL
    hangup = 1, // SIGHUP
    user1 = 10, // SIGUSR1
    user2 = 12, // SIGUSR2

    /// Convert to platform-specific signal number
    pub fn toNative(self: Signal) u8 {
        return @intFromEnum(self);
    }
};

/// Send a signal to a process (POSIX only, no-op on Windows)
pub fn kill(pid: Pid, sig: Signal) !void {
    if (comptime is_wasm) return error.Unsupported;

    if (comptime builtin.os.tag == .windows) {
        // Windows: TerminateProcess for kill signal
        if (sig == .kill or sig == .terminate) {
            const handle = windows.kernel32.OpenProcess(
                windows.PROCESS_TERMINATE,
                windows.FALSE,
                pid,
            ) orelse return error.AccessDenied;
            defer _ = windows.kernel32.CloseHandle(handle);

            if (windows.kernel32.TerminateProcess(handle, 1) == 0) {
                return error.OperationFailed;
            }
            return;
        }
        return error.Unsupported;
    }

    // POSIX
    const result = linux.kill(pid, sig.toNative());
    if (result != 0) {
        return error.OperationFailed;
    }
}

// ============================================================================
// Shell Command Execution
// ============================================================================

/// Result of a shell command
pub const CommandResult = struct {
    stdout: []u8,
    stderr: []u8,
    exit_code: u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CommandResult) void {
        self.allocator.free(self.stdout);
        self.allocator.free(self.stderr);
    }

    pub fn success(self: *const CommandResult) bool {
        return self.exit_code == 0;
    }
};

/// Execute a shell command and capture output
pub fn exec(allocator: std.mem.Allocator, command: []const u8) !CommandResult {
    if (comptime is_wasm) {
        return .{
            .stdout = try allocator.dupe(u8, ""),
            .stderr = try allocator.dupe(u8, "command execution not supported on WASM"),
            .exit_code = 1,
            .allocator = allocator,
        };
    }

    const shell = if (comptime builtin.os.tag == .windows) "cmd.exe" else "/bin/sh";
    const shell_arg = if (comptime builtin.os.tag == .windows) "/c" else "-c";

    var child = std.process.Child.init(
        &[_][]const u8{ shell, shell_arg, command },
        allocator,
    );
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout = try child.stdout.?.reader().readAllAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(stdout);

    const stderr = try child.stderr.?.reader().readAllAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(stderr);

    const term = try child.wait();

    return .{
        .stdout = stdout,
        .stderr = stderr,
        .exit_code = switch (term) {
            .Exited => |code| code,
            else => 1,
        },
        .allocator = allocator,
    };
}

/// Execute a command with arguments (no shell)
pub fn execArgs(allocator: std.mem.Allocator, argv: []const []const u8) !CommandResult {
    if (comptime is_wasm) {
        return .{
            .stdout = try allocator.dupe(u8, ""),
            .stderr = try allocator.dupe(u8, "command execution not supported on WASM"),
            .exit_code = 1,
            .allocator = allocator,
        };
    }

    var child = std.process.Child.init(argv, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout = try child.stdout.?.reader().readAllAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(stdout);

    const stderr = try child.stderr.?.reader().readAllAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(stderr);

    const term = try child.wait();

    return .{
        .stdout = stdout,
        .stderr = stderr,
        .exit_code = switch (term) {
            .Exited => |code| code,
            else => 1,
        },
        .allocator = allocator,
    };
}

// ============================================================================
// Clipboard Operations (for UI)
// ============================================================================

/// Clipboard operations (platform-specific)
pub const Clipboard = struct {
    /// Copy text to clipboard
    pub fn copy(allocator: std.mem.Allocator, text: []const u8) !void {
        if (comptime is_wasm) return error.Unsupported;

        if (comptime builtin.os.tag == .windows) {
            // Use clip.exe for simplicity
            var child = std.process.Child.init(
                &[_][]const u8{"clip.exe"},
                allocator,
            );
            child.stdin_behavior = .Pipe;

            try child.spawn();
            try child.stdin.?.writeAll(text);
            child.stdin.?.close();
            child.stdin = null;
            _ = try child.wait();
            return;
        }

        if (comptime builtin.os.tag == .macos) {
            var child = std.process.Child.init(
                &[_][]const u8{"pbcopy"},
                allocator,
            );
            child.stdin_behavior = .Pipe;

            try child.spawn();
            try child.stdin.?.writeAll(text);
            child.stdin.?.close();
            child.stdin = null;
            _ = try child.wait();
            return;
        }

        // Linux/BSD: try xclip, xsel, or wl-copy
        const clipboard_cmds = [_][]const u8{
            "wl-copy",
            "xclip -selection clipboard",
            "xsel --clipboard --input",
        };

        for (clipboard_cmds) |cmd| {
            var child = std.process.Child.init(
                &[_][]const u8{ "/bin/sh", "-c", cmd },
                allocator,
            );
            child.stdin_behavior = .Pipe;
            child.stderr_behavior = .Ignore;

            child.spawn() catch continue;
            child.stdin.?.writeAll(text) catch continue;
            child.stdin.?.close();
            child.stdin = null;
            const result = child.wait() catch continue;
            if (result == .Exited and result.Exited == 0) return;
        }

        return error.ClipboardUnavailable;
    }

    /// Paste text from clipboard
    pub fn paste(allocator: std.mem.Allocator) ![]u8 {
        if (comptime is_wasm) return error.Unsupported;

        if (comptime builtin.os.tag == .windows) {
            var result = try exec(allocator, "powershell.exe -command Get-Clipboard");
            defer allocator.free(result.stderr);
            if (result.success()) {
                return result.stdout;
            }
            allocator.free(result.stdout);
            return error.ClipboardUnavailable;
        }

        if (comptime builtin.os.tag == .macos) {
            var result = try execArgs(allocator, &[_][]const u8{"pbpaste"});
            defer allocator.free(result.stderr);
            if (result.success()) {
                return result.stdout;
            }
            allocator.free(result.stdout);
            return error.ClipboardUnavailable;
        }

        // Linux/BSD
        const clipboard_cmds = [_][]const u8{
            "wl-paste",
            "xclip -selection clipboard -o",
            "xsel --clipboard --output",
        };

        for (clipboard_cmds) |cmd| {
            var result = exec(allocator, cmd) catch continue;
            defer allocator.free(result.stderr);
            if (result.success()) {
                return result.stdout;
            }
            allocator.free(result.stdout);
        }

        return error.ClipboardUnavailable;
    }
};

// ============================================================================
// System Notifications (for agent actions)
// ============================================================================

/// Send a system notification
pub fn notify(allocator: std.mem.Allocator, title: []const u8, message: []const u8) !void {
    if (comptime is_wasm) return;

    if (comptime builtin.os.tag == .windows) {
        // Use PowerShell toast notification
        const ps_cmd = try std.fmt.allocPrint(
            allocator,
            "powershell.exe -command \"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; $template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template); $xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('{s}')); $xml.GetElementsByTagName('text')[1].AppendChild($xml.CreateTextNode('{s}')); $toast = [Windows.UI.Notifications.ToastNotification]::new($xml); [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ABI').Show($toast)\"",
            .{ title, message },
        );
        defer allocator.free(ps_cmd);

        var result = try exec(allocator, ps_cmd);
        result.deinit();
        return;
    }

    if (comptime builtin.os.tag == .macos) {
        const cmd = try std.fmt.allocPrint(
            allocator,
            "osascript -e 'display notification \"{s}\" with title \"{s}\"'",
            .{ message, title },
        );
        defer allocator.free(cmd);

        var result = try exec(allocator, cmd);
        result.deinit();
        return;
    }

    // Linux/BSD: use notify-send
    const cmd = try std.fmt.allocPrint(
        allocator,
        "notify-send \"{s}\" \"{s}\"",
        .{ title, message },
    );
    defer allocator.free(cmd);

    var result = exec(allocator, cmd) catch return;
    result.deinit();
}

// ============================================================================
// Terminal Detection
// ============================================================================

/// Check if stdin is a TTY
pub fn isatty() bool {
    if (comptime is_wasm) return false;

    if (comptime builtin.os.tag == .windows) {
        const handle = windows.kernel32.GetStdHandle(windows.STD_INPUT_HANDLE) orelse return false;
        var mode: u32 = 0;
        return windows.kernel32.GetConsoleMode(handle, &mode) != 0;
    }

    return posix.isatty(posix.STDIN_FILENO);
}

/// Check if stdout is a TTY
pub fn isattyStdout() bool {
    if (comptime is_wasm) return false;

    if (comptime builtin.os.tag == .windows) {
        const handle = windows.kernel32.GetStdHandle(windows.STD_OUTPUT_HANDLE) orelse return false;
        var mode: u32 = 0;
        return windows.kernel32.GetConsoleMode(handle, &mode) != 0;
    }

    return posix.isatty(posix.STDOUT_FILENO);
}

/// Detect if running in a CI environment
pub fn isCI() bool {
    // Common CI environment variables
    const ci_vars = [_][]const u8{
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "BUILDKITE",
        "DRONE",
        "TEAMCITY_VERSION",
    };

    for (ci_vars) |v| {
        if (Env.exists(v)) return true;
    }
    return false;
}

// ============================================================================
// File Permissions (POSIX)
// ============================================================================

/// File permission bits
pub const FileMode = struct {
    pub const owner_read: u32 = 0o400;
    pub const owner_write: u32 = 0o200;
    pub const owner_exec: u32 = 0o100;
    pub const group_read: u32 = 0o040;
    pub const group_write: u32 = 0o020;
    pub const group_exec: u32 = 0o010;
    pub const other_read: u32 = 0o004;
    pub const other_write: u32 = 0o002;
    pub const other_exec: u32 = 0o001;

    /// Common permission combinations
    pub const rwx_owner: u32 = owner_read | owner_write | owner_exec;
    pub const rw_owner: u32 = owner_read | owner_write;
    pub const rx_all: u32 = owner_read | owner_exec | group_read | group_exec | other_read | other_exec;
    pub const rw_all: u32 = owner_read | owner_write | group_read | group_write | other_read | other_write;

    /// 0755 - rwxr-xr-x (directories, executables)
    pub const default_dir: u32 = 0o755;
    /// 0644 - rw-r--r-- (regular files)
    pub const default_file: u32 = 0o644;
    /// 0600 - rw------- (private files)
    pub const private_file: u32 = 0o600;
};

/// Set file permissions (POSIX only)
pub fn chmod(path: []const u8, mode: u32) !void {
    if (comptime is_wasm) return error.Unsupported;
    if (comptime builtin.os.tag == .windows) return; // No-op on Windows

    const path_z = try std.fs.path.joinZ(std.heap.page_allocator, &.{path});
    defer std.heap.page_allocator.free(path_z);

    const result = posix.system.chmod(path_z, @intCast(mode));
    if (result != 0) {
        return error.PermissionDenied;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "platform detection" {
    const testing = std.testing;

    try testing.expect(current_os != .other or builtin.os.tag == .other);
    try testing.expect(getOsName().len > 0);
}

test "path operations" {
    const testing = std.testing;

    try testing.expectEqualStrings("file.txt", Path.basename("/path/to/file.txt"));
    try testing.expectEqualStrings("/path/to", Path.dirname("/path/to/file.txt"));
    try testing.expectEqualStrings(".txt", Path.extension("file.txt"));
    try testing.expectEqualStrings("", Path.extension("file"));

    if (comptime builtin.os.tag == .windows) {
        try testing.expect(Path.isAbsolute("C:\\Windows\\System32"));
        try testing.expect(!Path.isAbsolute("relative\\path"));
    } else {
        try testing.expect(Path.isAbsolute("/usr/bin"));
        try testing.expect(!Path.isAbsolute("relative/path"));
    }
}

test "environment operations" {
    const testing = std.testing;

    // PATH should exist on all platforms
    if (!is_wasm) {
        try testing.expect(Env.exists("PATH"));
    }
}

test "system info" {
    const testing = std.testing;

    try testing.expect(getCpuCount() >= 1);
    try testing.expect(getPageSize() > 0);
}
