const std = @import("std");
const builtin = @import("builtin");
const env = @import("env.zig");

pub const Limits = struct {
    max_memory_mb: u32 = 1024,
    max_cpu_percent: u8 = 50,
};

pub const OsPlatform = enum {
    macos,
    linux,
    windows,
    freebsd,
    unknown,
};

pub const Arch = enum {
    x86_64,
    aarch64,
    arm,
    riscv64,
    wasm32,
    unknown,
};

pub const FileInfo = struct {
    size: u64,
    is_dir: bool,
    is_file: bool,
    path: []const u8,
};

pub const ProcessInfo = struct {
    pid: u32,
    ppid: u32,
    name: []const u8,
};

pub const SystemInfo = struct {
    platform: OsPlatform,
    arch: Arch,
    hostname: []const u8,
    cpu_count: u32,
};

pub const OSController = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) OSController {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn getPid(self: *OSController) u32 {
        _ = self;
        // Portable PID without forcing libc on Linux/Windows: Linux uses the raw
        // syscall wrapper, Windows the Win32 API, macOS/BSD libc (always linked).
        return switch (builtin.target.os.tag) {
            .windows => std.os.windows.GetCurrentProcessId(),
            .linux => @intCast(std.os.linux.getpid()),
            else => @intCast(std.c.getpid()),
        };
    }

    pub fn getParentPid(self: *OSController) u32 {
        _ = self;
        return switch (builtin.target.os.tag) {
            // Windows has no cheap getppid; report 0 (unknown). Documented gap.
            .windows => 0,
            .linux => @intCast(std.os.linux.getppid()),
            else => @intCast(std.c.getppid()),
        };
    }

    pub fn readPath(self: *OSController, path: []const u8) ![]const u8 {
        const file = try std.Io.Dir.openFileAbsolute(self.io, path, .{});
        defer file.close(self.io);

        const stat = try file.stat(self.io);
        const buf = try self.allocator.alloc(u8, @intCast(stat.size));
        errdefer self.allocator.free(buf);
        const bytes_read = try file.readPositionalAll(self.io, buf, 0);
        if (bytes_read < stat.size) return error.UnexpectedEOF;
        return buf;
    }

    pub fn writeFile(self: *OSController, path: []const u8, content: []const u8) !void {
        const file = try std.Io.Dir.createFileAbsolute(self.io, path, .{});
        defer file.close(self.io);

        try std.Io.File.writeStreamingAll(file, self.io, content);
    }

    pub fn statFile(self: *OSController, path: []const u8) !FileInfo {
        const stat = try std.Io.Dir.statFile(std.Io.Dir.cwd(), self.io, path, .{});
        const path_copy = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_copy);

        return FileInfo{
            .size = stat.size,
            .is_dir = stat.kind == .directory,
            .is_file = stat.kind == .file,
            .path = path_copy,
        };
    }

    pub fn getEnvVar(self: *OSController, key: []const u8) ![]const u8 {
        // Portable, libc-free lookup from the captured process environment;
        // returns an owned copy (caller frees) to preserve the prior contract.
        const value = env.get(key) orelse return error.EnvVarNotFound;
        return try self.allocator.dupe(u8, value);
    }

    pub fn getCwd(self: *OSController) ![]const u8 {
        const cwd = try std.process.currentPathAlloc(self.io, self.allocator);
        defer self.allocator.free(cwd);
        return try self.allocator.dupe(u8, cwd);
    }

    pub fn getPlatform(self: *OSController) OsPlatform {
        _ = self;
        return switch (builtin.target.os.tag) {
            .macos => .macos,
            .linux => .linux,
            .windows => .windows,
            .freebsd => .freebsd,
            else => .unknown,
        };
    }

    pub fn getArch(self: *OSController) Arch {
        _ = self;
        return switch (builtin.target.cpu.arch) {
            .x86_64 => .x86_64,
            .aarch64 => .aarch64,
            .arm => .arm,
            .riscv64 => .riscv64,
            .wasm32 => .wasm32,
            else => .unknown,
        };
    }

    pub fn getHostname(self: *OSController) ![]const u8 {
        // Windows has no POSIX gethostname / HOST_NAME_MAX; resolve via the
        // COMPUTERNAME env var (owned copy) and fall back to "unknown". The
        // comptime branch keeps Windows from ever instantiating HOST_NAME_MAX.
        if (builtin.target.os.tag == .windows) {
            const name = env.get("COMPUTERNAME") orelse "unknown";
            return try self.allocator.dupe(u8, name);
        }
        var buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        const hostname = try std.posix.gethostname(&buf);
        return try self.allocator.dupe(u8, hostname);
    }

    pub fn getSystemInfo(self: *OSController) !SystemInfo {
        const hostname = try self.getHostname();
        errdefer self.allocator.free(hostname);

        return SystemInfo{
            .platform = self.getPlatform(),
            .arch = self.getArch(),
            .hostname = hostname,
            .cpu_count = @as(u32, @intCast(try std.Thread.getCpuCount())),
        };
    }

    pub fn listDirectory(self: *OSController, path: []const u8) !std.ArrayListUnmanaged([]const u8) {
        var dir = try std.Io.Dir.openDirAbsolute(self.io, path, .{ .iterate = true });
        defer dir.close(self.io);

        var entries: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (entries.items) |entry| {
                self.allocator.free(entry);
            }
            entries.deinit(self.allocator);
        }

        var iter = dir.iterate();
        while (try iter.next(self.io)) |entry| {
            const name_copy = try self.allocator.dupe(u8, entry.name);
            errdefer self.allocator.free(name_copy);
            try entries.append(self.allocator, name_copy);
        }

        return entries;
    }
};

test {
    const testing = std.testing;
    testing.refAllDecls(@This());
}

test "OSController init" {
    const controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    _ = controller;
}

test "OSController getPid" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const pid = controller.getPid();
    try std.testing.expect(pid > 0);
}

test "OSController getParentPid" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const ppid = controller.getParentPid();
    try std.testing.expect(ppid > 0);
}

test "OSController getPlatform" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const platform = controller.getPlatform();
    try std.testing.expect(platform != .unknown);
}

test "OSController getArch" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const arch = controller.getArch();
    try std.testing.expect(arch != .unknown);
}

test "OSController getHostname" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const hostname = try controller.getHostname();
    defer std.testing.allocator.free(hostname);

    try std.testing.expect(hostname.len > 0);
}

test "OSController getCwd" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const cwd = try controller.getCwd();
    defer std.testing.allocator.free(cwd);

    try std.testing.expect(cwd.len > 0);
}

test "OSController writeFile and readPath" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_os_test_{d}.txt", .{controller.getPid()});
    defer std.testing.allocator.free(test_path);
    const test_content = "hello from abi os controller";

    try controller.writeFile(test_path, test_content);
    defer std.Io.Dir.deleteFileAbsolute(controller.io, test_path) catch |err| std.log.warn("cleanup failed: {s}", .{@errorName(err)});

    const content = try controller.readPath(test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings(test_content, content);
}

test "OSController statFile" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_os_stat_test_{d}.txt", .{controller.getPid()});
    defer std.testing.allocator.free(test_path);
    const test_content = "stat test content";

    try controller.writeFile(test_path, test_content);
    defer std.Io.Dir.deleteFileAbsolute(controller.io, test_path) catch |err| std.log.warn("cleanup failed: {s}", .{@errorName(err)});

    const info = try controller.statFile(test_path);
    defer std.testing.allocator.free(info.path);

    try std.testing.expect(info.is_file);
    try std.testing.expect(!info.is_dir);
    try std.testing.expect(info.size > 0);
}

test "OSController listDirectory" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    var entries = try controller.listDirectory("/tmp");
    defer {
        for (entries.items) |entry| {
            std.testing.allocator.free(entry);
        }
        entries.deinit(std.testing.allocator);
    }

    try std.testing.expect(entries.items.len > 0);
}

test "OSController getSystemInfo" {
    var controller = OSController.init(std.testing.allocator, std.Options.debug_io);

    const info = try controller.getSystemInfo();
    defer std.testing.allocator.free(info.hostname);

    try std.testing.expect(info.cpu_count > 0);
    try std.testing.expect(info.hostname.len > 0);
}
