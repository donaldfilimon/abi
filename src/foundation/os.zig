const std = @import("std");
const os_config = @import("os_config.zig");

pub const OSController = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) OSController {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: *OSController, command: os_config.Command) !void {
        _ = self;
        switch (command) {
            .list_processes => std.log.info("Executing list_processes...", .{}),
            .get_cpu_usage => std.log.info("Executing get_cpu_usage...", .{}),
            .get_process_info,
            .get_memory_usage,
            .get_system_info,
            .read_file,
            .write_file,
            .stat_file,
            .list_directory,
            .get_env_var,
            .get_env_vars,
            .get_cwd,
            .get_platform,
            .get_arch,
            .get_hostname,
            .get_pid,
            .get_parent_pid,
            => std.log.info("Executing {any}...", .{command}),
        }
    }

    pub fn getPid(self: *OSController) u32 {
        _ = self;
        return @as(u32, @intCast(std.c.getpid()));
    }

    pub fn getParentPid(self: *OSController) u32 {
        _ = self;
        return @as(u32, @intCast(std.c.getppid()));
    }

    pub fn readPath(self: *OSController, path: []const u8) ![]const u8 {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();

        const stat = try file.stat();
        const buf = try self.allocator.alloc(u8, @intCast(stat.size));
        errdefer self.allocator.free(buf);

        const bytes_read = try file.readAll(buf);
        if (bytes_read < stat.size) {
            self.allocator.free(buf);
            return error.UnexpectedEOF;
        }

        return buf;
    }

    pub fn writeFile(self: *OSController, path: []const u8, content: []const u8) !void {
        _ = self;
        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        try file.writeAll(content);
    }

    pub fn statFile(self: *OSController, path: []const u8) !os_config.FileInfo {
        const stat = try std.fs.cwd().statFile(path);
        const path_copy = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_copy);

        return os_config.FileInfo{
            .size = stat.size,
            .is_dir = stat.kind == .directory,
            .is_file = stat.kind == .file,
            .path = path_copy,
        };
    }

    pub fn getEnvVar(self: *OSController, key: []const u8) ![]const u8 {
        const value = std.posix.getenv(key) orelse return error.EnvVarNotFound;
        return try self.allocator.dupe(u8, value);
    }

    pub fn getCwd(self: *OSController) ![]const u8 {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const cwd = try std.fs.getCwd().realpath(".", &buf);
        return try self.allocator.dupe(u8, cwd);
    }

    pub fn getPlatform(self: *OSController) os_config.Platform {
        _ = self;
        return switch (std.Target.current.os.tag) {
            .macos => .macos,
            .linux => .linux,
            .windows => .windows,
            .freebsd => .freebsd,
            else => .unknown,
        };
    }

    pub fn getArch(self: *OSController) os_config.Arch {
        _ = self;
        return switch (std.Target.current.cpu.arch) {
            .x86_64 => .x86_64,
            .aarch64 => .aarch64,
            .arm => .arm,
            .riscv64 => .riscv64,
            .wasm32 => .wasm32,
            else => .unknown,
        };
    }

    pub fn getHostname(self: *OSController) ![]const u8 {
        var buf: [256]u8 = undefined;
        _ = std.posix.gethostname(&buf);
        const len = std.mem.indexOfScalar(u8, &buf, 0) orelse buf.len;
        return try self.allocator.dupe(u8, buf[0..len]);
    }

    pub fn getSystemInfo(self: *OSController) !os_config.SystemInfo {
        var hostname_buf: [256]u8 = undefined;
        const hostname_len = blk: {
            _ = std.posix.gethostname(&hostname_buf);
            break :blk std.mem.indexOfScalar(u8, &hostname_buf, 0) orelse hostname_buf.len;
        };
        const hostname = try self.allocator.dupe(u8, hostname_buf[0..hostname_len]);
        errdefer self.allocator.free(hostname);

        return os_config.SystemInfo{
            .platform = self.getPlatform(),
            .arch = self.getArch(),
            .hostname = hostname,
            .cpu_count = try std.Thread.getCpuCount(),
            .total_memory_mb = 0,
        };
    }

    pub fn listDirectory(self: *OSController, path: []const u8) !std.ArrayListUnmanaged([]const u8) {
        var dir = try std.fs.openDirAbsolute(path, .{ .iterate = true });
        defer dir.close();

        var entries = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (entries.items) |entry| {
                self.allocator.free(entry);
            }
            entries.deinit(self.allocator);
        }

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
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
    const controller = OSController.init(std.testing.allocator);

    try std.testing.expect(controller.allocator.ptr != null);
}

test "OSController getPid" {
    const controller = OSController.init(std.testing.allocator);

    const pid = controller.getPid();
    try std.testing.expect(pid > 0);
}

test "OSController getParentPid" {
    const controller = OSController.init(std.testing.allocator);

    const ppid = controller.getParentPid();
    try std.testing.expect(ppid > 0);
}

test "OSController getPlatform" {
    const controller = OSController.init(std.testing.allocator);

    const platform = controller.getPlatform();
    try std.testing.expect(platform != .unknown);
}

test "OSController getArch" {
    const controller = OSController.init(std.testing.allocator);

    const arch = controller.getArch();
    try std.testing.expect(arch != .unknown);
}

test "OSController getHostname" {
    const controller = OSController.init(std.testing.allocator);

    const hostname = try controller.getHostname();
    defer std.testing.allocator.free(hostname);

    try std.testing.expect(hostname.len > 0);
}

test "OSController getCwd" {
    const controller = OSController.init(std.testing.allocator);

    const cwd = try controller.getCwd();
    defer std.testing.allocator.free(cwd);

    try std.testing.expect(cwd.len > 0);
}

test "OSController writeFile and readPath" {
    const controller = OSController.init(std.testing.allocator);

    const test_path = "/tmp/abi_os_test.txt";
    const test_content = "hello from abi os controller";

    try controller.writeFile(test_path, test_content);
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const content = try controller.readPath(test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings(test_content, content);
}

test "OSController statFile" {
    const controller = OSController.init(std.testing.allocator);

    const test_path = "/tmp/abi_os_stat_test.txt";
    const test_content = "stat test content";

    try controller.writeFile(test_path, test_content);
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const info = try controller.statFile(test_path);
    defer std.testing.allocator.free(info.path);

    try std.testing.expect(info.is_file);
    try std.testing.expect(!info.is_dir);
    try std.testing.expect(info.size > 0);
}

test "OSController listDirectory" {
    const controller = OSController.init(std.testing.allocator);

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
    const controller = OSController.init(std.testing.allocator);

    const info = try controller.getSystemInfo();
    defer std.testing.allocator.free(info.hostname);

    try std.testing.expect(info.cpu_count > 0);
    try std.testing.expect(info.hostname.len > 0);
}
