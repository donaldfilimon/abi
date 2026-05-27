const std = @import("std");
const builtin = @import("builtin");

pub const Command = enum {
    list_processes,
    get_process_info,
    get_pid,
    get_parent_pid,
    get_cpu_usage,
    get_memory_usage,
    get_system_info,
    read_file,
    write_file,
    stat_file,
    list_directory,
    get_env_var,
    get_env_vars,
    get_cwd,
    get_platform,
    get_arch,
    get_hostname,
};

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
    total_memory_mb: u64,
};

pub const OSController = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) OSController {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: *OSController, command: Command) !void {
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
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        const buf = try self.allocator.alloc(u8, @intCast(stat.size));
        errdefer self.allocator.free(buf);
        const bytes_read = try file.readPositionalAll(io, buf, 0);
        if (bytes_read < stat.size) return error.UnexpectedEOF;
        return buf;
    }

    pub fn writeFile(self: *OSController, path: []const u8, content: []const u8) !void {
        _ = self;
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.createFileAbsolute(io, path, .{});
        defer file.close(io);

        try std.Io.File.writeStreamingAll(file, io, content);
    }

    pub fn statFile(self: *OSController, path: []const u8) !FileInfo {
        const io = std.Options.debug_io;
        const stat = try std.Io.Dir.statFile(std.Io.Dir.cwd(), io, path, .{});
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
        const value = std.process.Environ.getAlloc(.{ .block = .global }, self.allocator, key) catch |err| switch (err) {
            error.EnvironmentVariableMissing => return error.EnvVarNotFound,
            else => return err,
        };
        return value;
    }

    pub fn getCwd(self: *OSController) ![]const u8 {
        const cwd = try std.process.currentPathAlloc(std.Options.debug_io, self.allocator);
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
        var buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        const hostname = try std.posix.gethostname(&buf);
        return try self.allocator.dupe(u8, hostname);
    }

    pub fn getSystemInfo(self: *OSController) !SystemInfo {
        var hostname_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        const hostname_len = blk: {
            const hostname_slice = try std.posix.gethostname(&hostname_buf);
            break :blk std.mem.indexOfScalar(u8, &hostname_buf, 0) orelse hostname_slice.len;
        };
        const hostname = try self.allocator.dupe(u8, hostname_buf[0..hostname_len]);
        errdefer self.allocator.free(hostname);

        return SystemInfo{
            .platform = self.getPlatform(),
            .arch = self.getArch(),
            .hostname = hostname,
            .cpu_count = @as(u32, @intCast(try std.Thread.getCpuCount())),
            .total_memory_mb = 0,
        };
    }

    pub fn listDirectory(self: *OSController, path: []const u8) !std.ArrayList([]const u8) {
        const io = std.Options.debug_io;
        var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true });
        defer dir.close(io);

        var entries: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (entries.items) |entry| {
                self.allocator.free(entry);
            }
            entries.deinit(self.allocator);
        }

        var iter = dir.iterate();
        while (try iter.next(io)) |entry| {
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

    _ = controller;
}

test "OSController getPid" {
    var controller = OSController.init(std.testing.allocator);

    const pid = controller.getPid();
    try std.testing.expect(pid > 0);
}

test "OSController getParentPid" {
    var controller = OSController.init(std.testing.allocator);

    const ppid = controller.getParentPid();
    try std.testing.expect(ppid > 0);
}

test "OSController getPlatform" {
    var controller = OSController.init(std.testing.allocator);

    const platform = controller.getPlatform();
    try std.testing.expect(platform != .unknown);
}

test "OSController getArch" {
    var controller = OSController.init(std.testing.allocator);

    const arch = controller.getArch();
    try std.testing.expect(arch != .unknown);
}

test "OSController getHostname" {
    var controller = OSController.init(std.testing.allocator);

    const hostname = try controller.getHostname();
    defer std.testing.allocator.free(hostname);

    try std.testing.expect(hostname.len > 0);
}

test "OSController getCwd" {
    var controller = OSController.init(std.testing.allocator);

    const cwd = try controller.getCwd();
    defer std.testing.allocator.free(cwd);

    try std.testing.expect(cwd.len > 0);
}

test "OSController writeFile and readPath" {
    var controller = OSController.init(std.testing.allocator);

    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_os_test_{d}.txt", .{controller.getPid()});
    defer std.testing.allocator.free(test_path);
    const test_content = "hello from abi os controller";

    try controller.writeFile(test_path, test_content);
    defer std.Io.Dir.deleteFileAbsolute(std.Options.debug_io, test_path) catch |err| std.log.warn("cleanup failed: {s}", .{@errorName(err)});

    const content = try controller.readPath(test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings(test_content, content);
}

test "OSController statFile" {
    var controller = OSController.init(std.testing.allocator);

    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_os_stat_test_{d}.txt", .{controller.getPid()});
    defer std.testing.allocator.free(test_path);
    const test_content = "stat test content";

    try controller.writeFile(test_path, test_content);
    defer std.Io.Dir.deleteFileAbsolute(std.Options.debug_io, test_path) catch |err| std.log.warn("cleanup failed: {s}", .{@errorName(err)});

    const info = try controller.statFile(test_path);
    defer std.testing.allocator.free(info.path);

    try std.testing.expect(info.is_file);
    try std.testing.expect(!info.is_dir);
    try std.testing.expect(info.size > 0);
}

test "OSController listDirectory" {
    var controller = OSController.init(std.testing.allocator);

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
    var controller = OSController.init(std.testing.allocator);

    const info = try controller.getSystemInfo();
    defer std.testing.allocator.free(info.hostname);

    try std.testing.expect(info.cpu_count > 0);
    try std.testing.expect(info.hostname.len > 0);
}
