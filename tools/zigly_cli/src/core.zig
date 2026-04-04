const std = @import("std");
const builtin = @import("builtin");
const download = @import("download.zig");
const archive = @import("archive.zig");

pub const Config = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    environ_map: *std.process.Environ.Map,
    home_dir: []const u8,
    zigly_dir: []const u8,
    project_version: ?[]const u8,

    pub fn deinit(self: *Config) void {
        if (self.project_version) |pv| {
            self.allocator.free(pv);
        }
        self.allocator.free(self.zigly_dir);
    }
};

pub fn initConfig(allocator: std.mem.Allocator, io: std.Io, environ_map: *std.process.Environ.Map) !Config {
    const home = environ_map.get("HOME") orelse return error.NoHomeDir;
    const zigly_dir = try std.fs.path.join(allocator, &[_][]const u8{ home, ".zigly" });

    // Create base directories
    try std.Io.Dir.cwd().createDirPath(io, zigly_dir);
    const tmp_dir = try std.fs.path.join(allocator, &[_][]const u8{ zigly_dir, "tmp" });
    defer allocator.free(tmp_dir);
    try std.Io.Dir.cwd().createDirPath(io, tmp_dir);

    const versions_dir = try std.fs.path.join(allocator, &[_][]const u8{ zigly_dir, "versions" });
    defer allocator.free(versions_dir);
    try std.Io.Dir.cwd().createDirPath(io, versions_dir);

    var project_version: ?[]const u8 = null;
    const file = std.Io.Dir.cwd().openFile(io, ".zigversion", .{}) catch null;
    if (file) |f| {
        defer f.close(io);
        if (f.stat(io)) |stat| {
            if (allocator.alloc(u8, stat.size)) |content| {
                if (f.readPositionalAll(io, content, 0)) |bytes_read| {
                    const trimmed = std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
                    if (trimmed.len == 0) {
                        allocator.free(content);
                    } else {
                        project_version = allocator.dupe(u8, trimmed) catch null;
                        allocator.free(content);
                    }
                } else |_| {
                    allocator.free(content);
                }
            } else |_| {}
        } else |_| {}
    }

    return Config{
        .allocator = allocator,
        .io = io,
        .environ_map = environ_map,
        .home_dir = home,
        .zigly_dir = zigly_dir,
        .project_version = project_version,
    };
}
