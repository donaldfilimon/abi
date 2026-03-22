//! Local Backend
//!
//! Filesystem-backed storage with in-memory cache. Writes are dual-write
//! (memory + disk), reads hit the cache first with disk fallback.

const std = @import("std");
const types = @import("types.zig");
const backend_mod = @import("backend.zig");
const memory_backend_mod = @import("memory_backend.zig");

const StorageError = types.StorageError;
const StorageObject = types.StorageObject;
const ObjectMetadata = types.ObjectMetadata;
const StorageStats = types.StorageStats;
const Backend = backend_mod.Backend;
const MemoryBackend = memory_backend_mod.MemoryBackend;

pub const LocalBackend = struct {
    allocator: std.mem.Allocator,
    inner: *MemoryBackend,
    base_path: []const u8, // owned
    io_backend: std.Io.Threaded,

    pub fn create(allocator: std.mem.Allocator, max_mb: u32, base_path: []const u8) !*LocalBackend {
        const lb = try allocator.create(LocalBackend);
        errdefer allocator.destroy(lb);

        const inner = try MemoryBackend.create(allocator, max_mb);
        errdefer inner.destroy();

        const owned_path = try allocator.dupe(u8, base_path);
        errdefer allocator.free(owned_path);

        lb.* = .{
            .allocator = allocator,
            .inner = inner,
            .base_path = owned_path,
            .io_backend = std.Io.Threaded.init(allocator, .{}),
        };

        // Ensure the base directory exists on disk.
        const io = lb.io_backend.io();
        const cwd = std.Io.Dir.cwd();
        _ = cwd.createDirPathOpen(io, owned_path, .{}) catch |err| {
            std.log.warn("Failed to create storage base directory '{s}': {t}", .{ owned_path, err });
        };

        return lb;
    }

    pub fn destroy(self: *LocalBackend) void {
        self.inner.destroy();
        self.io_backend.deinit();
        self.allocator.free(self.base_path);
        self.allocator.destroy(self);
    }

    /// Open the base_path as a Dir handle (relative to cwd).
    fn openBaseDir(self: *LocalBackend) ?std.Io.Dir {
        const io = self.io_backend.io();
        const cwd = std.Io.Dir.cwd();
        return cwd.openDir(io, self.base_path, .{}) catch null;
    }

    /// Persist data to disk at `{base_path}/{key}`.  Creates parent
    /// directories as needed via atomic file creation.
    fn persistToDisk(self: *LocalBackend, key: []const u8, data: []const u8) void {
        const io = self.io_backend.io();
        const base_dir = self.openBaseDir() orelse return;
        defer base_dir.close(io);

        // Use createFileAtomic with make_path to handle subdirectories.
        var atomic_file = base_dir.createFileAtomic(io, key, .{
            .make_path = true,
            .replace = true,
        }) catch return;
        defer atomic_file.deinit(io);

        atomic_file.file.writeStreamingAll(io, data) catch return;
        atomic_file.replace(io) catch return;
    }

    /// Read data from disk at `{base_path}/{key}`.
    fn readFromDisk(self: *LocalBackend, allocator: std.mem.Allocator, key: []const u8) ?[]u8 {
        const io = self.io_backend.io();
        const base_dir = self.openBaseDir() orelse return null;
        defer base_dir.close(io);

        return base_dir.readFileAlloc(io, key, allocator, .limited(64 * 1024 * 1024)) catch null;
    }

    /// Remove a file from disk at `{base_path}/{key}`.  Silently ignores
    /// missing files.
    fn removeFromDisk(self: *LocalBackend, key: []const u8) void {
        const io = self.io_backend.io();
        const base_dir = self.openBaseDir() orelse return;
        defer base_dir.close(io);

        base_dir.deleteFile(io, key) catch {};
    }

    /// Check whether a file exists on disk at `{base_path}/{key}`.
    fn existsOnDisk(self: *LocalBackend, key: []const u8) bool {
        const io = self.io_backend.io();
        const base_dir = self.openBaseDir() orelse return false;
        defer base_dir.close(io);

        var file = base_dir.openFile(io, key, .{}) catch return false;
        file.close(io);
        return true;
    }

    fn putImpl(ctx: *anyopaque, key: []const u8, data: []const u8, meta: ?ObjectMetadata) StorageError!void {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        // Write to in-memory cache first.
        try MemoryBackend.putImpl(@ptrCast(self.inner), key, data, meta);
        // Best-effort persist to disk.
        self.persistToDisk(key, data);
    }

    fn getImpl(ctx: *anyopaque, allocator: std.mem.Allocator, key: []const u8) StorageError![]u8 {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        // Fast path: serve from memory cache.
        return MemoryBackend.getImpl(@ptrCast(self.inner), allocator, key) catch |err| switch (err) {
            error.ObjectNotFound => {
                // Cold-start recovery: try reading from disk.
                const disk_data = self.readFromDisk(allocator, key) orelse
                    return error.ObjectNotFound;
                // Re-populate memory cache (best-effort).
                MemoryBackend.putImpl(@ptrCast(self.inner), key, disk_data, null) catch |cache_err| {
                    std.log.debug("Cache warm-up skipped for '{s}': {t}", .{ key, cache_err });
                };
                return disk_data;
            },
            else => return err,
        };
    }

    fn deleteImpl(ctx: *anyopaque, key: []const u8) StorageError!bool {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        const was_in_memory = try MemoryBackend.deleteImpl(@ptrCast(self.inner), key);
        // Best-effort remove from disk.
        self.removeFromDisk(key);
        return was_in_memory;
    }

    fn listImpl(ctx: *anyopaque, allocator: std.mem.Allocator, prefix: []const u8) StorageError![]StorageObject {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        return MemoryBackend.listImpl(@ptrCast(self.inner), allocator, prefix);
    }

    fn existsImpl(ctx: *anyopaque, key: []const u8) bool {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        // Check memory first, then fall back to disk.
        if (MemoryBackend.existsImpl(@ptrCast(self.inner), key)) return true;
        return self.existsOnDisk(key);
    }

    fn statsImpl(ctx: *anyopaque) StorageStats {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        var s = MemoryBackend.statsImpl(@ptrCast(self.inner));
        s.backend = .local;
        return s;
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *LocalBackend = @ptrCast(@alignCast(ctx));
        self.destroy();
    }

    pub const vtable = Backend.VTable{
        .put = putImpl,
        .get = getImpl,
        .delete = deleteImpl,
        .list = listImpl,
        .exists = existsImpl,
        .getStats = statsImpl,
        .deinitFn = deinitImpl,
    };

    pub fn backend(self: *LocalBackend) Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }
};
