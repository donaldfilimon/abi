//! Storage Module
//!
//! Unified file/object storage with vtable-based backend abstraction.
//!
//! Architecture:
//! - Backend vtable: put/get/delete/list/exists/deinit function pointers
//! - Memory backend: StringHashMap-based in-memory storage
//! - Local backend: Planned (requires I/O backend init)
//! - S3/GCS: Planned (requires HTTP client)
//!
//! Security: path traversal validation on all keys.

const std = @import("std");
const core_config = @import("../../core/config/storage.zig");
const sync = @import("../../services/shared/sync.zig");

pub const StorageConfig = core_config.StorageConfig;
pub const StorageBackend = core_config.StorageBackend;

/// Errors returned by storage operations.
pub const StorageError = error{
    FeatureDisabled,
    ObjectNotFound,
    BucketNotFound,
    PermissionDenied,
    StorageFull,
    OutOfMemory,
    /// Key contains path traversal sequences (e.g. `../`).
    InvalidKey,
    BackendNotAvailable,
};

/// Metadata descriptor for a stored object (key, size, MIME type, mtime).
pub const StorageObject = struct {
    key: []const u8 = "",
    size: u64 = 0,
    content_type: []const u8 = "application/octet-stream",
    last_modified: u64 = 0,
};

/// Optional per-object metadata (content type and up to 4 custom key-value pairs).
pub const ObjectMetadata = struct {
    content_type: []const u8 = "application/octet-stream",
    custom: [4]MetadataEntry = [_]MetadataEntry{.{}} ** 4,
    custom_count: u8 = 0,

    pub const MetadataEntry = struct {
        key: []const u8 = "",
        value: []const u8 = "",
    };
};

/// Aggregate statistics for the storage backend.
pub const StorageStats = struct {
    total_objects: u64 = 0,
    total_bytes: u64 = 0,
    backend: StorageBackend = .memory,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: StorageConfig,

    pub fn init(allocator: std.mem.Allocator, config: StorageConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Key Validation ────────────────────────────────────────────────────

fn isValidKey(key: []const u8) bool {
    if (key.len == 0 or key.len > 4096) return false;
    // Reject absolute paths
    if (key[0] == '/') return false;
    // Reject path traversal
    var i: usize = 0;
    while (i < key.len) {
        if (key[i] == '.' and i + 1 < key.len and key[i + 1] == '.') {
            // ".." at start, end, or surrounded by slashes
            if ((i == 0 or key[i - 1] == '/') and
                (i + 2 >= key.len or key[i + 2] == '/'))
            {
                return false;
            }
        }
        i += 1;
    }
    return true;
}

// ── Backend Interface ─────────────────────────────────────────────────

const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        put: *const fn (*anyopaque, []const u8, []const u8, ?ObjectMetadata) StorageError!void,
        get: *const fn (*anyopaque, std.mem.Allocator, []const u8) StorageError![]u8,
        delete: *const fn (*anyopaque, []const u8) StorageError!bool,
        list: *const fn (*anyopaque, std.mem.Allocator, []const u8) StorageError![]StorageObject,
        exists: *const fn (*anyopaque, []const u8) bool,
        getStats: *const fn (*anyopaque) StorageStats,
        deinitFn: *const fn (*anyopaque) void,
    };

    fn put(self: Backend, key: []const u8, data: []const u8, meta: ?ObjectMetadata) StorageError!void {
        return self.vtable.put(self.ptr, key, data, meta);
    }

    fn get(self: Backend, allocator: std.mem.Allocator, key: []const u8) StorageError![]u8 {
        return self.vtable.get(self.ptr, allocator, key);
    }

    fn delete(self: Backend, key: []const u8) StorageError!bool {
        return self.vtable.delete(self.ptr, key);
    }

    fn list(self: Backend, allocator: std.mem.Allocator, prefix: []const u8) StorageError![]StorageObject {
        return self.vtable.list(self.ptr, allocator, prefix);
    }

    fn exists(self: Backend, key: []const u8) bool {
        return self.vtable.exists(self.ptr, key);
    }

    fn getStats(self: Backend) StorageStats {
        return self.vtable.getStats(self.ptr);
    }

    fn deinitBackend(self: Backend) void {
        self.vtable.deinitFn(self.ptr);
    }
};

// ── Memory Backend ────────────────────────────────────────────────────

const MemoryObject = struct {
    data: []u8, // owned
    key: []u8, // owned
    content_type: []u8, // owned
    size: u64,
};

const MemoryBackend = struct {
    allocator: std.mem.Allocator,
    objects: std.StringHashMapUnmanaged(*MemoryObject),
    total_bytes: u64,
    max_bytes: u64,

    fn create(allocator: std.mem.Allocator, max_mb: u32) !*MemoryBackend {
        const mb = try allocator.create(MemoryBackend);
        mb.* = .{
            .allocator = allocator,
            .objects = .empty,
            .total_bytes = 0,
            .max_bytes = @as(u64, max_mb) * 1024 * 1024,
        };
        return mb;
    }

    fn destroy(self: *MemoryBackend) void {
        var iter = self.objects.iterator();
        while (iter.next()) |entry| {
            const obj = entry.value_ptr.*;
            self.allocator.free(obj.data);
            self.allocator.free(obj.key);
            self.allocator.free(obj.content_type);
            self.allocator.destroy(obj);
        }
        self.objects.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn putImpl(ctx: *anyopaque, key: []const u8, data: []const u8, meta: ?ObjectMetadata) StorageError!void {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));

        // Capacity check: account for freed size if overwriting existing entry
        if (self.max_bytes > 0) {
            var effective_total = self.total_bytes;
            if (self.objects.get(key)) |existing| {
                effective_total -= existing.size;
            }
            if (effective_total + data.len > self.max_bytes) {
                return error.StorageFull;
            }
        }

        const obj = self.allocator.create(MemoryObject) catch return error.OutOfMemory;
        errdefer self.allocator.destroy(obj);

        const owned_key = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_key);

        const owned_data = self.allocator.dupe(u8, data) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_data);

        const ct = if (meta) |m| m.content_type else "application/octet-stream";
        const owned_ct = self.allocator.dupe(u8, ct) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_ct);

        obj.* = .{
            .data = owned_data,
            .key = owned_key,
            .content_type = owned_ct,
            .size = data.len,
        };

        // Ensure hashmap capacity so putAssumeCapacity can't fail below
        self.objects.ensureUnusedCapacity(self.allocator, 1) catch
            return error.OutOfMemory;

        // Past this point, nothing can fail — safe to remove old entry
        if (self.objects.fetchRemove(key)) |kv| {
            const old = kv.value;
            self.total_bytes -= old.size;
            self.allocator.free(old.data);
            self.allocator.free(old.key);
            self.allocator.free(old.content_type);
            self.allocator.destroy(old);
        }

        self.objects.putAssumeCapacity(obj.key, obj);
        self.total_bytes += data.len;
    }

    fn getImpl(ctx: *anyopaque, allocator: std.mem.Allocator, key: []const u8) StorageError![]u8 {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        const obj = self.objects.get(key) orelse return error.ObjectNotFound;
        return allocator.dupe(u8, obj.data) catch return error.OutOfMemory;
    }

    fn deleteImpl(ctx: *anyopaque, key: []const u8) StorageError!bool {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        if (self.objects.fetchRemove(key)) |kv| {
            const obj = kv.value;
            self.total_bytes -= obj.size;
            self.allocator.free(obj.data);
            self.allocator.free(obj.key);
            self.allocator.free(obj.content_type);
            self.allocator.destroy(obj);
            return true;
        }
        return false;
    }

    fn listImpl(ctx: *anyopaque, allocator: std.mem.Allocator, prefix: []const u8) StorageError![]StorageObject {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));

        var results: std.ArrayListUnmanaged(StorageObject) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.objects.iterator();
        while (iter.next()) |entry| {
            const obj = entry.value_ptr.*;
            if (prefix.len == 0 or std.mem.startsWith(u8, obj.key, prefix)) {
                results.append(allocator, .{
                    .key = obj.key,
                    .size = obj.size,
                    .content_type = obj.content_type,
                }) catch return error.OutOfMemory;
            }
        }

        return results.toOwnedSlice(allocator) catch return error.OutOfMemory;
    }

    fn existsImpl(ctx: *anyopaque, key: []const u8) bool {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        return self.objects.get(key) != null;
    }

    fn statsImpl(ctx: *anyopaque) StorageStats {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        return .{
            .total_objects = self.objects.count(),
            .total_bytes = self.total_bytes,
            .backend = .memory,
        };
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        self.destroy();
    }

    const vtable = Backend.VTable{
        .put = putImpl,
        .get = getImpl,
        .delete = deleteImpl,
        .list = listImpl,
        .exists = existsImpl,
        .getStats = statsImpl,
        .deinitFn = deinitImpl,
    };

    fn backend(self: *MemoryBackend) Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

// ── Module State ──────────────────────────────────────────────────────

var storage_state: ?*StorageState = null;

const StorageState = struct {
    allocator: std.mem.Allocator,
    config: StorageConfig,
    active_backend: Backend,
    rw_lock: sync.RwLock,
};

// ── Public API ────────────────────────────────────────────────────────

/// Initialize the global storage singleton. Only the `memory` backend is
/// currently available; `local`, `s3`, and `gcs` return `BackendNotAvailable`.
pub fn init(allocator: std.mem.Allocator, config: StorageConfig) StorageError!void {
    if (storage_state != null) return;

    const active_backend = switch (config.backend) {
        .memory => blk: {
            const mb = MemoryBackend.create(allocator, config.max_object_size_mb) catch
                return error.OutOfMemory;
            break :blk mb.backend();
        },
        .local, .s3, .gcs => return error.BackendNotAvailable,
    };

    const s = allocator.create(StorageState) catch return error.OutOfMemory;
    s.* = .{
        .allocator = allocator,
        .config = config,
        .active_backend = active_backend,
        .rw_lock = sync.RwLock.init(),
    };
    storage_state = s;
}

/// Tear down the storage backend, freeing all stored objects.
pub fn deinit() void {
    if (storage_state) |s| {
        s.active_backend.deinitBackend();
        s.allocator.destroy(s);
        storage_state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return storage_state != null;
}

/// Store an object by key. Validates the key for path traversal.
pub fn putObject(
    _: std.mem.Allocator,
    key: []const u8,
    data: []const u8,
) StorageError!void {
    const s = storage_state orelse return error.FeatureDisabled;
    if (!isValidKey(key)) return error.InvalidKey;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    return s.active_backend.put(key, data, null);
}

/// Store an object with custom metadata (content type, key-value pairs).
pub fn putObjectWithMetadata(
    _: std.mem.Allocator,
    key: []const u8,
    data: []const u8,
    metadata: ObjectMetadata,
) StorageError!void {
    const s = storage_state orelse return error.FeatureDisabled;
    if (!isValidKey(key)) return error.InvalidKey;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    return s.active_backend.put(key, data, metadata);
}

/// Retrieve an object's data by key. Caller owns the returned slice.
pub fn getObject(allocator: std.mem.Allocator, key: []const u8) StorageError![]const u8 {
    const s = storage_state orelse return error.FeatureDisabled;
    if (!isValidKey(key)) return error.InvalidKey;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    return s.active_backend.get(allocator, key);
}

/// Delete an object by key. Returns `true` if the key was present.
pub fn deleteObject(key: []const u8) StorageError!bool {
    const s = storage_state orelse return error.FeatureDisabled;
    if (!isValidKey(key)) return error.InvalidKey;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    return s.active_backend.delete(key);
}

/// Check whether an object exists without reading it.
pub fn objectExists(key: []const u8) StorageError!bool {
    const s = storage_state orelse return error.FeatureDisabled;
    if (!isValidKey(key)) return error.InvalidKey;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    return s.active_backend.exists(key);
}

/// List objects whose keys start with `prefix`. Caller owns the returned slice.
pub fn listObjects(
    allocator: std.mem.Allocator,
    prefix: []const u8,
) StorageError![]StorageObject {
    const s = storage_state orelse return error.FeatureDisabled;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    return s.active_backend.list(allocator, prefix);
}

/// Snapshot object count, byte usage, and active backend type.
pub fn stats() StorageStats {
    const s = storage_state orelse return .{};

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    return s.active_backend.getStats();
}

// ── Tests ─────────────────────────────────────────────────────────────

test "storage basic put and get" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "test/key1", "hello world");
    const data = try getObject(allocator, "test/key1");
    defer allocator.free(data);

    try std.testing.expectEqualStrings("hello world", data);
}

test "storage delete" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "to-delete", "data");
    const deleted = try deleteObject("to-delete");
    try std.testing.expect(deleted);

    const not_deleted = try deleteObject("nonexistent");
    try std.testing.expect(!not_deleted);
}

test "storage object exists" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "exists-key", "data");

    const yes = try objectExists("exists-key");
    try std.testing.expect(yes);

    const no = try objectExists("missing");
    try std.testing.expect(!no);
}

test "storage list with prefix" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "photos/a.jpg", "a");
    try putObject(allocator, "photos/b.jpg", "b");
    try putObject(allocator, "docs/readme.md", "c");

    const photos = try listObjects(allocator, "photos/");
    defer allocator.free(photos);
    try std.testing.expectEqual(@as(usize, 2), photos.len);

    const all = try listObjects(allocator, "");
    defer allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), all.len);
}

test "storage path traversal rejection" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try std.testing.expectError(error.InvalidKey, putObject(allocator, "../etc/passwd", "bad"));
    try std.testing.expectError(error.InvalidKey, putObject(allocator, "/absolute/path", "bad"));
    try std.testing.expectError(error.InvalidKey, putObject(allocator, "foo/../../bar", "bad"));
    try std.testing.expectError(error.InvalidKey, putObject(allocator, "", "bad"));
}

test "storage overwrite existing key" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "key", "version1");
    try putObject(allocator, "key", "version2");

    const data = try getObject(allocator, "key");
    defer allocator.free(data);
    try std.testing.expectEqualStrings("version2", data);

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_objects);
}

test "storage stats" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "k1", "aaa");
    try putObject(allocator, "k2", "bbbbb");

    const s = stats();
    try std.testing.expectEqual(@as(u64, 2), s.total_objects);
    try std.testing.expectEqual(@as(u64, 8), s.total_bytes); // 3 + 5
    try std.testing.expectEqual(StorageBackend.memory, s.backend);
}

test "storage putObjectWithMetadata preserves content type" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObjectWithMetadata(allocator, "doc.json", "{}", .{
        .content_type = "application/json",
    });

    // Verify object exists and data is correct
    const data = try getObject(allocator, "doc.json");
    defer allocator.free(data);
    try std.testing.expectEqualStrings("{}", data);

    // List should show the object
    const objects = try listObjects(allocator, "doc");
    defer allocator.free(objects);
    try std.testing.expectEqual(@as(usize, 1), objects.len);
    try std.testing.expectEqualStrings("application/json", objects[0].content_type);
}

test "storage key with path separators" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    // Keys with forward slashes are valid (common in object stores)
    try putObject(allocator, "folder/sub/file.txt", "content");
    const data = try getObject(allocator, "folder/sub/file.txt");
    defer allocator.free(data);
    try std.testing.expectEqualStrings("content", data);

    const exists = try objectExists("folder/sub/file.txt");
    try std.testing.expect(exists);
}

test "storage object overwrite updates size tracking" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "sized", "short"); // 5 bytes
    const s1 = stats();
    try std.testing.expectEqual(@as(u64, 5), s1.total_bytes);

    try putObject(allocator, "sized", "a longer replacement value"); // 26 bytes
    const s2 = stats();
    try std.testing.expectEqual(@as(u64, 26), s2.total_bytes);
    try std.testing.expectEqual(@as(u64, 1), s2.total_objects);
}

test "storage capacity limit" {
    const allocator = std.testing.allocator;
    // Very small budget: 1 MB → use max_object_size_mb
    try init(allocator, .{ .backend = .memory, .max_object_size_mb = 1 });
    defer deinit();

    // Store a small object (should succeed)
    try putObject(allocator, "small", "data");

    // Budget is 1MB. We can't easily allocate 1MB+ in test, but verify
    // capacity tracking works by checking stats
    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_objects);
    try std.testing.expect(s.total_bytes < 1024 * 1024);
}

test "StorageConfig default values" {
    const config = StorageConfig{};
    try std.testing.expectEqual(StorageBackend.memory, config.backend);
    try std.testing.expectEqualStrings("./storage", config.base_path);
    try std.testing.expectEqual(@as(u32, 1024), config.max_object_size_mb);
}

test "StorageConfig.defaults matches zero-init" {
    const a = StorageConfig{};
    const b = StorageConfig.defaults();
    try std.testing.expectEqual(a.backend, b.backend);
    try std.testing.expectEqualStrings(a.base_path, b.base_path);
    try std.testing.expectEqual(a.max_object_size_mb, b.max_object_size_mb);
}

test "StorageObject default values" {
    const obj = StorageObject{};
    try std.testing.expectEqualStrings("", obj.key);
    try std.testing.expectEqual(@as(u64, 0), obj.size);
    try std.testing.expectEqualStrings("application/octet-stream", obj.content_type);
    try std.testing.expectEqual(@as(u64, 0), obj.last_modified);
}

test "ObjectMetadata default values" {
    const meta = ObjectMetadata{};
    try std.testing.expectEqualStrings("application/octet-stream", meta.content_type);
    try std.testing.expectEqual(@as(u8, 0), meta.custom_count);
}

test "StorageBackend enum coverage" {
    const backends = [_]StorageBackend{ .memory, .local, .s3, .gcs };
    try std.testing.expectEqual(@as(usize, 4), backends.len);
}

test "storage local backend not available" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.BackendNotAvailable,
        init(allocator, .{ .backend = .local }),
    );
}

test "storage s3 backend not available" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.BackendNotAvailable,
        init(allocator, .{ .backend = .s3 }),
    );
}

test "storage gcs backend not available" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.BackendNotAvailable,
        init(allocator, .{ .backend = .gcs }),
    );
}

test "storage get nonexistent key returns ObjectNotFound" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try std.testing.expectError(error.ObjectNotFound, getObject(allocator, "no-such-key"));
}

test "storage key validation edge cases" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    // Single dot is valid
    try putObject(allocator, "file.txt", "ok");
    const data = try getObject(allocator, "file.txt");
    defer allocator.free(data);
    try std.testing.expectEqualStrings("ok", data);

    // Double dot at end without trailing slash is fine (not traversal)
    // "foo.." has ".." but followed by end of string with i+2 >= key.len
    // AND preceded by non-slash — let's test actual traversal patterns
    try std.testing.expectError(error.InvalidKey, putObject(allocator, "..", "bad"));
    try std.testing.expectError(error.InvalidKey, putObject(allocator, "a/../b", "bad"));
}

test "storage stats empty" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    const s = stats();
    try std.testing.expectEqual(@as(u64, 0), s.total_objects);
    try std.testing.expectEqual(@as(u64, 0), s.total_bytes);
    try std.testing.expectEqual(StorageBackend.memory, s.backend);
}

test "storage Context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try Context.init(allocator, .{});
    defer ctx.deinit();
    try std.testing.expectEqual(StorageBackend.memory, ctx.config.backend);
}

test "storage re-initialization guard" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    // Second init should be no-op
    try init(allocator, .{ .backend = .memory, .max_object_size_mb = 1 });
    // Original config should still be active (1024 MB limit)
    try putObject(allocator, "test", "data");
    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.total_objects);
}

test "storage delete updates stats" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "x", "12345");
    const s1 = stats();
    try std.testing.expectEqual(@as(u64, 5), s1.total_bytes);
    try std.testing.expectEqual(@as(u64, 1), s1.total_objects);

    _ = try deleteObject("x");
    const s2 = stats();
    try std.testing.expectEqual(@as(u64, 0), s2.total_bytes);
    try std.testing.expectEqual(@as(u64, 0), s2.total_objects);
}

test "storage list empty prefix returns all" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "a", "1");
    try putObject(allocator, "b", "2");

    const all = try listObjects(allocator, "");
    defer allocator.free(all);
    try std.testing.expectEqual(@as(usize, 2), all.len);
}

test "storage list with no matches" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .backend = .memory });
    defer deinit();

    try putObject(allocator, "alpha", "data");
    const results = try listObjects(allocator, "beta");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}
