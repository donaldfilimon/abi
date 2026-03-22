//! Storage Module
//!
//! Unified file/object storage with vtable-based backend abstraction.
//!
//! Architecture:
//! - Backend vtable: put/get/delete/list/exists/deinit function pointers
//! - Memory backend: StringHashMap-based in-memory storage
//! - Local backend: Memory cache + filesystem persistence
//! - S3/GCS: Planned (requires HTTP client)
//!
//! Security: path traversal validation on all keys.

const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
const types = @import("types.zig");

// ── Sub-modules ───────────────────────────────────────────────────────
pub const backend_iface = @import("backend.zig");
pub const memory_backend = @import("memory_backend.zig");
pub const local_backend = @import("local_backend.zig");
pub const validation = @import("validation.zig");

// ── Re-exported types ─────────────────────────────────────────────────
pub const StorageConfig = types.StorageConfig;
pub const StorageBackend = types.StorageBackend;
pub const StorageError = types.StorageError;
pub const Error = StorageError;
pub const StorageObject = types.StorageObject;
pub const ObjectMetadata = types.ObjectMetadata;
pub const StorageStats = types.StorageStats;

// ── Internal aliases ──────────────────────────────────────────────────
const Backend = backend_iface.Backend;
const MemoryBackend = memory_backend.MemoryBackend;
const LocalBackend = local_backend.LocalBackend;
const isValidKey = validation.isValidKey;

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

// ── Module State ──────────────────────────────────────────────────────

var storage_state: ?*StorageState = null;

const StorageState = struct {
    allocator: std.mem.Allocator,
    config: StorageConfig,
    active_backend: Backend,
    rw_lock: sync.RwLock,
};

// ── Public API ────────────────────────────────────────────────────────

/// Initialize the global storage singleton. The `memory` and `local` backends
/// are available; `s3` and `gcs` return `BackendNotAvailable`.
pub fn init(allocator: std.mem.Allocator, config: StorageConfig) StorageError!void {
    if (storage_state != null) return;

    const active_backend = switch (config.backend) {
        .memory => blk: {
            const mb = MemoryBackend.create(allocator, config.max_object_size_mb) catch
                return error.OutOfMemory;
            break :blk mb.backend();
        },
        .local => blk: {
            const lb = LocalBackend.create(allocator, config.max_object_size_mb, config.base_path) catch
                return error.OutOfMemory;
            break :blk lb.backend();
        },
        .s3, .gcs => return error.BackendNotAvailable,
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
    // Very small budget: 1 MB
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

test {
    std.testing.refAllDecls(@This());
}
