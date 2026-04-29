//! Integration Tests: Storage Module
//!
//! Tests the unified file/object storage module exports, backend types,
//! config defaults, context lifecycle, and basic put/get/delete contracts.

const std = @import("std");
const abi = @import("abi");

const storage = abi.storage;

// ── Type Export Tests ──────────────────────────────────────────────────

test "storage: module exports expected types" {
    // Verify all public types are accessible through abi.storage
    const _config = storage.StorageConfig{};
    _ = _config;

    const _err: storage.StorageError = error.ObjectNotFound;
    _ = _err;

    const _obj = storage.StorageObject{};
    try std.testing.expectEqualStrings("application/octet-stream", _obj.content_type);

    const _meta = storage.ObjectMetadata{};
    try std.testing.expectEqualStrings("application/octet-stream", _meta.content_type);
    try std.testing.expectEqual(@as(u8, 0), _meta.custom_count);

    const _stats = storage.StorageStats{};
    try std.testing.expectEqual(@as(u64, 0), _stats.total_objects);
    try std.testing.expectEqual(@as(u64, 0), _stats.total_bytes);
}

test "storage: backend enum variants" {
    const backend_memory = storage.StorageBackend.memory;
    const backend_local = storage.StorageBackend.local;
    const backend_s3 = storage.StorageBackend.s3;
    const backend_gcs = storage.StorageBackend.gcs;

    try std.testing.expect(backend_memory != backend_local);
    try std.testing.expect(backend_s3 != backend_gcs);
}

test "storage: config default values" {
    const config = storage.StorageConfig{};
    try std.testing.expectEqual(storage.StorageBackend.memory, config.backend);
    try std.testing.expectEqualStrings("./storage", config.base_path);
    try std.testing.expectEqual(@as(u32, 1024), config.max_object_size_mb);
}

// ── Context Lifecycle Tests ────────────────────────────────────────────

test "storage: context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try storage.Context.init(allocator, .{});
    defer ctx.deinit();

    try std.testing.expectEqual(storage.StorageBackend.memory, ctx.config.backend);
}

test "storage: context with custom config" {
    const allocator = std.testing.allocator;
    const ctx = try storage.Context.init(allocator, .{
        .backend = .local,
        .base_path = "/tmp/test-storage",
        .max_object_size_mb = 64,
    });
    defer ctx.deinit();

    try std.testing.expectEqual(storage.StorageBackend.local, ctx.config.backend);
    try std.testing.expectEqualStrings("/tmp/test-storage", ctx.config.base_path);
    try std.testing.expectEqual(@as(u32, 64), ctx.config.max_object_size_mb);
}

// ── Module API Tests ───────────────────────────────────────────────────

test "storage: isEnabled returns true" {
    try std.testing.expect(storage.isEnabled());
}

test "storage: init deinit lifecycle" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try std.testing.expect(storage.isInitialized());
}

test "storage: put and get round-trip" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObject(allocator, "integration/test-key", "test-value");
    const data = try storage.getObject(allocator, "integration/test-key");
    defer allocator.free(data);

    try std.testing.expectEqualStrings("test-value", data);
}

test "storage: delete returns correct boolean" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObject(allocator, "to-remove", "data");
    const deleted = try storage.deleteObject("to-remove");
    try std.testing.expect(deleted);

    const not_deleted = try storage.deleteObject("nonexistent-key");
    try std.testing.expect(!not_deleted);
}

test "storage: objectExists contract" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObject(allocator, "present", "value");

    try std.testing.expect(try storage.objectExists("present"));
    try std.testing.expect(!try storage.objectExists("absent"));
}

test "storage: stats reflect stored objects" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObject(allocator, "s1", "aaa");
    try storage.putObject(allocator, "s2", "bbbbb");

    const s = storage.stats();
    try std.testing.expectEqual(@as(u64, 2), s.total_objects);
    try std.testing.expectEqual(@as(u64, 8), s.total_bytes);
    try std.testing.expectEqual(storage.StorageBackend.memory, s.backend);
}

test "storage: path traversal rejection" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try std.testing.expectError(error.InvalidKey, storage.putObject(allocator, "../etc/passwd", "bad"));
    try std.testing.expectError(error.InvalidKey, storage.putObject(allocator, "/absolute", "bad"));
    try std.testing.expectError(error.InvalidKey, storage.putObject(allocator, "", "bad"));
}

test "storage: putObjectWithMetadata preserves content type" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObjectWithMetadata(allocator, "doc.json", "{}", .{
        .content_type = "application/json",
    });

    const objects = try storage.listObjects(allocator, "doc");
    defer allocator.free(objects);
    try std.testing.expectEqual(@as(usize, 1), objects.len);
    try std.testing.expectEqualStrings("application/json", objects[0].content_type);
}

test "storage: list objects with prefix" {
    const allocator = std.testing.allocator;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    try storage.putObject(allocator, "img/a.png", "a");
    try storage.putObject(allocator, "img/b.png", "b");
    try storage.putObject(allocator, "txt/c.md", "c");

    const imgs = try storage.listObjects(allocator, "img/");
    defer allocator.free(imgs);
    try std.testing.expectEqual(@as(usize, 2), imgs.len);

    const all = try storage.listObjects(allocator, "");
    defer allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), all.len);
}

test "storage: error type is aliased as Error" {
    // Verify that storage.Error is the same as storage.StorageError
    const err1: storage.Error = error.ObjectNotFound;
    const err2: storage.StorageError = err1;
    _ = err2;
}

test "storage: s3 backend returns BackendNotAvailable" {
    const allocator = std.testing.allocator;
    const result = storage.init(allocator, .{ .backend = .s3 });
    try std.testing.expectError(error.BackendNotAvailable, result);
}

test {
    std.testing.refAllDecls(@This());
}
