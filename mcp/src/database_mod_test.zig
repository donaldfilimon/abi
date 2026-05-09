//! Focused database unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const build_options = @import("build_options");
const database = @import("features/database/mod.zig");

test {
    std.testing.refAllDecls(database);
}

// ── Feature gate tests ─────────────────────────────────────────────────

test "database isEnabled reflects build option" {
    try std.testing.expectEqual(build_options.feat_database, database.isEnabled());
}

// ── Store open and deinit ──────────────────────────────────────────────

test "database store open and deinit" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-open");
    defer store.deinit();

    const s = store.stats();
    try std.testing.expectEqual(@as(usize, 0), s.count);
}

// ── Insert and search round-trip ───────────────────────────────────────

test "database store insert and search" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-insert");
    defer store.deinit();

    try store.insert(1, &.{ 1.0, 0.0, 0.0 }, "first");
    try store.insert(2, &.{ 0.0, 1.0, 0.0 }, "second");

    const results = try store.search(&.{ 1.0, 0.0, 0.0 }, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    // Nearest to [1,0,0] should be id=1
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

// ── Get vector by ID ───────────────────────────────────────────────────

test "database store get existing vector" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-get");
    defer store.deinit();

    try store.insert(42, &.{ 0.5, 0.5, 0.5 }, "meta42");
    const view = store.get(42);
    try std.testing.expect(view != null);
}

test "database store get missing vector returns null" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-get-miss");
    defer store.deinit();

    const view = store.get(9999);
    try std.testing.expect(view == null);
}

// ── Remove vector ──────────────────────────────────────────────────────

test "database store remove vector" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-remove");
    defer store.deinit();

    try store.insert(10, &.{ 1.0, 1.0, 1.0 }, null);
    try std.testing.expect(store.remove(10));
    // Removing again should return false
    try std.testing.expect(!store.remove(10));
    try std.testing.expect(store.get(10) == null);
}

// ── Update vector ──────────────────────────────────────────────────────

test "database store update existing vector" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-update");
    defer store.deinit();

    try store.insert(5, &.{ 1.0, 0.0, 0.0 }, "original");
    const updated = try store.update(5, &.{ 0.0, 1.0, 0.0 });
    try std.testing.expect(updated);
}

// ── Stats ──────────────────────────────────────────────────────────────

test "database store stats reflect insertions" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var store = try database.Store.open(allocator, "test-stats");
    defer store.deinit();

    try store.insert(1, &.{ 0.1, 0.2 }, null);
    try store.insert(2, &.{ 0.3, 0.4 }, null);

    const s = store.stats();
    try std.testing.expectEqual(@as(usize, 2), s.count);
}

// ── Context tests ──────────────────────────────────────────────────────

test "database context init with memory path" {
    if (!build_options.feat_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const ctx = try database.Context.init(allocator, .{ .path = ":memory:" });
    defer ctx.deinit();

    try std.testing.expect(ctx.store != null);
}
