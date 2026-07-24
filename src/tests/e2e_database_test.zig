//! End-to-end WDBX database journey: insert vectors + key/value entries through
//! a durable `Session` (the same path the CLI and MCP server use), search,
//! checkpoint, close, and reopen through crash recovery to verify the whole
//! chain — not just an individual module — round-trips correctly.
//!
//! Note: the `Store` vector index has no delete/update-in-place API (only
//! `putVector`/`search`/`getVector`); the generic key/value side supports
//! upsert via `store()`. This test exercises what the store actually
//! supports rather than asserting a vector-delete path that does not exist.

const std = @import("std");
const durable_store = @import("../features/wdbx/durable_store.zig");

const testing = std.testing;

fn deleteIfExists(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(testing.io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.debug.print("e2e db cleanup: failed to delete '{s}': {s}\n", .{ path, @errorName(err) }),
    };
}

/// `zig-out` only exists after an install step has run, so a bare
/// `zig build test-integration` on a clean tree would otherwise fail with
/// FileNotFound before reaching a single assertion.
fn ensureOutDir() !void {
    std.Io.Dir.createDirPath(.cwd(), testing.io, "zig-out") catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

test "e2e: vector DB insert -> search -> update -> recover across a restart" {
    try ensureOutDir();
    const base = "zig-out/e2e-database.jsonl";
    const wp = "zig-out/e2e-database.jsonl.wal";
    const manifest = "zig-out/e2e-database.jsonl.manifest";
    const seg0 = "zig-out/e2e-database.jsonl.seg.0.jsonl";
    defer {
        deleteIfExists(base);
        deleteIfExists(wp);
        deleteIfExists(manifest);
        deleteIfExists(seg0);
    }
    deleteIfExists(base);
    deleteIfExists(wp);
    deleteIfExists(manifest);
    deleteIfExists(seg0);

    var v1: u32 = undefined;
    var v2: u32 = undefined;

    // --- Phase 1: insert vectors + kv, search, checkpoint on close. ---
    {
        var session = try durable_store.Session.openAt(testing.io, testing.allocator, base);
        defer session.deinit(); // folds the WAL into a checkpoint on close

        const store = session.storePtr();
        v1 = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
        v2 = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
        _ = try store.putVector(&.{ 0.9, 0.1, 0.0, 0.0 });
        try testing.expectEqual(@as(usize, 3), store.vectorCount());

        try store.store("session:profile", "abbey");

        const results = try store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 2);
        defer testing.allocator.free(results);
        try testing.expect(results.len == 2);
        try testing.expect(results[0].score >= results[1].score);
        try testing.expectEqual(v1, results[0].id);
    }

    // --- Phase 2: reopen through recovery, verify persisted state, "update" ---
    // the kv entry (upsert, since vectors have no update-in-place API).
    {
        var reopened = try durable_store.Session.openAt(testing.io, testing.allocator, base);
        defer reopened.deinit();

        const store = reopened.storePtr();
        try testing.expectEqual(@as(usize, 3), store.vectorCount());
        try testing.expectEqualStrings("abbey", store.get("session:profile").?);

        // getVector confirms the recovered vector index still answers by id,
        // not just by count.
        const recovered_v1 = store.getVector(v1).?;
        try testing.expectEqualSlices(f32, &.{ 1.0, 0.0, 0.0, 0.0 }, recovered_v1);

        const recovered_v2 = store.getVector(v2).?;
        try testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 0.0, 0.0 }, recovered_v2);

        // "update" == upsert on the same key.
        try store.store("session:profile", "aviva");
        try testing.expectEqualStrings("aviva", store.get("session:profile").?);

        // Search still works post-recovery against the recovered index.
        const results = try store.search(&.{ 0.0, 1.0, 0.0, 0.0 }, 1);
        defer testing.allocator.free(results);
        try testing.expectEqual(@as(usize, 1), results.len);
        try testing.expectEqual(v2, results[0].id);
    }

    // --- Phase 3: reopen once more, confirm the update from phase 2 survived
    // its own checkpoint. ---
    {
        var final = try durable_store.Session.openAt(testing.io, testing.allocator, base);
        defer final.deinit();
        try testing.expectEqualStrings("aviva", final.storePtr().get("session:profile").?);
        try testing.expectEqual(@as(usize, 3), final.storePtr().vectorCount());
    }
}

test {
    std.testing.refAllDecls(@This());
}
