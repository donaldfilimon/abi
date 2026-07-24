//! Bounded correctness-under-load checks. These insert enough records to
//! exercise growth/rehashing paths that small unit tests never touch, but
//! stay small enough (low thousands of ops, sub-second) to run alongside the
//! rest of the integration suite rather than needing a separate opt-in gate.
//!
//! These assert correctness only, never wall-clock timing — a latency
//! threshold would be flaky across CI hardware. (`test_helpers.bench` is
//! deliberately not used: it is currently uncompilable dead code, since
//! `std.time.Instant` no longer exists on the pinned toolchain.)

const std = @import("std");
const wdbx = @import("../features/wdbx/mod.zig");

const testing = std.testing;

const VECTOR_STRESS_COUNT: u32 = 2000;
const KV_STRESS_COUNT: usize = 2000;

test "stress: many vector inserts and searches stay correct under load" {
    const a = testing.allocator;
    var store = wdbx.Store.init(a);
    defer store.deinit();

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    var ids: [VECTOR_STRESS_COUNT]u32 = undefined;

    var inserted: u32 = 0;
    while (inserted < VECTOR_STRESS_COUNT) : (inserted += 1) {
        var vec: [8]f32 = undefined;
        for (&vec) |*component| component.* = rand.float(f32);
        ids[inserted] = try store.putVector(&vec);
    }

    try testing.expectEqual(@as(usize, VECTOR_STRESS_COUNT), store.vectorCount());

    // Every inserted id must resolve back to a vector (no silent drops from
    // whatever growth/rehash strategy the index uses internally).
    for (ids) |id| {
        try testing.expect(store.getVector(id) != null);
    }

    // Repeated searches against a large index must stay well-formed: correct
    // result count, and scores in non-increasing order.
    const query = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var search_round: usize = 0;
    while (search_round < 50) : (search_round += 1) {
        const results = try store.search(&query, 10);
        defer a.free(results);
        try testing.expectEqual(@as(usize, 10), results.len);
        var i: usize = 1;
        while (i < results.len) : (i += 1) {
            try testing.expect(results[i - 1].score >= results[i].score);
        }
    }
}

test "stress: many key/value upserts stay consistent under load" {
    const a = testing.allocator;
    var store = wdbx.Store.init(a);
    defer store.deinit();

    var buf: [32]u8 = undefined;

    var inserted: usize = 0;
    while (inserted < KV_STRESS_COUNT) : (inserted += 1) {
        const key = try std.fmt.bufPrint(&buf, "stress:key:{d}", .{inserted});
        try store.store(key, "v1");
    }

    try testing.expectEqual(@as(usize, KV_STRESS_COUNT), store.count());

    // Re-upsert every other key with a new value; verify both the updated
    // and untouched keys read back correctly (no cross-contamination).
    var i: usize = 0;
    while (i < KV_STRESS_COUNT) : (i += 2) {
        const key = try std.fmt.allocPrint(a, "stress:key:{d}", .{i});
        defer a.free(key);
        try store.store(key, "v2");
    }
    try testing.expectEqual(@as(usize, KV_STRESS_COUNT), store.count());

    i = 0;
    while (i < KV_STRESS_COUNT) : (i += 1) {
        const key = try std.fmt.allocPrint(a, "stress:key:{d}", .{i});
        defer a.free(key);
        const expected = if (i % 2 == 0) "v2" else "v1";
        try testing.expectEqualStrings(expected, store.get(key).?);
    }
}

test {
    std.testing.refAllDecls(@This());
}
