//! Automatic startup recovery for a WDBX Store (Storage Layer).
//!
//! A durable store is a JSONL snapshot plus a sidecar write-ahead log
//! (`<path>.wal`). The snapshot is a periodic checkpoint; the WAL is the
//! authoritative append-only record of every mutation. On open we reconcile the
//! two so a process resumes from the most complete committed state — including
//! mutations that were logged but not yet folded into a snapshot (e.g. a crash
//! between the WAL append and the next checkpoint).
//!
//! Selection rule: whichever source holds more committed blocks wins; a WAL that
//! ties the snapshot wins, since it is the authoritative log. A corrupt WAL
//! surfaces its error rather than silently falling back to a stale snapshot.

const std = @import("std");
const wdbx_mod = @import("mod.zig");
const persistence = @import("persistence.zig");
const wal = @import("wal.zig");

/// Which durable source the recovered Store was reconstructed from.
pub const Source = enum { empty, snapshot, wal };

pub const Opened = struct {
    store: wdbx_mod.Store,
    source: Source,
};

/// The sidecar WAL path for a snapshot at `path` (`"<path>.wal"`).
pub fn walPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.wal", .{path});
}

/// Open a Store from `path`, automatically recovering from its sidecar WAL.
/// Returns the reconstructed Store (caller owns it) plus the `Source` it came
/// from. A missing snapshot and/or WAL is treated as empty, not an error; a
/// corrupt WAL is propagated.
pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Opened {
    const wp = try walPath(allocator, path);
    defer allocator.free(wp);

    var have_snapshot = true;
    var snapshot = persistence.loadFromPath(io, allocator, path) catch |err| switch (err) {
        error.FileNotFound => blk: {
            have_snapshot = false;
            break :blk wdbx_mod.Store.init(allocator);
        },
        else => return err,
    };
    errdefer snapshot.deinit();

    var have_wal = true;
    var wal_store = wal.replay(io, allocator, wp) catch |err| switch (err) {
        error.FileNotFound => blk: {
            have_wal = false;
            break :blk wdbx_mod.Store.init(allocator);
        },
        else => return err,
    };
    errdefer wal_store.deinit();

    // WAL wins when it exists and holds at least as many committed blocks as the
    // snapshot (it is the authoritative append-only log). Deinit the loser; move
    // the winner out to the caller.
    if (have_wal and (!have_snapshot or wal_store.blockCount() >= snapshot.blockCount())) {
        snapshot.deinit();
        return .{ .store = wal_store, .source = .wal };
    }
    wal_store.deinit();
    return .{ .store = snapshot, .source = if (have_snapshot) .snapshot else .empty };
}

const testing = std.testing;

fn writeSnapshot(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profiles: []const []const u8) !void {
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();
    for (profiles) |p| _ = try store.appendBlock(p, 0, 0, "{\"t\":1}");
    try persistence.saveToPath(io, allocator, &store, path);
}

test "recovery: WAL ahead of snapshot wins (un-checkpointed blocks recovered)" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-ahead.jsonl";
    const wp = "zig-out/wdbx-recovery-ahead.jsonl.wal";
    defer std.Io.Dir.cwd().deleteFile(testing.io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(testing.io, wp) catch {};
    std.Io.Dir.cwd().deleteFile(testing.io, path) catch {};
    std.Io.Dir.cwd().deleteFile(testing.io, wp) catch {};

    // Snapshot checkpointed 1 block; the WAL logged 2 (one beyond the snapshot).
    try writeSnapshot(testing.io, allocator, path, &.{"abbey"});
    try wal.appendBlock(testing.io, allocator, wp, "abbey", 0, 0, "{\"t\":1}", 1000);
    try wal.appendBlock(testing.io, allocator, wp, "aviva", 0, 0, "{\"t\":2}", 2000);

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.wal, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: snapshot only when no WAL" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-snaponly.jsonl";
    defer std.Io.Dir.cwd().deleteFile(testing.io, path) catch {};
    std.Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    try writeSnapshot(testing.io, allocator, path, &.{ "abbey", "aviva" });

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.snapshot, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: empty when neither snapshot nor WAL exists" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-absent.jsonl";

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.empty, opened.source);
    try testing.expectEqual(@as(usize, 0), opened.store.blockCount());
}

test {
    testing.refAllDecls(@This());
}
