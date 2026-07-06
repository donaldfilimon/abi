//! Automatic startup recovery for a WDBX Store (Storage Layer).
//!
//! A durable store is a segment checkpoint (with a legacy JSONL snapshot mirror)
//! plus a sidecar write-ahead log (`<path>.wal`). The checkpoint is periodic;
//! the WAL is the authoritative append-only record of every mutation. On open we
//! reconcile the two so a process resumes from the most complete committed state
//! — including mutations that were logged but not yet folded into a checkpoint
//! (e.g. a crash between the WAL append and the next checkpoint).
//!
//! Reconciliation: the checkpoint is loaded as the baseline, then the sidecar
//! WAL — a post-checkpoint delta tagged with the checkpoint epoch it was written
//! against — is folded on top when its `base_epoch` matches the checkpoint's
//! current epoch. A WAL whose epoch is older (a crash after the new checkpoint's
//! manifest committed but before the old WAL was cleared) is superseded and
//! discarded. Because the segment manifest write is the atomic commit point,
//! the epoch comparison alone makes recovery crash-safe without double-applying.
//! A corrupt WAL surfaces its error rather than silently falling back.

const std = @import("std");
const wdbx_mod = @import("mod.zig");
const persistence = @import("persistence.zig");
const segments = @import("segments.zig");
const wal = @import("wal.zig");
const test_helpers = @import("../../testing/test_helpers.zig");

/// Which durable source the recovered Store was reconstructed from. `.merged`
/// means a checkpoint plus a folded-in WAL delta. `.wal` is retained for
/// backward compatibility but is no longer produced by `open`.
pub const Source = enum { empty, snapshot, segment, wal, merged };

pub const Opened = struct {
    store: wdbx_mod.Store,
    source: Source,
    /// Manifest epoch the checkpoint was loaded at (0 for legacy snapshot/empty).
    checkpoint_epoch: u64 = 0,
};

/// The sidecar WAL path for a snapshot at `path` (`"<path>.wal"`).
pub fn walPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.wal", .{path});
}

/// Open the best checkpoint for `path` without replaying the sidecar WAL.
/// Segment checkpoints are the default runtime checkpoint source when their
/// manifest exists; legacy monolithic snapshots remain readable as fallback.
pub fn openCheckpoint(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Opened {
    var segment_store = segments.SegmentStore.init(allocator, io, path);
    if (try segment_store.latestEpoch()) |epoch| {
        return .{ .store = try segment_store.loadLatest(), .source = .segment, .checkpoint_epoch = epoch };
    }

    const snapshot = persistence.loadFromPath(io, allocator, path) catch |err| switch (err) {
        error.FileNotFound => return .{ .store = wdbx_mod.Store.init(allocator), .source = .empty },
        else => return err,
    };
    return .{ .store = snapshot, .source = .snapshot };
}

/// Open a Store from `path`, automatically recovering from its sidecar WAL.
/// Returns the reconstructed Store (caller owns it) plus the `Source` it came
/// from. A missing snapshot and/or WAL is treated as empty, not an error; a
/// corrupt WAL is propagated.
pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Opened {
    const wp = try walPath(allocator, path);
    defer allocator.free(wp);

    var loaded = try openCheckpoint(io, allocator, path);
    errdefer loaded.store.deinit();

    // No sidecar WAL: the checkpoint is the whole story.
    if (!try wal.exists(io, allocator, wp)) return loaded;

    const wal_base = try wal.readBaseEpoch(io, allocator, wp);
    if (wal_base == loaded.checkpoint_epoch) {
        // Delta belongs to this checkpoint: fold it on top. Vector-id continuity
        // holds because the loaded checkpoint's counter is already at the delta's
        // first id. Corruption (CRC / CorruptVectorId) propagates, never a silent
        // fallback to the pre-delta checkpoint.
        _ = try wal.replayOnto(io, allocator, wp, &loaded.store);
        loaded.source = .merged;
        return loaded;
    }

    // Epoch mismatch: the WAL was written against a different checkpoint — most
    // commonly a crash after a newer checkpoint's manifest committed but before
    // its predecessor WAL was cleared (the double-apply hazard). The committed
    // checkpoint already contains those mutations, so the stale WAL is discarded
    // and removed, leaving a clean slate for the next delta.
    std.Io.Dir.cwd().deleteFile(io, wp) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    return loaded;
}

const testing = std.testing;

const deleteTestFileIfExists = test_helpers.deleteTestFileIfExists;

fn writeSnapshot(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profiles: []const []const u8) !void {
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();
    for (profiles) |p| _ = try store.appendBlock(p, 0, 0, "{\"t\":1}");
    try persistence.saveToPath(io, allocator, &store, path);
}

test "recovery: WAL delta merges onto snapshot (un-checkpointed block recovered)" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-ahead.jsonl";
    const wp = "zig-out/wdbx-recovery-ahead.jsonl.wal";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(wp);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);

    // Snapshot checkpointed 1 block (epoch 0); the WAL is a delta with the one
    // un-checkpointed block beyond it. Merge folds the delta onto the checkpoint.
    try writeSnapshot(testing.io, allocator, path, &.{"abbey"});
    try wal.appendBlock(testing.io, allocator, wp, .{ .profile = "aviva", .query_id = 0, .response_id = 0, .metadata = "{\"t\":2}", .timestamp_ms = 2000 });

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.merged, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: snapshot only when no WAL" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-snaponly.jsonl";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try writeSnapshot(testing.io, allocator, path, &.{ "abbey", "aviva" });

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.snapshot, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: segment checkpoint wins over legacy snapshot when no WAL" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-segment.jsonl";
    const manifest = "zig-out/wdbx-recovery-segment.jsonl.manifest";
    const seg0 = "zig-out/wdbx-recovery-segment.jsonl.seg.0.jsonl";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(manifest);
    defer deleteTestFileIfExists(seg0);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(manifest);
    deleteTestFileIfExists(seg0);

    try writeSnapshot(testing.io, allocator, path, &.{"abbey"});

    var segment_source = wdbx_mod.Store.init(allocator);
    defer segment_source.deinit();
    _ = try segment_source.appendBlock("abbey", 0, 0, "{\"t\":1}");
    _ = try segment_source.appendBlock("aviva", 0, 0, "{\"t\":2}");
    var segment_store = segments.SegmentStore.init(allocator, testing.io, path);
    _ = try segment_store.flush(&segment_source);

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.segment, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: WAL delta merges onto segment checkpoint" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-segment-wal.jsonl";
    const wp = "zig-out/wdbx-recovery-segment-wal.jsonl.wal";
    const manifest = "zig-out/wdbx-recovery-segment-wal.jsonl.manifest";
    const seg0 = "zig-out/wdbx-recovery-segment-wal.jsonl.seg.0.jsonl";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(wp);
    defer deleteTestFileIfExists(manifest);
    defer deleteTestFileIfExists(seg0);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);
    deleteTestFileIfExists(manifest);
    deleteTestFileIfExists(seg0);

    var checkpoint = wdbx_mod.Store.init(allocator);
    defer checkpoint.deinit();
    _ = try checkpoint.appendBlock("abbey", 0, 0, "{\"t\":1}");
    var segment_store = segments.SegmentStore.init(allocator, testing.io, path);
    _ = try segment_store.flush(&checkpoint); // epoch 0

    // Delta WAL at epoch 0: only the one un-checkpointed block.
    try wal.createWithEpoch(testing.io, allocator, wp, 0);
    try wal.appendBlock(testing.io, allocator, wp, .{ .profile = "abi", .query_id = 0, .response_id = 0, .metadata = "{\"t\":2}", .timestamp_ms = 2000 });

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.merged, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
}

test "recovery: superseded WAL (older epoch) is discarded after a newer checkpoint" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-stale-wal.jsonl";
    const wp = "zig-out/wdbx-recovery-stale-wal.jsonl.wal";
    const manifest = "zig-out/wdbx-recovery-stale-wal.jsonl.manifest";
    const seg0 = "zig-out/wdbx-recovery-stale-wal.jsonl.seg.0.jsonl";
    const seg1 = "zig-out/wdbx-recovery-stale-wal.jsonl.seg.1.jsonl";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(wp);
    defer deleteTestFileIfExists(manifest);
    defer deleteTestFileIfExists(seg0);
    defer deleteTestFileIfExists(seg1);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);
    deleteTestFileIfExists(manifest);
    deleteTestFileIfExists(seg0);
    deleteTestFileIfExists(seg1);

    var segment_store = segments.SegmentStore.init(allocator, testing.io, path);

    // Checkpoint epoch 0 (1 block) and a delta WAL tagged to epoch 0.
    var s0 = wdbx_mod.Store.init(allocator);
    _ = try s0.appendBlock("abbey", 0, 0, "{\"t\":1}");
    _ = try segment_store.flush(&s0); // epoch 0
    s0.deinit();
    try wal.createWithEpoch(testing.io, allocator, wp, 0);
    try wal.appendBlock(testing.io, allocator, wp, .{ .profile = "ghost", .query_id = 0, .response_id = 0, .metadata = "{\"t\":9}", .timestamp_ms = 9000 });

    // A newer checkpoint commits at epoch 1 (manifest advanced) — the crash-at-C2
    // state where the old WAL was not yet cleared. The epoch-1 checkpoint already
    // contains everything, so the epoch-0 WAL must be discarded (no double-apply).
    var s1 = wdbx_mod.Store.init(allocator);
    _ = try s1.appendBlock("abbey", 0, 0, "{\"t\":1}");
    _ = try s1.appendBlock("aviva", 0, 0, "{\"t\":2}");
    _ = try segment_store.flush(&s1); // epoch 1
    s1.deinit();

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.segment, opened.source);
    try testing.expectEqual(@as(usize, 2), opened.store.blockCount());
    // The stale WAL was removed so the next mutation starts a clean delta.
    try testing.expect(!try wal.exists(testing.io, allocator, wp));
}

test "recovery: vector delta merges onto checkpoint preserving id continuity" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-vec.jsonl";
    const wp = "zig-out/wdbx-recovery-vec.jsonl.wal";
    const manifest = "zig-out/wdbx-recovery-vec.jsonl.manifest";
    const seg0 = "zig-out/wdbx-recovery-vec.jsonl.seg.0.jsonl";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(wp);
    defer deleteTestFileIfExists(manifest);
    defer deleteTestFileIfExists(seg0);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);
    deleteTestFileIfExists(manifest);
    deleteTestFileIfExists(seg0);

    // Checkpoint with 2 vectors -> counter baseline is 3.
    var base = wdbx_mod.Store.init(allocator);
    _ = try base.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try base.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    var segment_store = segments.SegmentStore.init(allocator, testing.io, path);
    _ = try segment_store.flush(&base); // epoch 0
    base.deinit();

    // Delta WAL (epoch 0) logging the next vector at its absolute id 3.
    try wal.createWithEpoch(testing.io, allocator, wp, 0);
    try wal.appendVector(testing.io, allocator, wp, 3, &.{ 0.0, 0.0, 1.0, 0.0 });

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.merged, opened.source);
    try testing.expectEqual(@as(usize, 3), opened.store.vectorCount());
}

test "recovery: corrupt WAL frame propagates rather than silently dropping" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-corrupt.jsonl";
    const wp = "zig-out/wdbx-recovery-corrupt.jsonl.wal";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(wp);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);

    try writeSnapshot(testing.io, allocator, path, &.{"abbey"});
    try wal.createWithEpoch(testing.io, allocator, wp, 0);
    try wal.appendBlock(testing.io, allocator, wp, .{ .profile = "aviva", .query_id = 0, .response_id = 0, .metadata = "{\"turn\":2}", .timestamp_ms = 2000 });

    // Flip a byte inside the framed JSON payload; recovery must surface it.
    const content = try std.Io.Dir.cwd().readFileAlloc(testing.io, wp, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    const idx = std.mem.indexOf(u8, content, "turn").?;
    content[idx] = 'X';
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = wp, .data = content });

    try testing.expectError(error.WalCorruption, open(testing.io, allocator, path));
}

test "recovery: an orphan segment file without a manifest is ignored (crash before manifest commit)" {
    const allocator = testing.allocator;
    const path = "zig-out/wdbx-recovery-orphan.jsonl";
    const seg0 = "zig-out/wdbx-recovery-orphan.jsonl.seg.0.jsonl";
    defer deleteTestFileIfExists(path);
    defer deleteTestFileIfExists(seg0);
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(seg0);

    // Crash point C1: the segment file is on disk but the manifest that lists it
    // was never committed. The manifest is the source of truth, so recovery must
    // ignore the orphan rather than load partial/uncommitted state.
    var store = wdbx_mod.Store.init(allocator);
    _ = try store.appendBlock("abbey", 0, 0, "{\"t\":1}");
    try persistence.saveToPath(testing.io, allocator, &store, seg0);
    store.deinit();

    var opened = try open(testing.io, allocator, path);
    defer opened.store.deinit();
    try testing.expectEqual(Source.empty, opened.source);
    try testing.expectEqual(@as(usize, 0), opened.store.blockCount());
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
