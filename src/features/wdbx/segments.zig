//! Multi-segment durable storage with epoch reclamation (Storage Layer).
//!
//! Instead of a single monolithic snapshot, a store is checkpointed as a series
//! of immutable, epoch-numbered segment files (`<base>.seg.<epoch>.jsonl`) named
//! in a small manifest (`<base>.manifest`). Each `flush` writes a new segment at
//! a monotonic epoch; `reclaim` drops segments below a watermark to bound disk
//! use; `loadLatest` reconstructs from the highest active epoch. This keeps the
//! Store itself unchanged — segmentation is a persistence-orchestration layer on
//! top of the existing JSONL snapshot codec.
//!
//! The manifest is the source of truth for which segments are live, so no
//! directory scan is needed and a half-written flush never corrupts the set.

const std = @import("std");
const wdbx_mod = @import("mod.zig");
const persistence = @import("persistence.zig");
const test_helpers = @import("../../testing/test_helpers.zig");

pub const MANIFEST_HEADER = "# ABI-WDBX-SEGMENTS v1";

pub const SegmentError = error{ InvalidManifest, InvalidCompactionPolicy };

pub const CompactionResult = struct {
    before: usize,
    after: usize,
    deleted: usize,
    keep_latest: usize,
    latest_epoch: ?u64,
    watermark_epoch: ?u64,
};

const Manifest = struct {
    next_epoch: u64 = 0,
    active: std.ArrayListUnmanaged(u64) = .empty,

    fn deinit(self: *Manifest, allocator: std.mem.Allocator) void {
        self.active.deinit(allocator);
    }
};

pub const SegmentStore = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    base: []const u8, // borrowed path prefix

    pub fn init(allocator: std.mem.Allocator, io: std.Io, base: []const u8) SegmentStore {
        return .{ .allocator = allocator, .io = io, .base = base };
    }

    fn manifestPath(self: *const SegmentStore) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}.manifest", .{self.base});
    }

    fn segmentPath(self: *const SegmentStore, epoch: u64) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}.seg.{d}.jsonl", .{ self.base, epoch });
    }

    fn readManifest(self: *const SegmentStore) !Manifest {
        const mp = try self.manifestPath();
        defer self.allocator.free(mp);

        const content = std.Io.Dir.cwd().readFileAlloc(self.io, mp, self.allocator, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => return .{},
            else => return err,
        };
        defer self.allocator.free(content);

        var m = Manifest{};
        errdefer m.deinit(self.allocator);

        var lines = std.mem.splitScalar(u8, content, '\n');
        const header = lines.next() orelse return SegmentError.InvalidManifest;
        if (!std.mem.eql(u8, std.mem.trim(u8, header, " \t\r"), MANIFEST_HEADER)) return SegmentError.InvalidManifest;

        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0) continue;
            if (std.mem.startsWith(u8, line, "next_epoch=")) {
                m.next_epoch = std.fmt.parseInt(u64, line["next_epoch=".len..], 10) catch return SegmentError.InvalidManifest;
            } else if (std.mem.startsWith(u8, line, "active=")) {
                const list = line["active=".len..];
                var it = std.mem.splitScalar(u8, list, ',');
                while (it.next()) |tok| {
                    const t = std.mem.trim(u8, tok, " ");
                    if (t.len == 0) continue;
                    const e = std.fmt.parseInt(u64, t, 10) catch return SegmentError.InvalidManifest;
                    try m.active.append(self.allocator, e);
                }
            }
        }
        return m;
    }

    fn writeManifest(self: *const SegmentStore, m: Manifest) !void {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        try out.writer.writeAll(MANIFEST_HEADER);
        try out.writer.print("\nnext_epoch={d}\nactive=", .{m.next_epoch});
        for (m.active.items, 0..) |e, i| {
            if (i > 0) try out.writer.writeAll(",");
            try out.writer.print("{d}", .{e});
        }
        try out.writer.writeAll("\n");

        const mp = try self.manifestPath();
        defer self.allocator.free(mp);
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = mp, .data = out.written() });
    }

    /// Checkpoint `store` as a new immutable segment; returns its epoch.
    pub fn flush(self: *SegmentStore, store: *const wdbx_mod.Store) !u64 {
        var m = try self.readManifest();
        defer m.deinit(self.allocator);

        const epoch = m.next_epoch;
        const sp = try self.segmentPath(epoch);
        defer self.allocator.free(sp);

        try persistence.saveToPath(self.io, self.allocator, store, sp);

        try m.active.append(self.allocator, epoch);
        m.next_epoch += 1;
        try self.writeManifest(m);
        return epoch;
    }

    /// Highest active epoch, or null if no segments exist.
    pub fn latestEpoch(self: *const SegmentStore) !?u64 {
        var m = try self.readManifest();
        defer m.deinit(self.allocator);
        if (m.active.items.len == 0) return null;
        var max = m.active.items[0];
        for (m.active.items) |e| max = @max(max, e);
        return max;
    }

    /// The active segment epochs in ascending order (caller owns the slice).
    pub fn activeEpochs(self: *const SegmentStore, allocator: std.mem.Allocator) ![]u64 {
        var m = try self.readManifest();
        defer m.deinit(self.allocator);
        const out = try allocator.dupe(u64, m.active.items);
        std.mem.sort(u64, out, {}, std.sort.asc(u64));
        return out;
    }

    /// Reconstruct the Store from the highest active segment (empty if none).
    pub fn loadLatest(self: *const SegmentStore) !wdbx_mod.Store {
        const latest = (try self.latestEpoch()) orelse return wdbx_mod.Store.init(self.allocator);
        const sp = try self.segmentPath(latest);
        defer self.allocator.free(sp);
        return persistence.loadFromPath(self.io, self.allocator, sp);
    }

    /// Delete every segment whose epoch is below `keep_from_epoch`, bounding disk
    /// use once older checkpoints are superseded. Returns the number reclaimed.
    pub fn reclaim(self: *SegmentStore, keep_from_epoch: u64) !usize {
        var m = try self.readManifest();
        defer m.deinit(self.allocator);

        var kept: std.ArrayListUnmanaged(u64) = .empty;
        defer kept.deinit(self.allocator);

        var deleted: usize = 0;
        for (m.active.items) |e| {
            if (e < keep_from_epoch) {
                const sp = try self.segmentPath(e);
                defer self.allocator.free(sp);
                std.Io.Dir.cwd().deleteFile(self.io, sp) catch |err| switch (err) {
                    error.FileNotFound => {},
                    else => return err,
                };
                deleted += 1;
            } else {
                try kept.append(self.allocator, e);
            }
        }

        m.active.clearRetainingCapacity();
        try m.active.appendSlice(self.allocator, kept.items);
        try self.writeManifest(m);
        return deleted;
    }

    /// Larger-store compaction policy: retain the newest `keep_latest` active
    /// checkpoint epochs and reclaim every older manifest-listed segment. The
    /// latest checkpoint is always preserved so recovery still has a durable
    /// baseline for any sidecar WAL delta.
    pub fn compactRetainingLatest(self: *SegmentStore, keep_latest: usize) !CompactionResult {
        if (keep_latest == 0) return SegmentError.InvalidCompactionPolicy;

        const active = try self.activeEpochs(self.allocator);
        defer self.allocator.free(active);

        const before = active.len;
        if (before == 0) {
            return .{
                .before = 0,
                .after = 0,
                .deleted = 0,
                .keep_latest = keep_latest,
                .latest_epoch = null,
                .watermark_epoch = null,
            };
        }

        const latest = active[before - 1];
        if (before <= keep_latest) {
            return .{
                .before = before,
                .after = before,
                .deleted = 0,
                .keep_latest = keep_latest,
                .latest_epoch = latest,
                .watermark_epoch = active[0],
            };
        }

        const watermark = active[before - keep_latest];
        const deleted = try self.reclaim(watermark);
        const after = before - deleted;
        return .{
            .before = before,
            .after = after,
            .deleted = deleted,
            .keep_latest = keep_latest,
            .latest_epoch = latest,
            .watermark_epoch = watermark,
        };
    }

    /// Remove all manifest-listed segments and the manifest itself. This is
    /// used by `wdbx db init` to make reinitialization deterministic without a
    /// directory scan.
    pub fn reset(self: *SegmentStore) !void {
        var m = try self.readManifest();
        defer m.deinit(self.allocator);

        for (m.active.items) |e| {
            const sp = try self.segmentPath(e);
            defer self.allocator.free(sp);
            std.Io.Dir.cwd().deleteFile(self.io, sp) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
        }

        const mp = try self.manifestPath();
        defer self.allocator.free(mp);
        std.Io.Dir.cwd().deleteFile(self.io, mp) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
    }
};

const testing = std.testing;

fn cleanup(base: []const u8) void {
    var buf: [256]u8 = undefined;
    const mp = std.fmt.bufPrint(&buf, "{s}.manifest", .{base}) catch return;
    deleteTestFileIfExists(mp);
    var e: u64 = 0;
    while (e < 8) : (e += 1) {
        const sp = std.fmt.bufPrint(&buf, "{s}.seg.{d}.jsonl", .{ base, e }) catch continue;
        deleteTestFileIfExists(sp);
    }
}

const deleteTestFileIfExists = test_helpers.deleteTestFileIfExists;

test "segments: flush assigns monotonic epochs and loadLatest reads the newest" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-rt";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);

    var s0 = wdbx_mod.Store.init(allocator);
    _ = try s0.appendBlock("abbey", 0, 0, "{\"t\":1}");
    try testing.expectEqual(@as(u64, 0), try ss.flush(&s0));
    s0.deinit();

    var s1 = wdbx_mod.Store.init(allocator);
    _ = try s1.appendBlock("abbey", 0, 0, "{\"t\":1}");
    _ = try s1.appendBlock("aviva", 0, 0, "{\"t\":2}");
    try testing.expectEqual(@as(u64, 1), try ss.flush(&s1));
    s1.deinit();

    try testing.expectEqual(@as(?u64, 1), try ss.latestEpoch());

    var loaded = try ss.loadLatest();
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 2), loaded.blockCount());
}

test "segments: reclaim drops epochs below the watermark" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-reclaim";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var s = wdbx_mod.Store.init(allocator);
        _ = try s.appendBlock("p", 0, 0, "{\"t\":1}");
        _ = try ss.flush(&s);
        s.deinit();
    }

    // Keep epoch >= 2; epochs 0 and 1 are reclaimed.
    try testing.expectEqual(@as(usize, 2), try ss.reclaim(2));

    const active = try ss.activeEpochs(allocator);
    defer allocator.free(active);
    try testing.expectEqual(@as(usize, 1), active.len);
    try testing.expectEqual(@as(u64, 2), active[0]);

    // The surviving segment still loads.
    var loaded = try ss.loadLatest();
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 1), loaded.blockCount());
}

test "segments: compactRetainingLatest keeps the newest checkpoints" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-compact";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        var s = wdbx_mod.Store.init(allocator);
        var block: usize = 0;
        while (block <= i) : (block += 1) {
            _ = try s.appendBlock("p", 0, 0, "{\"t\":1}");
        }
        _ = try ss.flush(&s);
        s.deinit();
    }

    const result = try ss.compactRetainingLatest(2);
    try testing.expectEqual(@as(usize, 4), result.before);
    try testing.expectEqual(@as(usize, 2), result.after);
    try testing.expectEqual(@as(usize, 2), result.deleted);
    try testing.expectEqual(@as(?u64, 3), result.latest_epoch);
    try testing.expectEqual(@as(?u64, 2), result.watermark_epoch);

    const active = try ss.activeEpochs(allocator);
    defer allocator.free(active);
    try testing.expectEqualSlices(u64, &.{ 2, 3 }, active);

    var loaded = try ss.loadLatest();
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 4), loaded.blockCount());
}

test "segments: compactRetainingLatest with single segment is a no-op" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-single";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    var s = wdbx_mod.Store.init(allocator);
    defer s.deinit();
    _ = try s.appendBlock("p", 0, 0, "{\"t\":1}");
    _ = try ss.flush(&s);

    const result = try ss.compactRetainingLatest(1);
    try testing.expectEqual(@as(usize, 1), result.before);
    try testing.expectEqual(@as(usize, 1), result.after);
    try testing.expectEqual(@as(usize, 0), result.deleted);

    const active = try ss.activeEpochs(allocator);
    defer allocator.free(active);
    try testing.expectEqualSlices(u64, &.{0}, active);
}

test "segments: compactRetainingLatest on empty store succeeds" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-empty";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);

    const result = try ss.compactRetainingLatest(1);
    try testing.expectEqual(@as(usize, 0), result.before);
    try testing.expectEqual(@as(usize, 0), result.after);
    try testing.expectEqual(@as(usize, 0), result.deleted);
    try testing.expectEqual(@as(?u64, null), result.latest_epoch);
    try testing.expectEqual(@as(?u64, null), result.watermark_epoch);
}

test "segments: compactRetainingLatest with keep_latest=0 returns error" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-err";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    try testing.expectError(SegmentError.InvalidCompactionPolicy, ss.compactRetainingLatest(0));
}

test "segments: reset removes manifest-listed checkpoints" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-reset";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    var s = wdbx_mod.Store.init(allocator);
    defer s.deinit();
    _ = try s.appendBlock("p", 0, 0, "{\"t\":1}");
    _ = try ss.flush(&s);

    try ss.reset();
    try testing.expectEqual(@as(?u64, null), try ss.latestEpoch());

    var loaded = try ss.loadLatest();
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 0), loaded.blockCount());
}

test "segments: readManifest rejects corrupt manifests rather than mis-parsing" {
    const allocator = testing.allocator;
    const base = "zig-out/wdbx-seg-corrupt";
    cleanup(base);
    defer cleanup(base);

    var ss = SegmentStore.init(allocator, testing.io, base);
    const mp = base ++ ".manifest";

    // A garbled manifest must surface InvalidManifest — silently mis-parsing it
    // (e.g. dropping a live `active=` epoch) would lose committed segments.
    const corrupt = [_][]const u8{
        "# BOGUS HEADER\n", // wrong/missing header
        MANIFEST_HEADER ++ "\nnext_epoch=abc\n", // non-numeric next_epoch
        MANIFEST_HEADER ++ "\nnext_epoch=1\nactive=0,xx\n", // non-numeric active token
    };
    for (corrupt) |bad| {
        try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = mp, .data = bad });
        try testing.expectError(SegmentError.InvalidManifest, ss.latestEpoch());
    }
}

test {
    testing.refAllDecls(@This());
}
