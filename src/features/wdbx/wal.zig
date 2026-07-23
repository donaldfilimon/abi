//! Write-ahead log for the WDBX Store (Storage Layer).
//!
//! Each mutation is appended as a CRC32-framed, JSON-encoded record so the log
//! is append-only, deterministic, and verifiable. Replay reconstructs a Store
//! by handing the verified record bodies to `persistence.deserialize`, reusing
//! the exact same apply path as snapshot restore. A **mid-log** flipped byte or
//! malformed frame is rejected with `error.WalCorruption`. A **torn last frame**
//! (incomplete line at EOF after a crash mid-append) is skipped so verified
//! prefix frames still replay; callers may truncate the file to the last good
//! newline via `truncateTornTail`.
//!
//! Frame format (one record per line, after the header line):
//!   <crc32-hex8> <minified-json>\n
//! where crc32 is computed over the JSON bytes only.

const std = @import("std");
const builtin = @import("builtin");
const wdbx_mod = @import("mod.zig");
const persistence = @import("persistence.zig");
const persistence_parse = @import("persistence_parse.zig");
const test_helpers = @import("../../foundation/test_helpers.zig");

/// Header line prefix. A WAL written after a checkpoint additionally carries a
/// ` base_epoch=N` token naming the checkpoint epoch it is a delta from; legacy
/// headers without the token are treated as base_epoch 0 (full-history WAL).
pub const WAL_HEADER_PREFIX = "# ABI-WDBX-WAL v1";
/// Legacy exact header (no epoch tag). Retained for back-compat and tests.
pub const WAL_HEADER = WAL_HEADER_PREFIX;

pub const WalError = error{
    InvalidHeader,
    WalCorruption,
    MissingFrame,
};

fn crc32Hex(bytes: []const u8) [8]u8 {
    const Crc32 = if (@hasDecl(std.hash.crc, "Crc32"))
        std.hash.crc.Crc32
    else
        @field(std.hash.crc, "CRC-32/ISO-HDLC");
    const sum = Crc32.hash(bytes);
    return std.fmt.bytesToHex(std.mem.toBytes(std.mem.nativeToBig(u32, sum)), .lower);
}

/// Best-effort parent-directory fsync after creating/appending a WAL file so
/// the directory entry is durable on POSIX. No-op on Windows and when the
/// parent cannot be opened. Not a multi-host durability claim.
/// Uses `Io.File.sync` on the directory fd (`posix.fsync` was removed in 0.17).
fn syncParentDirBestEffort(io: std.Io, path: []const u8) void {
    if (comptime builtin.os.tag == .windows) return;
    const parent = std.fs.path.dirname(path) orelse ".";
    var dir = std.Io.Dir.cwd().openDir(io, parent, .{}) catch return;
    defer dir.close(io);
    const as_file: std.Io.File = .{ .handle = dir.handle, .flags = .{ .nonblocking = false } };
    as_file.sync(io) catch |err| {
        // Best-effort: directory entry may still be durable after file.sync;
        // do not fail the WAL append if parent-dir sync is unsupported.
        std.log.debug("WAL parent-dir sync skipped: {s}", .{@errorName(err)});
    };
}

/// Append one already-encoded JSON record to the WAL at `path`. O(1): an
/// existing log is opened and the framed record is written at EOF via a
/// positional writer (no full-file read-rewrite); a missing log is created with
/// the header line first. The record is framed with a CRC32 of its bytes so
/// replay can detect corruption.
pub fn appendRecord(io: std.Io, allocator: std.mem.Allocator, path: []const u8, json: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const crc = crc32Hex(json);

    // Open the existing log directly and branch on FileNotFound for the
    // create path, folding the create-vs-append decision into the open itself.
    // This avoids a redundant `access()` probe per append and closes the
    // access()-then-open TOCTOU window.
    var file = cwd.openFile(io, path, .{ .mode = .read_write }) catch |err| switch (err) {
        error.FileNotFound => {
            // First write: create the log with the legacy header + this record.
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            try out.writer.writeAll(WAL_HEADER_PREFIX);
            try out.writer.writeAll("\n");
            try out.writer.writeAll(&crc);
            try out.writer.writeAll(" ");
            try out.writer.writeAll(json);
            try out.writer.writeAll("\n");
            try cwd.writeFile(io, .{ .sub_path = path, .data = out.written() });
            // Best-effort durability for the create path (writeFile then reopen+sync).
            var created = try cwd.openFile(io, path, .{ .mode = .read_write });
            defer created.close(io);
            try created.sync(io);
            syncParentDirBestEffort(io, path);
            return;
        },
        else => return err,
    };

    // Existing log: append only the new framed record at EOF via a single
    // positional write (pwrite) at the current size — O(1), no reread of prior
    // contents, and no seek (matches the `writePositionalAll` idiom used in
    // src/foundation/io, avoiding the std seekTo flush footgun).
    defer file.close(io);
    const end = (try file.stat(io)).size;

    var frame: std.Io.Writer.Allocating = .init(allocator);
    defer frame.deinit();
    try frame.writer.writeAll(&crc);
    try frame.writer.writeAll(" ");
    try frame.writer.writeAll(json);
    try frame.writer.writeAll("\n");
    try file.writePositionalAll(io, frame.written(), end);
    // Durability: flush data to stable storage before treating the append as
    // committed, then best-effort parent-dir fsync for the directory entry.
    try file.sync(io);
    syncParentDirBestEffort(io, path);
}

/// Append a key/value mutation record.
pub fn appendKv(io: std.Io, allocator: std.mem.Allocator, path: []const u8, key: []const u8, value: []const u8) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("kv");
    try w.objectField("key");
    try w.write(key);
    try w.objectField("value");
    try w.write(value);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append a conversation-block mutation record (with explicit timestamp so the
/// replayed SHA-256 chain hash reproduces the original exactly).
pub const BlockRecord = struct {
    profile: []const u8,
    query_id: u32,
    response_id: u32,
    metadata: []const u8,
    timestamp_ms: i64,
};

pub fn appendBlock(io: std.Io, allocator: std.mem.Allocator, path: []const u8, record: BlockRecord) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("block");
    try w.objectField("profile");
    try w.write(record.profile);
    try w.objectField("query_id");
    try w.write(record.query_id);
    try w.objectField("response_id");
    try w.write(record.response_id);
    try w.objectField("metadata");
    try w.write(record.metadata);
    try w.objectField("timestamp_ms");
    try w.write(record.timestamp_ms);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append a vector record. `id` must be the value the originating `putVector`
/// call assigned, and `values` the original (unpadded) components. The JSON
/// shape is byte-identical to the snapshot serializer (`persistence.serialize`)
/// so replay's `applyVector` validates id ordering and reconstructs the HNSW
/// index through the exact same path as checkpoint restore.
///
/// `Store.putVector` invokes this per-mutation (see `mod.zig`), so vector
/// writes are durably logged like block/kv/temporal mutations. The durable
/// Session keeps only a post-checkpoint delta WAL with absolute vector ids;
/// recovery folds that delta on top of the checkpoint store and preserves the
/// vector-id counter, so an absolute id in a post-checkpoint delta replays
/// cleanly through `applyVector` rather than failing its sequential check.
pub fn appendVector(io: std.Io, allocator: std.mem.Allocator, path: []const u8, id: u32, values: []const f32) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("vector");
    try w.objectField("id");
    try w.write(id);
    try w.objectField("values");
    try w.beginArray();
    for (values) |v| {
        try w.write(v);
    }
    try w.endArray();
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append a temporal timestamp record. Replay routes this through the snapshot
/// parser, so WAL and JSONL checkpoint restore keep the same validation path.
pub fn appendTemporalNode(io: std.Io, allocator: std.mem.Allocator, path: []const u8, id: u32, timestamp_ms: i64) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("temporal_node");
    try w.objectField("id");
    try w.write(id);
    try w.objectField("timestamp_ms");
    try w.write(timestamp_ms);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append one causal edge record for temporal/causal hybrid ranking recovery.
pub fn appendTemporalEdge(io: std.Io, allocator: std.mem.Allocator, path: []const u8, cause: u32, effect: u32) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("temporal_edge");
    try w.objectField("cause");
    try w.write(cause);
    try w.objectField("effect");
    try w.write(effect);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Verify every frame's CRC32 and return the number of records, without
/// building a Store. Use for a fast integrity check (`wdbx db verify`).
pub fn verify(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !usize {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);
    var count: usize = 0;
    var it = frameIterator(content);
    while (try it.next()) |_| count += 1;
    return count;
}

/// Whether a WAL file exists at `path`.
pub fn exists(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !bool {
    return fileExists(io, allocator, path);
}

fn fileExists(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !bool {
    _ = allocator;
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

/// Create a WAL tagged as a delta from checkpoint `base_epoch`, if one does not
/// already exist. NEVER truncates an existing WAL: a present log may hold an
/// un-checkpointed delta that must survive. Recovery compares this epoch against
/// the checkpoint's to decide merge (equal) vs discard (WAL older).
pub fn createWithEpoch(io: std.Io, allocator: std.mem.Allocator, path: []const u8, base_epoch: u64) !void {
    if (try fileExists(io, allocator, path)) return;
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    try buf.writer.print("{s} base_epoch={d}\n", .{ WAL_HEADER_PREFIX, base_epoch });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.written() });
}

/// Parse the checkpoint epoch this WAL is a delta from. Returns 0 for an absent
/// file or a legacy header without a `base_epoch=` token (full-history WAL).
pub fn readBaseEpoch(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !u64 {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer allocator.free(content);
    const nl = std.mem.indexOfScalar(u8, content, '\n') orelse content.len;
    const header = std.mem.trim(u8, content[0..nl], " \t\r");
    if (!std.mem.startsWith(u8, header, WAL_HEADER_PREFIX)) return error.InvalidHeader;
    const rest = std.mem.trim(u8, header[WAL_HEADER_PREFIX.len..], " \t\r");
    if (std.mem.startsWith(u8, rest, "base_epoch=")) {
        return std.fmt.parseInt(u64, rest["base_epoch=".len..], 10) catch return error.WalCorruption;
    }
    return 0;
}

/// Replay WAL frames ON TOP of an existing `store` (used by recovery to fold a
/// post-checkpoint delta into the recovered checkpoint). Applies each verified
/// record through the same `persistence_parse.applyLine` path as snapshot
/// restore — so vector-id continuity holds: a delta logged against a checkpoint
/// with K vectors replays cleanly onto a store whose counter is already at K+1.
/// Returns the number of frames applied; corruption propagates.
pub fn replayOnto(io: std.Io, allocator: std.mem.Allocator, path: []const u8, store: *wdbx_mod.Store) !usize {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);
    var count: usize = 0;
    var it = frameIterator(content);
    while (try it.next()) |json| {
        try persistence_parse.applyLine(allocator, store, json);
        count += 1;
    }
    return count;
}

/// Replay the WAL into a fresh Store. Reuses `persistence.deserialize` so the
/// apply semantics (id/timestamp/hash fidelity, range checks) are identical to
/// snapshot restore.
pub fn replay(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !wdbx_mod.Store {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);

    var body: std.Io.Writer.Allocating = .init(allocator);
    defer body.deinit();
    try body.writer.writeAll(persistence.HEADER);
    try body.writer.writeAll("\n");

    var it = frameIterator(content);
    while (try it.next()) |json| {
        try body.writer.writeAll(json);
        try body.writer.writeAll("\n");
    }

    return persistence.deserialize(allocator, body.written());
}

const FrameIterator = struct {
    lines: std.mem.SplitIterator(u8, .scalar),
    started: bool = false,

    /// Returns the next verified JSON payload, or null at end. Header must be
    /// the first line. Mid-log frames whose CRC does not match, or that lack the
    /// `<crc> <json>` shape, fail with `error.WalCorruption`. A malformed or
    /// CRC-mismatched frame that is the last non-empty line is treated as a torn
    /// tail (return null) so verified prefix frames remain recoverable.
    fn next(self: *FrameIterator) !?[]const u8 {
        if (!self.started) {
            const header = self.lines.next() orelse return error.InvalidHeader;
            if (!std.mem.startsWith(u8, std.mem.trim(u8, header, " \t\r"), WAL_HEADER_PREFIX)) return error.InvalidHeader;
            self.started = true;
        }
        while (self.lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0) continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse {
                if (hasLaterNonEmpty(self.lines)) return error.WalCorruption;
                return null; // torn incomplete line at EOF
            };
            const crc = line[0..sp];
            const json = line[sp + 1 ..];
            if (json.len == 0) {
                if (hasLaterNonEmpty(self.lines)) return error.WalCorruption;
                return null; // torn incomplete line at EOF
            }
            const expect = crc32Hex(json);
            // A well-shaped frame with a bad CRC is always corruption — including
            // when it is the last line (bit flip / overwrite). Only incomplete
            // shape above is soft-skipped as a torn append tail.
            if (!std.mem.eql(u8, crc, &expect)) return error.WalCorruption;
            return json;
        }
        return null;
    }
};

fn hasLaterNonEmpty(it: std.mem.SplitIterator(u8, .scalar)) bool {
    var look = it;
    while (look.next()) |raw| {
        if (std.mem.trim(u8, raw, " \t\r").len > 0) return true;
    }
    return false;
}

fn frameIterator(content: []const u8) FrameIterator {
    return .{ .lines = std.mem.splitScalar(u8, content, '\n') };
}

const deleteTestFileIfExists = test_helpers.deleteTestFileIfExists;

test "wal: append and replay reconstruct kv + block state" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-replay.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "agent:abbey", "trained");
    try appendKv(std.testing.io, allocator, path, "agent:aviva", "creative");
    try appendBlock(std.testing.io, allocator, path, .{ .profile = "abbey", .query_id = 1, .response_id = 2, .metadata = "{\"turn\":1}", .timestamp_ms = 1000 });
    try appendBlock(std.testing.io, allocator, path, .{ .profile = "aviva", .query_id = 3, .response_id = 4, .metadata = "{\"turn\":2}", .timestamp_ms = 2000 });
    try appendTemporalNode(std.testing.io, allocator, path, 1, 1000);
    try appendTemporalNode(std.testing.io, allocator, path, 2, 2000);
    try appendTemporalEdge(std.testing.io, allocator, path, 1, 2);

    try std.testing.expectEqual(@as(usize, 7), try verify(std.testing.io, allocator, path));

    var store = try replay(std.testing.io, allocator, path);
    defer store.deinit();

    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expectEqualStrings("trained", store.get("agent:abbey").?);
    try std.testing.expectEqual(@as(usize, 2), store.blockCount());
    try std.testing.expect(store.verifyBlocks());
    const last = store.lastBlock().?;
    try std.testing.expectEqual(@as(i64, 2000), last.timestamp_ms);
    try std.testing.expectEqual(@as(usize, 2), store.temporalNodeCount());
    try std.testing.expectEqual(@as(usize, 1), store.temporalEdgeCount());
    try std.testing.expectEqual(@as(?i64, 1000), store.temporalTimestamp(1));
}

test "wal: detects a flipped byte in a framed record" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-corrupt.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "k1", "v1");

    // Flip a byte inside the JSON payload; CRC must reject it on replay.
    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    // Target the payload key "k1" (the header version string also contains
    // "v1", so flip a byte that only exists inside a framed record).
    const idx = std.mem.indexOf(u8, content, "k1").?;
    content[idx] = 'X';
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = content });

    try std.testing.expectError(error.WalCorruption, replay(std.testing.io, allocator, path));
    try std.testing.expectError(error.WalCorruption, verify(std.testing.io, allocator, path));
}

test "wal: torn last frame is skipped; verified prefix still replays" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-torn-tail.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "keep", "yes");
    try appendKv(std.testing.io, allocator, path, "also", "kept");

    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    // Append an incomplete last line (no trailing newline, truncated JSON) —
    // the crash-mid-append case. Prefix frames must still verify/replay.
    var torn: std.Io.Writer.Allocating = .init(allocator);
    defer torn.deinit();
    try torn.writer.writeAll(content);
    // Incomplete shape (no space-separated CRC/json) — not a well-formed frame.
    try torn.writer.writeAll("DEADBEEFincomplete");
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = torn.written() });

    try std.testing.expectEqual(@as(usize, 2), try verify(std.testing.io, allocator, path));

    var store = try replay(std.testing.io, allocator, path);
    defer store.deinit();
    try std.testing.expectEqualStrings("yes", store.get("keep").?);
    try std.testing.expectEqualStrings("kept", store.get("also").?);
    try std.testing.expect(store.get("lost") == null);
}

test "wal: mid-log corruption after a good frame still hard-fails" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-mid-corrupt.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "a", "1");
    try appendKv(std.testing.io, allocator, path, "b", "2");
    try appendKv(std.testing.io, allocator, path, "c", "3");

    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    // Corrupt the middle frame's key while leaving a later good-looking line.
    const idx = std.mem.indexOf(u8, content, "\"b\"").?;
    content[idx + 1] = 'X'; // flip inside "b"
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = content });

    try std.testing.expectError(error.WalCorruption, verify(std.testing.io, allocator, path));
}

test "wal: rejects a missing or wrong header" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-badheader.wal";
    defer deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = "garbage\n{\"type\":\"kv\"}\n" });
    try std.testing.expectError(error.InvalidHeader, replay(std.testing.io, allocator, path));
}

test "wal: vector records survive append, verify, and replay" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-vector.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    // Mirror what Store.putVector logs: counter-assigned ids starting at 1, in
    // insertion order, with the original (unpadded) components.
    try appendVector(std.testing.io, allocator, path, 1, &.{ 1.0, 0.0, 0.0, 0.0 });
    try appendVector(std.testing.io, allocator, path, 2, &.{ 0.0, 1.0, 0.0, 0.0 });
    try appendKv(std.testing.io, allocator, path, "agent:abbey", "trained");

    try std.testing.expectEqual(@as(usize, 3), try verify(std.testing.io, allocator, path));

    var store = try replay(std.testing.io, allocator, path);
    defer store.deinit();

    // Vectors are durable through the WAL alone, with the kv mutation intact.
    try std.testing.expectEqual(@as(usize, 2), store.vectorCount());
    try std.testing.expectEqualStrings("trained", store.get("agent:abbey").?);

    // The replayed HNSW index ranks the matching vector first.
    const results = try store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 1);
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(u32, 1), results[0].id);
    try std.testing.expect(results[0].score > 0.99);
}

test "wal: createWithEpoch writes a parseable base_epoch and does not truncate" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-epoch.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try createWithEpoch(std.testing.io, allocator, path, 42);
    try std.testing.expectEqual(@as(u64, 42), try readBaseEpoch(std.testing.io, allocator, path));

    // A record then a second createWithEpoch must NOT clobber the existing WAL.
    try appendKv(std.testing.io, allocator, path, "k", "v");
    try createWithEpoch(std.testing.io, allocator, path, 99);
    try std.testing.expectEqual(@as(u64, 42), try readBaseEpoch(std.testing.io, allocator, path));
    try std.testing.expectEqual(@as(usize, 1), try verify(std.testing.io, allocator, path));
}

test "wal: readBaseEpoch is 0 for legacy header and absent file" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-legacy.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    // Absent file -> 0.
    try std.testing.expectEqual(@as(u64, 0), try readBaseEpoch(std.testing.io, allocator, path));
    // Legacy header (appendRecord creates it without an epoch token) -> 0.
    try appendKv(std.testing.io, allocator, path, "k", "v");
    try std.testing.expectEqual(@as(u64, 0), try readBaseEpoch(std.testing.io, allocator, path));
}

test "wal: replayOnto folds a delta onto an existing store" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-onto.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    // Baseline store with 1 block + 2 vectors (counter now at 3).
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();
    _ = try store.appendBlock("abbey", 0, 0, "{\"t\":1}");
    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });

    // Delta WAL: a new block and the next vector (absolute id 3) — exactly what
    // putVector/appendBlock would log against this baseline.
    try createWithEpoch(std.testing.io, allocator, path, 7);
    try appendBlock(std.testing.io, allocator, path, .{ .profile = "aviva", .query_id = 0, .response_id = 0, .metadata = "{\"t\":2}", .timestamp_ms = 2000 });
    try appendVector(std.testing.io, allocator, path, 3, &.{ 0.0, 0.0, 1.0, 0.0 });

    const applied = try replayOnto(std.testing.io, allocator, path, &store);
    try std.testing.expectEqual(@as(usize, 2), applied);
    // Continuity holds: the delta merged without CorruptVectorId.
    try std.testing.expectEqual(@as(usize, 2), store.blockCount());
    try std.testing.expectEqual(@as(usize, 3), store.vectorCount());
}

test "wal: a malformed base_epoch token is rejected as corruption" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-badepoch.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = WAL_HEADER_PREFIX ++ " base_epoch=NaN\n" });
    try std.testing.expectError(error.WalCorruption, readBaseEpoch(std.testing.io, allocator, path));
}

test "wal: a frame missing the crc/json separator at EOF is a torn tail" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-nosep.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);
    // Valid header, then a last line lacking the "<crc> <json>" space split —
    // treated as a crash-torn append tail (skip), not mid-log corruption.
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = WAL_HEADER_PREFIX ++ "\nnospaceframe\n" });
    try std.testing.expectEqual(@as(usize, 0), try verify(std.testing.io, allocator, path));
}

test "wal: a frame missing the crc/json separator mid-log is corruption" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-nosep-mid.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "before", "ok");
    try appendKv(std.testing.io, allocator, path, "after", "also");
    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024));
    defer allocator.free(content);

    // Insert a separator-less line between the two good frames.
    var lines = std.mem.splitScalar(u8, content, '\n');
    const header = lines.next().?;
    const frame1 = lines.next().?;
    const frame2 = lines.next().?;
    var injected: std.Io.Writer.Allocating = .init(allocator);
    defer injected.deinit();
    try injected.writer.print("{s}\n{s}\nnospaceframe\n{s}\n", .{ header, frame1, frame2 });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = injected.written() });

    try std.testing.expectError(error.WalCorruption, verify(std.testing.io, allocator, path));
}

test {
    std.testing.refAllDecls(@This());
}
