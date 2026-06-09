const std = @import("std");
const features = @import("../../features/mod.zig");
const foundation_time = @import("../../foundation/time.zig");

const wdbx = features.wdbx;

fn deleteWalIfPresent(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !void {
    const wp = try wdbx.recovery.walPath(allocator, path);
    defer allocator.free(wp);
    std.Io.Dir.cwd().deleteFile(io, wp) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

fn checkpointStore(io: std.Io, allocator: std.mem.Allocator, path: []const u8, store: *const wdbx.Store) !void {
    var segment_store = wdbx.segments.SegmentStore.init(allocator, io, path);
    const new_epoch = try segment_store.flush(store);

    // Compatibility mirror for existing tooling that still opens the monolithic
    // snapshot path directly. Runtime open/verify prefers the segment manifest.
    try wdbx.persistence.saveToPath(io, allocator, store, path);

    // Fold the WAL into the new checkpoint and start a fresh delta tagged to the
    // new epoch (delta semantics, matching the durable Session). Recovery merges
    // a same-epoch WAL on top of the checkpoint and discards an older one.
    const wp = try wdbx.recovery.walPath(allocator, path);
    defer allocator.free(wp);
    std.Io.Dir.cwd().deleteFile(io, wp) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    try wdbx.wal.createWithEpoch(io, allocator, wp, new_epoch);
}

fn openRecovered(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !wdbx.recovery.Opened {
    return wdbx.recovery.open(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

pub fn initDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var segment_store = wdbx.segments.SegmentStore.init(allocator, io, path);
    try segment_store.reset();
    try checkpointStore(io, allocator, path, &store);
    try deleteWalIfPresent(io, allocator, path);
    std.debug.print("initialized empty WDBX segment checkpoint at {s}\n", .{path});
    return 0;
}

pub fn verifyDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var opened = wdbx.recovery.openCheckpoint(io, allocator, path) catch |err| {
        std.debug.print("verify FAILED: checkpoint {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer opened.store.deinit();

    const blocks_ok = opened.store.verifyBlocks();
    const s = opened.store.stats();
    std.debug.print(
        "checkpoint OK: source={s} epoch={d} kv={d} vectors={d} blocks={d} spatial={d} temporal_nodes={d} temporal_edges={d} chain_valid={any}\n",
        .{ @tagName(opened.source), opened.checkpoint_epoch, s.kv_entries, s.vectors, s.blocks, s.spatial_records, s.temporal_nodes, s.temporal_edges, blocks_ok },
    );

    const wp = try wdbx.recovery.walPath(allocator, path);
    defer allocator.free(wp);
    if (!try wdbx.wal.exists(io, allocator, wp)) return if (blocks_ok) 0 else 1;

    // The WAL is a post-checkpoint delta, not a divergent full history. First
    // verify per-frame CRC integrity (corruption/tamper detection)...
    const frames = wdbx.wal.verify(io, allocator, wp) catch |err| {
        std.debug.print("WAL verify FAILED: {s}: {s}\n", .{ wp, @errorName(err) });
        return 1;
    };
    const wal_base = wdbx.wal.readBaseEpoch(io, allocator, wp) catch |err| {
        std.debug.print("WAL verify FAILED: {s}: {s}\n", .{ wp, @errorName(err) });
        return 1;
    };

    // ...then confirm the delta folds cleanly onto a fresh checkpoint copy and
    // the merged chain is intact. A WAL whose epoch predates the checkpoint is a
    // superseded log that recovery will discard — reported, not an error.
    if (wal_base != opened.checkpoint_epoch) {
        std.debug.print("WAL note: frames={d} base_epoch={d} predates checkpoint epoch={d}; discarded on recovery\n", .{ frames, wal_base, opened.checkpoint_epoch });
        return if (blocks_ok) 0 else 1;
    }

    var merged = wdbx.recovery.openCheckpoint(io, allocator, path) catch {
        std.debug.print("verify FAILED: checkpoint reload {s}\n", .{path});
        return 1;
    };
    defer merged.store.deinit();
    _ = wdbx.wal.replayOnto(io, allocator, wp, &merged.store) catch |err| {
        std.debug.print("WAL replay FAILED: {s}: {s}\n", .{ wp, @errorName(err) });
        return 1;
    };
    const merged_ok = merged.store.verifyBlocks();
    std.debug.print("WAL OK: frames={d} merged_blocks={d} merged_chain_valid={any}\n", .{ frames, merged.store.blockCount(), merged_ok });
    return if (blocks_ok and merged_ok) 0 else 1;
}

pub fn blockInsert(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profile: []const u8, metadata: []const u8) anyerror!u8 {
    var opened = try openRecovered(io, allocator, path);
    defer opened.store.deinit();

    _ = try opened.store.appendBlock(profile, 0, 0, metadata);
    const last = opened.store.lastBlock().?;

    const wp = try wdbx.recovery.walPath(allocator, path);
    defer allocator.free(wp);
    // Ensure the WAL is tagged to the current checkpoint epoch before appending.
    // If recovery discarded a stale WAL, a bare appendBlock would create a
    // legacy (base_epoch=0) header; a crash before the next checkpoint would
    // then make recovery discard this block as superseded. Mirrors Session.openAt.
    try wdbx.wal.createWithEpoch(io, allocator, wp, opened.checkpoint_epoch);
    try wdbx.wal.appendBlock(io, allocator, wp, profile, 0, 0, metadata, last.timestamp_ms);
    try checkpointStore(io, allocator, path, &opened.store);

    std.debug.print("appended block: profile={s} blocks={d} hash={s}\n", .{ profile, opened.store.blockCount(), std.fmt.bytesToHex(last.id, .lower) });
    return 0;
}

pub fn blockGet(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var opened = openRecovered(io, allocator, path) catch return 1;
    defer opened.store.deinit();

    const last = opened.store.lastBlock() orelse {
        std.debug.print("no blocks in {s}\n", .{path});
        return 0;
    };
    std.debug.print(
        "block: profile={s} query_id={d} response_id={d} timestamp_ms={d}\n  hash={s}\n  metadata={s}\n",
        .{ last.profile, last.query_id, last.response_id, last.timestamp_ms, std.fmt.bytesToHex(last.id, .lower), last.metadata },
    );
    return 0;
}

/// `abi wdbx query <path> [text]`. With no text, prints the store-stats
/// manifest (unchanged legacy behavior). With text, embeds the query and
/// returns hybrid-ranked (semantic × temporal × causal × persona) results
/// over the recovered store's vectors.
pub fn query(io: std.Io, allocator: std.mem.Allocator, path: []const u8, text: ?[]const u8) anyerror!u8 {
    var opened = openRecovered(io, allocator, path) catch return 1;
    defer opened.store.deinit();

    const q = text orelse {
        const manifest = try opened.store.exportManifest(allocator);
        defer allocator.free(manifest);
        std.debug.print("{s}\n", .{manifest});
        return 0;
    };

    const store = &opened.store;
    const stats = store.stats();
    if (stats.vectors == 0) {
        std.debug.print("no vectors in {s}; nothing to rank (populate with `abi complete`)\n", .{path});
        return 0;
    }

    const query_vec = features.ai.textEmbedding(q);
    const scorer = wdbx.temporal.HybridScorer{ .now_ms = foundation_time.unixMs(), .half_life_ms = 24 * 60 * 60 * 1000 };
    // Anchor causal proximity on the most recent vector when present.
    const focus_id: u32 = if (stats.next_vector_id > 1) stats.next_vector_id - 1 else 1;
    const ranked = try wdbx.retrieval.hybridSearch(allocator, store, &query_vec, 10, &store.temporal_graph, scorer, focus_id, constPersona);
    defer allocator.free(ranked);

    if (ranked.len == 0) {
        std.debug.print("no matches for \"{s}\"\n", .{q});
        return 0;
    }
    std.debug.print("query \"{s}\" -> {d} ranked result(s) over {d} vectors (ranking=hybrid):\n", .{ q, ranked.len, stats.vectors });
    for (ranked, 0..) |r, i| {
        std.debug.print(
            "  {d}. vector_id={d} score={d:.4} semantic={d:.4} temporal={d:.4} causal={d:.4} persona={d:.4}\n",
            .{ i + 1, r.id, r.score, r.components.semantic, r.components.temporal, r.components.causal, r.components.persona },
        );
    }
    return 0;
}

/// Neutral persona weight for the path-addressed CLI query (no conversation
/// persona context, unlike the MCP tool which weights by routed profile).
fn constPersona(id: u32) f32 {
    _ = id;
    return 0.5;
}

test {
    std.testing.refAllDecls(@This());
}
