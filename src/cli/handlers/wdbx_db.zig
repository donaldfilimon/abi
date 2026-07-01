const std = @import("std");
const build_options = @import("build_options");
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
    // Shared with the durable Session so both keep identical checkpoint semantics
    // (segment flush + monolithic mirror + delta-WAL reset tagged to the new epoch).
    _ = try wdbx.durable_store.checkpointAt(io, allocator, path, store);
}

fn openRecovered(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !wdbx.recovery.Opened {
    return wdbx.recovery.open(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// `abi wdbx db init <path>`: initialize an empty WDBX segment checkpoint at
/// `path`, resetting the segment store and discarding any stale WAL. Returns the
/// process exit code.
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

/// `abi wdbx db verify <path>`: verify the checkpoint at `path` and, when a
/// post-checkpoint delta WAL is present, verify per-frame CRC integrity and that
/// the delta folds cleanly onto a fresh checkpoint copy. Returns 0 when all
/// checks pass and 1 on any verification failure.
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

/// `abi wdbx block insert <path> <profile> <metadata>`: append a block to the
/// recovered store at `path`, mirror it to the WAL tagged to the current
/// checkpoint epoch, and checkpoint. Returns the process exit code.
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

/// `abi wdbx block get <path>`: print the most recent block in the recovered
/// store at `path` (profile, query/response ids, timestamp, hash, and metadata),
/// or a notice when the store is empty. Returns the process exit code.
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

/// `abi wdbx query <path> [text] [persona]`. With no text, prints the
/// store-stats manifest (unchanged legacy behavior). With text, embeds the query
/// and returns hybrid-ranked (semantic × temporal × causal × persona) results.
/// With a persona, results are ISOLATED to that persona's memories rather than
/// blended — multi-persona memory routing over the recovered store's vectors.
pub fn query(io: std.Io, allocator: std.mem.Allocator, path: []const u8, text: ?[]const u8, persona: ?[]const u8) anyerror!u8 {
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

    // One pass over the chain so persona resolution (scoped filtering below and
    // the result display) is O(1) per vector instead of an O(blocks) chain scan.
    var persona_cache = try buildPersonaCache(allocator, store);
    defer persona_cache.deinit(allocator);

    const ranked = if (persona) |p| blk: {
        // Isolation: only this persona's memories, resolved via the durable
        // profile tags / block-backed cache — keeps persona semantics out of the
        // storage layer.
        const scope = PersonaScope{ .store = store, .cache = &persona_cache, .target = p };
        break :blk try wdbx.retrieval.hybridSearchScoped(allocator, store, &query_vec, 10, &store.temporal_graph, scorer, focus_id, &scope, personaScopeKeep);
    } else try wdbx.retrieval.hybridSearch(allocator, store, &query_vec, 10, &store.temporal_graph, scorer, focus_id, constPersona);
    defer allocator.free(ranked);

    const scope_label = persona orelse "all";
    if (ranked.len == 0) {
        std.debug.print("no matches for \"{s}\" (persona={s})\n", .{ q, scope_label });
        return 0;
    }
    std.debug.print("query \"{s}\" persona={s} -> {d} ranked result(s) over {d} vectors (ranking=hybrid):\n", .{ q, scope_label, ranked.len, stats.vectors });
    for (ranked, 0..) |r, i| {
        std.debug.print(
            "  {d}. vector_id={d} persona={s} score={d:.4} semantic={d:.4} temporal={d:.4} causal={d:.4} persona_w={d:.4}\n",
            .{ i + 1, r.id, resolvePersona(store, &persona_cache, r.id), r.score, r.components.semantic, r.components.temporal, r.components.causal, r.components.persona },
        );
    }
    return 0;
}

/// Neutral persona weight for the blended (non-scoped) path-addressed CLI query.
fn constPersona(id: u32) f32 {
    _ = id;
    return 0.5;
}

/// vector_id -> persona label, built once per query from the conversation chain.
const PersonaCache = std.AutoHashMapUnmanaged(u32, []const u8);

/// Build the persona cache in a single pass over the chain. Each block records
/// the profile for its query and response vectors, so this resolves BOTH without
/// a per-vector scan — turning the prior O(results · blocks) display/filter cost
/// into O(blocks) once. Caller owns and deinits the map. Profile slices are
/// borrowed from the store and valid for its lifetime.
fn buildPersonaCache(allocator: std.mem.Allocator, store: *const wdbx.Store) !PersonaCache {
    var cache: PersonaCache = .{};
    errdefer cache.deinit(allocator);
    var it = store.chain.iterator();
    defer store.chain.releaseIterator();
    while (it.next()) |node| {
        try cache.put(allocator, node.data.query_id, node.data.profile);
        try cache.put(allocator, node.data.response_id, node.data.profile);
    }
    return cache;
}

/// Resolve a vector's persona in O(1): a seeded `wdbx:profile:{id}` KV tag first
/// (MCP persona prototypes), else the block-backed cache (covers query AND
/// response vectors of every completion), else "unknown".
fn resolvePersona(store: *const wdbx.Store, cache: *const PersonaCache, id: u32) []const u8 {
    var key_buf: [64]u8 = undefined;
    if (std.fmt.bufPrint(&key_buf, "wdbx:profile:{d}", .{id})) |k| {
        if (store.get(k)) |label| return label;
    } else |_| {}
    return cache.get(id) orelse "unknown";
}

const PersonaScope = struct {
    store: *const wdbx.Store,
    cache: *const PersonaCache,
    target: []const u8,
};

fn personaScopeKeep(ctx: *const anyopaque, id: u32) bool {
    const scope: *const PersonaScope = @ptrCast(@alignCast(ctx));
    return std.mem.eql(u8, resolvePersona(scope.store, scope.cache, id), scope.target);
}

test "resolvePersona resolves query AND response vectors via the block cache" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const qid = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    const rid = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    _ = try store.appendBlock("abbey", qid, rid, "{\"profile\":\"abbey\"}");

    var cache = try buildPersonaCache(allocator, &store);
    defer cache.deinit(allocator);

    // No wdbx:profile KV tags set — both ids resolve to the persona via the
    // block-backed cache (response vector included). Unknown id -> "unknown".
    try std.testing.expectEqualStrings("abbey", resolvePersona(&store, &cache, qid));
    try std.testing.expectEqualStrings("abbey", resolvePersona(&store, &cache, rid));
    try std.testing.expectEqualStrings("unknown", resolvePersona(&store, &cache, 9999));
}

test {
    std.testing.refAllDecls(@This());
}
