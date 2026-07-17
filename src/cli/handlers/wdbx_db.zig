const std = @import("std");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");
const foundation_time = @import("../../foundation/time.zig");
const foundation_json = @import("../../foundation/json.zig");

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

/// `abi wdbx db compact <path> [keep]`: retain the newest `keep` segment
/// checkpoints and reclaim older manifest-listed segments. This bounds disk use
/// for larger stores without touching the latest checkpoint or sidecar WAL.
pub fn compactDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8, keep_latest: usize) anyerror!u8 {
    var segment_store = wdbx.segments.SegmentStore.init(allocator, io, path);
    const result = segment_store.compactRetainingLatest(keep_latest) catch |err| switch (err) {
        wdbx.segments.SegmentError.InvalidCompactionPolicy => {
            std.debug.print("compact FAILED: keep must be >= 1\n", .{});
            return 1;
        },
        else => {
            std.debug.print("compact FAILED: {s}: {s}\n", .{ path, @errorName(err) });
            return 1;
        },
    };

    std.debug.print(
        "compacted WDBX segments: path={s} keep_latest={d} before={d} after={d} deleted={d}",
        .{ path, result.keep_latest, result.before, result.after, result.deleted },
    );
    if (result.latest_epoch) |latest| {
        std.debug.print(" latest_epoch={d}", .{latest});
    } else {
        std.debug.print(" latest_epoch=none", .{});
    }
    if (result.watermark_epoch) |watermark| {
        std.debug.print(" watermark_epoch={d}", .{watermark});
    }
    std.debug.print("\n", .{});
    return 0;
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
    try wdbx.wal.appendBlock(io, allocator, wp, .{ .profile = profile, .query_id = 0, .response_id = 0, .metadata = metadata, .timestamp_ms = last.timestamp_ms });
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

/// Options for `abi wdbx query`. Positionals (`text`, `persona`) remain for
/// compatibility; flags (`--text`, `--persona`, `--limit`, `--json`) are preferred.
pub const QueryOptions = struct {
    path: []const u8,
    text: ?[]const u8 = null,
    persona: ?[]const u8 = null,
    limit: usize = 10,
    json: bool = false,
};

pub const ParseQueryError = error{Usage};

/// Parse `abi wdbx query` args after the `query` token.
/// Accepts: `<path> [text] [persona] [--limit N] [--json] [--text T] [--persona P]`
pub fn parseQueryArgs(args: []const []const u8) ParseQueryError!QueryOptions {
    if (args.len == 0) return error.Usage;
    var opts = QueryOptions{ .path = args[0] };
    var positionals: [2]?[]const u8 = .{ null, null };
    var positional_count: usize = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const tok = args[i];
        if (std.mem.eql(u8, tok, "--json")) {
            opts.json = true;
        } else if (std.mem.eql(u8, tok, "--limit")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            opts.limit = std.fmt.parseInt(usize, args[i], 10) catch return error.Usage;
            if (opts.limit == 0) return error.Usage;
        } else if (std.mem.eql(u8, tok, "--text")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            opts.text = args[i];
        } else if (std.mem.eql(u8, tok, "--persona")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            opts.persona = args[i];
        } else if (std.mem.startsWith(u8, tok, "--")) {
            return error.Usage;
        } else {
            if (positional_count >= positionals.len) return error.Usage;
            positionals[positional_count] = tok;
            positional_count += 1;
        }
    }
    if (opts.text == null and positionals[0] != null) opts.text = positionals[0];
    if (opts.persona == null and positionals[1] != null) opts.persona = positionals[1];
    // Legacy: single positional after path with --persona flag already set is text.
    if (opts.text == null and opts.persona != null and positionals[0] != null and positionals[1] == null) {
        opts.text = positionals[0];
    }
    return opts;
}

/// `abi wdbx query <path> [text] [persona] [--limit N] [--json] ...`.
/// With no text, prints the store-stats manifest (or a JSON wrapper when `--json`).
/// With text, embeds the query and returns hybrid-ranked
/// (semantic × temporal × causal × persona) results. With a persona, results are
/// ISOLATED to that persona's memories rather than blended.
pub fn query(io: std.Io, allocator: std.mem.Allocator, opts: QueryOptions) anyerror!u8 {
    var opened = openRecovered(io, allocator, opts.path) catch return 1;
    defer opened.store.deinit();

    const q = opts.text orelse {
        const manifest = try opened.store.exportManifest(allocator);
        defer allocator.free(manifest);
        if (opts.json) {
            const path_json = try jsonStringAlloc(allocator, opts.path);
            defer allocator.free(path_json);
            std.debug.print("{{\"path\":{s},\"mode\":\"stats\",\"ranking\":null,\"manifest\":{s}}}\n", .{ path_json, manifest });
        } else {
            std.debug.print("{s}\n", .{manifest});
        }
        return 0;
    };

    return queryWithText(allocator, &opened.store, opts, q);
}

fn queryWithText(allocator: std.mem.Allocator, store: *wdbx.Store, opts: QueryOptions, q: []const u8) anyerror!u8 {
    const stats = store.stats();
    if (stats.vectors == 0) {
        if (opts.json) {
            const path_json = try jsonStringAlloc(allocator, opts.path);
            defer allocator.free(path_json);
            const q_json = try jsonStringAlloc(allocator, q);
            defer allocator.free(q_json);
            std.debug.print("{{\"path\":{s},\"query\":{s},\"persona\":\"all\",\"ranking\":\"hybrid\",\"limit\":{d},\"vectors\":0,\"results\":[]}}\n", .{
                path_json, q_json, opts.limit,
            });
        } else {
            std.debug.print("no vectors in {s}; nothing to rank (populate with `abi complete`)\n", .{opts.path});
        }
        return 0;
    }

    const query_vec = features.ai.textEmbedding(q);
    const scorer = wdbx.temporal.HybridScorer{ .now_ms = foundation_time.unixMs(), .half_life_ms = 24 * 60 * 60 * 1000 };
    const focus_id: u32 = if (stats.next_vector_id > 1) stats.next_vector_id - 1 else 1;

    var persona_cache = try buildPersonaCache(allocator, store);
    defer persona_cache.deinit(allocator);

    const ranked = if (opts.persona) |p| blk: {
        const scope = PersonaScope{ .store = store, .cache = &persona_cache, .target = p };
        break :blk try wdbx.retrieval.hybridSearchScoped(allocator, store, &query_vec, opts.limit, &store.temporal_graph, scorer, focus_id, &scope, personaScopeKeep);
    } else try wdbx.retrieval.hybridSearch(allocator, store, &query_vec, opts.limit, &store.temporal_graph, scorer, focus_id, constPersona);
    defer allocator.free(ranked);

    const scope_label = opts.persona orelse "all";
    if (opts.json) {
        return printQueryJson(allocator, opts, q, scope_label, store, &persona_cache, ranked, stats.vectors);
    }

    if (ranked.len == 0) {
        std.debug.print("no matches for \"{s}\" (persona={s})\n", .{ q, scope_label });
        return 0;
    }
    std.debug.print("query \"{s}\" persona={s} -> {d} ranked result(s) over {d} vectors (ranking=hybrid limit={d}):\n", .{
        q, scope_label, ranked.len, stats.vectors, opts.limit,
    });
    for (ranked, 0..) |r, i| {
        // Prefer RankedNode.vector (attached zero-copy view); fall back to getVector.
        const dims: usize = if (r.vector) |view| view.len else if (store.getVector(r.id)) |view| view.len else 0;
        std.debug.print(
            "  {d}. vector_id={d} persona={s} score={d:.4} semantic={d:.4} temporal={d:.4} causal={d:.4} persona_w={d:.4} dims={d}\n",
            .{ i + 1, r.id, resolvePersona(store, &persona_cache, r.id), r.score, r.components.semantic, r.components.temporal, r.components.causal, r.components.persona, dims },
        );
    }
    return 0;
}
const jsonStringAlloc = foundation_json.jsonStringAlloc;

fn printQueryJson(
    allocator: std.mem.Allocator,
    opts: QueryOptions,
    q: []const u8,
    scope_label: []const u8,
    store: *const wdbx.Store,
    persona_cache: *const PersonaCache,
    ranked: []const wdbx.temporal.RankedNode,
    vector_count: usize,
) !u8 {
    const path_json = try jsonStringAlloc(allocator, opts.path);
    defer allocator.free(path_json);
    const q_json = try jsonStringAlloc(allocator, q);
    defer allocator.free(q_json);
    const persona_json = try jsonStringAlloc(allocator, scope_label);
    defer allocator.free(persona_json);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;

    try w.print("{{\"path\":{s},\"query\":{s},\"persona\":{s},\"ranking\":\"hybrid\",\"limit\":{d},\"vectors\":{d},\"results\":[", .{
        path_json, q_json, persona_json, opts.limit, vector_count,
    });
    for (ranked, 0..) |r, i| {
        if (i > 0) try w.writeAll(",");
        const label = resolvePersona(store, persona_cache, r.id);
        const label_json = try jsonStringAlloc(allocator, label);
        defer allocator.free(label_json);
        // Prefer RankedNode.vector (attached zero-copy); fall back to getVector.
        const dims: usize = if (r.vector) |view| view.len else if (store.getVector(r.id)) |view| view.len else 0;
        try w.print("{{\"vector_id\":{d},\"persona\":{s},\"score\":{d:.6},\"dims\":{d},\"vector_view\":\"borrowed\",\"components\":{{\"semantic\":{d:.6},\"temporal\":{d:.6},\"causal\":{d:.6},\"persona\":{d:.6}}}}}", .{
            r.id,
            label_json,
            r.score,
            dims,
            r.components.semantic,
            r.components.temporal,
            r.components.causal,
            r.components.persona,
        });
    }
    try w.writeAll("]}\n");
    std.debug.print("{s}", .{out.written()});
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
    } else |_| std.log.warn("resolvePersona bufPrint overflow for id={d}", .{id});
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

test "parseQueryArgs accepts flags and legacy positionals" {
    const a = try parseQueryArgs(&.{"/tmp/store.jsonl"});
    try std.testing.expectEqualStrings("/tmp/store.jsonl", a.path);
    try std.testing.expect(a.text == null);
    try std.testing.expectEqual(@as(usize, 10), a.limit);
    try std.testing.expect(!a.json);

    const b = try parseQueryArgs(&.{ "/tmp/s", "hello", "abbey", "--limit", "3", "--json" });
    try std.testing.expectEqualStrings("hello", b.text.?);
    try std.testing.expectEqualStrings("abbey", b.persona.?);
    try std.testing.expectEqual(@as(usize, 3), b.limit);
    try std.testing.expect(b.json);

    const c = try parseQueryArgs(&.{ "/tmp/s", "--text", "hi", "--persona", "aviva", "--limit", "1" });
    try std.testing.expectEqualStrings("hi", c.text.?);
    try std.testing.expectEqualStrings("aviva", c.persona.?);
    try std.testing.expectEqual(@as(usize, 1), c.limit);

    try std.testing.expectError(error.Usage, parseQueryArgs(&.{}));
    try std.testing.expectError(error.Usage, parseQueryArgs(&.{ "/tmp/s", "--limit", "0" }));
    try std.testing.expectError(error.Usage, parseQueryArgs(&.{ "/tmp/s", "--unknown" }));
}

test {
    std.testing.refAllDecls(@This());
}
