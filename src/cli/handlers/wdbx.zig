const std = @import("std");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");
const db_commands = @import("wdbx_db.zig");
const runtime_commands = @import("wdbx_runtime.zig");
const test_helpers = @import("abi").foundation.test_helpers;
const usage_mod = @import("../usage.zig");
const env = @import("../../foundation/env.zig");

const wdbx = features.wdbx;

/// `abi wdbx <db|block|query|benchmark|cluster|compute|secure|gpu|api> ...`
///
/// A WDBX runtime control surface backed by the in-process store, JSONL
/// snapshot persistence, and the write-ahead log. Cluster/api report honest
/// single-node / loopback state; they do not claim distributed or REST
/// capabilities the repo does not yet implement.
pub fn handleWdbx(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    if (build_options.feat_wdbx) {
        return run(io, allocator, args);
    } else {
        std.debug.print("wdbx feature is disabled in this build (build with -Dfeat-wdbx=true)\n", .{});
        return 1;
    }
}

fn usageCode(code: u8) u8 {
    std.debug.print(
        \\abi wdbx <command> ...
        \\
        \\  db init <path>                 Create an empty WDBX snapshot
        \\  db verify <path>               Verify snapshot integrity + block chain (+ WAL if present)
        \\  db compact <path> [keep]       Retain the newest segment checkpoints (default keep=2)
        \\  block insert <path> <profile> <metadata>   Append a conversation block (snapshot + WAL)
        \\  block get <path>               Print the most recent block
        \\  query <path> [text] [persona] [--limit N] [--json]  Store stats; hybrid semantic search (flags: --text/--persona/--limit/--json)
        \\  benchmark [count]              Measure local insert/search timing
        \\  cluster status                 Report cluster topology (single-node default)
        \\  cluster demo [nodes]           Run in-process consensus: elect, replicate, fail over
        \\  cluster serve <port> [node] [host]  Serve consensus RPC. host defaults to 127.0.0.1; non-loopback requires ABI_WDBX_CLUSTER_TOKEN. Optional ABI_WDBX_CLUSTER_PEERS. Front multi-host with TLS/mTLS proxy — not production sharding.
        \\  compute info                   Report CPU/GPU/NPU/TPU backends and dynamic selection
        \\  secure demo                    Demonstrate int8 + Huffman/rANS entropy + autoencoder + HE/FHE demos (not SOTA / not audited)
        \\  gpu info                       Report GPU backend capabilities
        \\  api serve [port]               Serve the REST API (POST /insert /query /verify, GET /health /stats)
        \\
    , .{});
    return code;
}

fn usage() u8 {
    return usageCode(2);
}

fn help() u8 {
    return usageCode(0);
}

fn run(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    if (args.len < 3) return usage();
    const sub = args[2];
    if (usage_mod.isHelpToken(sub)) return help();

    if (std.mem.eql(u8, sub, "db")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxDbHelp();
        if (args.len < 4) return usage();
        const op = args[3];
        if (std.mem.eql(u8, op, "init")) {
            if (args.len != 5) return usage();
            return db_commands.initDb(io, allocator, args[4]);
        } else if (std.mem.eql(u8, op, "verify")) {
            if (args.len != 5) return usage();
            return db_commands.verifyDb(io, allocator, args[4]);
        } else if (std.mem.eql(u8, op, "compact")) {
            if (args.len < 5 or args.len > 6) return usage();
            const keep: usize = if (args.len == 6) (std.fmt.parseInt(usize, args[5], 10) catch return usage()) else 2;
            if (keep == 0) return usage();
            return db_commands.compactDb(io, allocator, args[4], keep);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "block")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxBlockHelp();
        if (args.len < 4) return usage();
        const op = args[3];
        if (std.mem.eql(u8, op, "insert")) {
            if (args.len != 7) return usage();
            return db_commands.blockInsert(io, allocator, args[4], args[5], args[6]);
        } else if (std.mem.eql(u8, op, "get")) {
            if (args.len != 5) return usage();
            return db_commands.blockGet(io, allocator, args[4]);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "query")) {
        if (args.len >= 4 and usage_mod.isHelpToken(args[3])) return wdbxQueryHelp();
        if (args.len < 4) return usage();
        const opts = db_commands.parseQueryArgs(args[3..]) catch return usage();
        return db_commands.query(io, allocator, opts);
    }

    if (std.mem.eql(u8, sub, "benchmark")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxBenchmarkHelp();
        if (args.len > 4) return usage();
        // A malformed count is a user typo, not a request for the default —
        // surface usage rather than silently running 256 inserts.
        const count: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch return usage()) else 256;
        return runtime_commands.benchmark(allocator, count);
    }

    if (std.mem.eql(u8, sub, "cluster")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxClusterHelp();
        if (args.len < 4) return usage();
        if (std.mem.eql(u8, args[3], "status")) {
            if (args.len != 4) return usage();
            // Report the real consensus state machine's view rather than a fixed
            // string: a single-node cluster that elects itself leader. Honest
            // single-process state — networked multi-host RPC is still Phase-2.
            var cluster = wdbx.cluster.Cluster.init(allocator, 1) catch |err| {
                std.debug.print("cluster status failed: {s}\n", .{@errorName(err)});
                return 1;
            };
            defer cluster.deinit();
            _ = cluster.startElection(0) catch |err|
                std.debug.print("cluster election failed: {s}\n", .{@errorName(err)});
            const line = cluster.statusLine(allocator) catch |err| {
                std.debug.print("cluster status failed: {s}\n", .{@errorName(err)});
                return 1;
            };
            defer allocator.free(line);
            std.debug.print(
                "cluster: {s}\n(single-node default; in-process multi-node consensus is available — run `abi wdbx cluster demo`)\n",
                .{line},
            );
            std.debug.print("north-star status: single-node/in-process (Phase 1 landed); multi-host production cluster Proposed (Phase 2) (docs/spec/wdbx-north-star.mdx §2/§3.5)\n", .{});
            return 0;
        } else if (std.mem.eql(u8, args[3], "demo")) {
            if (args.len > 5) return usage();
            const nodes: usize = if (args.len >= 5) (std.fmt.parseInt(usize, args[4], 10) catch return usage()) else 3;
            if (nodes < 1) return usage();
            return runtime_commands.clusterDemo(allocator, nodes);
        } else if (std.mem.eql(u8, args[3], "serve")) {
            if (args.len < 5 or args.len > 7) return usage();
            const port: u16 = std.fmt.parseInt(u16, args[4], 10) catch return usage();
            const node_id: u32 = if (args.len >= 6) (std.fmt.parseInt(u32, args[5], 10) catch return usage()) else 0;
            // Optional bind host (default loopback). Use "0.0.0.0" or a specific
            // routable IPv4/IPv6 to serve a multi-host cluster node.
            const host: []const u8 = if (args.len >= 7) args[6] else "127.0.0.1";
            return runtime_commands.clusterServe(io, allocator, host, port, node_id);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "compute")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxComputeHelp();
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return runtime_commands.computeInfo();
    }

    if (std.mem.eql(u8, sub, "secure")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxSecureHelp();
        if (args.len != 4 or !std.mem.eql(u8, args[3], "demo")) return usage();
        return runtime_commands.secureDemo(allocator);
    }

    if (std.mem.eql(u8, sub, "gpu")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxGpuHelp();
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return runtime_commands.gpuInfo(allocator);
    }

    if (std.mem.eql(u8, sub, "api")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return wdbxApiHelp();
        if (args.len < 4 or args.len > 5 or !std.mem.eql(u8, args[3], "serve")) return usage();
        // A bad port is a typo, not the default — fail loudly with usage.
        const port: u16 = if (args.len >= 5) (std.fmt.parseInt(u16, args[4], 10) catch return usage()) else 8081;
        return runtime_commands.serveApi(io, allocator, port);
    }

    return usage();
}

fn wdbxDbHelp() u8 {
    std.debug.print("usage: abi wdbx db <init|verify|compact> <path> [keep]\n\nManage segment checkpoints, WAL recovery, and snapshot integrity.\n", .{});
    return 0;
}

fn wdbxBlockHelp() u8 {
    std.debug.print("usage: abi wdbx block <insert|get> <path> ...\n\nAppend or inspect SHA-linked conversation blocks in a WDBX checkpoint.\n", .{});
    return 0;
}

fn wdbxQueryHelp() u8 {
    std.debug.print(
        \\usage: abi wdbx query <path> [text] [persona] [--limit N] [--json] [--text T] [--persona P]
        \\
        \\Print store stats (no text) or run hybrid semantic retrieval (semantic × temporal × causal × persona).
        \\Persona isolates results to that persona's memories. --limit defaults to 10. --json emits a machine-
        \\readable result list (ranking=hybrid) with borrowed vector dims (zero-copy getVector view).
        \\
    , .{});
    return 0;
}

fn wdbxBenchmarkHelp() u8 {
    std.debug.print("usage: abi wdbx benchmark [count]\n\nMeasure local insert/search timing for the in-process vector store.\n", .{});
    return 0;
}

fn wdbxClusterHelp() u8 {
    std.debug.print("usage: abi wdbx cluster <status|demo|serve> ...\n\nRun single-node status, in-process consensus demo, or authenticated cluster RPC serving.\n", .{});
    return 0;
}

fn wdbxComputeHelp() u8 {
    std.debug.print("usage: abi wdbx compute info\n\nReport CPU/GPU/NPU/TPU backend selection and fallback state.\n", .{});
    return 0;
}

fn wdbxSecureHelp() u8 {
    std.debug.print("usage: abi wdbx secure demo\n\nDemonstrate local compression plus reference homomorphic aggregation; not security-audited FHE.\n", .{});
    return 0;
}

fn wdbxGpuHelp() u8 {
    std.debug.print("usage: abi wdbx gpu info\n\nReport GPU backend capability and native-kernel status.\n", .{});
    return 0;
}

fn wdbxApiHelp() u8 {
    std.debug.print("usage: abi wdbx api serve [port]\n\nServe the loopback WDBX REST API.\n\nEnv:\n  {s}     Optional bearer token for request auth.\n  ABI_WDBX_TLS_CERT       Path to PEM certificate (TLS config / proxy deployment).\n  ABI_WDBX_TLS_KEY        Path to PEM private key (TLS config / proxy deployment).\n\nTLS: native termination is not linked; deploy behind nginx/Caddy/haproxy.\n", .{env.WDBX_REST_TOKEN_ENV});
    return 0;
}

fn cleanupTestDb(path: []const u8) void {
    var buf: [256]u8 = undefined;
    deleteTestFileIfExists(path);
    const wp = std.fmt.bufPrint(&buf, "{s}.wal", .{path}) catch return;
    deleteTestFileIfExists(wp);
    const manifest = std.fmt.bufPrint(&buf, "{s}.manifest", .{path}) catch return;
    deleteTestFileIfExists(manifest);
    var epoch: u64 = 0;
    while (epoch < 8) : (epoch += 1) {
        const segment = std.fmt.bufPrint(&buf, "{s}.seg.{d}.jsonl", .{ path, epoch }) catch continue;
        deleteTestFileIfExists(segment);
    }
}

const deleteTestFileIfExists = test_helpers.deleteTestFileIfExists;

test "wdbx handler usage returns non-zero without args" {
    // db/block require args; bare invocation prints usage with exit code 2.
    const args = [_][]const u8{ "abi", "wdbx" };
    const code = try handleWdbx(std.testing.io, std.testing.allocator, &args);
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "wdbx handler help returns success" {
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "db", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "cluster", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, std.testing.allocator, &.{ "abi", "wdbx", "api", "help" }));
}

test "wdbx rejects malformed numeric args with usage instead of silent defaults" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    // A typo'd count/port/node must surface usage (exit 2), not silently run
    // the default value — this guards the parseInt-catch-return-usage paths.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "benchmark", "notanumber" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "compact", "zig-out/nope.jsonl", "notanumber" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "compact", "zig-out/nope.jsonl", "0" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "cluster", "demo", "notanumber" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "api", "serve", "99999" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "cluster", "serve", "7000", "notanode" }));
    // Non-loopback binds must fail before serving when no cluster shared secret
    // is configured; this covers the host argument without blocking on a listener.
    try std.testing.expectEqual(@as(u8, 1), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "cluster", "serve", "7000", "0", "0.0.0.0" }));
    try std.testing.expectEqual(@as(u8, 1), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "cluster", "serve", "7000", "0", "not.a.valid.host" }));
}

test "wdbx db init + block insert + verify + query round-trip" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-rt.jsonl";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "aviva", "{\"turn\":2}" }));
    // Snapshot integrity + WAL replay consistency must both hold.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "get", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path }));
}

test "wdbx db verify detects a corrupted WAL frame" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-corrupt.jsonl";
    const wp = "zig-out/wdbx-cli-corrupt.jsonl.wal";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));

    // A valid post-checkpoint delta (one logged block) is NOT divergence — it
    // folds onto the checkpoint and verifies clean.
    try wdbx.wal.appendBlock(std.testing.io, allocator, wp, .{ .profile = "aviva", .query_id = 0, .response_id = 0, .metadata = "{\"turn\":2}", .timestamp_ms = 4242 });
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));

    // Flip a byte inside the WAL's framed JSON: verify must surface the
    // corruption (exit 1), not silently trust a damaged log.
    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, wp, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    const idx = std.mem.indexOf(u8, content, "turn").?;
    content[idx] = 'X';
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = wp, .data = content });
    try std.testing.expectEqual(@as(u8, 1), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
}

test "wdbx query runs scoped to a persona over a recovered store" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-persona.jsonl";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    // A store with two persona-tagged vectors (the durable convention written at
    // insert time): id 1 -> abbey, id 2 -> aviva.
    {
        var s = wdbx.Store.init(allocator);
        defer s.deinit();
        // Insert vectors in the same EMBED_DIM space the query path embeds into
        // (textEmbedding), so the recovered store's vector_dimensions matches the
        // query vector instead of tripping DimensionMismatch on a toy 4-dim vector.
        const v_abbey = features.ai.textEmbedding("abbey memory");
        _ = try s.putVector(&v_abbey);
        try s.store("wdbx:profile:1", "abbey");
        const v_aviva = features.ai.textEmbedding("aviva memory");
        _ = try s.putVector(&v_aviva);
        try s.store("wdbx:profile:2", "aviva");
        try wdbx.persistence.saveToPath(std.testing.io, allocator, &s, path);
    }

    // Persona-scoped query and unscoped query both succeed over the recovery.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path, "hello", "abbey" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path, "hello" }));
    // Flag form: --limit / --json / --persona must accept the same recovered store.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path, "--text", "hello", "--persona", "abbey", "--limit", "2", "--json" }));
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path, "--limit", "0" }));
}

test "wdbx CLI keeps the live WAL tagged to the checkpoint epoch" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-epoch.jsonl";
    const wp = "zig-out/wdbx-cli-epoch.jsonl.wal";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));

    // The live WAL must carry the current checkpoint epoch; otherwise a
    // crash-pending delta appended to it would be discarded as superseded.
    var cp = try wdbx.recovery.openCheckpoint(std.testing.io, allocator, path);
    defer cp.store.deinit();
    const wal_base = try wdbx.wal.readBaseEpoch(std.testing.io, allocator, wp);
    try std.testing.expectEqual(cp.checkpoint_epoch, wal_base);
}

test "wdbx runtime commands recover WAL-ahead state before reading or writing" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-recovery.jsonl";
    const wp = "zig-out/wdbx-cli-recovery.jsonl.wal";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));

    // Simulate a crash after the WAL append but before checkpointing: the WAL
    // holds a valid post-checkpoint delta. verify reports it clean (it folds
    // onto the checkpoint), and runtime commands recover the merged block.
    try wdbx.wal.appendBlock(std.testing.io, allocator, wp, .{ .profile = "aviva", .query_id = 0, .response_id = 0, .metadata = "{\"turn\":2}", .timestamp_ms = 4242 });
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "get", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path }));

    // A subsequent normal write starts from recovered state and checkpoints it.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abi", "{\"turn\":3}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));

    var recovered_snapshot = try wdbx.persistence.loadFromPath(std.testing.io, allocator, path);
    defer recovered_snapshot.deinit();
    try std.testing.expectEqual(@as(usize, 3), recovered_snapshot.blockCount());
}

test "wdbx runtime commands use segment checkpoints without snapshot mirror" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-segment-default.jsonl";
    const wp = "zig-out/wdbx-cli-segment-default.jsonl.wal";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));

    // Remove legacy mirrors; the segment manifest remains the runtime
    // checkpoint source.
    deleteTestFileIfExists(path);
    deleteTestFileIfExists(wp);

    var opened = try wdbx.recovery.open(std.testing.io, allocator, path);
    defer opened.store.deinit();
    try std.testing.expectEqual(wdbx.recovery.Source.segment, opened.source);
    try std.testing.expectEqual(@as(usize, 1), opened.store.blockCount());

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path }));
}

test "wdbx db compact reclaims older segment checkpoints and preserves recovery" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-compact.jsonl";
    defer cleanupTestDb(path);
    cleanupTestDb(path);

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "aviva", "{\"turn\":2}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abi", "{\"turn\":3}" }));

    var ss = wdbx.segments.SegmentStore.init(allocator, std.testing.io, path);
    {
        const before = try ss.activeEpochs(allocator);
        defer allocator.free(before);
        try std.testing.expect(before.len >= 4);
    }

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "compact", path, "1" }));
    {
        const after = try ss.activeEpochs(allocator);
        defer allocator.free(after);
        try std.testing.expectEqual(@as(usize, 1), after.len);
        try std.testing.expectEqual(@as(u64, 3), after[0]);
    }

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
    var opened = try wdbx.recovery.open(std.testing.io, allocator, path);
    defer opened.store.deinit();
    try std.testing.expectEqual(@as(usize, 3), opened.store.blockCount());
}

test {
    std.testing.refAllDecls(@This());
}
