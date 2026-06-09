const std = @import("std");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");
const foundation_time = @import("../../foundation/time.zig");

const wdbx = features.wdbx;

/// `abi wdbx <db|block|query|benchmark|cluster|gpu|api> ...`
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

fn usage() u8 {
    std.debug.print(
        \\abi wdbx <command> ...
        \\
        \\  db init <path>                 Create an empty WDBX snapshot
        \\  db verify <path>               Verify snapshot integrity + block chain (+ WAL if present)
        \\  block insert <path> <profile> <metadata>   Append a conversation block (snapshot + WAL)
        \\  block get <path>               Print the most recent block
        \\  query <path>                   Print store statistics
        \\  benchmark [count]              Measure local insert/search timing
        \\  cluster status                 Report cluster topology (single-node default)
        \\  cluster demo [nodes]           Run in-process consensus: elect, replicate, fail over
        \\  compute info                   Report CPU/GPU/NPU/TPU backends and dynamic selection
        \\  secure demo                    Demonstrate embedding compression + homomorphic aggregation
        \\  gpu info                       Report GPU backend capabilities
        \\  api serve [port]               Serve the REST API (POST /insert /query /verify, GET /health /stats)
        \\
    , .{});
    return 2;
}

fn walPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.wal", .{path});
}

fn run(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    if (args.len < 3) return usage();
    const sub = args[2];

    if (std.mem.eql(u8, sub, "db")) {
        if (args.len < 4) return usage();
        const op = args[3];
        if (std.mem.eql(u8, op, "init")) {
            if (args.len != 5) return usage();
            var store = wdbx.Store.init(allocator);
            defer store.deinit();
            try wdbx.persistence.saveToPath(io, allocator, &store, args[4]);
            std.debug.print("initialized empty WDBX snapshot at {s}\n", .{args[4]});
            return 0;
        } else if (std.mem.eql(u8, op, "verify")) {
            if (args.len != 5) return usage();
            return verifyDb(io, allocator, args[4]);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "block")) {
        if (args.len < 4) return usage();
        const op = args[3];
        if (std.mem.eql(u8, op, "insert")) {
            if (args.len != 7) return usage();
            return blockInsert(io, allocator, args[4], args[5], args[6]);
        } else if (std.mem.eql(u8, op, "get")) {
            if (args.len != 5) return usage();
            return blockGet(io, allocator, args[4]);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "query")) {
        if (args.len != 4) return usage();
        return query(io, allocator, args[3]);
    }

    if (std.mem.eql(u8, sub, "benchmark")) {
        const count: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 256) else 256;
        return benchmark(allocator, count);
    }

    if (std.mem.eql(u8, sub, "cluster")) {
        if (args.len < 4) return usage();
        if (std.mem.eql(u8, args[3], "status")) {
            std.debug.print(
                \\cluster: nodes=1 role=standalone replication=none quorum=n/a
                \\(single-node default; in-process multi-node consensus is available — run `abi wdbx cluster demo`)
                \\
            , .{});
            return 0;
        } else if (std.mem.eql(u8, args[3], "demo")) {
            const nodes: usize = if (args.len >= 5) (std.fmt.parseInt(usize, args[4], 10) catch 3) else 3;
            return clusterDemo(allocator, nodes);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "compute")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return computeInfo();
    }

    if (std.mem.eql(u8, sub, "secure")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "demo")) return usage();
        return secureDemo(allocator);
    }

    if (std.mem.eql(u8, sub, "gpu")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return gpuInfo(allocator);
    }

    if (std.mem.eql(u8, sub, "api")) {
        if (args.len < 4 or !std.mem.eql(u8, args[3], "serve")) return usage();
        const port: u16 = if (args.len >= 5) (std.fmt.parseInt(u16, args[4], 10) catch 8081) else 8081;
        var store = wdbx.Store.init(allocator);
        defer store.deinit();
        std.debug.print("serving WDBX REST on http://127.0.0.1:{d} (Ctrl-C to stop)\n", .{port});
        wdbx.rest.serve(allocator, io, &store, port) catch |err| {
            std.debug.print("REST server error: {s}\n", .{@errorName(err)});
            return 1;
        };
        return 0;
    }

    return usage();
}

fn verifyDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("verify FAILED: snapshot {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const blocks_ok = store.verifyBlocks();
    const s = store.stats();
    std.debug.print("snapshot OK: kv={d} vectors={d} blocks={d} spatial={d} chain_valid={any}\n", .{ s.kv_entries, s.vectors, s.blocks, s.spatial_records, blocks_ok });

    // If a sidecar WAL exists, replay it and cross-check the block count so a
    // divergence between durable log and snapshot is surfaced, not hidden.
    const wp = try walPath(allocator, path);
    defer allocator.free(wp);
    var wal_store = wdbx.wal.replay(io, allocator, wp) catch |err| switch (err) {
        error.FileNotFound => return if (blocks_ok) 0 else 1,
        else => {
            std.debug.print("WAL verify FAILED: {s}: {s}\n", .{ wp, @errorName(err) });
            return 1;
        },
    };
    defer wal_store.deinit();

    const wal_blocks = wal_store.blockCount();
    const consistent = wal_blocks == s.blocks and wal_store.verifyBlocks();
    std.debug.print("WAL replay OK: blocks={d} consistent_with_snapshot={any}\n", .{ wal_blocks, consistent });
    return if (blocks_ok and consistent) 0 else 1;
}

fn blockInsert(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profile: []const u8, metadata: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| switch (err) {
        error.FileNotFound => wdbx.Store.init(allocator),
        else => return err,
    };
    defer store.deinit();

    _ = try store.appendBlock(profile, 0, 0, metadata);
    const last = store.lastBlock().?;

    try wdbx.persistence.saveToPath(io, allocator, &store, path);

    // Mirror the mutation into the durable WAL with the block's exact timestamp
    // so replay reproduces the identical SHA-256 chain.
    const wp = try walPath(allocator, path);
    defer allocator.free(wp);
    try wdbx.wal.appendBlock(io, allocator, wp, profile, 0, 0, metadata, last.timestamp_ms);

    std.debug.print("appended block: profile={s} blocks={d} hash={s}\n", .{ profile, store.blockCount(), std.fmt.bytesToHex(last.id, .lower) });
    return 0;
}

fn blockGet(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const last = store.lastBlock() orelse {
        std.debug.print("no blocks in {s}\n", .{path});
        return 0;
    };
    std.debug.print(
        "block: profile={s} query_id={d} response_id={d} timestamp_ms={d}\n  hash={s}\n  metadata={s}\n",
        .{ last.profile, last.query_id, last.response_id, last.timestamp_ms, std.fmt.bytesToHex(last.id, .lower), last.metadata },
    );
    return 0;
}

fn query(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const manifest = try store.exportManifest(allocator);
    defer allocator.free(manifest);
    std.debug.print("{s}\n", .{manifest});
    return 0;
}

fn benchmark(allocator: std.mem.Allocator, count: usize) anyerror!u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const insert_start = foundation_time.monotonicNs();
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var v: [4]f32 = .{ 0, 0, 0, 0 };
        v[0] = @floatFromInt(i % 97);
        v[1] = @floatFromInt(i % 31);
        _ = try store.putVector(&v);
    }
    const insert_ns: u64 = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - insert_start));

    const search_start = foundation_time.monotonicNs();
    const queries: usize = @min(count, 200);
    var j: usize = 0;
    while (j < queries) : (j += 1) {
        const r = try store.search(&.{ 1, 0, 0, 0 }, 10);
        allocator.free(r);
    }
    const search_ns: u64 = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - search_start));

    const ins_avg = if (count > 0) insert_ns / count else 0;
    const srch_avg = if (queries > 0) search_ns / queries else 0;
    std.debug.print(
        \\benchmark (local, in-memory; not a published throughput claim):
        \\  inserts: {d} in {d} ns  (avg {d} ns/op; includes per-op acceleration-kernel dispatch)
        \\  searches: {d} in {d} ns (avg {d} ns/op, k=10 over {d} vectors)
        \\
    , .{ count, insert_ns, ins_avg, queries, search_ns, srch_avg, store.vectorCount() });
    return 0;
}

fn clusterDemo(allocator: std.mem.Allocator, nodes: usize) anyerror!u8 {
    if (nodes < 1) return usage();
    var c = try wdbx.cluster.Cluster.init(allocator, nodes);
    defer c.deinit();

    const elected = try c.startElection(0);
    std.debug.print("election(node 0): leader_elected={any}\n", .{elected});
    {
        const line = try c.statusLine(allocator);
        defer allocator.free(line);
        std.debug.print("  status: {s}\n", .{line});
    }

    const acks = c.replicate("set k=v") catch 0;
    std.debug.print("replicate(\"set k=v\"): acks={d} quorum={d}\n", .{ acks, c.quorum() });

    // Fail the current leader and re-elect from the survivors.
    const old = c.leader().?.id;
    try c.failNode(old);
    std.debug.print("failover: downed leader node {d}\n", .{old});
    const next: u32 = if (old == 0) 1 else 0;
    const re = c.startElection(next) catch false;
    std.debug.print("re-election(node {d}): leader_elected={any}\n", .{ next, re });
    {
        const line = try c.statusLine(allocator);
        defer allocator.free(line);
        std.debug.print("  status: {s}\n", .{line});
    }
    std.debug.print("(in-process Raft-style consensus; networked RPC transport is a Phase-2 item)\n", .{});
    return 0;
}

fn computeInfo() anyerror!u8 {
    const caps = wdbx.compute.capabilities();
    std.debug.print("compute backends (native dispatch not linked in this build; CPU fallback active):\n", .{});
    for (caps) |cap| {
        std.debug.print("  {s:<10} class={s:<3} available={any} native={any}\n", .{ cap.backend.name(), cap.backend.class(), cap.available, cap.native });
    }
    const best = wdbx.compute.bestCpuBackend();
    const sel = wdbx.compute.select(.npu_ane);
    std.debug.print("dynamic selection: best_cpu={s}; request npu-ane -> effective={s} ({s})\n", .{ best.name(), sel.effective.name(), sel.message });
    return 0;
}

fn secureDemo(allocator: std.mem.Allocator) anyerror!u8 {
    // Embedding compression round-trip.
    var vec: [128]f32 = undefined;
    for (&vec, 0..) |*v, i| v.* = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
    var q = try wdbx.compression.quantize(allocator, &vec);
    defer q.deinit(allocator);
    const back = try wdbx.compression.dequantize(allocator, q);
    defer allocator.free(back);
    std.debug.print("compression: {d} f32 -> int8 codes, ratio={d:.2}x, max_error={d:.5}\n", .{ vec.len, q.compressionRatio(), wdbx.compression.maxError(&vec, back) });

    // Additively homomorphic aggregation: sum ciphertexts without decrypting.
    const key = wdbx.crypto_he.Key.init(0xABCDEF);
    var acc = try key.encrypt(allocator, 0, 0);
    defer acc.deinit(allocator);
    var plain_sum: u64 = 0;
    var i: u64 = 1;
    while (i <= 5) : (i += 1) {
        var ci = try key.encrypt(allocator, i * 100, 1000 + i);
        defer ci.deinit(allocator);
        const next = try wdbx.crypto_he.add(allocator, acc, ci);
        acc.deinit(allocator);
        acc = next;
        plain_sum += i * 100;
    }
    const decrypted = try key.decrypt(acc);
    std.debug.print("homomorphic add: sum of 5 encrypted values decrypts to {d} (expected {d}, match={any})\n", .{ decrypted, plain_sum, decrypted == plain_sum });
    std.debug.print("(additive single-key homomorphism; full FHE with multiplication is a research-horizon item)\n", .{});
    return 0;
}

fn gpuInfo(allocator: std.mem.Allocator) anyerror!u8 {
    const status = features.gpu.detectBackend();
    const native = features.gpu.nativeKernelStatus();
    const report = try features.gpu.backendStatusReport(allocator);
    defer allocator.free(report);
    std.debug.print("GPU: {s} available={any} accelerated={any}\n{s}\nnative_kernels: linked={any} ({s})\n", .{
        features.gpu.backendName(status.backend),
        status.available,
        status.accelerated,
        report,
        native.linked,
        native.message,
    });
    return 0;
}

test "wdbx handler usage returns non-zero without args" {
    // db/block require args; bare invocation prints usage with exit code 2.
    const args = [_][]const u8{ "abi", "wdbx" };
    const code = try handleWdbx(std.testing.io, std.testing.allocator, &args);
    try std.testing.expectEqual(@as(u8, 2), code);
}

test "wdbx db init + block insert + verify + query round-trip" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-rt.jsonl";
    const wp = "zig-out/wdbx-cli-rt.jsonl.wal";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, wp) catch {};
    std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    std.Io.Dir.cwd().deleteFile(std.testing.io, wp) catch {};

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "aviva", "{\"turn\":2}" }));
    // Snapshot integrity + WAL replay consistency must both hold.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "get", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "query", path }));
}

test "wdbx db verify detects WAL/snapshot divergence" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-cli-divergence.jsonl";
    const wp = "zig-out/wdbx-cli-divergence.jsonl.wal";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, wp) catch {};
    std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    std.Io.Dir.cwd().deleteFile(std.testing.io, wp) catch {};

    // A consistent snapshot+WAL with one block verifies clean.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "init", path }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "block", "insert", path, "abbey", "{\"turn\":1}" }));
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));

    // Append a block to the durable WAL only, leaving the snapshot behind. The
    // log now records more history than the snapshot — verify must surface the
    // divergence (exit 1), not silently trust the snapshot.
    try wdbx.wal.appendBlock(std.testing.io, allocator, wp, "aviva", 0, 0, "{\"turn\":2}", 4242);
    try std.testing.expectEqual(@as(u8, 1), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "db", "verify", path }));
}

test {
    std.testing.refAllDecls(@This());
}
