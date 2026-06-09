const std = @import("std");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");
const db_commands = @import("wdbx_db.zig");
const runtime_commands = @import("wdbx_runtime.zig");

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

fn run(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    if (args.len < 3) return usage();
    const sub = args[2];

    if (std.mem.eql(u8, sub, "db")) {
        if (args.len < 4) return usage();
        const op = args[3];
        if (std.mem.eql(u8, op, "init")) {
            if (args.len != 5) return usage();
            return db_commands.initDb(io, allocator, args[4]);
        } else if (std.mem.eql(u8, op, "verify")) {
            if (args.len != 5) return usage();
            return db_commands.verifyDb(io, allocator, args[4]);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "block")) {
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
        if (args.len != 4) return usage();
        return db_commands.query(io, allocator, args[3]);
    }

    if (std.mem.eql(u8, sub, "benchmark")) {
        const count: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 256) else 256;
        return runtime_commands.benchmark(allocator, count);
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
            if (nodes < 1) return usage();
            return runtime_commands.clusterDemo(allocator, nodes);
        }
        return usage();
    }

    if (std.mem.eql(u8, sub, "compute")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return runtime_commands.computeInfo();
    }

    if (std.mem.eql(u8, sub, "secure")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "demo")) return usage();
        return runtime_commands.secureDemo(allocator);
    }

    if (std.mem.eql(u8, sub, "gpu")) {
        if (args.len != 4 or !std.mem.eql(u8, args[3], "info")) return usage();
        return runtime_commands.gpuInfo(allocator);
    }

    if (std.mem.eql(u8, sub, "api")) {
        if (args.len < 4 or !std.mem.eql(u8, args[3], "serve")) return usage();
        const port: u16 = if (args.len >= 5) (std.fmt.parseInt(u16, args[4], 10) catch 8081) else 8081;
        return runtime_commands.serveApi(io, allocator, port);
    }

    return usage();
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
