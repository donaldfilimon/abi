const std = @import("std");
const features = @import("../../features/mod.zig");
const foundation_time = @import("../../foundation/time.zig");

const wdbx = features.wdbx;

/// Nearest-rank percentile (p in [0,100]) over an already-sorted ascending
/// slice. Returns 0 for empty input.
fn percentileSorted(sorted: []const u64, p: u8) u64 {
    if (sorted.len == 0) return 0;
    const idx_f = @ceil(@as(f64, @floatFromInt(p)) / 100.0 * @as(f64, @floatFromInt(sorted.len)));
    const raw_idx = @as(usize, @intFromFloat(idx_f));
    const clamped = if (raw_idx == 0) 0 else @min(raw_idx, sorted.len) - 1;
    return sorted[clamped];
}

pub fn benchmark(allocator: std.mem.Allocator, count: usize) anyerror!u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    // Per-op latency samples so we can report P50/P95/P99, not just averages.
    const insert_samples = try allocator.alloc(u64, count);
    defer allocator.free(insert_samples);
    const queries: usize = @min(count, 200);
    const search_samples = try allocator.alloc(u64, queries);
    defer allocator.free(search_samples);

    const insert_start = foundation_time.monotonicNs();
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var v: [4]f32 = .{ 0, 0, 0, 0 };
        v[0] = @floatFromInt(i % 97);
        v[1] = @floatFromInt(i % 31);
        const op_start = foundation_time.monotonicNs();
        _ = try store.putVector(&v);
        insert_samples[i] = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - op_start));
    }
    const insert_ns: u64 = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - insert_start));

    const search_start = foundation_time.monotonicNs();
    var j: usize = 0;
    while (j < queries) : (j += 1) {
        const op_start = foundation_time.monotonicNs();
        const r = try store.search(&.{ 1, 0, 0, 0 }, 10);
        search_samples[j] = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - op_start));
        allocator.free(r);
    }
    const search_ns: u64 = @intCast(@max(@as(i64, 0), foundation_time.monotonicNs() - search_start));

    std.mem.sort(u64, insert_samples, {}, std.sort.asc(u64));
    std.mem.sort(u64, search_samples, {}, std.sort.asc(u64));

    const ins_avg = if (count > 0) insert_ns / count else 0;
    const srch_avg = if (queries > 0) search_ns / queries else 0;
    std.debug.print(
        \\benchmark (local, in-memory; not a published throughput claim):
        \\  inserts: {d} in {d} ns  (avg {d} ns/op; includes per-op acceleration-kernel dispatch)
        \\    p50={d} ns  p95={d} ns  p99={d} ns
        \\  searches: {d} in {d} ns (avg {d} ns/op, k=10 over {d} vectors)
        \\    p50={d} ns  p95={d} ns  p99={d} ns
        \\
    , .{
        count,                                insert_ns,
        ins_avg,                              percentileSorted(insert_samples, 50),
        percentileSorted(insert_samples, 95), percentileSorted(insert_samples, 99),
        queries,                              search_ns,
        srch_avg,                             store.vectorCount(),
        percentileSorted(search_samples, 50), percentileSorted(search_samples, 95),
        percentileSorted(search_samples, 99),
    });
    return 0;
}

pub fn clusterDemo(allocator: std.mem.Allocator, nodes: usize) anyerror!u8 {
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

pub fn clusterServe(io: std.Io, allocator: std.mem.Allocator, port: u16, node_id: u32) anyerror!u8 {
    var node = wdbx.cluster.Node{ .id = node_id };
    defer {
        for (node.log.items) |e| allocator.free(e.data);
        node.log.deinit(allocator);
    }

    var server = wdbx.cluster_rpc.listen(io, port) catch |err| {
        std.debug.print("cluster serve: bind 127.0.0.1:{d} failed: {s}\n", .{ port, @errorName(err) });
        return 1;
    };
    defer server.deinit(io);

    std.debug.print(
        "cluster node {d} serving consensus RPC on 127.0.0.1:{d} (RequestVote/AppendEntries); peers connect via the cluster_rpc transport. Ctrl-C to stop.\n",
        .{ node_id, port },
    );
    try wdbx.cluster_rpc.serveLoop(io, &server, &node, allocator);
    return 0;
}

pub fn computeInfo() anyerror!u8 {
    const caps = wdbx.compute.capabilities();
    std.debug.print("compute backends (native dispatch not linked in this build; CPU fallback active):\n", .{});
    for (caps) |cap| {
        std.debug.print("  {s:<10} class={s:<3} available={any} native={any}\n", .{ cap.backend.name(), cap.backend.class(), cap.available, cap.native });
    }
    const best = wdbx.compute.bestCpuBackend();
    const sel = wdbx.compute.select(.npu_ane);
    std.debug.print("dynamic selection: best_cpu={s}; request npu-ane -> effective={s} ({s})\n", .{ best.name(), sel.effective.name(), sel.message });
    std.debug.print("apple neural engine: hardware_present={any} native_dispatch=false (CoreML/ANE path requires Apple frameworks, not linked; CPU fallback)\n", .{wdbx.compute.aneHardwarePresent()});
    const remote_ep = wdbx.remote_compute.endpoint();
    std.debug.print("remote compute dispatch: endpoint={s} (set {s}=host:port to route ops to a remote TPU/GPU service; CPU fallback otherwise)\n", .{ remote_ep orelse "none", wdbx.remote_compute.ENDPOINT_ENV });
    return 0;
}

pub fn secureDemo(allocator: std.mem.Allocator) anyerror!u8 {
    var vec: [128]f32 = undefined;
    for (&vec, 0..) |*v, i| v.* = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
    var q = try wdbx.compression.quantize(allocator, &vec);
    defer q.deinit(allocator);
    const back = try wdbx.compression.dequantize(allocator, q);
    defer allocator.free(back);
    std.debug.print("compression: {d} f32 -> int8 codes, ratio={d:.2}x, max_error={d:.5}\n", .{ vec.len, q.compressionRatio(), wdbx.compression.maxError(&vec, back) });

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
    std.debug.print("additive HE: sum of 5 encrypted values decrypts to {d} (expected {d}, match={any})\n", .{ decrypted, plain_sum, decrypted == plain_sum });

    // Somewhat-homomorphic encryption supporting BOTH add (XOR) and multiply
    // (AND) on encrypted bits — evaluate a small circuit entirely on ciphertexts.
    var prng = std.Random.DefaultPrng.init(0x5EEDF00DC0FFEE11);
    const rand = prng.random();
    const kp = wdbx.fhe.keygen(rand);
    const e1 = wdbx.fhe.encrypt(kp, rand, 1);
    const e1b = wdbx.fhe.encrypt(kp, rand, 1);
    const e0 = wdbx.fhe.encrypt(kp, rand, 0);
    const e_eval = wdbx.fhe.add(kp, wdbx.fhe.mul(kp, e1, e1b), e0); // (1 AND 1) XOR 0
    const eval_bit = wdbx.fhe.decrypt(kp, e_eval);
    std.debug.print("homomorphic eval: enc((1 AND 1) XOR 0) decrypts to {d} (expected 1, match={any})\n", .{ eval_bit, eval_bit == 1 });
    std.debug.print("(DGHV somewhat-homomorphic scheme: real encrypted add+multiply on ciphertexts, reference parameters / bounded depth — not security-audited)\n", .{});
    return 0;
}

pub fn gpuInfo(allocator: std.mem.Allocator) anyerror!u8 {
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

pub fn serveApi(io: std.Io, allocator: std.mem.Allocator, port: u16) anyerror!u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    std.debug.print("serving WDBX REST on http://127.0.0.1:{d} (Ctrl-C to stop)\n", .{port});
    wdbx.rest.serve(allocator, io, &store, port) catch |err| {
        std.debug.print("REST server error: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
