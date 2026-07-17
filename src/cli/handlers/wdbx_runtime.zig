const std = @import("std");
const features = @import("../../features/mod.zig");
const env = @import("../../foundation/env.zig");
const foundation_time = @import("../../foundation/time.zig");

const wdbx = features.wdbx;

pub const CLUSTER_TOKEN_ENV = env.WDBX_CLUSTER_TOKEN_ENV;
pub const CLUSTER_PEERS_ENV = "ABI_WDBX_CLUSTER_PEERS";

const ClusterConfig = struct {
    policy: wdbx.cluster_rpc.ClusterPolicy = .{},
    peers_owned: ?[]u32 = null,

    fn deinit(self: *ClusterConfig, allocator: std.mem.Allocator) void {
        if (self.peers_owned) |peers| allocator.free(peers);
    }
};

/// Nearest-rank percentile (p in [0,100]) over an already-sorted ascending
/// slice. Returns 0 for empty input.
fn percentileSorted(sorted: []const u64, p: u8) u64 {
    if (sorted.len == 0) return 0;
    const idx_f = @ceil(@as(f64, @floatFromInt(p)) / 100.0 * @as(f64, @floatFromInt(sorted.len)));
    const raw_idx = @as(usize, @intFromFloat(idx_f));
    const clamped = if (raw_idx == 0) 0 else @min(raw_idx, sorted.len) - 1;
    return sorted[clamped];
}

fn containsWhitespace(value: []const u8) bool {
    for (value) |byte| {
        if (std.ascii.isWhitespace(byte)) return true;
    }
    return false;
}

fn parseClusterPeers(allocator: std.mem.Allocator, raw: []const u8) ![]u32 {
    var peers: std.ArrayListUnmanaged(u32) = .empty;
    errdefer peers.deinit(allocator);

    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const token = std.mem.trim(u8, part, " \t\r\n");
        if (token.len == 0) return error.InvalidClusterPeers;
        const id = std.fmt.parseInt(u32, token, 10) catch return error.InvalidClusterPeers;
        try peers.append(allocator, id);
    }
    if (peers.items.len == 0) return error.InvalidClusterPeers;
    return peers.toOwnedSlice(allocator);
}

pub fn clusterConfigFromValues(allocator: std.mem.Allocator, token_raw: ?[]const u8, peers_raw: ?[]const u8) !ClusterConfig {
    var config = ClusterConfig{};
    errdefer config.deinit(allocator);

    if (token_raw) |token| {
        if (token.len == 0 or containsWhitespace(token)) return error.InvalidClusterToken;
        config.policy.auth = .{ .token = token };
    }
    if (peers_raw) |peers| {
        const owned = try parseClusterPeers(allocator, peers);
        config.peers_owned = owned;
        config.policy.peers = owned;
    }
    return config;
}

fn clusterConfigFromEnv(allocator: std.mem.Allocator) !ClusterConfig {
    return clusterConfigFromValues(allocator, env.get(CLUSTER_TOKEN_ENV), env.get(CLUSTER_PEERS_ENV));
}

/// `abi wdbx benchmark [count]`: run a local, in-memory insert/search benchmark
/// over `count` vectors and report totals plus P50/P95/P99 per-op latencies. This
/// is a local microbenchmark, not a published throughput claim. Returns the exit code.
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

/// `abi wdbx cluster demo [nodes]`: run an in-process Raft-style consensus demo
/// over `nodes` nodes — leader election, log replication with quorum, leader
/// failover, and re-election. Networked RPC serving is exposed separately by
/// `cluster serve`.
/// Returns the process exit code.
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
    std.debug.print("(in-process Raft-style consensus; networked RPC serving is available via `cluster serve`)\n", .{});
    std.debug.print("north-star status: in-process (Phase 1 landed); multi-host production cluster Proposed (Phase 2) (docs/spec/wdbx-north-star.mdx §2/§3.5)\n", .{});
    return 0;
}

/// Loopback hosts need no exposure warning. Treats the IPv4 loopback block
/// (`127.0.0.0/8`), the IPv6 loopback `::1`, and the `localhost` name as local.
fn isLoopbackHost(host: []const u8) bool {
    return std.mem.eql(u8, host, "localhost") or
        std.mem.eql(u8, host, "::1") or
        std.mem.startsWith(u8, host, "127.");
}

/// `abi wdbx cluster serve <host> <port> <node_id>`: bind the consensus RPC
/// transport (RequestVote/AppendEntries) on `host:port` and serve as `node_id`
/// until interrupted. Non-loopback binds require a shared-secret token.
pub fn clusterServe(io: std.Io, allocator: std.mem.Allocator, host: []const u8, port: u16, node_id: u32) anyerror!u8 {
    var node = wdbx.cluster.Node{ .id = node_id };
    defer {
        for (node.log.items) |e| allocator.free(e.data);
        node.log.deinit(allocator);
    }

    var config = clusterConfigFromEnv(allocator) catch |err| {
        std.debug.print("cluster serve: invalid cluster config ({s}); {s} must be non-empty with no whitespace and {s} must be comma-separated u32 node ids\n", .{ @errorName(err), CLUSTER_TOKEN_ENV, CLUSTER_PEERS_ENV });
        return 1;
    };
    defer config.deinit(allocator);

    if (!isLoopbackHost(host) and !config.policy.auth.enabled()) {
        std.debug.print("cluster serve: refusing non-loopback bind {s}:{d} without {s}; set a shared secret or bind 127.0.0.1\n", .{ host, port, CLUSTER_TOKEN_ENV });
        return 1;
    }

    // Bind the requested host: "127.0.0.1" (loopback, default), "0.0.0.0" (all
    // interfaces for multi-host), or a specific routable IPv4/IPv6 address.
    var server = wdbx.cluster_rpc.listenAddr(io, host, port) catch |err| {
        std.debug.print("cluster serve: bind {s}:{d} failed: {s}\n", .{ host, port, @errorName(err) });
        return 1;
    };
    defer server.deinit(io);

    std.debug.print(
        "cluster node {d} serving consensus RPC on {s}:{d} (RequestVote/AppendEntries; auth={s}; peers={s}). Ctrl-C to stop.\n",
        .{ node_id, host, port, if (config.policy.auth.enabled()) "shared-secret" else "off", if (config.policy.peers == null) "any" else "configured" },
    );
    if (!isLoopbackHost(host)) {
        std.debug.print(
            "ops note: non-loopback bind is authenticated TCP only — front with TLS/mTLS (nginx/caddy/envoy) for transit encryption; native TLS is not linked. Env: {s}=shared-secret, {s}=comma-separated node ids. This is not production multi-host/sharding.\n",
            .{ CLUSTER_TOKEN_ENV, CLUSTER_PEERS_ENV },
        );
    }
    try wdbx.cluster_rpc.serveLoopAuth(io, &server, &node, allocator, config.policy);
    return 0;
}

/// `abi wdbx compute info`: report the available compute backends and dynamic
/// selection (best CPU backend, ANE/remote-dispatch availability). Native
/// dispatch is not linked, so the CPU fallback is active. Returns the exit code.
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

/// `abi wdbx secure demo`: demonstrate the WDBX security primitives — int8
/// vector quantization, additive homomorphic encryption summed over ciphertexts,
/// and a small DGHV somewhat-homomorphic add+multiply circuit. These use
/// reference parameters and are not security-audited. Returns the exit code.
pub fn secureDemo(allocator: std.mem.Allocator) anyerror!u8 {
    var vec: [128]f32 = undefined;
    for (&vec, 0..) |*v, i| v.* = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
    var q = try wdbx.compression.quantize(allocator, &vec);
    defer q.deinit(allocator);
    const back = try wdbx.compression.dequantize(allocator, q);
    defer allocator.free(back);
    std.debug.print("compression: {d} f32 -> int8 codes, ratio={d:.2}x, max_error={d:.5}\n", .{ vec.len, q.compressionRatio(), wdbx.compression.maxError(&vec, back) });

    const entropy_src = "WDBX-entropy-demo-aaaaaaaaaa-bbbbbbbbbb-cccccccccc-HELLO";
    var huff = try wdbx.entropy.encode(allocator, entropy_src);
    defer huff.deinit(allocator);
    const huff_back = try wdbx.entropy.decode(allocator, huff);
    defer allocator.free(huff_back);
    std.debug.print("entropy Huffman: mode={s} {d}B -> serialized {d}B ratio={d:.2}x roundtrip={any}\n", .{
        @tagName(huff.mode), entropy_src.len, huff.serializedLen(), huff.compressionRatio(), std.mem.eql(u8, entropy_src, huff_back),
    });

    var rans0 = try wdbx.ans.encode(allocator, entropy_src);
    defer rans0.deinit(allocator);
    const rans0_back = try wdbx.ans.decode(allocator, rans0);
    defer allocator.free(rans0_back);
    std.debug.print("entropy rANS0: mode={s} {d}B -> serialized {d}B ratio={d:.2}x roundtrip={any}\n", .{
        @tagName(rans0.mode), entropy_src.len, rans0.serializedLen(), rans0.compressionRatio(), std.mem.eql(u8, entropy_src, rans0_back),
    });

    const o1_src = "the the the cat sat on the mat the the cat sat";
    var rans1 = try wdbx.ans.encodeOrder1(allocator, o1_src);
    defer rans1.deinit(allocator);
    const rans1_back = try wdbx.ans.decode(allocator, rans1);
    defer allocator.free(rans1_back);
    std.debug.print("entropy rANS1: mode={s} {d}B -> serialized {d}B ratio={d:.2}x roundtrip={any} (demo; not SOTA)\n", .{
        @tagName(rans1.mode), o1_src.len, rans1.serializedLen(), rans1.compressionRatio(), std.mem.eql(u8, o1_src, rans1_back),
    });

    var ae = try wdbx.neural_compress.Autoencoder.init(allocator, 8, 4, 0xC0DEC0DE);
    defer ae.deinit();
    var sample: [8]f32 = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    _ = ae.trainStep(&sample, 0.05);
    var latent: [4]f32 = undefined;
    var recon: [8]f32 = undefined;
    ae.encode(&sample, &latent);
    ae.decode(&latent, &recon);
    std.debug.print("neural_compress: autoencoder 8->4->8 recon[0]={d:.4} (reference demo, not SOTA)\n", .{recon[0]});

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
    std.debug.print("north-star status: Partial — int8 + Huffman + rANS/order-1 demos + autoencoder + additive HE + reference DGHV SHE (not audited, not SOTA); production FHE/SOTA codecs remain Proposed\n", .{});
    return 0;
}

/// `abi wdbx gpu info`: report the detected GPU backend, whether it is
/// accelerated, the backend status report, and native-kernel link status.
/// Returns the process exit code.
pub fn gpuInfo(allocator: std.mem.Allocator) anyerror!u8 {
    // Probe Metal (or CPU fallback) before reporting so linked/accelerated
    // flags reflect a real init attempt on this host.
    _ = features.gpu.vectorOps();
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

/// `abi wdbx api serve [port]`: serve the WDBX REST API over an in-memory store
/// on `127.0.0.1:<port>` until interrupted. Returns the process exit code.
/// TLS: when ABI_WDBX_TLS_CERT and ABI_WDBX_TLS_KEY are set, the server validates
/// cert/key files and advises proxy-based HTTPS deployment (native TLS not linked).
pub fn serveApi(io: std.Io, allocator: std.mem.Allocator, port: u16) anyerror!u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    const tls_str = if (wdbx.tls_config.TlsConfig.fromEnv(io) != null) " (TLS configured)" else "";
    std.debug.print("serving WDBX REST on http://127.0.0.1:{d}{s} (Ctrl-C to stop)\n", .{ port, tls_str });
    wdbx.rest.serve(allocator, io, &store, port) catch |err| {
        std.debug.print("REST server error: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}

test "wdbx cluster config rejects invalid token and peer values" {
    try std.testing.expectError(error.InvalidClusterToken, clusterConfigFromValues(std.testing.allocator, "", null));
    try std.testing.expectError(error.InvalidClusterToken, clusterConfigFromValues(std.testing.allocator, "has whitespace", null));
    try std.testing.expectError(error.InvalidClusterPeers, clusterConfigFromValues(std.testing.allocator, null, "0,,2"));
    try std.testing.expectError(error.InvalidClusterPeers, clusterConfigFromValues(std.testing.allocator, null, "0,nope,2"));
}

test "wdbx cluster config accepts shared secret and peer allowlist" {
    var config = try clusterConfigFromValues(std.testing.allocator, "cluster-secret", "0,1,2");
    defer config.deinit(std.testing.allocator);

    try std.testing.expect(config.policy.auth.enabled());
    try std.testing.expect(config.policy.peers != null);
    try std.testing.expectEqual(@as(usize, 3), config.policy.peers.?.len);
    try std.testing.expectEqual(@as(u32, 2), config.policy.peers.?[2]);
}

test "wdbx cluster serve exits on invalid token env before binding" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put(CLUSTER_TOKEN_ENV, "has whitespace");
    env.install(&environ);
    defer env.resetForTesting();

    try std.testing.expectEqual(@as(u8, 1), try clusterServe(std.testing.io, std.testing.allocator, "127.0.0.1", 39991, 0));
}

test "wdbx cluster serve exits on invalid peer env before binding" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put(CLUSTER_PEERS_ENV, "0,,2");
    env.install(&environ);
    defer env.resetForTesting();

    try std.testing.expectEqual(@as(u8, 1), try clusterServe(std.testing.io, std.testing.allocator, "127.0.0.1", 39992, 0));
}
