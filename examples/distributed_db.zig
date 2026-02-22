//! Database + Network/Raft Cross-Module Integration Example
//!
//! Demonstrates how a distributed database cluster would be configured
//! using multiple ABI subsystems working together:
//! - Vector database creation and indexing
//! - Raft consensus for leader election and replication
//! - Network node registry and cluster topology
//! - Circuit breakers for fault tolerance
//!
//! Run with: `zig build run-distributed-db`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Distributed Database Integration Example ===\n\n", .{});

    // ── Step 1: Database Configuration ─────────────────────────────────────
    std.debug.print("--- Step 1: Vector Database Setup ---\n", .{});

    const db_enabled = abi.database.isEnabled();
    std.debug.print("  Database module: {s}\n", .{if (db_enabled) "enabled" else "disabled (stub)"});

    if (db_enabled) {
        var handle = abi.database.openOrCreate(allocator, "distributed-vectors") catch |err| {
            std.debug.print("  Failed to open database: {t}\n", .{err});
            return;
        };
        defer abi.database.close(&handle);

        // Insert sample vectors representing node embeddings
        const embeddings = [_][4]f32{
            .{ 0.9, 0.1, 0.0, 0.0 }, // Node 1 workload profile
            .{ 0.1, 0.8, 0.1, 0.0 }, // Node 2 workload profile
            .{ 0.0, 0.1, 0.9, 0.0 }, // Node 3 workload profile
        };
        for (embeddings, 1..) |vec, i| {
            abi.database.insert(&handle, @intCast(i), &vec, null) catch |err| {
                std.debug.print("  Insert failed for node {d}: {t}\n", .{ i, err });
                return;
            };
        }

        const db_stats = abi.database.stats(&handle);
        std.debug.print("  Vectors stored: {d} (dim={d})\n", .{ db_stats.count, db_stats.dimension });

        // Find similar workload profiles
        const query = [_]f32{ 0.8, 0.2, 0.0, 0.0 };
        const results = abi.database.search(&handle, allocator, &query, 2) catch |err| {
            std.debug.print("  Search failed: {t}\n", .{err});
            return;
        };
        defer allocator.free(results);

        std.debug.print("  Nearest workload profiles to query:\n", .{});
        for (results) |r| {
            std.debug.print("    Node {d}: similarity={d:.4}\n", .{ r.id, r.score });
        }
    } else {
        std.debug.print("  (Database operations skipped — feature disabled)\n", .{});
    }

    // ── Step 2: Raft Consensus Configuration ───────────────────────────────
    std.debug.print("\n--- Step 2: Raft Consensus Topology ---\n", .{});

    const net_enabled = abi.network.isEnabled();
    std.debug.print("  Network module: {s}\n", .{if (net_enabled) "enabled" else "disabled (stub)"});

    // Show Raft configuration (types are always available from the module)
    const raft_config = abi.network.RaftConfig{
        .election_timeout_min_ms = 150,
        .election_timeout_max_ms = 300,
        .heartbeat_interval_ms = 50,
        .max_entries_per_append = 100,
        .enable_pre_vote = true,
        .enable_leader_lease = true,
        .leader_lease_ms = 200,
    };

    std.debug.print("  Raft consensus config:\n", .{});
    std.debug.print("    Election timeout:   {d}-{d} ms\n", .{
        raft_config.election_timeout_min_ms,
        raft_config.election_timeout_max_ms,
    });
    std.debug.print("    Heartbeat interval: {d} ms\n", .{raft_config.heartbeat_interval_ms});
    std.debug.print("    Max entries/append: {d}\n", .{raft_config.max_entries_per_append});
    std.debug.print("    Pre-vote:           {}\n", .{raft_config.enable_pre_vote});
    std.debug.print("    Leader lease:       {} ({d} ms)\n", .{
        raft_config.enable_leader_lease,
        raft_config.leader_lease_ms,
    });

    // ── Step 3: Network Node Registry ──────────────────────────────────────
    std.debug.print("\n--- Step 3: Cluster Node Registry ---\n", .{});

    if (net_enabled) {
        try abi.network.initWithConfig(allocator, .{
            .cluster_id = "db-cluster-east",
            .heartbeat_timeout_ms = 15_000,
            .max_nodes = 16,
        });
        defer abi.network.deinit();

        const registry = abi.network.defaultRegistry() catch |err| {
            std.debug.print("  Registry unavailable: {t}\n", .{err});
            return;
        };

        // Register database nodes
        const node_addrs = [_]struct { id: []const u8, addr: []const u8 }{
            .{ .id = "db-node-1", .addr = "10.0.1.10:5432" },
            .{ .id = "db-node-2", .addr = "10.0.1.11:5432" },
            .{ .id = "db-node-3", .addr = "10.0.1.12:5432" },
        };
        for (node_addrs) |n| {
            registry.register(n.id, n.addr) catch |err| {
                std.debug.print("  Failed to register {s}: {t}\n", .{ n.id, err });
                return;
            };
        }

        // Mark leader and follower status
        _ = registry.touch("db-node-1");
        _ = registry.setStatus("db-node-2", .degraded);

        const nodes = registry.list();
        std.debug.print("  Cluster topology ({d} nodes):\n", .{nodes.len});
        for (nodes) |node| {
            std.debug.print("    {s} @ {s} — status: {t}\n", .{ node.id, node.address, node.status });
        }
    } else {
        std.debug.print("  (Node registry skipped — network disabled)\n", .{});
    }

    // ── Step 4: Circuit Breaker for Replication ────────────────────────────
    std.debug.print("\n--- Step 4: Circuit Breaker Config ---\n", .{});

    const cb_config = abi.network.CircuitConfig{
        .failure_threshold = 5,
        .success_threshold = 3,
        .timeout_ms = 30_000,
        .half_open_max_calls = 2,
    };

    std.debug.print("  Replication circuit breaker:\n", .{});
    std.debug.print("    Failure threshold:    {d}\n", .{cb_config.failure_threshold});
    std.debug.print("    Success threshold:    {d}\n", .{cb_config.success_threshold});
    std.debug.print("    Timeout:              {d} ms\n", .{cb_config.timeout_ms});
    std.debug.print("    Half-open max calls:  {d}\n", .{cb_config.half_open_max_calls});

    // ── Summary ────────────────────────────────────────────────────────────
    std.debug.print("\n--- Integration Summary ---\n", .{});
    std.debug.print("  A distributed vector database cluster combines:\n", .{});
    std.debug.print("  • abi.database   — vector storage + HNSW similarity search\n", .{});
    std.debug.print("  • abi.network    — node registry + service discovery\n", .{});
    std.debug.print("  • network.raft   — consensus for leader election\n", .{});
    std.debug.print("  • circuit_breaker — fault tolerance on replication paths\n", .{});

    std.debug.print("\n=== Distributed Database Integration Complete ===\n", .{});
}
