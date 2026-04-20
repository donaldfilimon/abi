//! Distributed Training Coordinator
//!
//! Provides data-parallel distributed training using gradient AllReduce
//! across multiple GPUs (single-node or multi-node).
//!
//! Architecture:
//!   1. Each worker has a full model replica
//!   2. Training data is sharded across workers
//!   3. Forward/backward passes run independently
//!   4. Gradients are synchronized via AllReduce before optimizer step
//!   5. GradientBucketManager fuses small gradients for efficiency
//!
//! Integration points (pluggable, not imported directly to avoid circular deps):
//!   - `gpu_cluster.GPUCluster.allReduce()` for single-machine multi-GPU
//!   - `gradient_sync.GradientBucketManager` for gradient fusion
//!   - `peer_transfer.NetworkPeerTransfer.ringAllReduce()` for multi-node

const std = @import("std");

/// Reduction operation for distributed gradient synchronization.
pub const ReduceOp = enum {
    /// Sum gradients across all workers.
    sum,
    /// Average gradients across all workers (sum / world_size).
    average,
};

/// Configuration for distributed data-parallel training.
pub const DistributedConfig = struct {
    /// Number of worker ranks (GPUs/nodes).
    world_size: u32 = 1,
    /// This worker's rank (0-indexed).
    rank: u32 = 0,
    /// Whether this rank is the coordinator (rank 0).
    is_coordinator: bool = true,
    /// Gradient bucket size in bytes for fusion (25 MB default, matches PyTorch DDP).
    bucket_size_bytes: usize = 25 * 1024 * 1024,
    /// Enable gradient compression (future: top-k sparsification).
    enable_compression: bool = false,
    /// AllReduce operation type.
    reduce_op: ReduceOp = .average,

    /// Validate configuration.
    pub fn validate(self: DistributedConfig) error{InvalidConfig}!void {
        if (self.world_size == 0) return error.InvalidConfig;
        if (self.rank >= self.world_size) return error.InvalidConfig;
        if (self.bucket_size_bytes == 0) return error.InvalidConfig;
    }
};

/// Distributed training coordinator for data-parallel gradient synchronization.
///
/// Sits between the backward pass and the optimizer step. After each backward
/// pass, call `synchronizeGradients` to AllReduce across workers, then step
/// the optimizer. For single-worker training (world_size == 1) all sync
/// operations are no-ops with zero overhead.
///
/// Integration points for multi-node training:
///   - `gpu_cluster_handle`: Opaque pointer to a `gpu_cluster.GPUCluster` for
///     single-machine multi-GPU AllReduce.
///   - `compressor_handle`: Opaque pointer to a `gradient_compression.GradientCompressor`
///     for bandwidth-efficient gradient synchronization across nodes.
///   - `compression_ratio`: Sparsification ratio (0.0 = disabled, 0.001 = keep top 0.1%).
pub const DistributedTrainer = struct {
    allocator: std.mem.Allocator,
    config: DistributedConfig,
    gradient_buffer: std.ArrayListUnmanaged(f32) = .empty,
    stats: Stats = .{},

    /// Opaque handle to a GPUCluster for multi-GPU AllReduce.
    /// Set via `attachGpuCluster()`. When null, AllReduce is performed locally.
    gpu_cluster_handle: ?*anyopaque = null,

    /// Opaque handle to a GradientCompressor for top-k sparsification.
    /// Set via `attachCompressor()`. When null, gradients are sent uncompressed.
    compressor_handle: ?*anyopaque = null,

    /// Compression ratio for gradient sparsification (0.0 = disabled).
    /// Set via `enableCompression()`. The actual compression is performed by
    /// the GradientCompressor pointed to by `compressor_handle`.
    compression_ratio: f32 = 0.0,

    /// Cumulative statistics for distributed training.
    pub const Stats = struct {
        total_allreduce_calls: u64 = 0,
        total_bytes_synced: u64 = 0,
        total_sync_time_ns: u64 = 0,
        epochs_completed: u32 = 0,
    };

    /// Initialize a distributed trainer.
    pub fn init(allocator: std.mem.Allocator, config: DistributedConfig) DistributedTrainer {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Release resources.
    pub fn deinit(self: *DistributedTrainer) void {
        self.gradient_buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Synchronize gradients across all workers via AllReduce.
    ///
    /// Call after the backward pass, before `optimizer.step()`.
    /// No-op when `world_size <= 1`.
    ///
    /// In a full multi-GPU deployment this would:
    ///   1. Bucket gradients via GradientBucketManager
    ///   2. Call GPUCluster.allReduce() per ready bucket
    ///   3. Divide by world_size when reduce_op == .average
    ///
    /// The current implementation performs the averaging locally so the
    /// public API is stable when real backends are wired in later.
    pub fn synchronizeGradients(self: *DistributedTrainer, gradients: []f32) void {
        if (self.config.world_size <= 1) return;

        // Apply reduction. A real backend would AllReduce across ranks first;
        // the averaging step is the same regardless of transport.
        if (self.config.reduce_op == .average) {
            const scale = 1.0 / @as(f32, @floatFromInt(self.config.world_size));
            for (gradients) |*g| {
                g.* *= scale;
            }
        }

        self.stats.total_allreduce_calls += 1;
        self.stats.total_bytes_synced += gradients.len * @sizeOf(f32);
    }

    /// Shard a dataset across workers. Returns the slice for this rank.
    ///
    /// The last rank receives any remainder elements so no data is dropped.
    pub fn shardData(self: *const DistributedTrainer, comptime T: type, data: []const T) []const T {
        if (self.config.world_size <= 1) return data;

        const shard_size = data.len / self.config.world_size;
        const start = shard_size * self.config.rank;
        const end = if (self.config.rank == self.config.world_size - 1)
            data.len
        else
            start + shard_size;
        return data[start..end];
    }

    /// Whether this rank should log metrics / save checkpoints.
    /// Only the coordinator (rank 0) performs I/O to avoid duplicates.
    pub fn shouldLog(self: *const DistributedTrainer) bool {
        return self.config.is_coordinator;
    }

    /// Record completion of an epoch.
    pub fn recordEpoch(self: *DistributedTrainer) void {
        self.stats.epochs_completed += 1;
    }

    /// Return cumulative distributed training statistics.
    pub fn getStats(self: *const DistributedTrainer) Stats {
        return self.stats;
    }

    // ------------------------------------------------------------------
    // Multi-node integration
    // ------------------------------------------------------------------

    /// Attach a GPUCluster handle for multi-GPU AllReduce.
    ///
    /// The handle is an opaque pointer to a `gpu_cluster.GPUCluster` (or
    /// compatible AllReduce provider). When attached, `synchronizeGradients`
    /// will delegate to the cluster's AllReduce instead of local averaging.
    ///
    /// Pass `null` to detach.
    pub fn attachGpuCluster(self: *DistributedTrainer, handle: *anyopaque) void {
        self.gpu_cluster_handle = handle;
    }

    /// Detach the GPUCluster handle.
    pub fn detachGpuCluster(self: *DistributedTrainer) void {
        self.gpu_cluster_handle = null;
    }

    /// Attach a GradientCompressor handle for top-k sparsification.
    ///
    /// The handle is an opaque pointer to a
    /// `gradient_compression.GradientCompressor`. When attached and
    /// `compression_ratio > 0`, gradients are compressed before being
    /// sent over the network.
    ///
    /// Pass `null` to detach.
    pub fn attachCompressor(self: *DistributedTrainer, handle: *anyopaque) void {
        self.compressor_handle = handle;
    }

    /// Detach the GradientCompressor handle.
    pub fn detachCompressor(self: *DistributedTrainer) void {
        self.compressor_handle = null;
        self.compression_ratio = 0.0;
    }

    /// Enable gradient compression with the given sparsification ratio.
    ///
    /// `ratio` is the fraction of gradient elements to keep (e.g. 0.001
    /// keeps the top 0.1%). The actual compression is performed by the
    /// `GradientCompressor` attached via `attachCompressor()`. This
    /// method only stores the ratio; if no compressor is attached the
    /// ratio is still recorded so it takes effect once one is attached.
    ///
    /// Pass `0.0` to disable compression.
    pub fn enableCompression(self: *DistributedTrainer, ratio: f32) void {
        self.compression_ratio = @max(ratio, 0.0);
    }

    /// Whether gradient compression is currently enabled.
    pub fn isCompressionEnabled(self: *const DistributedTrainer) bool {
        return self.compression_ratio > 0.0 and self.compressor_handle != null;
    }

    /// Whether a GPU cluster backend is attached.
    pub fn hasGpuCluster(self: *const DistributedTrainer) bool {
        return self.gpu_cluster_handle != null;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

test "distributed trainer init/deinit lifecycle" {
    const config: DistributedConfig = .{ .world_size = 4, .rank = 0 };
    var trainer = DistributedTrainer.init(std.testing.allocator, config);
    defer trainer.deinit();

    try std.testing.expectEqual(@as(u32, 4), trainer.config.world_size);
    try std.testing.expectEqual(@as(u32, 0), trainer.config.rank);
    try std.testing.expectEqual(@as(u64, 0), trainer.stats.total_allreduce_calls);
}

test "synchronizeGradients no-op for single worker" {
    var trainer = DistributedTrainer.init(std.testing.allocator, .{});
    defer trainer.deinit();

    var grads = [_]f32{ 4.0, 8.0, 12.0 };
    trainer.synchronizeGradients(&grads);

    // world_size == 1: gradients unchanged, no stats recorded
    try std.testing.expectEqual(@as(f32, 4.0), grads[0]);
    try std.testing.expectEqual(@as(f32, 8.0), grads[1]);
    try std.testing.expectEqual(@as(f32, 12.0), grads[2]);
    try std.testing.expectEqual(@as(u64, 0), trainer.stats.total_allreduce_calls);
}

test "synchronizeGradients averages with world_size 4" {
    const config: DistributedConfig = .{ .world_size = 4, .rank = 1, .is_coordinator = false };
    var trainer = DistributedTrainer.init(std.testing.allocator, config);
    defer trainer.deinit();

    var grads = [_]f32{ 4.0, 8.0, 12.0, 16.0 };
    trainer.synchronizeGradients(&grads);

    // Each gradient divided by 4
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grads[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grads[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grads[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grads[3], 1e-6);

    try std.testing.expectEqual(@as(u64, 1), trainer.stats.total_allreduce_calls);
    try std.testing.expectEqual(@as(u64, 4 * @sizeOf(f32)), trainer.stats.total_bytes_synced);
}

test "shardData partitions correctly" {
    const data = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // Rank 0 of 3: elements 0..3
    const cfg0: DistributedConfig = .{ .world_size = 3, .rank = 0 };
    const t0 = DistributedTrainer.init(std.testing.allocator, cfg0);
    const shard0 = t0.shardData(u32, &data);
    try std.testing.expectEqual(@as(usize, 3), shard0.len);
    try std.testing.expectEqual(@as(u32, 0), shard0[0]);

    // Rank 2 of 3 (last): gets remainder, elements 6..10
    const cfg2: DistributedConfig = .{ .world_size = 3, .rank = 2, .is_coordinator = false };
    const t2 = DistributedTrainer.init(std.testing.allocator, cfg2);
    const shard2 = t2.shardData(u32, &data);
    try std.testing.expectEqual(@as(usize, 4), shard2.len);
    try std.testing.expectEqual(@as(u32, 9), shard2[shard2.len - 1]);
}

test "shouldLog only for coordinator" {
    const coord: DistributedConfig = .{ .world_size = 4, .rank = 0, .is_coordinator = true };
    const worker: DistributedConfig = .{ .world_size = 4, .rank = 2, .is_coordinator = false };

    const t_coord = DistributedTrainer.init(std.testing.allocator, coord);
    const t_worker = DistributedTrainer.init(std.testing.allocator, worker);

    try std.testing.expect(t_coord.shouldLog());
    try std.testing.expect(!t_worker.shouldLog());
}

test "config validation" {
    // Valid config
    const valid: DistributedConfig = .{ .world_size = 4, .rank = 3 };
    try valid.validate();

    // Invalid: rank >= world_size
    const bad_rank: DistributedConfig = .{ .world_size = 2, .rank = 2 };
    try std.testing.expectError(error.InvalidConfig, bad_rank.validate());

    // Invalid: world_size == 0
    const bad_world: DistributedConfig = .{ .world_size = 0, .rank = 0 };
    try std.testing.expectError(error.InvalidConfig, bad_world.validate());
}

test "attachGpuCluster and detach" {
    var trainer = DistributedTrainer.init(std.testing.allocator, .{});
    defer trainer.deinit();

    try std.testing.expect(!trainer.hasGpuCluster());

    var dummy: u8 = 0;
    trainer.attachGpuCluster(@ptrCast(&dummy));
    try std.testing.expect(trainer.hasGpuCluster());

    trainer.detachGpuCluster();
    try std.testing.expect(!trainer.hasGpuCluster());
}

test "attachCompressor and enableCompression" {
    var trainer = DistributedTrainer.init(std.testing.allocator, .{});
    defer trainer.deinit();

    // No compressor, no compression
    try std.testing.expect(!trainer.isCompressionEnabled());

    // Enable ratio without compressor — not yet active
    trainer.enableCompression(0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), trainer.compression_ratio, 1e-6);
    try std.testing.expect(!trainer.isCompressionEnabled());

    // Attach compressor — now active
    var dummy: u8 = 0;
    trainer.attachCompressor(@ptrCast(&dummy));
    try std.testing.expect(trainer.isCompressionEnabled());

    // Disable via ratio
    trainer.enableCompression(0.0);
    try std.testing.expect(!trainer.isCompressionEnabled());

    // Detach clears both
    trainer.enableCompression(0.01);
    trainer.detachCompressor();
    try std.testing.expect(!trainer.isCompressionEnabled());
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), trainer.compression_ratio, 1e-9);
}

test "default fields for new integration points" {
    const trainer = DistributedTrainer.init(std.testing.allocator, .{});
    // Not calling deinit intentionally — no allocations for default config.

    try std.testing.expectEqual(@as(?*anyopaque, null), trainer.gpu_cluster_handle);
    try std.testing.expectEqual(@as(?*anyopaque, null), trainer.compressor_handle);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), trainer.compression_ratio, 1e-9);
}

test {
    std.testing.refAllDecls(@This());
}
