//! GPU Cluster - Multi-GPU Context Management.
//!
//! Provides AllReduce, scatter/gather, model partitioning,
//! and tensor parallelism across multiple GPU devices.

const std = @import("std");
const unified = @import("unified.zig");
const peer_transfer = @import("peer_transfer/mod.zig");
const device_group_mod = @import("device_group.zig");

const sync = @import("../../services/shared/sync.zig");
const Mutex = sync.Mutex;

// Re-import types from device_group for use in this module
const DeviceId = device_group_mod.DeviceId;
const DeviceGroup = device_group_mod.DeviceGroup;
const MultiDeviceConfig = device_group_mod.MultiDeviceConfig;
const WorkDistribution = device_group_mod.WorkDistribution;
const DeviceBarrier = device_group_mod.DeviceBarrier;

/// Reduction operation for AllReduce.
pub const ReduceOp = enum {
    sum,
    product,
    min,
    max,
    avg,
};

/// Parallelism strategy for model distribution.
pub const ParallelismStrategy = enum {
    /// Split data across devices (batch dimension).
    data_parallel,
    /// Split model across devices (layer dimension).
    model_parallel,
    /// Split within attention heads (tensor parallelism).
    tensor_parallel,
    /// Combination of data and model parallelism.
    hybrid,
};

/// Model partition for model parallelism.
pub const ModelPartition = struct {
    device_id: DeviceId,
    layer_start: usize,
    layer_end: usize,
    is_first: bool,
    is_last: bool,
};

/// GPU Cluster configuration.
pub const GPUClusterConfig = struct {
    /// Device group configuration.
    device_config: MultiDeviceConfig = .{},
    /// Parallelism strategy.
    parallelism: ParallelismStrategy = .data_parallel,
    /// AllReduce algorithm.
    allreduce_algo: AllReduceAlgorithm = .ring,
    /// Enable gradient compression.
    gradient_compression: bool = false,
    /// Compression threshold (only compress tensors larger than this).
    compression_threshold: usize = 1024 * 1024,
    /// Enable overlap of compute and communication.
    overlap_comm: bool = true,
    /// Number of communication streams per device.
    comm_streams: u32 = 2,
};

/// AllReduce algorithm selection.
pub const AllReduceAlgorithm = enum {
    /// Ring AllReduce - good for large tensors.
    ring,
    /// Tree AllReduce - good for many devices.
    tree,
    /// Direct AllReduce - good for small clusters.
    direct,
    /// Bucket AllReduce - fuse small tensors.
    bucket,
};

/// Warning flags for simulated operations.
var warned_simulated_allreduce: bool = false;
var warned_simulated_peer_transfer: bool = false;

/// Log a warning once about simulated multi-GPU operations.
fn warnSimulatedOnce(comptime msg: []const u8, warned: *bool) void {
    if (!warned.*) {
        warned.* = true;
        std.log.warn("[multi_device] " ++ msg, .{});
    }
}

/// GPU Cluster managing multiple GPU contexts.
pub const GPUCluster = struct {
    allocator: std.mem.Allocator,
    config: GPUClusterConfig,
    device_group: DeviceGroup,
    gpu_contexts: std.AutoHashMapUnmanaged(DeviceId, *unified.Gpu),
    comm_buffers: std.AutoHashMapUnmanaged(DeviceId, CommBuffer),
    barrier: ?DeviceBarrier,
    mutex: Mutex,
    /// Peer transfer manager for real GPU-to-GPU transfers.
    peer_manager: ?peer_transfer.PeerTransferManager,

    /// Communication buffer for a device.
    const CommBuffer = struct {
        send_buffer: []f32,
        recv_buffer: []f32,
        temp_buffer: []f32,
    };

    /// Initialize GPU cluster with configuration.
    pub fn init(allocator: std.mem.Allocator, config: GPUClusterConfig) !GPUCluster {
        var cluster = GPUCluster{
            .allocator = allocator,
            .config = config,
            .device_group = try DeviceGroup.init(allocator, config.device_config),
            .gpu_contexts = .{},
            .comm_buffers = .{},
            .barrier = null,
            .mutex = .{},
            .peer_manager = null,
        };

        // Initialize GPU contexts for each device
        for (cluster.device_group.active_devices.items) |device_id| {
            try cluster.initDeviceContext(device_id);
        }

        // Create barrier for synchronization
        if (cluster.device_group.active_devices.items.len > 0) {
            cluster.barrier = DeviceBarrier.init(cluster.device_group.active_devices.items);
        }

        // Initialize peer transfer manager for real GPU-to-GPU transfers
        if (config.device_config.enable_peer_transfer) {
            cluster.peer_manager = peer_transfer.PeerTransferManager.init(allocator, &cluster.device_group) catch null;
        }

        return cluster;
    }

    /// Initialize context for a specific device.
    fn initDeviceContext(self: *GPUCluster, device_id: DeviceId) !void {
        if (self.gpu_contexts.get(device_id) != null) return;

        // Create GPU context for this device
        const gpu = try self.allocator.create(unified.Gpu);
        gpu.* = try unified.Gpu.init(self.allocator, .{});

        try self.gpu_contexts.put(self.allocator, device_id, gpu);
    }

    /// Deinitialize the cluster.
    pub fn deinit(self: *GPUCluster) void {
        // Clean up peer transfer manager
        if (self.peer_manager) |*pm| {
            pm.deinit();
        }

        // Clean up GPU contexts
        var ctx_iter = self.gpu_contexts.iterator();
        while (ctx_iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.gpu_contexts.deinit(self.allocator);

        // Clean up communication buffers
        var buf_iter = self.comm_buffers.iterator();
        while (buf_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.send_buffer);
            self.allocator.free(entry.value_ptr.recv_buffer);
            self.allocator.free(entry.value_ptr.temp_buffer);
        }
        self.comm_buffers.deinit(self.allocator);

        self.device_group.deinit();
        self.* = undefined;
    }

    /// Get GPU context for a device.
    pub fn getContext(self: *GPUCluster, device_id: DeviceId) ?*unified.Gpu {
        return self.gpu_contexts.get(device_id);
    }

    /// Get number of devices in cluster.
    pub fn deviceCount(self: *const GPUCluster) usize {
        return self.device_group.activeDeviceCount();
    }

    /// Distribute work for data parallelism.
    pub fn distributeWork(self: *GPUCluster, total_work: usize) ![]WorkDistribution {
        return self.device_group.distributeWork(total_work);
    }

    /// Create model partitions for model parallelism.
    pub fn partitionModel(self: *GPUCluster, num_layers: usize) ![]ModelPartition {
        const device_count = self.device_group.activeDeviceCount();
        if (device_count == 0) return &.{};

        const layers_per_device = (num_layers + device_count - 1) / device_count;
        var partitions = try self.allocator.alloc(ModelPartition, device_count);

        var layer_idx: usize = 0;
        for (self.device_group.active_devices.items, 0..) |device_id, i| {
            const start = layer_idx;
            const end = @min(layer_idx + layers_per_device, num_layers);

            partitions[i] = .{
                .device_id = device_id,
                .layer_start = start,
                .layer_end = end,
                .is_first = i == 0,
                .is_last = i == device_count - 1,
            };

            layer_idx = end;
        }

        return partitions;
    }

    /// Create tensor parallel partitions for a dimension.
    pub fn partitionTensor(
        self: *GPUCluster,
        tensor_size: usize,
        split_dim_size: usize,
    ) ![]TensorPartition {
        const device_count = self.device_group.activeDeviceCount();
        if (device_count == 0) return &.{};

        const elements_per_device = (split_dim_size + device_count - 1) / device_count;
        var partitions = try self.allocator.alloc(TensorPartition, device_count);

        _ = tensor_size;

        var offset: usize = 0;
        for (self.device_group.active_devices.items, 0..) |device_id, i| {
            const size = @min(elements_per_device, split_dim_size - offset);
            partitions[i] = .{
                .device_id = device_id,
                .offset = offset,
                .size = size,
            };
            offset += size;
        }

        return partitions;
    }

    /// AllReduce operation across all devices.
    /// Synchronizes and reduces data from all devices.
    /// Uses real peer-to-peer transfers when available.
    pub fn allReduce(self: *GPUCluster, data: []f32, op: ReduceOp) !void {
        const device_count = self.device_group.activeDeviceCount();
        if (device_count <= 1) return; // No reduction needed

        // Use peer transfer manager for real GPU-to-GPU transfers if available
        if (self.peer_manager) |*pm| {
            // Create device buffers for all active devices
            var buffers = try self.allocator.alloc(peer_transfer.DeviceBuffer, device_count);
            defer self.allocator.free(buffers);

            for (self.device_group.active_devices.items, 0..) |dev_id, i| {
                buffers[i] = .{
                    .device_id = dev_id,
                    .data = data, // In a real impl, each device would have its own copy
                };
            }

            try pm.allReduceAsync(buffers, op, .{});
            try pm.waitAll();
            return;
        }

        // Fallback to simulated AllReduce
        switch (self.config.allreduce_algo) {
            .ring => try self.ringAllReduce(data, op),
            .tree => try self.treeAllReduce(data, op),
            .direct => try self.directAllReduce(data, op),
            .bucket => try self.bucketAllReduce(data, op),
        }
    }

    /// AllReduce operation with per-device buffers.
    /// This is the preferred API when each device has its own buffer.
    pub fn allReduceBuffers(self: *GPUCluster, buffers: []peer_transfer.DeviceBuffer, op: ReduceOp) !void {
        if (buffers.len <= 1) return; // No reduction needed

        // Use peer transfer manager for real GPU-to-GPU transfers if available
        if (self.peer_manager) |*pm| {
            try pm.allReduceAsync(buffers, op, .{});
            try pm.waitAll();
            return;
        }

        // Fallback: reduce to first buffer using local operations
        const first = buffers[0].data;
        for (buffers[1..]) |buf| {
            for (first, 0..) |*val, i| {
                if (i < buf.data.len) {
                    val.* = applyOp(val.*, buf.data[i], op);
                }
            }
        }

        // Handle average
        if (op == .avg) {
            const scale = 1.0 / @as(f32, @floatFromInt(buffers.len));
            for (first) |*val| {
                val.* *= scale;
            }
        }

        // Copy result back to all buffers
        for (buffers[1..]) |buf| {
            @memcpy(buf.data, first);
        }
    }

    /// Get the peer transfer manager for advanced operations.
    pub fn getPeerManager(self: *GPUCluster) ?*peer_transfer.PeerTransferManager {
        if (self.peer_manager) |*pm| {
            return pm;
        }
        return null;
    }

    /// Get transfer statistics.
    pub fn getTransferStats(self: *const GPUCluster) ?peer_transfer.TransferStats {
        if (self.peer_manager) |pm| {
            return pm.getStats();
        }
        return null;
    }

    /// Ring AllReduce implementation.
    /// NOTE: Currently simulated locally. Real peer-to-peer transfers require
    /// backend-specific implementation (CUDA: cudaMemcpyPeer, Vulkan: VK_KHR_external_memory).
    fn ringAllReduce(self: *GPUCluster, data: []f32, op: ReduceOp) !void {
        const n = self.device_group.activeDeviceCount();
        if (n <= 1) return;

        // Warn once that this is simulated
        warnSimulatedOnce("AllReduce is simulated locally - no real peer transfers. " ++
            "For production multi-GPU, implement backend-specific peer transfers.", &warned_simulated_allreduce);

        const chunk_size = (data.len + n - 1) / n;

        // Phase 1: Reduce-scatter
        // Each device sends its chunk to the next device in the ring
        var phase: usize = 0;
        while (phase < n - 1) : (phase += 1) {
            // Simulate ring communication
            for (0..n) |i| {
                const send_chunk = i;
                const recv_chunk = (i + n - 1) % n;

                const send_start = send_chunk * chunk_size;
                const recv_start = recv_chunk * chunk_size;

                const send_end = @min(send_start + chunk_size, data.len);
                const recv_end = @min(recv_start + chunk_size, data.len);

                // In a real implementation, this would use peer-to-peer transfers
                // For now, simulate the reduction locally
                if (send_start < data.len and recv_start < data.len) {
                    for (recv_start..recv_end) |j| {
                        const src_idx = send_start + (j - recv_start);
                        if (src_idx < send_end) {
                            data[j] = applyOp(data[j], data[src_idx], op);
                        }
                    }
                }
            }
        }

        // Phase 2: Allgather
        // Each device broadcasts its reduced chunk
        // (In simulation, data is already available locally)

        // Synchronize all devices
        if (self.barrier) |*barrier| {
            barrier.wait();
        }
    }

    /// Tree AllReduce implementation (for many devices).
    fn treeAllReduce(self: *GPUCluster, data: []f32, op: ReduceOp) !void {
        const n = self.device_group.activeDeviceCount();
        if (n <= 1) return;

        // Reduce phase: binary tree reduction
        var stride: usize = 1;
        while (stride < n) : (stride *= 2) {
            for (0..n) |i| {
                if (i % (stride * 2) == 0 and i + stride < n) {
                    // Reduce from device i+stride to device i
                    // In real implementation, use peer transfer
                    for (data) |*d| {
                        d.* = applyOp(d.*, d.*, op);
                    }
                }
            }
        }

        // Broadcast phase: reverse tree
        stride = n / 2;
        while (stride >= 1) : (stride /= 2) {
            for (0..n) |i| {
                if (i % (stride * 2) == 0 and i + stride < n) {
                    // Broadcast from device i to device i+stride
                    // In real implementation, use peer transfer
                }
            }
            if (stride == 1) break;
        }

        if (self.barrier) |*barrier| {
            barrier.wait();
        }
    }

    /// Direct AllReduce (for small clusters).
    fn directAllReduce(self: *GPUCluster, data: []f32, op: ReduceOp) !void {
        // All devices send to device 0, which reduces and broadcasts back
        const n = self.device_group.activeDeviceCount();
        if (n <= 1) return;

        // In real implementation:
        // 1. All devices send to device 0
        // 2. Device 0 reduces all data
        // 3. Device 0 broadcasts result

        // For simulation, apply reduction locally
        for (data) |*d| {
            // Simulate n-way reduction
            var val = d.*;
            for (1..n) |_| {
                val = applyOp(val, d.*, op);
            }
            if (op == .avg) {
                val /= @as(f32, @floatFromInt(n));
            }
            d.* = val;
        }

        if (self.barrier) |*barrier| {
            barrier.wait();
        }
    }

    /// Bucket AllReduce (fuses small tensors).
    fn bucketAllReduce(self: *GPUCluster, data: []f32, op: ReduceOp) !void {
        // For small tensors, just use direct method
        if (data.len < self.config.compression_threshold) {
            try self.directAllReduce(data, op);
            return;
        }

        // For larger tensors, use ring
        try self.ringAllReduce(data, op);
    }

    /// Apply reduction operation.
    fn applyOp(a: f32, b: f32, op: ReduceOp) f32 {
        return switch (op) {
            .sum => a + b,
            .product => a * b,
            .min => @min(a, b),
            .max => @max(a, b),
            .avg => a + b, // Division happens at the end
        };
    }

    /// Broadcast data from source device to all devices.
    pub fn broadcast(self: *GPUCluster, data: []f32, source_device: DeviceId) !void {
        _ = source_device;

        // In real implementation, use peer transfers or staged copies
        // For now, data is already available locally

        if (self.barrier) |*barrier| {
            barrier.wait();
        }

        // Ensure data is visible to all devices
        std.atomic.fence(.seq_cst);
        _ = data;
    }

    /// Scatter data to all devices (each gets a chunk).
    pub fn scatter(
        self: *GPUCluster,
        data: []const f32,
        chunk_size: usize,
    ) ![]DeviceChunk {
        const n = self.device_group.activeDeviceCount();
        if (n == 0) return &.{};

        var chunks = try self.allocator.alloc(DeviceChunk, n);

        for (self.device_group.active_devices.items, 0..) |device_id, i| {
            const start = i * chunk_size;
            const end = @min(start + chunk_size, data.len);

            if (start < data.len) {
                const chunk_data = try self.allocator.alloc(f32, end - start);
                @memcpy(chunk_data, data[start..end]);

                chunks[i] = .{
                    .device_id = device_id,
                    .data = chunk_data,
                };
            } else {
                chunks[i] = .{
                    .device_id = device_id,
                    .data = &.{},
                };
            }
        }

        return chunks;
    }

    /// Gather data from all devices into single buffer.
    pub fn gather(
        self: *GPUCluster,
        chunks: []const DeviceChunk,
        output: []f32,
    ) !void {
        _ = self; // Used for future peer transfer optimization

        var offset: usize = 0;
        for (chunks) |chunk| {
            if (offset + chunk.data.len > output.len) break;
            @memcpy(output[offset..][0..chunk.data.len], chunk.data);
            offset += chunk.data.len;
        }
    }

    /// Synchronize all devices.
    pub fn synchronize(self: *GPUCluster) void {
        if (self.barrier) |*barrier| {
            barrier.wait();
        }
        std.atomic.fence(.seq_cst);
    }

    /// Get cluster statistics.
    pub fn getStats(self: *const GPUCluster) ClusterStats {
        const group_stats = self.device_group.getStats();

        return .{
            .device_count = group_stats.device_count,
            .active_device_count = group_stats.active_device_count,
            .total_memory_mb = group_stats.total_memory_mb,
            .available_memory_mb = group_stats.available_memory_mb,
            .total_compute_units = group_stats.total_compute_units,
            .parallelism = self.config.parallelism,
            .allreduce_algo = self.config.allreduce_algo,
        };
    }
};

/// Tensor partition for tensor parallelism.
pub const TensorPartition = struct {
    device_id: DeviceId,
    offset: usize,
    size: usize,
};

/// Device chunk for scatter/gather.
pub const DeviceChunk = struct {
    device_id: DeviceId,
    data: []f32,
};

/// Cluster statistics.
pub const ClusterStats = struct {
    device_count: usize,
    active_device_count: usize,
    total_memory_mb: u64,
    available_memory_mb: u64,
    total_compute_units: u32,
    parallelism: ParallelismStrategy,
    allreduce_algo: AllReduceAlgorithm,
};
