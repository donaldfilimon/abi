//! Multi-GPU device management and coordination.
//!
//! Provides support for managing multiple GPU devices, distributing
//! workloads across devices, coordinating memory transfers, and
//! synchronizing gradient updates for distributed training.
//!
//! ## Features
//!
//! - **Device Discovery**: Enumerate GPUs across all backends
//! - **Load Balancing**: Round-robin, memory-aware, capability-weighted
//! - **Work Distribution**: Data parallelism and model parallelism
//! - **Peer Transfers**: GPU-to-GPU memory copies
//! - **AllReduce**: Gradient synchronization for distributed training
//! - **Barriers**: Cross-device synchronization primitives
//!
//! ## Usage
//!
//! ```zig
//! const multi = @import("multi_device.zig");
//!
//! // Create GPU cluster
//! var cluster = try multi.GPUCluster.init(allocator, .{});
//! defer cluster.deinit();
//!
//! // Distribute work across devices
//! const work = try cluster.distributeWork(total_elements);
//! defer allocator.free(work);
//!
//! // AllReduce for gradient sync
//! try cluster.allReduce(gradients, .sum);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const unified = @import("unified.zig");
const peer_transfer = @import("peer_transfer/mod.zig");

const sync = @import("../../services/shared/sync.zig");
const Mutex = sync.Mutex;

/// Device identifier.
pub const DeviceId = u32;

/// Device type.
pub const DeviceType = enum {
    discrete_gpu,
    integrated_gpu,
    virtual_gpu,
    cpu_fallback,
    unknown,

    pub fn priority(self: DeviceType) u8 {
        return switch (self) {
            .discrete_gpu => 4,
            .integrated_gpu => 3,
            .virtual_gpu => 2,
            .cpu_fallback => 1,
            .unknown => 0,
        };
    }
};

/// Device capabilities.
pub const DeviceCapabilities = struct {
    compute_units: u32 = 0,
    max_memory_mb: u64 = 0,
    max_work_group_size: u32 = 256,
    supports_fp16: bool = false,
    supports_fp64: bool = false,
    supports_int8: bool = false,
    supports_atomics: bool = true,
    max_shared_memory_kb: u32 = 48,
    warp_size: u32 = 32,
};

/// Device information.
pub const DeviceInfo = struct {
    id: DeviceId,
    name: [128]u8 = [_]u8{0} ** 128,
    name_len: usize = 0,
    device_type: DeviceType = .unknown,
    capabilities: DeviceCapabilities = .{},
    available_memory_mb: u64 = 0,
    utilization_percent: u8 = 0,
    temperature_celsius: ?u8 = null,
    power_usage_watts: ?u32 = null,

    pub fn getName(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }
};

/// Load balancing strategy.
pub const LoadBalanceStrategy = enum {
    /// Round-robin distribution.
    round_robin,
    /// Distribute to device with most free memory.
    memory_aware,
    /// Distribute to least utilized device.
    least_loaded,
    /// Distribute based on device compute capability.
    capability_weighted,
    /// Pin to single device.
    pinned,
};

/// Multi-device configuration.
pub const MultiDeviceConfig = struct {
    /// Load balancing strategy.
    strategy: LoadBalanceStrategy = .memory_aware,
    /// Enable peer-to-peer transfers.
    enable_peer_transfer: bool = true,
    /// Minimum work size to distribute (smaller goes to single device).
    min_distribute_size: usize = 1024 * 1024,
    /// Enable automatic device selection.
    auto_select: bool = true,
    /// Preferred device IDs (empty = use all).
    preferred_devices: []const DeviceId = &.{},
};

/// Device group for multi-GPU operations.
pub const DeviceGroup = struct {
    allocator: std.mem.Allocator,
    config: MultiDeviceConfig,
    devices: std.ArrayListUnmanaged(DeviceInfo),
    active_devices: std.ArrayListUnmanaged(DeviceId),
    round_robin_counter: std.atomic.Value(u64),
    mutex: Mutex,

    /// Initialize a device group with discovered devices.
    pub fn init(allocator: std.mem.Allocator, config: MultiDeviceConfig) !DeviceGroup {
        var group = DeviceGroup{
            .allocator = allocator,
            .config = config,
            .devices = .{},
            .active_devices = .{},
            .round_robin_counter = std.atomic.Value(u64).init(0),
            .mutex = .{},
        };

        try group.discoverDevices();
        return group;
    }

    /// Deinitialize the device group.
    pub fn deinit(self: *DeviceGroup) void {
        self.devices.deinit(self.allocator);
        self.active_devices.deinit(self.allocator);
        self.* = undefined;
    }

    /// Discover available GPU devices.
    pub fn discoverDevices(self: *DeviceGroup) !void {
        self.devices.clearRetainingCapacity();
        self.active_devices.clearRetainingCapacity();

        // Always add CPU fallback
        const cpu_device = DeviceInfo{
            .id = 0,
            .name = blk: {
                var name: [128]u8 = [_]u8{0} ** 128;
                const src = "CPU Fallback";
                @memcpy(name[0..src.len], src);
                break :blk name;
            },
            .name_len = 12,
            .device_type = .cpu_fallback,
            .capabilities = .{
                .compute_units = 1,
                .max_memory_mb = 4096,
                .supports_fp64 = true,
            },
            .available_memory_mb = 4096,
        };
        try self.devices.append(self.allocator, cpu_device);
        try self.active_devices.append(self.allocator, 0);

        // In a real implementation, discover GPUs from backends
        // For now, add simulated devices for testing
        if (build_options.enable_gpu) {
            // Simulated GPU device
            const gpu_device = DeviceInfo{
                .id = 1,
                .name = blk: {
                    var name: [128]u8 = [_]u8{0} ** 128;
                    const src = "Simulated GPU";
                    @memcpy(name[0..src.len], src);
                    break :blk name;
                },
                .name_len = 13,
                .device_type = .discrete_gpu,
                .capabilities = .{
                    .compute_units = 32,
                    .max_memory_mb = 8192,
                    .supports_fp16 = true,
                    .supports_fp64 = true,
                },
                .available_memory_mb = 8192,
            };
            try self.devices.append(self.allocator, gpu_device);
            try self.active_devices.append(self.allocator, 1);
        }
    }

    /// Get total number of devices.
    pub fn deviceCount(self: *const DeviceGroup) usize {
        return self.devices.items.len;
    }

    /// Get number of active devices.
    pub fn activeDeviceCount(self: *const DeviceGroup) usize {
        return self.active_devices.items.len;
    }

    /// Get device info by ID.
    pub fn getDevice(self: *const DeviceGroup, id: DeviceId) ?*const DeviceInfo {
        for (self.devices.items) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    /// Get all devices.
    pub fn getAllDevices(self: *const DeviceGroup) []const DeviceInfo {
        return self.devices.items;
    }

    /// Select best device for a task.
    pub fn selectDevice(self: *DeviceGroup, work_size: usize) DeviceId {
        if (self.active_devices.items.len == 0) return 0;
        if (self.active_devices.items.len == 1) return self.active_devices.items[0];

        return switch (self.config.strategy) {
            .round_robin => self.selectRoundRobin(),
            .memory_aware => self.selectMemoryAware(work_size),
            .least_loaded => self.selectLeastLoaded(),
            .capability_weighted => self.selectCapabilityWeighted(),
            .pinned => self.active_devices.items[0],
        };
    }

    /// Distribute work across multiple devices.
    pub fn distributeWork(self: *DeviceGroup, total_work: usize) ![]WorkDistribution {
        if (self.active_devices.items.len == 0) {
            return &.{};
        }

        // If work is too small, use single device
        if (total_work < self.config.min_distribute_size) {
            var result = try self.allocator.alloc(WorkDistribution, 1);
            result[0] = .{
                .device_id = self.selectDevice(total_work),
                .offset = 0,
                .size = total_work,
            };
            return result;
        }

        // Distribute based on device capabilities
        var total_weight: u64 = 0;
        for (self.active_devices.items) |id| {
            if (self.getDevice(id)) |device| {
                total_weight += device.capabilities.compute_units;
            }
        }

        if (total_weight == 0) total_weight = self.active_devices.items.len;

        var result = try self.allocator.alloc(WorkDistribution, self.active_devices.items.len);
        var offset: usize = 0;

        for (self.active_devices.items, 0..) |id, i| {
            const weight: usize = if (self.getDevice(id)) |device|
                device.capabilities.compute_units
            else
                1;

            const share = (total_work * weight) / total_weight;
            const size = if (i == self.active_devices.items.len - 1)
                total_work - offset
            else
                share;

            result[i] = .{
                .device_id = id,
                .offset = offset,
                .size = size,
            };
            offset += size;
        }

        return result;
    }

    /// Enable a device.
    pub fn enableDevice(self: *DeviceGroup, id: DeviceId) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.active_devices.items) |active_id| {
            if (active_id == id) return; // Already active
        }

        try self.active_devices.append(self.allocator, id);
    }

    /// Disable a device.
    pub fn disableDevice(self: *DeviceGroup, id: DeviceId) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.active_devices.items, 0..) |active_id, i| {
            if (active_id == id) {
                _ = self.active_devices.swapRemove(i);
                return;
            }
        }
    }

    /// Get group statistics.
    pub fn getStats(self: *const DeviceGroup) GroupStats {
        var total_memory: u64 = 0;
        var available_memory: u64 = 0;
        var total_compute_units: u32 = 0;

        for (self.devices.items) |device| {
            total_memory += device.capabilities.max_memory_mb;
            available_memory += device.available_memory_mb;
            total_compute_units += device.capabilities.compute_units;
        }

        return .{
            .device_count = self.devices.items.len,
            .active_device_count = self.active_devices.items.len,
            .total_memory_mb = total_memory,
            .available_memory_mb = available_memory,
            .total_compute_units = total_compute_units,
        };
    }

    // Selection strategies
    fn selectRoundRobin(self: *DeviceGroup) DeviceId {
        const counter = self.round_robin_counter.fetchAdd(1, .monotonic);
        const idx = counter % self.active_devices.items.len;
        return self.active_devices.items[idx];
    }

    fn selectMemoryAware(self: *DeviceGroup, work_size: usize) DeviceId {
        _ = work_size;
        var best_id: DeviceId = self.active_devices.items[0];
        var best_memory: u64 = 0;

        for (self.active_devices.items) |id| {
            if (self.getDevice(id)) |device| {
                if (device.available_memory_mb > best_memory) {
                    best_memory = device.available_memory_mb;
                    best_id = id;
                }
            }
        }
        return best_id;
    }

    fn selectLeastLoaded(self: *DeviceGroup) DeviceId {
        var best_id: DeviceId = self.active_devices.items[0];
        var best_util: u8 = 100;

        for (self.active_devices.items) |id| {
            if (self.getDevice(id)) |device| {
                if (device.utilization_percent < best_util) {
                    best_util = device.utilization_percent;
                    best_id = id;
                }
            }
        }
        return best_id;
    }

    fn selectCapabilityWeighted(self: *DeviceGroup) DeviceId {
        var best_id: DeviceId = self.active_devices.items[0];
        var best_score: u32 = 0;

        for (self.active_devices.items) |id| {
            if (self.getDevice(id)) |device| {
                const score = device.capabilities.compute_units *
                    device.device_type.priority();
                if (score > best_score) {
                    best_score = score;
                    best_id = id;
                }
            }
        }
        return best_id;
    }
};

/// Work distribution for a device.
pub const WorkDistribution = struct {
    device_id: DeviceId,
    offset: usize,
    size: usize,
};

/// Group statistics.
pub const GroupStats = struct {
    device_count: usize,
    active_device_count: usize,
    total_memory_mb: u64,
    available_memory_mb: u64,
    total_compute_units: u32,
};

/// Peer transfer configuration.
pub const PeerTransferConfig = struct {
    source_device: DeviceId,
    target_device: DeviceId,
    size: usize,
    async_transfer: bool = true,
};

/// Cross-device memory transfer.
pub const PeerTransfer = struct {
    allocator: std.mem.Allocator,
    config: PeerTransferConfig,
    completed: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: PeerTransferConfig) PeerTransfer {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Execute the transfer (stub implementation).
    pub fn execute(self: *PeerTransfer) !void {
        // In a real implementation, this would perform GPU-to-GPU transfer
        // For now, just mark as completed
        self.completed = true;
    }

    /// Wait for transfer to complete.
    pub fn wait(self: *PeerTransfer) !void {
        while (!self.completed) {
            std.Thread.yield() catch |err| {
                std.log.debug("Thread.yield failed during PeerTransfer.wait: {t}", .{err});
            };
        }
    }

    /// Check if transfer is complete.
    pub fn isComplete(self: *const PeerTransfer) bool {
        return self.completed;
    }
};

/// Device synchronization barrier.
pub const DeviceBarrier = struct {
    devices: []const DeviceId,
    count: std.atomic.Value(usize),
    generation: std.atomic.Value(u64),

    pub fn init(devices: []const DeviceId) DeviceBarrier {
        return .{
            .devices = devices,
            .count = std.atomic.Value(usize).init(0),
            .generation = std.atomic.Value(u64).init(0),
        };
    }

    /// Wait at the barrier.
    pub fn wait(self: *DeviceBarrier) void {
        const gen = self.generation.load(.acquire);
        const arrived = self.count.fetchAdd(1, .acq_rel) + 1;

        if (arrived == self.devices.len) {
            // Last to arrive, release all
            self.count.store(0, .release);
            _ = self.generation.fetchAdd(1, .release);
        } else {
            // Wait for generation to change
            while (self.generation.load(.acquire) == gen) {
                std.atomic.spinLoopHint();
            }
        }
    }
};

// ============================================================================
// GPU Cluster - Multi-GPU Context Management
// ============================================================================

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
        gpu.* = try unified.Gpu.init(self.allocator, .{
            .device_index = device_id,
        });

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

// ============================================================================
// Gradient Bucket Manager for Efficient AllReduce
// ============================================================================

/// Gradient bucket for fusing small gradients.
pub const GradientBucket = struct {
    allocator: std.mem.Allocator,
    buffer: []f32,
    capacity: usize,
    used: usize,
    offsets: std.ArrayListUnmanaged(GradientOffset),
    ready: bool,

    const GradientOffset = struct {
        param_id: usize,
        offset: usize,
        size: usize,
    };

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !GradientBucket {
        return .{
            .allocator = allocator,
            .buffer = try allocator.alloc(f32, capacity),
            .capacity = capacity,
            .used = 0,
            .offsets = .{},
            .ready = false,
        };
    }

    pub fn deinit(self: *GradientBucket) void {
        self.allocator.free(self.buffer);
        self.offsets.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add gradient to bucket.
    pub fn add(self: *GradientBucket, param_id: usize, gradient: []const f32) !bool {
        if (self.used + gradient.len > self.capacity) {
            return false; // Bucket full
        }

        @memcpy(self.buffer[self.used..][0..gradient.len], gradient);

        try self.offsets.append(self.allocator, .{
            .param_id = param_id,
            .offset = self.used,
            .size = gradient.len,
        });

        self.used += gradient.len;
        return true;
    }

    /// Mark bucket as ready for AllReduce.
    pub fn markReady(self: *GradientBucket) void {
        self.ready = true;
    }

    /// Get gradient data for AllReduce.
    pub fn getData(self: *const GradientBucket) []f32 {
        return self.buffer[0..self.used];
    }

    /// Extract gradient after AllReduce.
    pub fn extractGradient(self: *const GradientBucket, param_id: usize) ?[]const f32 {
        for (self.offsets.items) |offset| {
            if (offset.param_id == param_id) {
                return self.buffer[offset.offset..][0..offset.size];
            }
        }
        return null;
    }

    /// Reset bucket for reuse.
    pub fn reset(self: *GradientBucket) void {
        self.used = 0;
        self.offsets.clearRetainingCapacity();
        self.ready = false;
    }
};

/// Gradient bucket manager for efficient AllReduce.
pub const GradientBucketManager = struct {
    allocator: std.mem.Allocator,
    buckets: std.ArrayListUnmanaged(GradientBucket),
    bucket_size: usize,
    current_bucket: usize,

    pub fn init(allocator: std.mem.Allocator, num_buckets: usize, bucket_size: usize) !GradientBucketManager {
        var manager = GradientBucketManager{
            .allocator = allocator,
            .buckets = .{},
            .bucket_size = bucket_size,
            .current_bucket = 0,
        };

        for (0..num_buckets) |_| {
            const bucket = try GradientBucket.init(allocator, bucket_size);
            try manager.buckets.append(allocator, bucket);
        }

        return manager;
    }

    pub fn deinit(self: *GradientBucketManager) void {
        for (self.buckets.items) |*bucket| {
            bucket.deinit();
        }
        self.buckets.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add gradient to appropriate bucket.
    pub fn addGradient(self: *GradientBucketManager, param_id: usize, gradient: []const f32) !void {
        // Try current bucket first
        if (self.current_bucket < self.buckets.items.len) {
            if (try self.buckets.items[self.current_bucket].add(param_id, gradient)) {
                return;
            }
            // Bucket full, mark ready and move to next
            self.buckets.items[self.current_bucket].markReady();
            self.current_bucket += 1;
        }

        // Try to add to new bucket
        if (self.current_bucket < self.buckets.items.len) {
            _ = try self.buckets.items[self.current_bucket].add(param_id, gradient);
        }
    }

    /// Get all ready buckets for AllReduce.
    pub fn getReadyBuckets(self: *GradientBucketManager) []GradientBucket {
        var ready_count: usize = 0;
        for (self.buckets.items) |bucket| {
            if (bucket.ready) ready_count += 1;
        }

        // Return slice of ready buckets
        return self.buckets.items[0..ready_count];
    }

    /// Reset all buckets for next iteration.
    pub fn reset(self: *GradientBucketManager) void {
        for (self.buckets.items) |*bucket| {
            bucket.reset();
        }
        self.current_bucket = 0;
    }
};

test "device group creation" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{});
    defer group.deinit();

    try std.testing.expect(group.deviceCount() >= 1);
    try std.testing.expect(group.activeDeviceCount() >= 1);
}

test "device selection" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{ .strategy = .round_robin });
    defer group.deinit();

    const id1 = group.selectDevice(1024);
    const id2 = group.selectDevice(1024);

    // With round robin, different calls should potentially select different devices
    // (depends on device count)
    _ = id1;
    _ = id2;
}

test "work distribution" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{ .min_distribute_size = 100 });
    defer group.deinit();

    // Small work - single device
    const dist1 = try group.distributeWork(50);
    defer allocator.free(dist1);
    try std.testing.expectEqual(@as(usize, 1), dist1.len);

    // Larger work - may be distributed
    const dist2 = try group.distributeWork(1000);
    defer allocator.free(dist2);
    try std.testing.expect(dist2.len >= 1);
}

test "group stats" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{});
    defer group.deinit();

    const stats = group.getStats();
    try std.testing.expect(stats.device_count >= 1);
    try std.testing.expect(stats.total_memory_mb > 0);
}

test "gpu cluster creation" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        // GPU initialization may fail in test environment
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    try std.testing.expect(cluster.deviceCount() >= 1);

    const stats = cluster.getStats();
    try std.testing.expect(stats.device_count >= 1);
}

test "model partitioning" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const partitions = try cluster.partitionModel(12);
    defer allocator.free(partitions);

    try std.testing.expect(partitions.len >= 1);

    // First partition should start at layer 0
    if (partitions.len > 0) {
        try std.testing.expectEqual(@as(usize, 0), partitions[0].layer_start);
        try std.testing.expect(partitions[0].is_first);
    }

    // Last partition should end at last layer
    if (partitions.len > 0) {
        try std.testing.expect(partitions[partitions.len - 1].is_last);
    }
}

test "tensor partitioning" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const partitions = try cluster.partitionTensor(1024, 256);
    defer allocator.free(partitions);

    try std.testing.expect(partitions.len >= 1);

    // Sum of partition sizes should equal split_dim_size
    var total_size: usize = 0;
    for (partitions) |p| {
        total_size += p.size;
    }
    try std.testing.expectEqual(@as(usize, 256), total_size);
}

test "allreduce operations" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{
        .allreduce_algo = .direct,
    }) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // AllReduce should work (though with 1 device it's a no-op)
    try cluster.allReduce(&data, .sum);
}

test "gradient bucket" {
    const allocator = std.testing.allocator;
    var bucket = try GradientBucket.init(allocator, 100);
    defer bucket.deinit();

    const grad1 = [_]f32{ 0.1, 0.2, 0.3 };
    const grad2 = [_]f32{ 0.4, 0.5 };

    try std.testing.expect(try bucket.add(0, &grad1));
    try std.testing.expect(try bucket.add(1, &grad2));

    try std.testing.expectEqual(@as(usize, 5), bucket.used);

    const extracted = bucket.extractGradient(0).?;
    try std.testing.expectEqual(@as(usize, 3), extracted.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), extracted[0], 1e-6);
}

test "gradient bucket manager" {
    const allocator = std.testing.allocator;
    var manager = try GradientBucketManager.init(allocator, 3, 50);
    defer manager.deinit();

    const grad1 = [_]f32{ 0.1, 0.2, 0.3 };
    try manager.addGradient(0, &grad1);

    // Reset and verify
    manager.reset();
    try std.testing.expectEqual(@as(usize, 0), manager.current_bucket);
}

test "scatter gather" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const chunks = try cluster.scatter(&data, 3);
    defer {
        for (chunks) |chunk| {
            if (chunk.data.len > 0) {
                allocator.free(chunk.data);
            }
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len >= 1);

    // Gather back
    var output: [6]f32 = undefined;
    try cluster.gather(chunks, &output);

    // Should have some data
    try std.testing.expect(output[0] == 1.0);
}
