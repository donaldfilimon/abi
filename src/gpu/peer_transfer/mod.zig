//! GPU Peer Transfer Module
//!
//! Provides real GPU-to-GPU peer transfers with backend-specific optimizations
//! and automatic fallback to host-staged transfers when direct P2P is unavailable.
//!
//! ## Supported Backends
//!
//! - **CUDA/NCCL**: Direct P2P via NVLink/PCIe, NCCL for optimized AllReduce
//! - **Vulkan**: VK_KHR_external_memory for zero-copy, compute shader reduce
//! - **Metal**: MTLSharedEvent for Apple Silicon unified memory
//! - **Host-Staged**: Universal fallback through pinned host memory
//!
//! ## Usage
//!
//! ```zig
//! const peer = @import("peer_transfer/mod.zig");
//!
//! var manager = try peer.PeerTransferManager.init(allocator, device_group);
//! defer manager.deinit();
//!
//! // Async transfer
//! const handle = try manager.transferAsync(src_device, dst_device, data, .{});
//! try manager.waitForTransfer(handle);
//!
//! // AllReduce
//! try manager.allReduceAsync(buffers, .sum, .{});
//! try manager.waitAll();
//! ```

const std = @import("std");
const build_options = @import("build_options");
const shared_utils = @import("../../shared/utils.zig");

const multi_device = @import("../multi_device.zig");
const stream_mod = @import("../stream.zig");
const backend_mod = @import("../backend.zig");
const device_mod = @import("../device.zig");

// Backend implementations
pub const host_staged = @import("host_staged.zig");
pub const cuda_backend = if (build_options.gpu_cuda) @import("cuda.zig") else struct {};
pub const vulkan_backend = if (build_options.gpu_vulkan) @import("vulkan.zig") else struct {};
pub const metal_backend = if (build_options.gpu_metal) @import("metal.zig") else struct {};

pub const DeviceId = multi_device.DeviceId;
pub const DeviceGroup = multi_device.DeviceGroup;
pub const ReduceOp = multi_device.ReduceOp;
pub const Backend = backend_mod.Backend;
pub const Stream = stream_mod.Stream;
pub const Event = stream_mod.Event;

/// Transfer capability between two devices.
pub const TransferCapability = enum {
    /// Direct P2P via NVLink or PCIe peer access
    direct_p2p,
    /// NVIDIA NCCL for optimized collective ops
    nccl,
    /// Vulkan external memory extension
    vulkan_external,
    /// Metal shared events (Apple Silicon unified memory)
    metal_shared,
    /// Fallback through host RAM (always available)
    host_staged,

    pub fn priority(self: TransferCapability) u8 {
        return switch (self) {
            .nccl => 5, // Highest priority - optimized for multi-GPU
            .direct_p2p => 4, // Direct hardware path
            .vulkan_external => 3, // Zero-copy when available
            .metal_shared => 3, // Unified memory on Apple Silicon
            .host_staged => 1, // Always works, lowest priority
        };
    }

    pub fn name(self: TransferCapability) []const u8 {
        return switch (self) {
            .direct_p2p => "Direct P2P",
            .nccl => "NCCL",
            .vulkan_external => "Vulkan External Memory",
            .metal_shared => "Metal Shared",
            .host_staged => "Host Staged",
        };
    }
};

/// Device pair key for capability lookups.
pub const DevicePair = struct {
    src: DeviceId,
    dst: DeviceId,

    pub fn hash(self: DevicePair) u64 {
        return @as(u64, self.src) << 32 | @as(u64, self.dst);
    }

    pub fn eql(a: DevicePair, b: DevicePair) bool {
        return a.src == b.src and a.dst == b.dst;
    }
};

/// Transfer status.
pub const TransferStatus = enum {
    pending,
    in_progress,
    completed,
    failed,
    cancelled,
};

/// Transfer handle for async operations.
pub const TransferHandle = struct {
    id: u64,
    src_device: DeviceId,
    dst_device: DeviceId,
    size: usize,
    capability: TransferCapability,
    status: std.atomic.Value(TransferStatus),
    error_info: ?TransferError = null,
    start_time: i64,
    completion_time: ?i64 = null,
    event: ?*Event = null,

    pub fn isComplete(self: *const TransferHandle) bool {
        const status = self.status.load(.acquire);
        return status == .completed or status == .failed or status == .cancelled;
    }

    pub fn succeeded(self: *const TransferHandle) bool {
        return self.status.load(.acquire) == .completed;
    }
};

/// Transfer error types.
pub const TransferError = error{
    DeviceNotFound,
    PeerAccessDenied,
    OutOfMemory,
    TransferTimeout,
    DeviceLost,
    PartialTransfer,
    InvalidBuffer,
    BackendError,
    StreamError,
    Cancelled,
};

/// Recovery strategy for transfer failures.
pub const RecoveryStrategy = enum {
    /// Fail immediately on any error
    abort,
    /// Retry failed transfers via host-staged (default)
    retry_with_fallback,
    /// Continue without failed device
    skip_failed_device,
};

/// Transfer priority.
pub const Priority = enum {
    low,
    normal,
    high,
    critical,

    pub fn value(self: Priority) i32 {
        return switch (self) {
            .low => -1,
            .normal => 0,
            .high => 1,
            .critical => 2,
        };
    }
};

/// Transfer options.
pub const TransferOptions = struct {
    /// Stream for async operations (null = default stream)
    stream: ?*Stream = null,
    /// Transfer priority
    priority: Priority = .normal,
    /// Allow compute to overlap with transfer
    overlap_compute: bool = true,
    /// Timeout in milliseconds (0 = no timeout)
    timeout_ms: u32 = 30000,
    /// Force use of specific capability (null = auto-select)
    force_capability: ?TransferCapability = null,
};

/// Device buffer descriptor for AllReduce operations.
pub const DeviceBuffer = struct {
    device_id: DeviceId,
    data: []f32,
    /// Backend-specific handle (GPU buffer pointer)
    handle: ?*anyopaque = null,
};

/// Peer transfer statistics.
pub const TransferStats = struct {
    total_transfers: u64 = 0,
    successful_transfers: u64 = 0,
    failed_transfers: u64 = 0,
    bytes_transferred: u64 = 0,
    fallback_count: u64 = 0,
    total_time_ns: u64 = 0,

    pub fn successRate(self: TransferStats) f64 {
        if (self.total_transfers == 0) return 1.0;
        return @as(f64, @floatFromInt(self.successful_transfers)) /
            @as(f64, @floatFromInt(self.total_transfers));
    }

    pub fn avgTransferTimeNs(self: TransferStats) u64 {
        if (self.successful_transfers == 0) return 0;
        return self.total_time_ns / self.successful_transfers;
    }

    pub fn throughputBytesPerSec(self: TransferStats) f64 {
        if (self.total_time_ns == 0) return 0.0;
        const seconds = @as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000_000.0;
        return @as(f64, @floatFromInt(self.bytes_transferred)) / seconds;
    }
};

/// Peer Transfer Manager
///
/// Manages GPU-to-GPU transfers with automatic capability detection,
/// backend selection, and fallback handling.
pub const PeerTransferManager = struct {
    allocator: std.mem.Allocator,
    device_group: *DeviceGroup,
    capabilities: std.AutoHashMap(u64, TransferCapability),
    active_transfers: std.ArrayListUnmanaged(TransferHandle),
    recovery_strategy: RecoveryStrategy,
    stats: TransferStats,
    mutex: std.Thread.Mutex,
    next_transfer_id: std.atomic.Value(u64),

    // Backend instances
    host_staged_backend: host_staged.HostStagedBackend,

    const Self = @This();

    /// Initialize the peer transfer manager.
    pub fn init(allocator: std.mem.Allocator, device_group: *DeviceGroup) !Self {
        var manager = Self{
            .allocator = allocator,
            .device_group = device_group,
            .capabilities = std.AutoHashMap(u64, TransferCapability).init(allocator),
            .active_transfers = .{},
            .recovery_strategy = .retry_with_fallback,
            .stats = .{},
            .mutex = .{},
            .next_transfer_id = std.atomic.Value(u64).init(1),
            .host_staged_backend = try host_staged.HostStagedBackend.init(allocator),
        };

        // Probe capabilities for all device pairs
        try manager.probeCapabilities();

        return manager;
    }

    /// Deinitialize the manager.
    pub fn deinit(self: *Self) void {
        self.host_staged_backend.deinit();
        self.capabilities.deinit();
        self.active_transfers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Probe peer transfer capabilities for all device pairs.
    fn probeCapabilities(self: *Self) !void {
        const devices = self.device_group.active_devices.items;

        for (devices) |src_id| {
            for (devices) |dst_id| {
                if (src_id == dst_id) continue;

                const capability = try self.probeDevicePair(src_id, dst_id);
                const pair = DevicePair{ .src = src_id, .dst = dst_id };
                try self.capabilities.put(pair.hash(), capability);
            }
        }
    }

    /// Probe capability for a specific device pair.
    fn probeDevicePair(self: *Self, src_id: DeviceId, dst_id: DeviceId) !TransferCapability {
        const src_info = self.device_group.getDevice(src_id);
        const dst_info = self.device_group.getDevice(dst_id);

        if (src_info == null or dst_info == null) {
            return .host_staged;
        }

        // Try backends in priority order
        if (comptime build_options.gpu_cuda) {
            if (cuda_backend.canAccessPeer(src_id, dst_id)) {
                if (cuda_backend.hasNCCL()) {
                    return .nccl;
                }
                return .direct_p2p;
            }
        }

        if (comptime build_options.gpu_vulkan) {
            if (vulkan_backend.hasExternalMemory(src_id, dst_id)) {
                return .vulkan_external;
            }
        }

        if (comptime build_options.gpu_metal) {
            if (metal_backend.hasSharedEvents(src_id, dst_id)) {
                return .metal_shared;
            }
        }

        // Fallback is always available
        return .host_staged;
    }

    /// Get the best capability for a device pair.
    pub fn getCapability(self: *Self, src_id: DeviceId, dst_id: DeviceId) TransferCapability {
        const pair = DevicePair{ .src = src_id, .dst = dst_id };
        return self.capabilities.get(pair.hash()) orelse .host_staged;
    }

    /// Initiate an async transfer between devices.
    pub fn transferAsync(
        self: *Self,
        src_device: DeviceId,
        dst_device: DeviceId,
        data: []u8,
        opts: TransferOptions,
    ) !*TransferHandle {
        const capability = opts.force_capability orelse self.getCapability(src_device, dst_device);
        const transfer_id = self.next_transfer_id.fetchAdd(1, .monotonic);
        const now = shared_utils.unixMs();

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.active_transfers.append(.{
            .id = transfer_id,
            .src_device = src_device,
            .dst_device = dst_device,
            .size = data.len,
            .capability = capability,
            .status = std.atomic.Value(TransferStatus).init(.pending),
            .start_time = now,
        });

        const handle = &self.active_transfers.items[self.active_transfers.items.len - 1];

        // Start transfer based on capability
        self.startTransfer(handle, data, opts) catch |err| {
            handle.status.store(.failed, .release);
            handle.error_info = err;

            // Try fallback if enabled
            if (self.recovery_strategy == .retry_with_fallback and capability != .host_staged) {
                self.stats.fallback_count += 1;
                try self.host_staged_backend.transfer(src_device, dst_device, data);
                handle.status.store(.completed, .release);
                handle.completion_time = shared_utils.unixMs();
            }
        };

        self.stats.total_transfers += 1;

        return handle;
    }

    /// Start a transfer using the appropriate backend.
    fn startTransfer(self: *Self, handle: *TransferHandle, data: []u8, opts: TransferOptions) !void {
        handle.status.store(.in_progress, .release);

        switch (handle.capability) {
            .direct_p2p => {
                if (comptime build_options.gpu_cuda) {
                    try cuda_backend.memcpyPeerAsync(
                        handle.src_device,
                        handle.dst_device,
                        data,
                        opts.stream,
                    );
                } else {
                    return error.BackendError;
                }
            },
            .nccl => {
                if (comptime build_options.gpu_cuda) {
                    try cuda_backend.ncclTransfer(
                        handle.src_device,
                        handle.dst_device,
                        data,
                    );
                } else {
                    return error.BackendError;
                }
            },
            .vulkan_external => {
                if (comptime build_options.gpu_vulkan) {
                    try vulkan_backend.externalMemoryTransfer(
                        handle.src_device,
                        handle.dst_device,
                        data,
                    );
                } else {
                    return error.BackendError;
                }
            },
            .metal_shared => {
                if (comptime build_options.gpu_metal) {
                    try metal_backend.sharedEventTransfer(
                        handle.src_device,
                        handle.dst_device,
                        data,
                    );
                } else {
                    return error.BackendError;
                }
            },
            .host_staged => {
                try self.host_staged_backend.transfer(
                    handle.src_device,
                    handle.dst_device,
                    data,
                );
            },
        }

        handle.status.store(.completed, .release);
        handle.completion_time = shared_utils.unixMs();

        self.stats.successful_transfers += 1;
        self.stats.bytes_transferred += handle.size;
        if (handle.completion_time) |end| {
            self.stats.total_time_ns += @intCast((end - handle.start_time) * 1_000_000);
        }
    }

    /// Perform AllReduce across all devices.
    pub fn allReduceAsync(
        self: *Self,
        buffers: []DeviceBuffer,
        op: ReduceOp,
        opts: TransferOptions,
    ) !void {
        if (buffers.len <= 1) return;

        const n = buffers.len;
        const data_len = if (buffers.len > 0) buffers[0].data.len else 0;

        // Check for NCCL availability first (most efficient for AllReduce)
        if (comptime build_options.gpu_cuda) {
            if (cuda_backend.hasNCCL()) {
                try cuda_backend.ncclAllReduce(buffers, op);
                return;
            }
        }

        // Fall back to ring AllReduce algorithm
        try self.ringAllReduce(buffers, op, opts, n, data_len);
    }

    /// Ring AllReduce implementation with real peer transfers.
    fn ringAllReduce(
        self: *Self,
        buffers: []DeviceBuffer,
        op: ReduceOp,
        opts: TransferOptions,
        n: usize,
        data_len: usize,
    ) !void {
        if (n <= 1 or data_len == 0) return;

        const chunk_size = (data_len + n - 1) / n;

        // Allocate temporary buffer for receiving data
        const recv_buffer = try self.allocator.alloc(f32, chunk_size);
        defer self.allocator.free(recv_buffer);

        // Phase 1: Reduce-scatter
        var phase: usize = 0;
        while (phase < n - 1) : (phase += 1) {
            for (0..n) |i| {
                const send_rank = i;
                const recv_rank = (i + 1) % n;

                const send_chunk_idx = (send_rank + n - phase - 1) % n;
                const recv_chunk_idx = (recv_rank + n - phase - 1) % n;

                const send_start = send_chunk_idx * chunk_size;
                const recv_start = recv_chunk_idx * chunk_size;

                const send_end = @min(send_start + chunk_size, data_len);
                const recv_end = @min(recv_start + chunk_size, data_len);

                if (send_start >= data_len or recv_start >= data_len) continue;

                const send_data = buffers[send_rank].data[send_start..send_end];
                const recv_data = buffers[recv_rank].data[recv_start..recv_end];

                // Perform peer transfer
                const send_bytes = std.mem.sliceAsBytes(send_data);
                _ = try self.transferAsync(
                    buffers[send_rank].device_id,
                    buffers[recv_rank].device_id,
                    @constCast(send_bytes),
                    opts,
                );

                // Apply reduction operation
                for (recv_data, 0..) |*val, j| {
                    if (j < send_data.len) {
                        val.* = applyOp(val.*, send_data[j], op);
                    }
                }
            }

            // Wait for all transfers in this phase
            try self.waitAll();
        }

        // Phase 2: Allgather
        phase = 0;
        while (phase < n - 1) : (phase += 1) {
            for (0..n) |i| {
                const send_rank = i;
                const recv_rank = (i + 1) % n;

                const chunk_idx = (send_rank + n - phase) % n;
                const chunk_start = chunk_idx * chunk_size;
                const chunk_end = @min(chunk_start + chunk_size, data_len);

                if (chunk_start >= data_len) continue;

                const chunk_data = buffers[send_rank].data[chunk_start..chunk_end];

                // Broadcast this chunk to next device
                const chunk_bytes = std.mem.sliceAsBytes(chunk_data);
                _ = try self.transferAsync(
                    buffers[send_rank].device_id,
                    buffers[recv_rank].device_id,
                    @constCast(chunk_bytes),
                    opts,
                );

                // Copy to receiver's buffer
                @memcpy(buffers[recv_rank].data[chunk_start..chunk_end], chunk_data);
            }

            try self.waitAll();
        }

        // Handle average operation
        if (op == .avg) {
            const scale = 1.0 / @as(f32, @floatFromInt(n));
            for (buffers) |buf| {
                for (buf.data) |*val| {
                    val.* *= scale;
                }
            }
        }
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

    /// Wait for a specific transfer to complete.
    pub fn waitForTransfer(self: *Self, handle: *TransferHandle) !void {
        _ = self;

        const timeout_ns: u64 = 30_000_000_000; // 30 seconds
        var timer = std.time.Timer.start() catch return error.TimerFailed;

        while (!handle.isComplete()) {
            if (timer.read() > timeout_ns) {
                handle.status.store(.failed, .release);
                handle.error_info = error.TransferTimeout;
                return error.TransferTimeout;
            }
            std.Thread.yield() catch {};
        }

        if (!handle.succeeded()) {
            if (handle.error_info) |err| {
                return err;
            }
            return error.BackendError;
        }
    }

    /// Poll transfer status without blocking.
    pub fn pollTransfer(self: *Self, handle: *const TransferHandle) TransferStatus {
        _ = self;
        return handle.status.load(.acquire);
    }

    /// Wait for all active transfers to complete.
    pub fn waitAll(self: *Self) !void {
        self.mutex.lock();
        const transfers = self.active_transfers.items;
        self.mutex.unlock();

        for (transfers) |*transfer| {
            if (!transfer.isComplete()) {
                // Cast away const for mutable handle
                const mutable_transfer: *TransferHandle = @constCast(transfer);
                self.waitForTransfer(mutable_transfer) catch |err| {
                    if (self.recovery_strategy == .abort) {
                        return err;
                    }
                    // Log and continue for other strategies
                    self.stats.failed_transfers += 1;
                };
            }
        }

        // Clean up completed transfers
        self.mutex.lock();
        defer self.mutex.unlock();

        var i: usize = 0;
        while (i < self.active_transfers.items.len) {
            if (self.active_transfers.items[i].isComplete()) {
                _ = self.active_transfers.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Cancel a pending transfer.
    pub fn cancelTransfer(self: *Self, handle: *TransferHandle) void {
        _ = self;
        const status = handle.status.load(.acquire);
        if (status == .pending or status == .in_progress) {
            handle.status.store(.cancelled, .release);
            handle.error_info = error.Cancelled;
        }
    }

    /// Get transfer statistics.
    pub fn getStats(self: *const Self) TransferStats {
        return self.stats;
    }

    /// Reset statistics.
    pub fn resetStats(self: *Self) void {
        self.stats = .{};
    }

    /// Set recovery strategy.
    pub fn setRecoveryStrategy(self: *Self, strategy: RecoveryStrategy) void {
        self.recovery_strategy = strategy;
    }

    /// Check if peer access is available between two devices.
    pub fn hasPeerAccess(self: *Self, src_id: DeviceId, dst_id: DeviceId) bool {
        const cap = self.getCapability(src_id, dst_id);
        return cap != .host_staged;
    }

    /// Get all capabilities for display/debugging.
    pub fn listCapabilities(self: *Self, writer: anytype) !void {
        const devices = self.device_group.active_devices.items;

        try writer.writeAll("Peer Transfer Capabilities:\n");
        try writer.writeAll("─────────────────────────────\n");

        for (devices) |src_id| {
            for (devices) |dst_id| {
                if (src_id == dst_id) continue;

                const cap = self.getCapability(src_id, dst_id);
                try writer.print("  Device {} → Device {}: {s}\n", .{
                    src_id,
                    dst_id,
                    cap.name(),
                });
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PeerTransferManager initialization" {
    const allocator = std.testing.allocator;

    var device_group = try DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_transfers);
}

test "TransferCapability priority ordering" {
    try std.testing.expect(TransferCapability.nccl.priority() > TransferCapability.direct_p2p.priority());
    try std.testing.expect(TransferCapability.direct_p2p.priority() > TransferCapability.host_staged.priority());
}

test "DevicePair hashing" {
    const pair1 = DevicePair{ .src = 0, .dst = 1 };
    const pair2 = DevicePair{ .src = 1, .dst = 0 };
    const pair3 = DevicePair{ .src = 0, .dst = 1 };

    try std.testing.expect(pair1.hash() != pair2.hash());
    try std.testing.expect(pair1.hash() == pair3.hash());
    try std.testing.expect(pair1.eql(pair3));
    try std.testing.expect(!pair1.eql(pair2));
}
