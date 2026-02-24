//! Device group management for multi-GPU operations.
//!
//! Contains device discovery, load balancing, work distribution,
//! peer transfers, and synchronization barriers.

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");

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
    backend: backend_mod.Backend = .stdgpu,
    backend_device_id: ?u32 = null,
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

        if (build_options.enable_gpu) {
            const discovered = device_mod.discoverDevices(self.allocator) catch |err| {
                std.log.warn("DeviceGroup discovery failed, using CPU fallback only: {t}", .{err});
                try self.appendCpuFallback(0);
                try self.active_devices.append(self.allocator, 0);
                return;
            };
            defer self.allocator.free(discovered);

            var next_fallback_id: DeviceId = 0;
            for (discovered) |dev| {
                next_fallback_id = @max(next_fallback_id, dev.id + 1);
                try self.appendDiscoveredDevice(dev);
            }

            if (self.devices.items.len == 0) {
                try self.appendCpuFallback(next_fallback_id);
            }
        } else {
            try self.appendCpuFallback(0);
        }

        try self.activateConfiguredDevices();
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

    fn appendDiscoveredDevice(self: *DeviceGroup, dev: device_mod.Device) !void {
        const memory_mb = resolveMemoryMb(dev.total_memory, dev.available_memory, dev.backend);
        const max_shared_kb: u32 = if (dev.capability.max_shared_memory_bytes) |bytes|
            @intCast(@max(bytes / 1024, 1))
        else
            48;
        const compute_units = dev.compute_units orelse defaultComputeUnits(dev.device_type);

        var info = DeviceInfo{
            .id = dev.id,
            .backend = dev.backend,
            .backend_device_id = dev.id,
            .device_type = mapDeviceType(dev.device_type),
            .capabilities = .{
                .compute_units = compute_units,
                .max_memory_mb = memory_mb,
                .max_work_group_size = dev.capability.max_threads_per_block orelse 256,
                .supports_fp16 = dev.capability.supports_fp16,
                .supports_fp64 = dev.device_type == .cpu,
                .supports_int8 = dev.capability.supports_int8,
                .supports_atomics = true,
                .max_shared_memory_kb = max_shared_kb,
                .warp_size = 32,
            },
            .available_memory_mb = memory_mb,
        };

        const copied = copyDeviceName(&info.name, dev.name);
        info.name_len = copied;

        try self.devices.append(self.allocator, info);
    }

    fn appendCpuFallback(self: *DeviceGroup, id: DeviceId) !void {
        var name: [128]u8 = [_]u8{0} ** 128;
        const src = "CPU Fallback";
        @memcpy(name[0..src.len], src);

        try self.devices.append(self.allocator, .{
            .id = id,
            .backend = .stdgpu,
            .backend_device_id = null,
            .name = name,
            .name_len = src.len,
            .device_type = .cpu_fallback,
            .capabilities = .{
                .compute_units = 1,
                .max_memory_mb = 4096,
                .supports_fp64 = true,
                .supports_int8 = true,
                .supports_atomics = true,
            },
            .available_memory_mb = 4096,
        });
    }

    fn activateConfiguredDevices(self: *DeviceGroup) !void {
        if (self.config.preferred_devices.len > 0) {
            for (self.config.preferred_devices) |device_id| {
                if (self.getDevice(device_id) != null and !isDeviceActive(self, device_id)) {
                    try self.active_devices.append(self.allocator, device_id);
                }
            }
        }

        if (self.active_devices.items.len == 0 and
            (self.config.auto_select or self.config.preferred_devices.len == 0))
        {
            for (self.devices.items) |device| {
                try self.active_devices.append(self.allocator, device.id);
            }
        }

        if (self.active_devices.items.len == 0 and self.devices.items.len > 0) {
            try self.active_devices.append(self.allocator, self.devices.items[0].id);
        }
    }

    fn isDeviceActive(self: *const DeviceGroup, id: DeviceId) bool {
        for (self.active_devices.items) |active_id| {
            if (active_id == id) return true;
        }
        return false;
    }
};

fn mapDeviceType(kind: device_mod.DeviceType) DeviceType {
    return switch (kind) {
        .discrete => .discrete_gpu,
        .integrated => .integrated_gpu,
        .virtual => .virtual_gpu,
        .cpu => .cpu_fallback,
        .other => .unknown,
    };
}

fn defaultComputeUnits(kind: device_mod.DeviceType) u32 {
    return switch (kind) {
        .discrete => 32,
        .integrated => 16,
        .virtual => 8,
        .cpu => 1,
        .other => 4,
    };
}

fn resolveMemoryMb(total_memory: ?u64, available_memory: ?u64, backend: backend_mod.Backend) u64 {
    const bytes = available_memory orelse total_memory orelse defaultMemoryBytes(backend);
    const mb = bytes / (1024 * 1024);
    return @max(mb, 1);
}

fn defaultMemoryBytes(backend: backend_mod.Backend) u64 {
    return switch (backend) {
        .cuda => 8 * 1024 * 1024 * 1024,
        .vulkan => 6 * 1024 * 1024 * 1024,
        .metal => 4 * 1024 * 1024 * 1024,
        .webgpu => 2 * 1024 * 1024 * 1024,
        .opengl, .opengles => 2 * 1024 * 1024 * 1024,
        .webgl2 => 1024 * 1024 * 1024,
        .fpga => 16 * 1024 * 1024 * 1024,
        .tpu => 16 * 1024 * 1024 * 1024,
        .stdgpu, .simulated => 4 * 1024 * 1024 * 1024,
    };
}

fn copyDeviceName(buffer: *[128]u8, name: []const u8) usize {
    const len = @min(name.len, buffer.len);
    @memcpy(buffer[0..len], name[0..len]);
    return len;
}

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

test {
    std.testing.refAllDecls(@This());
}
