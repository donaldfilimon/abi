//! GPU Device Abstraction
//!
//! Provides a unified device abstraction layer for the GPU API.
//! Handles device discovery, selection, and capability querying.

const std = @import("std");
const backend_mod = @import("backend.zig");

pub const Backend = backend_mod.Backend;
pub const DeviceCapability = backend_mod.DeviceCapability;

/// Device type classification for scoring and selection.
pub const DeviceType = enum {
    discrete,
    integrated,
    virtual,
    cpu,
    other,

    pub fn score(self: DeviceType) u32 {
        return switch (self) {
            .discrete => 1000,
            .integrated => 500,
            .virtual => 100,
            .cpu => 50,
            .other => 10,
        };
    }
};

/// Represents a GPU device with extended information.
pub const Device = struct {
    /// Unique device identifier within this session.
    id: u32,
    /// Backend this device belongs to.
    backend: Backend,
    /// Human-readable device name.
    name: []const u8,
    /// Device type classification.
    device_type: DeviceType,
    /// Total device memory in bytes (if known).
    total_memory: ?u64,
    /// Available device memory in bytes (if known).
    available_memory: ?u64,
    /// Whether this is an emulated/software device.
    is_emulated: bool,
    /// Device capabilities.
    capability: DeviceCapability,
    /// Compute units / streaming multiprocessors.
    compute_units: ?u32,
    /// Clock speed in MHz (if known).
    clock_mhz: ?u32,

    /// Calculate a score for device selection.
    pub fn score(self: Device) u32 {
        var total: u32 = self.device_type.score();

        // Bonus for real hardware
        if (!self.is_emulated) {
            total += 500;
        }

        // Bonus for memory (scaled)
        if (self.total_memory) |mem| {
            const gb: u64 = @min(mem / (1024 * 1024 * 1024), 32);
            total += @intCast(gb * 10); // 10 points per GB, max 320
        }

        // Bonus for compute units
        if (self.compute_units) |cu| {
            total += @min(cu, 100) * 2; // 2 points per CU, max 200
        }

        // Bonus for capabilities
        if (self.capability.supports_fp16) total += 50;
        if (self.capability.supports_int8) total += 30;
        if (self.capability.supports_async_transfers) total += 40;
        if (self.capability.unified_memory) total += 20;

        return total;
    }

    /// Check if this device supports a specific feature.
    pub fn supportsFeature(self: Device, feature: DeviceFeature) bool {
        return switch (feature) {
            .fp16 => self.capability.supports_fp16,
            .int8 => self.capability.supports_int8,
            .async_transfers => self.capability.supports_async_transfers,
            .unified_memory => self.capability.unified_memory,
            .compute_shaders => backend_mod.backendSupportsKernels(self.backend),
        };
    }

    /// Get maximum workgroup/block size.
    pub fn maxWorkgroupSize(self: Device) u32 {
        return self.capability.max_threads_per_block orelse 256;
    }

    /// Get maximum shared memory per workgroup.
    pub fn maxSharedMemory(self: Device) u32 {
        return self.capability.max_shared_memory_bytes orelse 16 * 1024;
    }
};

/// Features that can be queried on a device.
pub const DeviceFeature = enum {
    fp16,
    int8,
    async_transfers,
    unified_memory,
    compute_shaders,
};

/// Device selection criteria.
pub const DeviceSelector = union(enum) {
    /// Select the best device based on scoring.
    best: void,
    /// Select a specific device by ID.
    by_id: u32,
    /// Select by backend preference.
    by_backend: Backend,
    /// Select by device type preference.
    by_type: DeviceType,
    /// Select by minimum memory requirement.
    by_memory: u64,
    /// Select by required features.
    by_features: []const DeviceFeature,
    /// Custom selection function.
    custom: *const fn ([]const Device) ?Device,

    pub fn select(self: DeviceSelector, devices: []const Device) ?Device {
        if (devices.len == 0) return null;

        return switch (self) {
            .best => selectBest(devices),
            .by_id => |id| selectById(devices, id),
            .by_backend => |backend| selectByBackend(devices, backend),
            .by_type => |device_type| selectByType(devices, device_type),
            .by_memory => |min_memory| selectByMemory(devices, min_memory),
            .by_features => |features| selectByFeatures(devices, features),
            .custom => |func| func(devices),
        };
    }

    fn selectBest(devices: []const Device) ?Device {
        if (devices.len == 0) return null;

        var best = devices[0];
        var best_score = best.score();

        for (devices[1..]) |device| {
            const device_score = device.score();
            if (device_score > best_score) {
                best = device;
                best_score = device_score;
            }
        }

        return best;
    }

    fn selectById(devices: []const Device, id: u32) ?Device {
        for (devices) |device| {
            if (device.id == id) return device;
        }
        return null;
    }

    fn selectByBackend(devices: []const Device, backend: Backend) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.backend == backend) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }

    fn selectByType(devices: []const Device, device_type: DeviceType) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.device_type == device_type) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }

    fn selectByMemory(devices: []const Device, min_memory: u64) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.total_memory) |mem| {
                if (mem >= min_memory) {
                    const device_score = device.score();
                    if (best == null or device_score > best_score) {
                        best = device;
                        best_score = device_score;
                    }
                }
            }
        }

        return best;
    }

    fn selectByFeatures(devices: []const Device, features: []const DeviceFeature) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            var has_all = true;
            for (features) |feature| {
                if (!device.supportsFeature(feature)) {
                    has_all = false;
                    break;
                }
            }

            if (has_all) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }
};

/// Device manager for discovery and selection.
pub const DeviceManager = struct {
    allocator: std.mem.Allocator,
    devices: []Device,
    active_device: ?*const Device,

    pub fn init(allocator: std.mem.Allocator) !DeviceManager {
        const devices = try discoverDevices(allocator);
        return .{
            .allocator = allocator,
            .devices = devices,
            .active_device = null,
        };
    }

    pub fn deinit(self: *DeviceManager) void {
        self.allocator.free(self.devices);
        self.* = undefined;
    }

    /// Get all discovered devices.
    pub fn listDevices(self: *const DeviceManager) []const Device {
        return self.devices;
    }

    /// Get the currently active device.
    pub fn getActiveDevice(self: *const DeviceManager) ?*const Device {
        return self.active_device;
    }

    /// Select and activate a device based on criteria.
    pub fn selectDevice(self: *DeviceManager, selector: DeviceSelector) !*const Device {
        const selected = selector.select(self.devices);
        if (selected) |device| {
            // Find the pointer to the device in our slice
            for (self.devices) |*d| {
                if (d.id == device.id) {
                    self.active_device = d;
                    return d;
                }
            }
        }
        return error.DeviceNotFound;
    }

    /// Select the best available device.
    pub fn selectBestDevice(self: *DeviceManager) !*const Device {
        return self.selectDevice(.best);
    }

    /// Get device by ID.
    pub fn getDevice(self: *const DeviceManager, id: u32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    /// Get devices matching a backend.
    pub fn getDevicesForBackend(self: *const DeviceManager, allocator: std.mem.Allocator, backend: Backend) ![]const Device {
        var matching = std.ArrayListUnmanaged(Device).empty;
        errdefer matching.deinit(allocator);

        for (self.devices) |device| {
            if (device.backend == backend) {
                try matching.append(allocator, device);
            }
        }

        return matching.toOwnedSlice(allocator);
    }

    /// Check if any device is available.
    pub fn hasDevices(self: *const DeviceManager) bool {
        return self.devices.len > 0;
    }

    /// Get the number of devices.
    pub fn deviceCount(self: *const DeviceManager) usize {
        return self.devices.len;
    }
};

/// Discover all available GPU devices.
pub fn discoverDevices(allocator: std.mem.Allocator) ![]Device {
    const backend_devices = try backend_mod.listDevices(allocator);
    defer allocator.free(backend_devices);

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    for (backend_devices) |info| {
        const device_type = classifyDevice(info);

        try devices.append(allocator, .{
            .id = info.id,
            .backend = info.backend,
            .name = info.name,
            .device_type = device_type,
            .total_memory = info.total_memory_bytes,
            .available_memory = null, // Not tracked at discovery time
            .is_emulated = info.is_emulated,
            .capability = info.capability,
            .compute_units = null, // Would need deeper probing
            .clock_mhz = null, // Would need deeper probing
        });
    }

    return devices.toOwnedSlice(allocator);
}

/// Classify a device based on its properties.
fn classifyDevice(info: backend_mod.DeviceInfo) DeviceType {
    // Emulated devices are virtual
    if (info.is_emulated) {
        return .virtual;
    }

    // stdgpu is CPU-based
    if (info.backend == .stdgpu) {
        return .cpu;
    }

    // Real hardware - classify based on memory
    if (info.total_memory_bytes) |mem| {
        if (mem >= 4 * 1024 * 1024 * 1024) { // 4GB+
            return .discrete;
        } else {
            return .integrated;
        }
    }

    // Unknown - assume integrated for safety
    return .integrated;
}

/// Get the best available backend for kernels.
pub fn getBestKernelBackend(allocator: std.mem.Allocator) !Backend {
    const devices = try discoverDevices(allocator);
    defer allocator.free(devices);

    if (devices.len == 0) {
        return error.NoDevicesAvailable;
    }

    // Find best device that supports kernels
    var best: ?Device = null;
    var best_score: u32 = 0;

    for (devices) |device| {
        if (backend_mod.backendSupportsKernels(device.backend)) {
            const device_score = device.score();
            if (best == null or device_score > best_score) {
                best = device;
                best_score = device_score;
            }
        }
    }

    if (best) |device| {
        return device.backend;
    }

    return error.NoKernelBackendAvailable;
}

// ============================================================================
// Tests
// ============================================================================

test "DeviceType scoring" {
    try std.testing.expect(DeviceType.discrete.score() > DeviceType.integrated.score());
    try std.testing.expect(DeviceType.integrated.score() > DeviceType.virtual.score());
    try std.testing.expect(DeviceType.virtual.score() > DeviceType.cpu.score());
    try std.testing.expect(DeviceType.cpu.score() > DeviceType.other.score());
}

test "Device scoring" {
    const device1 = Device{
        .id = 0,
        .backend = .cuda,
        .name = "Test GPU",
        .device_type = .discrete,
        .total_memory = 8 * 1024 * 1024 * 1024,
        .available_memory = null,
        .is_emulated = false,
        .capability = .{
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
        },
        .compute_units = 40,
        .clock_mhz = null,
    };

    const device2 = Device{
        .id = 1,
        .backend = .stdgpu,
        .name = "CPU Fallback",
        .device_type = .cpu,
        .total_memory = 2 * 1024 * 1024 * 1024,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    try std.testing.expect(device1.score() > device2.score());
}

test "DeviceSelector best" {
    const devices = [_]Device{
        .{
            .id = 0,
            .backend = .vulkan,
            .name = "Device 0",
            .device_type = .integrated,
            .total_memory = 2 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = true,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "Device 1",
            .device_type = .discrete,
            .total_memory = 8 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{ .supports_fp16 = true },
            .compute_units = null,
            .clock_mhz = null,
        },
    };

    const selector = DeviceSelector{ .best = {} };
    const selected = selector.select(&devices);

    try std.testing.expect(selected != null);
    try std.testing.expect(selected.?.id == 1); // CUDA device should score higher
}

test "DeviceSelector by_backend" {
    const devices = [_]Device{
        .{
            .id = 0,
            .backend = .vulkan,
            .name = "Vulkan Device",
            .device_type = .discrete,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "CUDA Device",
            .device_type = .discrete,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
        },
    };

    const selector = DeviceSelector{ .by_backend = .vulkan };
    const selected = selector.select(&devices);

    try std.testing.expect(selected != null);
    try std.testing.expect(selected.?.backend == .vulkan);
}

test "DeviceManager init and deinit" {
    var manager = try DeviceManager.init(std.testing.allocator);
    defer manager.deinit();

    // Should at least not crash; device count depends on system
    _ = manager.deviceCount();
    _ = manager.hasDevices();
}
