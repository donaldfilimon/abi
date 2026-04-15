//! Core GPU device types shared across the device subsystem.

const std = @import("std");
const backend_mod = @import("../internal/backend.zig");
const vendor_mod = @import("vendor.zig");

pub const Backend = backend_mod.Backend;
pub const DeviceCapability = backend_mod.DeviceCapability;
pub const Vendor = vendor_mod.Vendor;

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

/// Features that can be queried on a device.
pub const DeviceFeature = enum {
    fp16,
    int8,
    async_transfers,
    unified_memory,
    compute_shaders,
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
    /// GPU vendor.
    vendor: Vendor,
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
    /// PCI bus ID (if available).
    pci_bus_id: ?[]const u8,
    /// Driver version (if available).
    driver_version: ?[]const u8,

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

        // Bonus for using vendor's native backend
        if (self.isUsingNativeBackend()) {
            total += 100;
        }

        return total;
    }

    /// Check if this device is using its vendor's native/optimal backend.
    pub fn isUsingNativeBackend(self: Device) bool {
        const recommended = self.vendor.recommendedBackend();
        return self.backend == recommended;
    }

    /// Get vendor display name.
    pub fn vendorName(self: Device) []const u8 {
        return self.vendor.displayName();
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

/// Structured device selection criteria (alternative to union-based DeviceSelector).
pub const DeviceSelectionCriteria = struct {
    prefer_discrete: bool = false,
    min_memory_gb: u64 = 0,
    required_features: []const DeviceFeature = &.{},
};

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
        .vendor = .nvidia,
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
        .pci_bus_id = null,
        .driver_version = null,
    };

    const device2 = Device{
        .id = 1,
        .backend = .stdgpu,
        .name = "CPU Fallback",
        .device_type = .cpu,
        .vendor = .unknown,
        .total_memory = 2 * 1024 * 1024 * 1024,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
        .pci_bus_id = null,
        .driver_version = null,
    };

    try std.testing.expect(device1.score() > device2.score());
}

test {
    std.testing.refAllDecls(@This());
}
