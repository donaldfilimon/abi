//! GPU Device Abstraction
//!
//! Provides a unified device abstraction layer for the GPU API.
//! Handles device discovery, selection, and capability querying.
//!
//! ## Memory Ownership
//!
//! Functions that return `[]Device` allocate memory that the **caller must free**:
//! - `enumerateAllDevices(allocator)` → caller owns returned slice
//! - `enumerateDevicesForBackend(allocator, backend)` → caller owns returned slice
//! - `discoverDevices(allocator)` → caller owns returned slice
//! - `DeviceManager.getDevicesForBackend(allocator, backend)` → caller owns returned slice
//!
//! Functions that return `?Device` (by value) do **not** require cleanup.
//!
//! Example:
//! ```zig
//! const devices = try device.enumerateAllDevices(allocator);
//! defer allocator.free(devices);  // Caller must free
//! ```

const std = @import("std");
const backend_mod = @import("backend.zig");
const build_options = @import("build_options");

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

/// GPU vendor identification.
pub const Vendor = enum {
    nvidia,
    amd,
    intel,
    apple,
    qualcomm,
    arm,
    mesa,
    microsoft,
    unknown,

    /// Get vendor from device name string.
    pub fn fromDeviceName(name: []const u8) Vendor {
        // Convert to lowercase for case-insensitive matching
        var lower_buf: [256]u8 = undefined;
        const len = @min(name.len, lower_buf.len);
        for (name[0..len], 0..) |c, i| {
            lower_buf[i] = std.ascii.toLower(c);
        }
        const lower = lower_buf[0..len];

        // NVIDIA detection
        if (std.mem.indexOf(u8, lower, "nvidia") != null or
            std.mem.indexOf(u8, lower, "geforce") != null or
            std.mem.indexOf(u8, lower, "quadro") != null or
            std.mem.indexOf(u8, lower, "tesla") != null or
            std.mem.indexOf(u8, lower, "rtx") != null or
            std.mem.indexOf(u8, lower, "gtx") != null)
        {
            return .nvidia;
        }

        // AMD detection
        if (std.mem.indexOf(u8, lower, "amd") != null or
            std.mem.indexOf(u8, lower, "radeon") != null or
            std.mem.indexOf(u8, lower, "vega") != null or
            std.mem.indexOf(u8, lower, "navi") != null or
            std.mem.indexOf(u8, lower, "polaris") != null or
            std.mem.indexOf(u8, lower, "rx ") != null)
        {
            return .amd;
        }

        // Intel detection
        if (std.mem.indexOf(u8, lower, "intel") != null or
            std.mem.indexOf(u8, lower, "iris") != null or
            std.mem.indexOf(u8, lower, "arc") != null or
            std.mem.indexOf(u8, lower, "uhd graphics") != null or
            std.mem.indexOf(u8, lower, "hd graphics") != null)
        {
            return .intel;
        }

        // Apple detection
        if (std.mem.indexOf(u8, lower, "apple") != null or
            std.mem.indexOf(u8, lower, "m1") != null or
            std.mem.indexOf(u8, lower, "m2") != null or
            std.mem.indexOf(u8, lower, "m3") != null or
            std.mem.indexOf(u8, lower, "m4") != null)
        {
            return .apple;
        }

        // Qualcomm detection (Adreno)
        if (std.mem.indexOf(u8, lower, "qualcomm") != null or
            std.mem.indexOf(u8, lower, "adreno") != null)
        {
            return .qualcomm;
        }

        // ARM Mali detection
        if (std.mem.indexOf(u8, lower, "mali") != null or
            std.mem.indexOf(u8, lower, "arm") != null)
        {
            return .arm;
        }

        // Mesa (open source) detection
        if (std.mem.indexOf(u8, lower, "llvmpipe") != null or
            std.mem.indexOf(u8, lower, "softpipe") != null or
            std.mem.indexOf(u8, lower, "mesa") != null or
            std.mem.indexOf(u8, lower, "swrast") != null)
        {
            return .mesa;
        }

        // Microsoft (WARP) detection
        if (std.mem.indexOf(u8, lower, "microsoft") != null or
            std.mem.indexOf(u8, lower, "warp") != null)
        {
            return .microsoft;
        }

        return .unknown;
    }

    /// Get recommended backend for this vendor.
    pub fn recommendedBackend(self: Vendor) Backend {
        return switch (self) {
            .nvidia => .cuda, // CUDA is optimal for NVIDIA
            .amd => .vulkan, // Vulkan works well on AMD
            .intel => .vulkan, // Vulkan or OpenCL for Intel
            .apple => .metal, // Metal is native for Apple
            .qualcomm => .vulkan, // Vulkan for mobile Qualcomm
            .arm => .vulkan, // Vulkan for ARM Mali
            .mesa, .microsoft => .vulkan, // Software rasterizers
            .unknown => .stdgpu, // Fall back to std.gpu
        };
    }

    /// Get vendor display name.
    pub fn displayName(self: Vendor) []const u8 {
        return switch (self) {
            .nvidia => "NVIDIA",
            .amd => "AMD",
            .intel => "Intel",
            .apple => "Apple",
            .qualcomm => "Qualcomm",
            .arm => "ARM",
            .mesa => "Mesa/Open Source",
            .microsoft => "Microsoft",
            .unknown => "Unknown",
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
    /// Select by vendor preference.
    by_vendor: Vendor,
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
            .by_vendor => |vendor| selectByVendor(devices, vendor),
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

    fn selectByVendor(devices: []const Device, vendor: Vendor) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.vendor == vendor) {
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
///
/// Returns a slice of Device structs. **Caller owns the returned memory**
/// and must free it with `allocator.free(devices)` when done.
pub fn discoverDevices(allocator: std.mem.Allocator) ![]Device {
    const backend_devices = try backend_mod.listDevices(allocator);
    defer allocator.free(backend_devices);

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    for (backend_devices) |info| {
        const device_type = classifyDevice(info);
        const vendor = Vendor.fromDeviceName(info.name);

        try devices.append(allocator, .{
            .id = info.id,
            .backend = info.backend,
            .name = info.name,
            .device_type = device_type,
            .vendor = vendor,
            .total_memory = info.total_memory_bytes,
            .available_memory = null, // Not tracked at discovery time
            .is_emulated = info.is_emulated,
            .capability = info.capability,
            .compute_units = null, // Would need deeper probing
            .clock_mhz = null, // Would need deeper probing
            .pci_bus_id = null, // Would need deeper probing
            .driver_version = null, // Would need deeper probing
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
// Multi-GPU Device Enumeration (Task 1.1)
// ============================================================================

/// Structured device selection criteria (alternative to union-based DeviceSelector).
pub const DeviceSelectionCriteria = struct {
    prefer_discrete: bool = false,
    min_memory_gb: u64 = 0,
    required_features: []const DeviceFeature = &.{},
};

/// Enumerate all available GPU devices across all backends.
///
/// Returns a slice of Device structs. **Caller owns the returned memory**
/// and must free it with `allocator.free(devices)` when done.
pub fn enumerateAllDevices(allocator: std.mem.Allocator) ![]Device {
    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    var device_id: u32 = 0;

    // Try each backend
    for (std.meta.tags(Backend)) |backend_tag| {
        const backend_devices = enumerateDevicesForBackend(allocator, backend_tag) catch continue;
        defer allocator.free(backend_devices);

        for (backend_devices) |dev| {
            var dev_copy = dev;
            dev_copy.id = device_id;
            device_id += 1;
            try devices.append(allocator, dev_copy);
        }
    }

    return devices.toOwnedSlice(allocator);
}

/// Enumerate devices for a specific backend.
///
/// Returns a slice of Device structs. **Caller owns the returned memory**
/// and must free it with `allocator.free(devices)` when done.
/// Returns an empty slice if the backend is unavailable.
pub fn enumerateDevicesForBackend(
    allocator: std.mem.Allocator,
    backend_type: Backend,
) ![]Device {
    if (!backend_mod.backendAvailability(backend_type).available) {
        return &[_]Device{};
    }

    return switch (backend_type) {
        .cuda => if (comptime build_options.gpu_cuda) try enumerateCudaDevices(allocator) else &[_]Device{},
        .vulkan => if (comptime build_options.gpu_vulkan) try enumerateVulkanDevices(allocator) else &[_]Device{},
        .stdgpu => if (comptime build_options.gpu_stdgpu) try enumerateStdgpuDevices(allocator) else &[_]Device{},
        .metal => if (comptime build_options.gpu_metal) try enumerateMetalDevices(allocator) else &[_]Device{},
        .webgpu => if (comptime build_options.gpu_webgpu) try enumerateWebGPUDevices(allocator) else &[_]Device{},
        .opengl => if (comptime build_options.gpu_opengl) try enumerateOpenGLDevices(allocator) else &[_]Device{},
        .opengles => if (comptime build_options.gpu_opengles) try enumerateOpenGLDevices(allocator) else &[_]Device{},
        .webgl2 => &[_]Device{}, // Not yet implemented
        .fpga => &[_]Device{}, // Not yet implemented
        .tpu => &[_]Device{}, // TPU runtime not yet linked
        .simulated => if (comptime build_options.enable_gpu) try enumerateStdgpuDevices(allocator) else &[_]Device{},
    };
}

/// Select the best device based on criteria.
pub fn selectBestDevice(
    allocator: std.mem.Allocator,
    criteria: DeviceSelectionCriteria,
) !?Device {
    const all_devices = try enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    if (all_devices.len == 0) return null;

    var best: ?Device = null;
    var best_score: u32 = 0;

    for (all_devices) |dev| {
        if (!meetsRequirements(dev, criteria)) continue;

        const score_val = dev.score();
        if (score_val > best_score) {
            best = dev;
            best_score = score_val;
        }
    }

    return best;
}

fn meetsRequirements(dev: Device, criteria: DeviceSelectionCriteria) bool {
    if (criteria.prefer_discrete and dev.device_type != .discrete) {
        if (dev.device_type != .integrated) return false;
    }

    if (criteria.min_memory_gb > 0) {
        if (dev.total_memory) |mem| {
            const gb = mem / (1024 * 1024 * 1024);
            if (gb < criteria.min_memory_gb) return false;
        } else {
            return false; // Unknown memory doesn't meet requirement
        }
    }

    for (criteria.required_features) |feature| {
        if (!dev.supportsFeature(feature)) return false;
    }

    return true;
}

// ============================================================================
// Per-backend device enumeration stubs
// ============================================================================

fn enumerateCudaDevices(allocator: std.mem.Allocator) ![]Device {
    const cuda = @import("backends/cuda/mod.zig");
    return cuda.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate CUDA devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateVulkanDevices(allocator: std.mem.Allocator) ![]Device {
    const vulkan = @import("backends/vulkan.zig");
    return vulkan.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate Vulkan devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateMetalDevices(allocator: std.mem.Allocator) ![]Device {
    const metal = @import("backends/metal.zig");
    return metal.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate Metal devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateWebGPUDevices(allocator: std.mem.Allocator) ![]Device {
    const webgpu = @import("backends/webgpu.zig");
    return webgpu.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate WebGPU devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateOpenGLDevices(allocator: std.mem.Allocator) ![]Device {
    const opengl = @import("backends/opengl.zig");
    return opengl.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate OpenGL devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateStdgpuDevices(allocator: std.mem.Allocator) ![]Device {
    const devices_slice = try allocator.alloc(Device, 1);
    devices_slice[0] = .{
        .id = 0,
        .backend = .stdgpu,
        .name = "CPU Fallback",
        .device_type = .cpu,
        .vendor = .unknown,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{
            .supports_fp16 = false,
            .supports_int8 = true,
            .supports_async_transfers = false,
            .unified_memory = true,
        },
        .compute_units = null,
        .clock_mhz = null,
        .pci_bus_id = null,
        .driver_version = null,
    };

    return devices_slice;
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

test "DeviceSelector best" {
    const devices = [_]Device{
        .{
            .id = 0,
            .backend = .vulkan,
            .name = "Device 0",
            .device_type = .integrated,
            .vendor = .intel,
            .total_memory = 2 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = true,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "Device 1",
            .device_type = .discrete,
            .vendor = .nvidia,
            .total_memory = 8 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{ .supports_fp16 = true },
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
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
            .vendor = .amd,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "CUDA Device",
            .device_type = .discrete,
            .vendor = .nvidia,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
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
