//! Device discovery and enumeration across GPU backends.

const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../internal/backend.zig");
const build_options = @import("build_options");
const policy = @import("../policy/mod.zig");
const android_probe = @import("android_probe.zig");
const types = @import("types.zig");

const Device = types.Device;
const DeviceType = types.DeviceType;
const Vendor = types.Vendor;
const Backend = types.Backend;

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
    if (android_probe.chooseAndroidPrimary()) |primary| {
        if (backend_mod.backendAvailability(primary).available and
            backend_mod.backendSupportsKernels(primary))
        {
            return primary;
        }
    }

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
// Multi-GPU Device Enumeration
// ============================================================================

/// Enumerate all available GPU devices across all backends.
///
/// Returns a slice of Device structs. **Caller owns the returned memory**
/// and must free it with `allocator.free(devices)` when done.
pub fn enumerateAllDevices(allocator: std.mem.Allocator) ![]Device {
    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    var device_id: u32 = 0;
    const order = preferredBackendOrder();

    // Try each backend in canonical platform order.
    for (order.slice()) |backend_tag| {
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

const BackendOrder = struct {
    items: [16]Backend = undefined,
    len: usize = 0,

    fn append(self: *BackendOrder, backend: Backend) void {
        for (self.items[0..self.len]) |existing| {
            if (existing == backend) return;
        }
        if (self.len >= self.items.len) return;
        self.items[self.len] = backend;
        self.len += 1;
    }

    fn slice(self: *const BackendOrder) []const Backend {
        return self.items[0..self.len];
    }
};

fn preferredBackendOrder() BackendOrder {
    var order = BackendOrder{};
    const platform = policy.classify(builtin.target.os.tag, builtin.abi);
    const android_primary_name: ?[]const u8 = if (android_probe.chooseAndroidPrimary()) |backend|
        backend.name()
    else
        null;

    const backend_names = policy.resolveAutoBackendNames(.{
        .platform = platform,
        .enable_gpu = build_options.feat_gpu,
        .enable_web = build_options.feat_web,
        .can_link_metal = true,
        .warn_if_metal_skipped = false,
        .allow_simulated = true,
        .android_primary = android_primary_name,
    });

    for (backend_names.slice()) |name| {
        if (backend_mod.backendFromString(name)) |backend| {
            order.append(backend);
        }
    }

    if (@hasDecl(build_options, "gpu_tpu") and build_options.gpu_tpu) order.append(.tpu);
    if (@hasDecl(build_options, "gpu_fpga") and build_options.gpu_fpga) order.append(.fpga);
    order.append(.stdgpu);
    order.append(.simulated);
    return order;
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
        .opengles => if (comptime build_options.gpu_opengles) try enumerateOpenGlesDevices(allocator) else &[_]Device{},
        .webgl2 => &[_]Device{}, // Not yet implemented
        .fpga => &[_]Device{}, // Not yet implemented
        .tpu => &[_]Device{}, // TPU runtime not yet linked
        .simulated => if (comptime build_options.feat_gpu) try enumerateStdgpuDevices(allocator) else &[_]Device{},
    };
}

// ============================================================================
// Per-backend device enumeration stubs
// ============================================================================

fn enumerateCudaDevices(allocator: std.mem.Allocator) ![]Device {
    const cuda = @import("../backends/cuda/mod.zig");
    return cuda.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate CUDA devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateVulkanDevices(allocator: std.mem.Allocator) ![]Device {
    const vulkan = @import("../backends/vulkan.zig");
    return vulkan.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate Vulkan devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateMetalDevices(allocator: std.mem.Allocator) ![]Device {
    const metal = @import("../backends/metal.zig");
    return metal.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate Metal devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateWebGPUDevices(allocator: std.mem.Allocator) ![]Device {
    const webgpu = @import("../backends/webgpu.zig");
    return webgpu.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate WebGPU devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateOpenGLDevices(allocator: std.mem.Allocator) ![]Device {
    const opengl = @import("../backends/opengl.zig");
    return opengl.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate OpenGL devices: {t}", .{err});
        return &[_]Device{};
    };
}

fn enumerateOpenGlesDevices(allocator: std.mem.Allocator) ![]Device {
    const opengles = @import("../backends/opengles.zig");
    return opengles.enumerateDevices(allocator) catch |err| {
        std.log.warn("Failed to enumerate OpenGL ES devices: {t}", .{err});
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

test {
    std.testing.refAllDecls(@This());
}
