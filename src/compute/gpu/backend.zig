//! GPU backend metadata, detection, and device enumeration.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const shared = @import("backends/shared.zig");

pub const Backend = enum {
    cuda,
    vulkan,
    metal,
    webgpu,
    opengl,
    opengles,
    webgl2,
};

pub const DetectionLevel = enum {
    none,
    loader,
    device_count,
};

pub const BackendAvailability = struct {
    enabled: bool,
    available: bool,
    reason: []const u8,
    device_count: usize,
    level: DetectionLevel,
};

pub const BackendInfo = struct {
    backend: Backend,
    name: []const u8,
    description: []const u8,
    enabled: bool,
    available: bool,
    availability: []const u8,
    device_count: usize,
    build_flag: []const u8,
};

pub const DeviceCapability = struct {
    unified_memory: bool = false,
    supports_fp16: bool = false,
    supports_int8: bool = false,
    supports_async_transfers: bool = false,
    max_threads_per_block: ?u32 = null,
    max_shared_memory_bytes: ?u32 = null,
};

pub const DeviceInfo = struct {
    id: u32,
    backend: Backend,
    name: []const u8,
    total_memory_bytes: ?u64 = null,
    is_emulated: bool = true,
    capability: DeviceCapability = .{},
};

pub const Summary = struct {
    module_enabled: bool,
    enabled_backend_count: usize,
    available_backend_count: usize,
    device_count: usize,
    emulated_devices: usize,
};

const GiB: u64 = 1024 * 1024 * 1024;

const opengles_aliases = [_][]const u8{ "gles" };
const webgl2_aliases = [_][]const u8{ "webgl" };

const BackendMeta = struct {
    name: []const u8,
    display_name: []const u8,
    description: []const u8,
    build_flag: []const u8,
    device_name: []const u8,
    device_name_emulated: []const u8,
    memory_bytes: ?u64,
    capability: DeviceCapability,
    supports_kernels: bool,
    aliases: []const []const u8,
};

const backend_meta = [_]BackendMeta{
    .{
        .name = "cuda",
        .display_name = "CUDA",
        .description = "NVIDIA CUDA backend",
        .build_flag = "-Dgpu-cuda",
        .device_name = "CUDA Adapter",
        .device_name_emulated = "CUDA Adapter (emulated)",
        .memory_bytes = 8 * GiB,
        .capability = .{
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
            .max_threads_per_block = 1024,
            .max_shared_memory_bytes = 48 * 1024,
        },
        .supports_kernels = true,
        .aliases = &.{},
    },
    .{
        .name = "vulkan",
        .display_name = "Vulkan",
        .description = "Cross-platform Vulkan backend",
        .build_flag = "-Dgpu-vulkan",
        .device_name = "Vulkan Adapter",
        .device_name_emulated = "Vulkan Adapter (emulated)",
        .memory_bytes = 6 * GiB,
        .capability = .{
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_int8 = false,
            .supports_async_transfers = true,
            .max_threads_per_block = 512,
            .max_shared_memory_bytes = 32 * 1024,
        },
        .supports_kernels = true,
        .aliases = &.{},
    },
    .{
        .name = "metal",
        .display_name = "Metal",
        .description = "Apple Metal backend",
        .build_flag = "-Dgpu-metal",
        .device_name = "Metal Adapter",
        .device_name_emulated = "Metal Adapter (emulated)",
        .memory_bytes = 4 * GiB,
        .capability = .{
            .unified_memory = true,
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
            .max_threads_per_block = 512,
            .max_shared_memory_bytes = 32 * 1024,
        },
        .supports_kernels = true,
        .aliases = &.{},
    },
    .{
        .name = "webgpu",
        .display_name = "WebGPU",
        .description = "WebGPU backend",
        .build_flag = "-Dgpu-webgpu",
        .device_name = "WebGPU Adapter",
        .device_name_emulated = "WebGPU Adapter (emulated)",
        .memory_bytes = 2 * GiB,
        .capability = .{
            .unified_memory = true,
            .supports_fp16 = false,
            .supports_int8 = false,
            .supports_async_transfers = false,
            .max_threads_per_block = 256,
            .max_shared_memory_bytes = 16 * 1024,
        },
        .supports_kernels = true,
        .aliases = &.{},
    },
    .{
        .name = "opengl",
        .display_name = "OpenGL",
        .description = "OpenGL backend",
        .build_flag = "-Dgpu-opengl",
        .device_name = "OpenGL Adapter",
        .device_name_emulated = "OpenGL Adapter (emulated)",
        .memory_bytes = null,
        .capability = .{},
        .supports_kernels = true,
        .aliases = &.{},
    },
    .{
        .name = "opengles",
        .display_name = "OpenGL ES",
        .description = "OpenGL ES backend",
        .build_flag = "-Dgpu-opengles",
        .device_name = "OpenGL ES Adapter",
        .device_name_emulated = "OpenGL ES Adapter (emulated)",
        .memory_bytes = null,
        .capability = .{},
        .supports_kernels = true,
        .aliases = opengles_aliases[0..],
    },
    .{
        .name = "webgl2",
        .display_name = "WebGL2",
        .description = "WebGL2 backend (browser)",
        .build_flag = "-Dgpu-webgl2",
        .device_name = "WebGL2 Adapter",
        .device_name_emulated = "WebGL2 Adapter (emulated)",
        .memory_bytes = null,
        .capability = .{},
        .supports_kernels = false,
        .aliases = webgl2_aliases[0..],
    },
};

fn meta(backend: Backend) BackendMeta {
    return backend_meta[@intFromEnum(backend)];
}

pub fn moduleEnabled() bool {
    return build_options.enable_gpu;
}

pub fn isEnabled(backend: Backend) bool {
    if (!moduleEnabled()) return false;
    return switch (backend) {
        .cuda => build_options.gpu_cuda,
        .vulkan => build_options.gpu_vulkan,
        .metal => build_options.gpu_metal,
        .webgpu => build_options.gpu_webgpu,
        .opengl => build_options.gpu_opengl,
        .opengles => build_options.gpu_opengles,
        .webgl2 => build_options.gpu_webgl2,
    };
}

pub fn backendSupportsKernels(backend: Backend) bool {
    return meta(backend).supports_kernels;
}

pub fn summary() Summary {
    if (!moduleEnabled()) {
        return .{
            .module_enabled = false,
            .enabled_backend_count = 0,
            .available_backend_count = 0,
            .device_count = 0,
            .emulated_devices = 0,
        };
    }

    var enabled_count: usize = 0;
    var available_count: usize = 0;
    var device_count: usize = 0;
    var emulated_devices: usize = 0;

    for (std.enums.values(Backend)) |backend| {
        if (isEnabled(backend)) {
            enabled_count += 1;
        }

        const availability = backendAvailability(backend);
        if (!availability.available) continue;

        available_count += 1;
        const count = if (availability.device_count > 0) availability.device_count else 1;
        device_count += count;
        if (availability.level != .device_count) {
            emulated_devices += count;
        }
    }

    return .{
        .module_enabled = true,
        .enabled_backend_count = enabled_count,
        .available_backend_count = available_count,
        .device_count = device_count,
        .emulated_devices = emulated_devices,
    };
}

pub fn backendName(backend: Backend) []const u8 {
    return meta(backend).name;
}

pub fn backendDisplayName(backend: Backend) []const u8 {
    return meta(backend).display_name;
}

pub fn backendDescription(backend: Backend) []const u8 {
    return meta(backend).description;
}

pub fn backendFlag(backend: Backend) []const u8 {
    return meta(backend).build_flag;
}

pub fn backendFromString(text: []const u8) ?Backend {
    for (backend_meta, 0..) |info, index| {
        if (std.ascii.eqlIgnoreCase(text, info.name)) {
            return @enumFromInt(index);
        }
        for (info.aliases) |alias| {
            if (std.ascii.eqlIgnoreCase(text, alias)) {
                return @enumFromInt(index);
            }
        }
    }
    return null;
}

pub fn backendAvailability(backend: Backend) BackendAvailability {
    if (!moduleEnabled()) {
        return .{
            .enabled = false,
            .available = false,
            .reason = "gpu module disabled",
            .device_count = 0,
            .level = .none,
        };
    }
    if (!isEnabled(backend)) {
        return .{
            .enabled = false,
            .available = false,
            .reason = "disabled at build time",
            .device_count = 0,
            .level = .none,
        };
    }

    return switch (backend) {
        .cuda => detectCuda(),
        .vulkan => detectVulkan(),
        .metal => detectMetal(),
        .webgpu => detectWebGpu(),
        .opengl => detectOpenGl(),
        .opengles => detectOpenGles(),
        .webgl2 => detectWebGl2(),
    };
}

pub fn listBackendInfo(allocator: std.mem.Allocator) ![]BackendInfo {
    var list = std.ArrayListUnmanaged(BackendInfo).empty;
    errdefer list.deinit(allocator);
    for (std.enums.values(Backend)) |backend| {
        const availability = backendAvailability(backend);
        try list.append(allocator, .{
            .backend = backend,
            .name = backendDisplayName(backend),
            .description = backendDescription(backend),
            .enabled = availability.enabled,
            .available = availability.available,
            .availability = availability.reason,
            .device_count = availability.device_count,
            .build_flag = backendFlag(backend),
        });
    }
    return list.toOwnedSlice(allocator);
}

pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    var list = std.ArrayListUnmanaged(Backend).empty;
    errdefer list.deinit(allocator);
    if (!moduleEnabled()) return list.toOwnedSlice(allocator);
    for (std.enums.values(Backend)) |backend| {
        const availability = backendAvailability(backend);
        if (availability.available) {
            try list.append(allocator, backend);
        }
    }
    return list.toOwnedSlice(allocator);
}

pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    var devices = std.ArrayListUnmanaged(DeviceInfo).empty;
    errdefer devices.deinit(allocator);
    if (!moduleEnabled()) return devices.toOwnedSlice(allocator);

    var id: u32 = 0;
    for (std.enums.values(Backend)) |backend| {
        const availability = backendAvailability(backend);
        if (!availability.available) continue;
        const count = if (availability.device_count > 0) availability.device_count else 1;
        const is_emulated = availability.level != .device_count;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            try devices.append(allocator, .{
                .id = id,
                .backend = backend,
                .name = backendDeviceName(backend, is_emulated),
                .total_memory_bytes = backendMemoryBytes(backend),
                .is_emulated = is_emulated,
                .capability = backendCapabilities(backend),
            });
            id += 1;
        }
    }
    return devices.toOwnedSlice(allocator);
}

pub fn defaultDevice(allocator: std.mem.Allocator) !?DeviceInfo {
    const devices = try listDevices(allocator);
    defer allocator.free(devices);
    return selectDefaultDevice(devices);
}

pub fn defaultDeviceLabel(allocator: std.mem.Allocator) !?[]u8 {
    const device = try defaultDevice(allocator);
    if (device == null) return null;
    const label = try std.fmt.allocPrint(
        allocator,
        "{s} ({s})",
        .{ device.?.name, backendDisplayName(device.?.backend) },
    );
    return label;
}

fn selectDefaultDevice(devices: []const DeviceInfo) ?DeviceInfo {
    if (devices.len == 0) return null;
    for (devices) |device| {
        if (!device.is_emulated) return device;
    }
    return devices[0];
}

fn detectCuda() BackendAvailability {
    if (cudaLibNames().len == 0) {
        return unavailableAvailability("cuda unsupported on this platform");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("cuda requires dynamic loader");
    }

    var lib = shared.openFirst(cudaLibNames()) orelse
        return unavailableAvailability("cuda driver not found");
    defer lib.close();

    const cuInit = lib.lookup(CuInitFn, "cuInit") orelse {
        return availableAvailability(.loader, 1, "cuda driver loaded");
    };
    if (cuInit(0) != 0) {
        return unavailableAvailability("cuda init failed");
    }

    const cuDeviceGetCount = lib.lookup(CuDeviceGetCountFn, "cuDeviceGetCount") orelse {
        return availableAvailability(.loader, 1, "cuda driver loaded");
    };

    var count: i32 = 0;
    if (cuDeviceGetCount(&count) != 0) {
        return availableAvailability(.loader, 1, "cuda driver loaded");
    }
    if (count <= 0) {
        return unavailableAvailability("cuda devices not found");
    }
    return availableAvailability(.device_count, @intCast(count), "cuda devices detected");
}

fn detectVulkan() BackendAvailability {
    if (vulkanLibNames().len == 0) {
        return unavailableAvailability("vulkan unsupported on this platform");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("vulkan requires dynamic loader");
    }
    if (!shared.tryLoadAny(vulkanLibNames())) {
        return unavailableAvailability("vulkan loader not found");
    }
    return availableAvailability(.loader, 1, "vulkan loader available");
}

fn detectMetal() BackendAvailability {
    if (builtin.target.os.tag != .macos) {
        return unavailableAvailability("metal requires macos");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("metal requires dynamic loader");
    }
    if (!shared.tryLoadAny(metalLibNames())) {
        return unavailableAvailability("metal framework not found");
    }
    return availableAvailability(.loader, 1, "metal framework available");
}

fn detectWebGpu() BackendAvailability {
    if (shared.isWebTarget()) {
        return availableAvailability(.loader, 1, "web target");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("webgpu requires dynamic loader");
    }
    if (!shared.tryLoadAny(webGpuLibNames())) {
        return unavailableAvailability("webgpu runtime not found");
    }
    return availableAvailability(.loader, 1, "webgpu runtime available");
}

fn detectOpenGl() BackendAvailability {
    if (openGlLibNames().len == 0) {
        return unavailableAvailability("opengl unsupported on this platform");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("opengl requires dynamic loader");
    }
    if (!shared.tryLoadAny(openGlLibNames())) {
        return unavailableAvailability("opengl runtime not found");
    }
    return availableAvailability(.loader, 1, "opengl runtime available");
}

fn detectOpenGles() BackendAvailability {
    if (openGlesLibNames().len == 0) {
        return unavailableAvailability("opengles unsupported on this platform");
    }
    if (!shared.canUseDynLib()) {
        return unavailableAvailability("opengles requires dynamic loader");
    }
    if (!shared.tryLoadAny(openGlesLibNames())) {
        return unavailableAvailability("opengles runtime not found");
    }
    return availableAvailability(.loader, 1, "opengles runtime available");
}

fn detectWebGl2() BackendAvailability {
    if (shared.isWebTarget()) {
        return availableAvailability(.loader, 1, "web target");
    }
    return unavailableAvailability("webgl2 requires web target");
}

fn availableAvailability(
    level: DetectionLevel,
    device_count: usize,
    reason: []const u8,
) BackendAvailability {
    return .{
        .enabled = true,
        .available = true,
        .reason = reason,
        .device_count = device_count,
        .level = level,
    };
}

fn unavailableAvailability(reason: []const u8) BackendAvailability {
    return .{
        .enabled = true,
        .available = false,
        .reason = reason,
        .device_count = 0,
        .level = .none,
    };
}

fn backendDeviceName(backend: Backend, emulated: bool) []const u8 {
    const info = meta(backend);
    return if (emulated) info.device_name_emulated else info.device_name;
}

fn backendMemoryBytes(backend: Backend) ?u64 {
    return meta(backend).memory_bytes;
}

fn backendCapabilities(backend: Backend) DeviceCapability {
    return meta(backend).capability;
}

fn cudaLibNames() []const []const u8 {
    return switch (builtin.target.os.tag) {
        .windows => cuda_windows[0..],
        .linux => cuda_linux[0..],
        else => &.{},
    };
}

fn vulkanLibNames() []const []const u8 {
    return switch (builtin.target.os.tag) {
        .windows => vulkan_windows[0..],
        .linux => vulkan_linux[0..],
        .macos => vulkan_macos[0..],
        else => &.{},
    };
}

fn metalLibNames() []const []const u8 {
    return metal_macos[0..];
}

fn webGpuLibNames() []const []const u8 {
    return switch (builtin.target.os.tag) {
        .windows => webgpu_windows[0..],
        .linux => webgpu_linux[0..],
        .macos => webgpu_macos[0..],
        else => &.{},
    };
}

fn openGlLibNames() []const []const u8 {
    return switch (builtin.target.os.tag) {
        .windows => opengl_windows[0..],
        .linux => opengl_linux[0..],
        .macos => opengl_macos[0..],
        else => &.{},
    };
}

fn openGlesLibNames() []const []const u8 {
    return switch (builtin.target.os.tag) {
        .windows => opengles_windows[0..],
        .linux => opengles_linux[0..],
        .macos => opengles_macos[0..],
        else => &.{},
    };
}

const cuda_windows = [_][]const u8{"nvcuda.dll"};
const cuda_linux = [_][]const u8{ "libcuda.so.1", "libcuda.so" };

const vulkan_windows = [_][]const u8{"vulkan-1.dll"};
const vulkan_linux = [_][]const u8{ "libvulkan.so.1", "libvulkan.so" };
const vulkan_macos = [_][]const u8{"libvulkan.dylib"};

const metal_macos = [_][]const u8{"/System/Library/Frameworks/Metal.framework/Metal"};

const webgpu_windows = [_][]const u8{ "wgpu_native.dll", "dawn_native.dll" };
const webgpu_linux = [_][]const u8{ "libwgpu_native.so", "libdawn_native.so" };
const webgpu_macos = [_][]const u8{ "libwgpu_native.dylib", "libdawn_native.dylib" };

const opengl_windows = [_][]const u8{"opengl32.dll"};
const opengl_linux = [_][]const u8{ "libGL.so.1", "libGL.so" };
const opengl_macos = [_][]const u8{"/System/Library/Frameworks/OpenGL.framework/OpenGL"};

const opengles_windows = [_][]const u8{ "libGLESv2.dll", "libEGL.dll" };
const opengles_linux = [_][]const u8{ "libGLESv2.so.2", "libGLESv2.so" };
const opengles_macos = [_][]const u8{"/System/Library/Frameworks/OpenGLES.framework/OpenGLES"};

const CuInitFn = *const fn (u32) callconv(.c) i32;
const CuDeviceGetCountFn = *const fn (*i32) callconv(.c) i32;
