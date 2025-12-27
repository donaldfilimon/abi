//! GPU backend detection and device enumeration.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const memory = @import("memory.zig");

pub const MemoryError = memory.MemoryError;
pub const BufferFlags = memory.BufferFlags;
pub const GPUBuffer = memory.GPUBuffer;
pub const GPUMemoryPool = memory.GPUMemoryPool;
pub const MemoryStats = memory.MemoryStats;
pub const AsyncTransfer = memory.AsyncTransfer;
pub const GpuError = memory.MemoryError || error{GpuDisabled};

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

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) GpuError!void {
    if (!moduleEnabled()) return error.GpuDisabled;
    initialized = true;
}

pub fn ensureInitialized(allocator: std.mem.Allocator) GpuError!void {
    if (!isInitialized()) {
        try init(allocator);
    }
}

pub fn deinit() void {
    initialized = false;
}

pub fn moduleEnabled() bool {
    return build_options.enable_gpu;
}

pub fn isInitialized() bool {
    return initialized;
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
    return switch (backend) {
        .cuda => "cuda",
        .vulkan => "vulkan",
        .metal => "metal",
        .webgpu => "webgpu",
        .opengl => "opengl",
        .opengles => "opengles",
        .webgl2 => "webgl2",
    };
}

pub fn backendDisplayName(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "CUDA",
        .vulkan => "Vulkan",
        .metal => "Metal",
        .webgpu => "WebGPU",
        .opengl => "OpenGL",
        .opengles => "OpenGL ES",
        .webgl2 => "WebGL2",
    };
}

pub fn backendDescription(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "NVIDIA CUDA backend",
        .vulkan => "Cross-platform Vulkan backend",
        .metal => "Apple Metal backend",
        .webgpu => "WebGPU backend",
        .opengl => "OpenGL backend",
        .opengles => "OpenGL ES backend",
        .webgl2 => "WebGL2 backend (browser)",
    };
}

pub fn backendFlag(backend: Backend) []const u8 {
    return switch (backend) {
        .cuda => "-Dgpu-cuda",
        .vulkan => "-Dgpu-vulkan",
        .metal => "-Dgpu-metal",
        .webgpu => "-Dgpu-webgpu",
        .opengl => "-Dgpu-opengl",
        .opengles => "-Dgpu-opengles",
        .webgl2 => "-Dgpu-webgl2",
    };
}

pub fn backendFromString(text: []const u8) ?Backend {
    if (std.ascii.eqlIgnoreCase(text, "cuda")) return .cuda;
    if (std.ascii.eqlIgnoreCase(text, "vulkan")) return .vulkan;
    if (std.ascii.eqlIgnoreCase(text, "metal")) return .metal;
    if (std.ascii.eqlIgnoreCase(text, "webgpu")) return .webgpu;
    if (std.ascii.eqlIgnoreCase(text, "opengl")) return .opengl;
    if (std.ascii.eqlIgnoreCase(text, "opengles")) return .opengles;
    if (std.ascii.eqlIgnoreCase(text, "gles")) return .opengles;
    if (std.ascii.eqlIgnoreCase(text, "webgl2")) return .webgl2;
    if (std.ascii.eqlIgnoreCase(text, "webgl")) return .webgl2;
    return null;
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
    if (!canUseDynLib()) {
        return unavailableAvailability("cuda requires dynamic loader");
    }

    var lib = openFirst(cudaLibNames()) orelse
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
    if (!canUseDynLib()) {
        return unavailableAvailability("vulkan requires dynamic loader");
    }
    if (!tryLoadAny(vulkanLibNames())) {
        return unavailableAvailability("vulkan loader not found");
    }
    return availableAvailability(.loader, 1, "vulkan loader available");
}

fn detectMetal() BackendAvailability {
    if (builtin.target.os.tag != .macos) {
        return unavailableAvailability("metal requires macos");
    }
    if (!canUseDynLib()) {
        return unavailableAvailability("metal requires dynamic loader");
    }
    if (!tryLoadAny(metalLibNames())) {
        return unavailableAvailability("metal framework not found");
    }
    return availableAvailability(.loader, 1, "metal framework available");
}

fn detectWebGpu() BackendAvailability {
    if (isWebTarget()) {
        return availableAvailability(.loader, 1, "web target");
    }
    if (!canUseDynLib()) {
        return unavailableAvailability("webgpu requires dynamic loader");
    }
    if (!tryLoadAny(webGpuLibNames())) {
        return unavailableAvailability("webgpu runtime not found");
    }
    return availableAvailability(.loader, 1, "webgpu runtime available");
}

fn detectOpenGl() BackendAvailability {
    if (openGlLibNames().len == 0) {
        return unavailableAvailability("opengl unsupported on this platform");
    }
    if (!canUseDynLib()) {
        return unavailableAvailability("opengl requires dynamic loader");
    }
    if (!tryLoadAny(openGlLibNames())) {
        return unavailableAvailability("opengl runtime not found");
    }
    return availableAvailability(.loader, 1, "opengl runtime available");
}

fn detectOpenGles() BackendAvailability {
    if (openGlesLibNames().len == 0) {
        return unavailableAvailability("opengles unsupported on this platform");
    }
    if (!canUseDynLib()) {
        return unavailableAvailability("opengles requires dynamic loader");
    }
    if (!tryLoadAny(openGlesLibNames())) {
        return unavailableAvailability("opengles runtime not found");
    }
    return availableAvailability(.loader, 1, "opengles runtime available");
}

fn detectWebGl2() BackendAvailability {
    if (isWebTarget()) {
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

fn canUseDynLib() bool {
    return switch (builtin.target.cpu.arch) {
        .wasm32, .wasm64 => false,
        else => true,
    };
}

fn isWebTarget() bool {
    return switch (builtin.target.cpu.arch) {
        .wasm32, .wasm64 => true,
        else => false,
    };
}

fn tryLoadAny(libs: []const []const u8) bool {
    var lib = openFirst(libs) orelse return false;
    lib.close();
    return true;
}

fn openFirst(libs: []const []const u8) ?std.DynLib {
    for (libs) |name| {
        if (std.DynLib.open(name)) |lib| {
            return lib;
        } else |_| {}
    }
    return null;
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

fn backendDeviceName(backend: Backend, emulated: bool) []const u8 {
    return switch (backend) {
        .cuda => if (emulated) "CUDA Adapter (emulated)" else "CUDA Adapter",
        .vulkan => if (emulated) "Vulkan Adapter (emulated)" else "Vulkan Adapter",
        .metal => if (emulated) "Metal Adapter (emulated)" else "Metal Adapter",
        .webgpu => if (emulated) "WebGPU Adapter (emulated)" else "WebGPU Adapter",
        .opengl => if (emulated) "OpenGL Adapter (emulated)" else "OpenGL Adapter",
        .opengles => if (emulated) "OpenGL ES Adapter (emulated)" else "OpenGL ES Adapter",
        .webgl2 => if (emulated) "WebGL2 Adapter (emulated)" else "WebGL2 Adapter",
    };
}

fn backendMemoryBytes(backend: Backend) ?u64 {
    return switch (backend) {
        .cuda => 8 * 1024 * 1024 * 1024,
        .vulkan => 6 * 1024 * 1024 * 1024,
        .metal => 4 * 1024 * 1024 * 1024,
        .webgpu => 2 * 1024 * 1024 * 1024,
        .opengl => null,
        .opengles => null,
        .webgl2 => null,
    };
}

fn backendCapabilities(backend: Backend) DeviceCapability {
    return switch (backend) {
        .cuda => .{
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
            .max_threads_per_block = 1024,
            .max_shared_memory_bytes = 48 * 1024,
        },
        .vulkan => .{
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_int8 = false,
            .supports_async_transfers = true,
            .max_threads_per_block = 512,
            .max_shared_memory_bytes = 32 * 1024,
        },
        .metal => .{
            .unified_memory = true,
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
            .max_threads_per_block = 512,
            .max_shared_memory_bytes = 32 * 1024,
        },
        .webgpu => .{
            .unified_memory = true,
            .supports_fp16 = false,
            .supports_int8 = false,
            .supports_async_transfers = false,
            .max_threads_per_block = 256,
            .max_shared_memory_bytes = 16 * 1024,
        },
        .opengl => .{},
        .opengles => .{},
        .webgl2 => .{},
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

test "backend parsing helpers" {
    try std.testing.expectEqual(@as(?Backend, .cuda), backendFromString("cuda"));
    try std.testing.expectEqual(@as(?Backend, .vulkan), backendFromString("VULKAN"));
    try std.testing.expectEqual(@as(?Backend, .opengl), backendFromString("OpenGL"));
    try std.testing.expectEqual(@as(?Backend, .opengles), backendFromString("GLES"));
    try std.testing.expectEqual(@as(?Backend, .webgl2), backendFromString("webgl"));
    try std.testing.expectEqual(@as(?Backend, null), backendFromString("unknown"));
    try std.testing.expect(backendName(.metal).len > 0);
    try std.testing.expect(backendDisplayName(.webgpu).len > 0);
}

test "backend flags are stable" {
    try std.testing.expectEqualStrings("-Dgpu-cuda", backendFlag(.cuda));
    try std.testing.expectEqualStrings("-Dgpu-webgpu", backendFlag(.webgpu));
    try std.testing.expectEqualStrings("-Dgpu-opengl", backendFlag(.opengl));
    try std.testing.expectEqualStrings("-Dgpu-opengles", backendFlag(.opengles));
    try std.testing.expectEqualStrings("-Dgpu-webgl2", backendFlag(.webgl2));
}

test "available backends reflect availability" {
    const allocator = std.testing.allocator;
    const backends = try availableBackends(allocator);
    defer allocator.free(backends);
    for (backends) |backend| {
        const availability = backendAvailability(backend);
        try std.testing.expect(availability.available);
    }
}

test "summary matches enabled backends" {
    const details = summary();
    try std.testing.expect(details.module_enabled == moduleEnabled());
    if (!details.module_enabled) return;

    var enabled_count: usize = 0;
    for (std.enums.values(Backend)) |backend| {
        if (isEnabled(backend)) enabled_count += 1;
    }
    try std.testing.expectEqual(enabled_count, details.enabled_backend_count);
}

test "default device prefers non-emulated" {
    const devices = [_]DeviceInfo{
        .{ .id = 0, .backend = .cuda, .name = "emu", .is_emulated = true },
        .{ .id = 1, .backend = .vulkan, .name = "real", .is_emulated = false },
    };
    const selected = selectDefaultDevice(&devices);
    try std.testing.expect(selected != null);
    try std.testing.expectEqual(@as(u32, 1), selected.?.id);
}

test "default device falls back to first when all emulated" {
    const devices = [_]DeviceInfo{
        .{ .id = 4, .backend = .opengl, .name = "emu0", .is_emulated = true },
        .{ .id = 7, .backend = .webgpu, .name = "emu1", .is_emulated = true },
    };
    const selected = selectDefaultDevice(&devices);
    try std.testing.expect(selected != null);
    try std.testing.expectEqual(@as(u32, 4), selected.?.id);
}
