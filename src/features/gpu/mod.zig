const std = @import("std");
const compute_gpu = @import("../../compute/gpu/mod.zig");

pub const MemoryError = compute_gpu.MemoryError;
pub const GpuError = compute_gpu.GpuError;
pub const Backend = compute_gpu.Backend;
pub const BackendInfo = compute_gpu.BackendInfo;
pub const BackendAvailability = compute_gpu.BackendAvailability;
pub const DetectionLevel = compute_gpu.DetectionLevel;
pub const DeviceCapability = compute_gpu.DeviceCapability;
pub const DeviceInfo = compute_gpu.DeviceInfo;
pub const Summary = compute_gpu.Summary;
pub const BufferFlags = compute_gpu.BufferFlags;
pub const Buffer = compute_gpu.GPUBuffer;
pub const MemoryPool = compute_gpu.GPUMemoryPool;
pub const MemoryStats = compute_gpu.MemoryStats;
pub const AsyncTransfer = compute_gpu.AsyncTransfer;

pub fn init(allocator: std.mem.Allocator) GpuError!void {
    return compute_gpu.init(allocator);
}

pub fn deinit() void {
    compute_gpu.deinit();
}

pub fn moduleEnabled() bool {
    return compute_gpu.moduleEnabled();
}

pub fn isInitialized() bool {
    return compute_gpu.isInitialized();
}

pub fn summary() Summary {
    return compute_gpu.summary();
}

pub fn backendName(backend: Backend) []const u8 {
    return compute_gpu.backendName(backend);
}

pub fn backendDisplayName(backend: Backend) []const u8 {
    return compute_gpu.backendDisplayName(backend);
}

pub fn backendDescription(backend: Backend) []const u8 {
    return compute_gpu.backendDescription(backend);
}

pub fn backendFlag(backend: Backend) []const u8 {
    return compute_gpu.backendFlag(backend);
}

pub fn backendFromString(text: []const u8) ?Backend {
    return compute_gpu.backendFromString(text);
}

pub fn isEnabled(backend: Backend) bool {
    return compute_gpu.isEnabled(backend);
}

pub fn backendAvailability(backend: Backend) BackendAvailability {
    return compute_gpu.backendAvailability(backend);
}

pub fn listBackendInfo(allocator: std.mem.Allocator) ![]BackendInfo {
    return compute_gpu.listBackendInfo(allocator);
}

pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    return compute_gpu.availableBackends(allocator);
}

pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    return compute_gpu.listDevices(allocator);
}

pub fn defaultDevice(allocator: std.mem.Allocator) !?DeviceInfo {
    return compute_gpu.defaultDevice(allocator);
}

pub fn createBuffer(allocator: std.mem.Allocator, size: usize) !Buffer {
    return Buffer.init(allocator, size, .{});
}

pub fn createPool(allocator: std.mem.Allocator, max_size: usize) MemoryPool {
    return MemoryPool.init(allocator, max_size);
}

test "backend parsing helpers" {
    try std.testing.expectEqual(@as(?Backend, .cuda), backendFromString("cuda"));
    try std.testing.expectEqual(@as(?Backend, .vulkan), backendFromString("VULKAN"));
    try std.testing.expectEqual(@as(?Backend, .opengl), backendFromString("OpenGL"));
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

test "buffer copy and fill" {
    var buffer = try Buffer.init(std.testing.allocator, 4, .{});
    defer buffer.deinit();

    try buffer.fill(0xaa);
    try std.testing.expectEqualSlices(u8, &.{ 0xaa, 0xaa, 0xaa, 0xaa }, buffer.bytes);

    try buffer.copyFrom(&.{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, buffer.bytes);
}
