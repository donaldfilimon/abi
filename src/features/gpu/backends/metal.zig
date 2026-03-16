//! Metal backend implementation with native GPU execution.
//!
//! This is the orchestrator module that delegates to specialized submodules:
//! - metal_device.zig  — device initialization, capabilities, selection, enumeration
//! - metal_buffers.zig — buffer creation, management, memory transfers
//! - metal_compute.zig — compute pipeline, kernel dispatch, execution
//! - metal_state.zig   — shared mutable state across all submodules
//!
//! Metal uses Objective-C runtime, so this module uses objc_msgSend for message dispatch.
//!
//! ## Features
//! - Full kernel dispatch with proper MTLSize struct handling
//! - Device property queries (memory, compute units, device name)
//! - NSString creation for runtime kernel compilation
//! - Proper synchronization via command buffer tracking
//! - Multi-device enumeration via MTLCopyAllDevices
//!
//! ## Architecture Support
//! - Apple Silicon (ARM64): Uses standard objc_msgSend for all calls
//! - Intel Macs (x86_64): Uses objc_msgSend_stret for struct returns

const std = @import("std");
const types = @import("../kernel_types.zig");

// Re-export extracted type definitions for build discovery
pub const metal_types = @import("metal_types.zig");

// Submodules
pub const metal_device = @import("metal_device.zig");
pub const metal_buffers = @import("metal_buffers.zig");
pub const metal_compute = @import("metal_compute.zig");
pub const metal_state = @import("metal_state.zig");

pub const MetalError = metal_types.MetalError;

// Objective-C runtime types (from metal_types.zig)
const SEL = metal_types.SEL;
const Class = metal_types.Class;
const ID = metal_types.ID;

// Metal struct types (re-exported from metal_types.zig)
pub const MTLSize = metal_types.MTLSize;
pub const MTLOrigin = metal_types.MTLOrigin;
pub const MTLRegion = metal_types.MTLRegion;

// Metal GPU Family / Feature detection
pub const gpu_family = @import("metal/gpu_family.zig");
pub const capabilities = @import("metal/capabilities.zig");
pub const MetalGpuFamily = gpu_family.MetalGpuFamily;
pub const MetalFeatureSet = gpu_family.MetalFeatureSet;
pub const MetalLevel = capabilities.MetalLevel;

// Re-export device info type
pub const DeviceInfo = metal_types.DeviceInfo;

// ============================================================================
// Public API — delegates to submodules
// ============================================================================

/// Initialize the Metal backend.
pub fn init() !void {
    return metal_device.init();
}

/// Deinitialize the Metal backend and release all resources.
pub fn deinit() void {
    metal_device.deinit();
}

/// Compile a Metal kernel from source.
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    return metal_compute.compileKernel(allocator, source);
}

/// Launch a compiled kernel synchronously.
pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    return metal_compute.launchKernel(allocator, kernel_handle, config, args);
}

/// Launch a kernel asynchronously without waiting for completion.
pub fn launchKernelAsync(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!ID {
    return metal_compute.launchKernelAsync(allocator, kernel_handle, config, args);
}

/// Destroy a compiled kernel and release its resources.
pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    metal_compute.destroyKernel(allocator, kernel_handle);
}

/// Allocate device memory (Metal buffer).
pub fn allocateDeviceMemory(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    return metal_buffers.allocateDeviceMemory(allocator, size);
}

/// Allocate device memory with a specific allocator.
pub fn allocateDeviceMemoryWithAllocator(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    return metal_buffers.allocateDeviceMemoryWithAllocator(allocator, size);
}

/// Free device memory (Metal buffer).
pub fn freeDeviceMemory(allocator: std.mem.Allocator, ptr: *anyopaque) void {
    metal_buffers.freeDeviceMemory(allocator, ptr);
}

/// Copy data from host to device.
pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    return metal_buffers.memcpyHostToDevice(dst, src, size);
}

/// Copy data from device to host.
pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    return metal_buffers.memcpyDeviceToHost(dst, src, size);
}

/// Copy data between device buffers.
pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    return metal_buffers.memcpyDeviceToDevice(dst, src, size);
}

/// Set the allocator to use for buffer metadata allocations.
pub fn setBufferAllocator(allocator: std.mem.Allocator) void {
    metal_device.setBufferAllocator(allocator);
}

/// Set the allocator to use for tracking pending command buffers.
pub fn setPendingBuffersAllocator(allocator: std.mem.Allocator) void {
    metal_device.setPendingBuffersAllocator(allocator);
}

/// Synchronize with the GPU. Blocks until all pending commands are complete.
pub fn synchronize() void {
    metal_compute.synchronize();
}

/// Wait for a specific command buffer to complete.
pub fn waitForCommandBuffer(cmd_buffer: ID) void {
    metal_compute.waitForCommandBuffer(cmd_buffer);
}

/// Check if Metal backend is available on this system.
pub fn isAvailable() bool {
    return metal_device.isAvailable();
}

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../device.zig").Device;

/// Enumerate all Metal devices available on this Mac.
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    return metal_device.enumerateDevices(allocator);
}

/// Get detailed information about the current default Metal device.
pub fn getDeviceInfo() ?DeviceInfo {
    return metal_device.getDeviceInfo();
}

/// Get the detected GPU feature set (populated during init).
pub fn getFeatureSet() ?MetalFeatureSet {
    return metal_device.getFeatureSet();
}

pub fn getMetalLevel() MetalLevel {
    return metal_device.getMetalLevel();
}

pub fn supportsMetal4() bool {
    return metal_device.supportsMetal4();
}

// ============================================================================
// Test discovery for extracted submodules
// ============================================================================

test {
    _ = @import("metal_types.zig");
    _ = @import("metal_test.zig");
    _ = @import("metal/gpu_family.zig");
    _ = @import("metal/capabilities.zig");
    _ = @import("metal/mps.zig");
    _ = @import("metal/coreml.zig");
    _ = @import("metal/mesh_shaders.zig");
    _ = @import("metal/ray_tracing.zig");
    _ = @import("metal_state.zig");
    _ = @import("metal_device.zig");
    _ = @import("metal_buffers.zig");
    _ = @import("metal_compute.zig");
}

test {
    std.testing.refAllDecls(@This());
}
