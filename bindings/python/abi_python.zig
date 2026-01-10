//! Python bindings for ABI framework using the C API.
//!
//! This module provides a Zig-based C extension that can be loaded by Python.
//! Build with: zig build-lib -dynamic -target x86_64-linux-gnu abi_python.zig

const std = @import("std");
const root = @import("../../src/abi.zig");

// Export version information
pub export fn abi_version() [*:0]const u8 {
    return root.version();
}

// Memory management exports
var global_allocator: std.mem.Allocator = std.heap.page_allocator;

pub export fn abi_alloc(size: usize) ?[*]u8 {
    const mem = global_allocator.alloc(u8, size) catch return null;
    return mem.ptr;
}

pub export fn abi_free(ptr: [*]u8, size: usize) void {
    global_allocator.free(ptr[0..size]);
}

// Framework initialization
var framework: ?root.Framework = null;

pub export fn abi_init() i32 {
    if (framework != null) return 0; // Already initialized

    framework = root.init(global_allocator, .{}) catch |err| {
        _ = err;
        return -1;
    };
    return 0;
}

pub export fn abi_shutdown() void {
    if (framework) |*fw| {
        root.shutdown(fw);
        framework = null;
    }
}

pub export fn abi_is_initialized() bool {
    return framework != null;
}

// Feature checking
pub export fn abi_has_feature(feature: u32) bool {
    if (framework) |fw| {
        const tag: root.Feature = @enumFromInt(feature);
        return fw.isFeatureEnabled(tag);
    }
    return false;
}

// SIMD operations
pub export fn abi_simd_available() bool {
    return root.hasSimdSupport();
}

pub export fn abi_vector_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return root.simd.vectorDot(a[0..len], b[0..len]);
}

pub export fn abi_cosine_similarity(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return root.simd.cosineSimilarity(a[0..len], b[0..len]);
}

pub export fn abi_l2_norm(vec: [*]const f32, len: usize) f32 {
    return root.simd.vectorL2Norm(vec[0..len]);
}

pub export fn abi_vector_add(a: [*]const f32, b: [*]const f32, result: [*]f32, len: usize) void {
    root.simd.vectorAdd(a[0..len], b[0..len], result[0..len]);
}

// Platform information
pub export fn abi_platform_os() u32 {
    const info = root.platform.platform.PlatformInfo.detect();
    return @intFromEnum(info.os);
}

pub export fn abi_platform_arch() u32 {
    const info = root.platform.platform.PlatformInfo.detect();
    return @intFromEnum(info.arch);
}

pub export fn abi_platform_max_threads() u32 {
    const info = root.platform.platform.PlatformInfo.detect();
    return @intCast(info.max_threads);
}

// Error handling
pub export fn abi_get_last_error() [*:0]const u8 {
    return "No error";
}

// Module handle for Python extension
pub export fn PyInit_abi() ?*anyopaque {
    // This would be filled in by the actual Python C extension code
    // For now, return null to indicate the module structure needs
    // to be created by the Python build system
    return null;
}
