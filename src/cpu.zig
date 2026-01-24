//! CPU Fallback and Host-side GPU Operations
//!
//! Provides CPU fallback implementations and host-side GPU utility functions.
//! Used when GPU is unavailable or when CPU-specific operations are required.

const std = @import("std");

/// CPU-specific memory operations and fallbacks
pub const CpuBackend = struct {
    /// Check if CPU fallback is available
    pub fn isAvailable() bool {
        return true;
    }

    /// Get CPU device count (always 1 for CPU)
    pub fn getDeviceCount() u32 {
        return 1;
    }

    /// Get CPU device capabilities
    pub fn getDeviceCaps(device_id: u32) struct {
        name: []const u8,
        total_memory: usize,
        compute_capability_major: u32,
        compute_capability_minor: u32,
    } {
        _ = device_id;
        return .{
            .name = "CPU Fallback",
            .total_memory = std.mem.total_size,
            .compute_capability_major = 1,
            .compute_capability_minor = 0,
        };
    }

    /// Execute compute operation on CPU
    pub fn execute(comptime Kernel: type, config: anytype, args: anytype) void {
        _ = Kernel;
        _ = config;
        _ = args;
        // CPU execution stub - would be implemented for specific kernels
    }
};

/// CPU memory buffer for fallback operations
pub const CpuBuffer = struct {
    data: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !CpuBuffer {
        return .{
            .data = try allocator.alloc(u8, size),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CpuBuffer) void {
        self.allocator.free(self.data);
    }

    pub fn read(self: *const CpuBuffer, comptime T: type, dest: []T) void {
        @memcpy(dest[0..@min(dest.len, self.data.len)], self.data[0..@min(dest.len, self.data.len)]);
    }

    pub fn write(self: *CpuBuffer, comptime T: type, src: []const T) void {
        @memcpy(self.data[0..@min(self.data.len, src.len)], src[0..@min(self.data.len, src.len)]);
    }
};

/// Test if running tests
pub fn isTest() bool {
    return @import("builtin").is_test;
}
