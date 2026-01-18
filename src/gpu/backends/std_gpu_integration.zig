//! Integration with Zig 0.16's std.gpu facilities
//!
//! Provides a bridge between our backend interface and Zig's standard library
//! GPU abstraction. This is a compatibility layer that gracefully handles
//! cases where std.gpu may not be fully available yet.

const std = @import("std");

pub const StdGpuError = error{
    DeviceInitFailed,
    QueueCreationFailed,
    BufferAllocationFailed,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    OutOfMemory,
    StdGpuNotAvailable,
};

/// Check if std.gpu is available in this Zig version
pub fn isStdGpuAvailable() bool {
    return @hasDecl(std, "gpu");
}

/// Wrapper around std.gpu.Device (compatibility layer)
pub const StdGpuDevice = struct {
    allocator: std.mem.Allocator,
    // Note: In future Zig versions with std.gpu, this would hold the actual device
    is_emulated: bool = true,

    pub fn deinit(self: *StdGpuDevice) void {
        _ = self;
        // Cleanup when std.gpu is available
    }

    pub fn createQueue(self: *StdGpuDevice) !StdGpuQueue {
        if (!isStdGpuAvailable()) {
            return StdGpuError.StdGpuNotAvailable;
        }

        return StdGpuQueue{
            .allocator = self.allocator,
        };
    }

    pub fn createBuffer(self: *StdGpuDevice, desc: BufferDescriptor) !StdGpuBuffer {
        if (!isStdGpuAvailable()) {
            return StdGpuError.StdGpuNotAvailable;
        }

        // Allocate CPU-side buffer as fallback
        const buffer_data = try self.allocator.alloc(u8, desc.size);
        errdefer self.allocator.free(buffer_data);

        return StdGpuBuffer{
            .data = buffer_data,
            .size = desc.size,
            .allocator = self.allocator,
        };
    }
};

pub const StdGpuQueue = struct {
    allocator: std.mem.Allocator,

    pub fn deinit(self: *StdGpuQueue) void {
        _ = self;
    }

    pub fn submit(self: *StdGpuQueue) !void {
        _ = self;
        // Submit commands when std.gpu is available
    }
};

pub const BufferDescriptor = struct {
    size: usize,
    usage: BufferUsage = .{},
};

pub const BufferUsage = struct {
    storage: bool = false,
    uniform: bool = false,
    copy_dst: bool = false,
    copy_src: bool = false,
};

pub const StdGpuBuffer = struct {
    data: []u8,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *StdGpuBuffer) void {
        self.allocator.free(self.data);
    }

    pub fn write(self: *StdGpuBuffer, offset: usize, data: []const u8) !void {
        if (offset + data.len > self.size) {
            return error.BufferTooSmall;
        }

        @memcpy(self.data[offset..][0..data.len], data);
    }

    pub fn read(self: *StdGpuBuffer, offset: usize, data: []u8) !void {
        if (offset + data.len > self.size) {
            return error.BufferTooSmall;
        }

        @memcpy(data, self.data[offset..][0..data.len]);
    }
};

/// Initialize a std.gpu device (or CPU fallback)
pub fn initStdGpuDevice(allocator: std.mem.Allocator) !StdGpuDevice {
    if (!isStdGpuAvailable()) {
        // Return CPU emulation device
        return StdGpuDevice{
            .allocator = allocator,
            .is_emulated = true,
        };
    }

    // Future: Initialize actual std.gpu device
    return StdGpuDevice{
        .allocator = allocator,
        .is_emulated = false,
    };
}

/// Compile SPIR-V shader using std.gpu (placeholder for future implementation)
pub fn compileShaderToSpirv(
    allocator: std.mem.Allocator,
    source: []const u8,
    entry_point: []const u8,
) ![]const u32 {
    _ = allocator;
    _ = source;
    _ = entry_point;

    if (!isStdGpuAvailable()) {
        return error.StdGpuNotAvailable;
    }

    // TODO: Integrate with std.gpu's shader compilation when available
    // For now, return minimal SPIR-V header
    const spirv_header = [_]u32{
        0x07230203, // SPIR-V magic
        0x00010000, // Version 1.0
        0x00000000, // Generator
        0x00000001, // Bound
        0x00000000, // Schema
    };

    const result = try allocator.alloc(u32, spirv_header.len);
    @memcpy(result, &spirv_header);
    return result;
}
