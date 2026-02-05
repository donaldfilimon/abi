const std = @import("std");

pub const Error = error{
    GpuDisabled,
    NoDeviceAvailable,
    InitializationFailed,
    InvalidConfig,
    OutOfMemory,
    KernelCompilationFailed,
    KernelExecutionFailed,
};

pub const Buffer = struct {
    pub fn read(_: *Buffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }

    pub fn readBytes(_: *Buffer, _: []u8) Error!void {
        return error.GpuDisabled;
    }

    pub fn write(_: *Buffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }

    pub fn writeBytes(_: *Buffer, _: []const u8) Error!void {
        return error.GpuDisabled;
    }

    pub fn size(_: *const Buffer) usize {
        return 0;
    }

    pub fn deinit(_: *Buffer) void {}
};

pub const UnifiedBuffer = struct {
    pub fn read(_: *UnifiedBuffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }

    pub fn write(_: *UnifiedBuffer, comptime _: type, _: anytype) Error!void {
        return error.GpuDisabled;
    }

    pub fn size(_: *const UnifiedBuffer) usize {
        return 0;
    }

    pub fn deinit(_: *UnifiedBuffer) void {}
};
pub const BufferFlags = packed struct { read: bool = true, write: bool = true };
pub const BufferOptions = struct {};
pub const BufferView = struct {};
pub const BufferStats = struct {};
pub const MappedBuffer = struct {};

pub const MemoryPool = struct {};
pub const MemoryStats = struct {};
pub const MemoryInfo = struct {
    used_bytes: usize = 0,
    peak_used_bytes: usize = 0,
    total_bytes: usize = 0,
};
pub const MemoryMode = enum { automatic, explicit, unified };
pub const MemoryLocation = enum { device, host };
