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
pub const AccessHint = enum { read_only, write_only, read_write };
pub const ElementType = enum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f16,
    f32,
    f64,

    pub fn size(self: ElementType) usize {
        return switch (self) {
            .u8, .i8 => 1,
            .u16, .i16, .f16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }
};
pub const AsyncTransfer = struct {
    pub fn isComplete(_: *const AsyncTransfer) bool {
        return true;
    }
    pub fn wait(_: *AsyncTransfer) Error!void {
        return error.GpuDisabled;
    }
};
