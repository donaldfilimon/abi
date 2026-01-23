//! FPGA VTable Backend Implementation
//!
//! Implements the Backend interface for FPGA accelerators.
//! Targeted for AMD Alveo and Intel Agilex platforms as per research docs.

const std = @import("std");
const interface = @import("../../interface.zig");
const kernels = @import("kernels.zig");

pub const FpgaBackend = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        const self = allocator.create(Self) catch return interface.BackendError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        return 0; // TODO: Detect FPGAs
    }

    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        _ = self;
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        return interface.DeviceCaps{
            .name = "FPGA Accelerator (Simulated)",
            .name_len = 26,
            .total_memory = 16 * 1024 * 1024 * 1024, // 16GB HBM
            .max_threads_per_block = 1, // Task parallelism
            .max_shared_memory = 32 * 1024 * 1024, // URAM size
            .warp_size = 1,
            .supports_fp16 = true,
            .supports_fp64 = false,
            .supports_int8 = true, // Native quantized support
            .unified_memory = false,
        };
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = self;
        _ = size;
        _ = flags;
        return interface.MemoryError.NotImplemented;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        _ = self;
        _ = ptr;
    }

    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        _ = stream;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        _ = stream;
        return interface.MemoryError.NotImplemented;
    }

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, name: []const u8) interface.KernelError!*anyopaque {
        _ = self;
        _ = allocator;
        _ = source;
        _ = name;
        return interface.KernelError.NotImplemented;
    }

    pub fn launchKernel(self: *Self, kernel: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        _ = self;
        _ = kernel;
        _ = config;
        _ = args;
        return interface.KernelError.NotImplemented;
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        _ = self;
        _ = kernel;
    }

    pub fn synchronize(self: *Self) interface.BackendError!void {
        _ = self;
        return {};
    }
};

pub fn createFpgaVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try FpgaBackend.init(allocator);
    return interface.createBackend(FpgaBackend, impl);
}
