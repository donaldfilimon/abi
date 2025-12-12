//! Hardware Accelerator Driver Interface
//!
//! Defines the interface that all hardware backends (CPU, GPU, etc.) must implement.

const std = @import("std");

/// Type of hardware accelerator
pub const AcceleratorType = enum {
    cpu,
    gpu,
    tpu,
    npu,

    pub fn toString(self: AcceleratorType) []const u8 {
        return switch (self) {
            .cpu => "CPU",
            .gpu => "GPU",
            .tpu => "TPU",
            .npu => "NPU",
        };
    }
};

/// Information about a specific device
pub const DeviceInfo = struct {
    /// Human-readable name of the device
    name: []const u8,
    /// Type of accelerator
    type: AcceleratorType,
    /// Total memory available in bytes
    total_memory: usize,
    /// Compute capability version (if applicable)
    compute_version: ?u32 = null,
    /// Max threads per block/group
    max_threads_per_block: u32 = 0,
};

/// Interface for hardware drivers
pub const Driver = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Initialize the driver and device
        init: *const fn (ctx: *anyopaque) anyerror!void,
        /// Clean up resources
        deinit: *const fn (ctx: *anyopaque) void,
        /// Get device information
        getDeviceInfo: *const fn (ctx: *anyopaque) DeviceInfo,
        /// Allocate memory on the device
        allocate: *const fn (ctx: *anyopaque, size: usize) anyerror![]u8,
        /// Free device memory
        free: *const fn (ctx: *anyopaque, ptr: []u8) void,
        /// Copy memory from host to device
        copyHostToDevice: *const fn (ctx: *anyopaque, dst: []u8, src: []const u8) anyerror!void,
        /// Copy memory from device to host
        copyDeviceToHost: *const fn (ctx: *anyopaque, dst: []u8, src: []const u8) anyerror!void,
        /// Synchronize execution
        synchronize: *const fn (ctx: *anyopaque) anyerror!void,
        /// Matrix multiplication: C = A * B
        matmul: *const fn (ctx: *anyopaque, c: []u8, a: []const u8, b: []const u8, m: usize, n: usize, k: usize) anyerror!void,
        /// 2D Convolution
        conv2d: *const fn (ctx: *anyopaque, output: []u8, input: []const u8, kernel: []const u8, input_dims: [3]usize, kernel_dims: [4]usize) anyerror!void,
    };

    pub fn init(self: Driver) !void {
        return self.vtable.init(self.ptr);
    }

    pub fn deinit(self: Driver) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn getDeviceInfo(self: Driver) DeviceInfo {
        return self.vtable.getDeviceInfo(self.ptr);
    }

    pub fn allocate(self: Driver, size: usize) ![]u8 {
        return self.vtable.allocate(self.ptr, size);
    }

    pub fn free(self: Driver, ptr: []u8) void {
        self.vtable.free(self.ptr, ptr);
    }

    pub fn copyHostToDevice(self: Driver, dst: []u8, src: []const u8) !void {
        return self.vtable.copyHostToDevice(self.ptr, dst, src);
    }

    pub fn copyDeviceToHost(self: Driver, dst: []u8, src: []const u8) !void {
        return self.vtable.copyDeviceToHost(self.ptr, dst, src);
    }

    pub fn synchronize(self: Driver) !void {
        return self.vtable.synchronize(self.ptr);
    }

    pub fn matmul(self: Driver, c: []u8, a: []const u8, b: []const u8, m: usize, n: usize, k: usize) !void {
        return self.vtable.matmul(self.ptr, c, a, b, m, n, k);
    }

    pub fn conv2d(self: Driver, output: []u8, input: []const u8, kernel: []const u8, input_dims: [3]usize, kernel_dims: [4]usize) !void {
        return self.vtable.conv2d(self.ptr, output, input, kernel, input_dims, kernel_dims);
    }
};
