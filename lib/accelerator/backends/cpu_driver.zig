//! CPU Accelerator Driver
//!
//! Reference implementation of the hardware accelerator interface for CPU.

const std = @import("std");

const driver = @import("../driver.zig");

pub const CpuDriver = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) driver.Driver {
        const self = allocator.create(CpuDriver) catch @panic("OOM");
        self.* = .{ .allocator = allocator };

        return driver.Driver{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn initFn(ctx: *anyopaque) anyerror!void {
        _ = ctx;
        // CPU driver is always ready
    }

    fn deinitFn(ctx: *anyopaque) void {
        const self: *CpuDriver = @ptrCast(@alignCast(ctx));
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    fn getDeviceInfoFn(_: *anyopaque) driver.DeviceInfo {
        return .{
            .name = "CPU Fallback",
            .type = .cpu,
            .total_memory = 0, // System memory
        };
    }

    fn allocateFn(ctx: *anyopaque, size: usize) anyerror![]u8 {
        const self: *CpuDriver = @ptrCast(@alignCast(ctx));
        return self.allocator.alloc(u8, size);
    }

    fn freeFn(ctx: *anyopaque, ptr: []u8) void {
        const self: *CpuDriver = @ptrCast(@alignCast(ctx));
        self.allocator.free(ptr);
    }

    fn copyHostToDeviceFn(_: *anyopaque, dst: []u8, src: []const u8) anyerror!void {
        @memcpy(dst, src);
    }

    fn copyDeviceToHostFn(_: *anyopaque, dst: []u8, src: []const u8) anyerror!void {
        @memcpy(dst, src);
    }

    fn synchronizeFn(_: *anyopaque) anyerror!void {
        // CPU is synchronous
    }

    fn matmulFn(_: *anyopaque, c: []u8, a: []const u8, b: []const u8, m: usize, n: usize, k: usize) anyerror!void {
        // Naive matrix multiplication implementation for standard f32
        // Assuming data is f32 (4 bytes)
        // A: m x k, B: k x n, C: m x n

        // Cast to f32 slices
        const a_f32 = std.mem.bytesAsSlice(f32, a);
        const b_f32 = std.mem.bytesAsSlice(f32, b);
        const c_f32 = std.mem.bytesAsSlice(f32, c);

        if (a_f32.len != m * k or b_f32.len != k * n or c_f32.len != m * n) {
            return error.DimensionMismatch;
        }

        @memset(c_f32, 0);

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |p| {
                    sum += a_f32[i * k + p] * b_f32[p * n + j];
                }
                c_f32[i * n + j] = sum;
            }
        }
    }

    fn conv2dFn(_: *anyopaque, output: []u8, input: []const u8, kernel: []const u8, input_dims: [3]usize, kernel_dims: [4]usize) anyerror!void {
        _ = output;
        _ = input;
        _ = kernel;
        _ = input_dims;
        _ = kernel_dims;
        // Placeholder for CPU conv2d
    }

    const vtable = driver.Driver.VTable{
        .init = initFn,
        .deinit = deinitFn,
        .getDeviceInfo = getDeviceInfoFn,
        .allocate = allocateFn,
        .free = freeFn,
        .copyHostToDevice = copyHostToDeviceFn,
        .copyDeviceToHost = copyDeviceToHostFn,
        .synchronize = synchronizeFn,
        .matmul = matmulFn,
        .conv2d = conv2dFn,
    };
};
