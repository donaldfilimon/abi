//! CPU-based simulation of GPU kernels for backend fallback and testing.
//!
//! This backend provides CPU-based implementations of common GPU operations
//! for testing and fallback scenarios when no GPU is available.

const std = @import("std");
const types = @import("../kernel_types.zig");

pub const SimulatedError = error{
    UnsupportedKernel,
    InvalidArguments,
    ArgumentCountMismatch,
};

pub const KernelHandle = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
};

pub fn compile(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) (types.KernelError || SimulatedError)!*anyopaque {
    const handle = try allocator.create(KernelHandle);
    errdefer allocator.destroy(handle);

    const name = try allocator.dupe(u8, source.name);
    errdefer allocator.free(name);

    handle.* = .{
        .allocator = allocator,
        .name = name,
    };

    std.log.debug("Simulated kernel compiled: {s}", .{name});
    return handle;
}

pub fn launch(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) (types.KernelError || SimulatedError)!void {
    _ = allocator;
    _ = config;

    const handle: *KernelHandle = @ptrCast(@alignCast(kernel_handle));

    if (std.ascii.eqlIgnoreCase(handle.name, "vector_add")) {
        try launchVectorAdd(args);
        std.log.debug("Simulated kernel launched: vector_add", .{});
        return;
    }
    if (std.ascii.eqlIgnoreCase(handle.name, "matmul")) {
        try launchMatMul(args);
        std.log.debug("Simulated kernel launched: matmul", .{});
        return;
    }
    if (std.ascii.eqlIgnoreCase(handle.name, "reduce_sum")) {
        try launchReduceSum(args);
        std.log.debug("Simulated kernel launched: reduce_sum", .{});
        return;
    }

    std.log.warn("Unsupported simulated kernel: {s}", .{handle.name});
    return SimulatedError.UnsupportedKernel;
}

pub fn destroy(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    const handle: *KernelHandle = @ptrCast(@alignCast(kernel_handle));
    allocator.free(handle.name);
    allocator.destroy(handle);
}

fn argPtrConst(
    comptime T: type,
    args: []const ?*const anyopaque,
    index: usize,
) SimulatedError!*const T {
    if (index >= args.len) return SimulatedError.ArgumentCountMismatch;
    const raw = args[index] orelse return SimulatedError.InvalidArguments;
    return @ptrCast(@alignCast(raw));
}

fn argPtrMut(
    comptime T: type,
    args: []const ?*const anyopaque,
    index: usize,
) SimulatedError!*T {
    const ptr = try argPtrConst(T, args, index);
    return @constCast(ptr);
}

fn launchVectorAdd(args: []const ?*const anyopaque) SimulatedError!void {
    const a_ptr = try argPtrConst(f32, args, 0);
    const b_ptr = try argPtrConst(f32, args, 1);
    const c_ptr = try argPtrMut(f32, args, 2);
    const n_ptr = try argPtrConst(u32, args, 3);

    const n = @as(usize, @intCast(n_ptr.*));
    const a_many: [*]const f32 = @ptrCast(@alignCast(a_ptr));
    const b_many: [*]const f32 = @ptrCast(@alignCast(b_ptr));
    const c_many: [*]f32 = @ptrCast(@alignCast(c_ptr));

    for (0..n) |i| {
        c_many[i] = a_many[i] + b_many[i];
    }
}

fn launchMatMul(args: []const ?*const anyopaque) SimulatedError!void {
    const a_ptr = try argPtrConst(f32, args, 0);
    const b_ptr = try argPtrConst(f32, args, 1);
    const c_ptr = try argPtrMut(f32, args, 2);
    const m_ptr = try argPtrConst(u32, args, 3);
    const n_ptr = try argPtrConst(u32, args, 4);
    const k_ptr = try argPtrConst(u32, args, 5);

    const m = @as(usize, @intCast(m_ptr.*));
    const n = @as(usize, @intCast(n_ptr.*));
    const k = @as(usize, @intCast(k_ptr.*));

    const a_many: [*]const f32 = @ptrCast(@alignCast(a_ptr));
    const b_many: [*]const f32 = @ptrCast(@alignCast(b_ptr));
    const c_many: [*]f32 = @ptrCast(@alignCast(c_ptr));

    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |idx| {
                sum += a_many[row * k + idx] * b_many[idx * n + col];
            }
            c_many[row * n + col] = sum;
        }
    }
}

fn launchReduceSum(args: []const ?*const anyopaque) SimulatedError!void {
    const input_ptr = try argPtrConst(f32, args, 0);
    const output_ptr = try argPtrMut(f32, args, 1);
    const n_ptr = try argPtrConst(u32, args, 2);

    const n = @as(usize, @intCast(n_ptr.*));
    const input_many: [*]const f32 = @ptrCast(@alignCast(input_ptr));
    const output_many: [*]f32 = @ptrCast(@alignCast(output_ptr));

    var sum: f32 = 0.0;
    for (0..n) |i| {
        sum += input_many[i];
    }
    output_many[0] = sum;
}
