//! CPU-based simulation of GPU kernels for backend fallback and testing.
const std = @import("std");
const types = @import("../kernel_types.zig");

pub const KernelHandle = struct {
    name: []const u8,
};

pub fn compile(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    const handle = allocator.create(KernelHandle) catch
        return types.KernelError.CompilationFailed;
    errdefer allocator.destroy(handle);

    handle.* = .{
        .name = allocator.dupe(u8, source.name) catch {
            return types.KernelError.CompilationFailed;
        },
    };

    return handle;
}

pub fn launch(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;
    _ = config;

    const handle: *KernelHandle = @ptrCast(@alignCast(kernel_handle));
    if (std.ascii.eqlIgnoreCase(handle.name, "vector_add")) {
        return launchVectorAdd(args);
    }
    if (std.ascii.eqlIgnoreCase(handle.name, "matmul")) {
        return launchMatMul(args);
    }
    if (std.ascii.eqlIgnoreCase(handle.name, "reduce_sum")) {
        return launchReduceSum(args);
    }
    return types.KernelError.LaunchFailed;
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
) types.KernelError!*const T {
    if (index >= args.len) return types.KernelError.InvalidArguments;
    const raw = args[index] orelse return types.KernelError.InvalidArguments;
    return @ptrCast(@alignCast(raw));
}

fn argPtrMut(
    comptime T: type,
    args: []const ?*const anyopaque,
    index: usize,
) types.KernelError!*T {
    const ptr = try argPtrConst(T, args, index);
    return @constCast(ptr);
}

fn launchVectorAdd(args: []const ?*const anyopaque) types.KernelError!void {
    const a_ptr = try argPtrConst(f32, args, 0);
    const b_ptr = try argPtrConst(f32, args, 1);
    const c_ptr = try argPtrMut(f32, args, 2);
    const n_ptr = try argPtrConst(u32, args, 3);

    const n = @as(usize, @intCast(n_ptr.*));
    const a_many: [*]const f32 = @ptrCast(@alignCast(a_ptr));
    const b_many: [*]const f32 = @ptrCast(@alignCast(b_ptr));
    var c_many: [*]f32 = @ptrCast(@alignCast(c_ptr));

    var i: usize = 0;
    while (i < n) : (i += 1) {
        c_many[i] = a_many[i] + b_many[i];
    }
}

fn launchMatMul(args: []const ?*const anyopaque) types.KernelError!void {
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
    var c_many: [*]f32 = @ptrCast(@alignCast(c_ptr));

    var row: usize = 0;
    while (row < m) : (row += 1) {
        var col: usize = 0;
        while (col < n) : (col += 1) {
            var sum: f32 = 0.0;
            var idx: usize = 0;
            while (idx < k) : (idx += 1) {
                sum += a_many[row * k + idx] * b_many[idx * n + col];
            }
            c_many[row * n + col] = sum;
        }
    }
}

fn launchReduceSum(args: []const ?*const anyopaque) types.KernelError!void {
    const input_ptr = try argPtrConst(f32, args, 0);
    const output_ptr = try argPtrMut(f32, args, 1);
    const n_ptr = try argPtrConst(u32, args, 2);

    const n = @as(usize, @intCast(n_ptr.*));
    const input_many: [*]const f32 = @ptrCast(@alignCast(input_ptr));
    var output_many: [*]f32 = @ptrCast(@alignCast(output_ptr));

    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += input_many[i];
    }
    output_many[0] = sum;
}
