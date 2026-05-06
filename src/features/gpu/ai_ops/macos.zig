//! macOS/Darwin AiOps Implementation
//!
//! Wraps the MacOSAccelerator into the unified AiOps interface.
//! Selects between Accelerate (AMX) and MPS (GPU) based on size.

const std = @import("std");
const ai_ops = @import("../ai_ops.zig");
const adapters = @import("adapters.zig");
const macos_accel = @import("../backends/metal/macos_accelerator.zig");
const accelerate = @import("../backends/metal/accelerate.zig");
const Transpose = ai_ops.Transpose;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;

pub const MacosAiOps = struct {
    allocator: std.mem.Allocator,
    accel: macos_accel.MacOSAccelerator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .accel = macos_accel.MacOSAccelerator.init(.{}),
        };
        // Attempt to init metal for MPS support
        self.accel.initMetal() catch |err| {
            std.log.warn("MacosAiOps: Metal/MPS init failed (falling back to Accelerate/CPU): {any}", .{err});
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn isAvailable(self: *Self) bool {
        _ = self;
        return true; // We always have Accelerate on Darwin
    }

    pub fn sgemm(
        self: *Self,
        _: Transpose,
        _: Transpose,
        m: i32,
        n: i32,
        k: i32,
        _: f32,
        a: *const anyopaque,
        _: i32,
        b: *const anyopaque,
        _: i32,
        _: f32,
        c: *anyopaque,
        _: i32,
    ) AiOpsError!void {
        const m_u = @as(usize, @intCast(m));
        const n_u = @as(usize, @intCast(n));
        const k_u = @as(usize, @intCast(k));

        const a_slice = @as([*]const f32, @ptrCast(@alignCast(a)))[0 .. m_u * k_u];
        const b_slice = @as([*]const f32, @ptrCast(@alignCast(b)))[0 .. k_u * n_u];
        const c_slice = @as([*]f32, @ptrCast(@alignCast(c)))[0 .. m_u * n_u];

        self.accel.matmul(
            a_slice,
            b_slice,
            c_slice,
            @intCast(m_u),
            @intCast(n_u),
            @intCast(k_u),
        ) catch |err| {
            std.log.err("MacosAiOps: sgemm failed: {any}", .{err});
            return AiOpsError.KernelFailed;
        };
    }

    pub fn sgemmStridedBatched(
        self: *Self,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const anyopaque,
        lda: i32,
        stride_a: i64,
        b: *const anyopaque,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: *anyopaque,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) AiOpsError!void {
        // MacOSAccelerator doesn't have strided batched sgemm directly yet,
        // so we loop using sgemm.
        for (0..@intCast(batch_count)) |i| {
            const a_offset = i * @as(usize, @intCast(stride_a));
            const b_offset = i * @as(usize, @intCast(stride_b));
            const c_offset = i * @as(usize, @intCast(stride_c));

            const a_ptr = @as([*]const f32, @ptrCast(@alignCast(a))) + a_offset;
            const b_ptr = @as([*]const f32, @ptrCast(@alignCast(b))) + b_offset;
            const c_ptr = @as([*]f32, @ptrCast(@alignCast(c))) + c_offset;

            try self.sgemm(
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a_ptr,
                lda,
                b_ptr,
                ldb,
                beta,
                c_ptr,
                ldc,
            );
        }
    }

    pub fn softmax(self: *Self, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const data_slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        // Use Accelerate optimized softmax
        accelerate.softmax(data_slice, data_slice, self.allocator) catch return AiOpsError.KernelFailed;
    }

    pub fn rmsnorm(self: *Self, x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, _: ?*anyopaque) AiOpsError!void {
        const x_slice = @as([*]f32, @ptrCast(@alignCast(x)))[0..len];
        const w_slice = @as([*]const f32, @ptrCast(@alignCast(weight)))[0..len];
        accelerate.rmsnorm(x_slice, w_slice, x_slice, eps, self.allocator) catch return AiOpsError.KernelFailed;
    }

    pub fn silu(self: *Self, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const data_slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        accelerate.silu(data_slice, data_slice, self.allocator) catch return AiOpsError.KernelFailed;
    }

    pub fn gelu(self: *Self, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const data_slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        accelerate.gelu(data_slice, data_slice, self.allocator) catch return AiOpsError.KernelFailed;
    }

    pub fn scale(self: *Self, data: *anyopaque, scalar: f32, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const data_slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        accelerate.sscal(scalar, data_slice) catch return AiOpsError.KernelFailed;
    }

    pub fn elementwiseMul(self: *Self, a: *anyopaque, b: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const a_slice = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
        const b_slice = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
        accelerate.vmul(a_slice, b_slice, a_slice) catch return AiOpsError.KernelFailed;
    }

    pub fn elementwiseAdd(self: *Self, a: *anyopaque, b: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const a_slice = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
        const b_slice = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
        accelerate.vadd(a_slice, b_slice, a_slice) catch return AiOpsError.KernelFailed;
    }

    pub fn allocDevice(self: *Self, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
        // On Apple Silicon, memory is unified. We can just use regular allocator.
        // For MPS, we might need MTLBuffer, but MacOSAccelerator handles those internally
        // when passed host pointers (it creates wraps).
        const ptr = allocator.alloc(u8, size) catch return AiOpsError.OutOfMemory;
        return DeviceBuffer{
            .ptr = ptr.ptr,
            .size = size,
            .allocator = allocator,
            .ops = self.asAiOps(),
        };
    }

    pub fn copyToDevice(_: *Self, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
        const dst_ptr = @as([*]u8, @ptrCast(@alignCast(dst)));
        @memcpy(dst_ptr[0..len], src[0..len]);
    }

    pub fn copyFromDevice(_: *Self, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
        const src_ptr = @as([*]const u8, @ptrCast(@alignCast(src)));
        @memcpy(dst[0..len], src_ptr[0..len]);
    }

    pub fn freeDevice(_: *Self, ptr: *anyopaque) void {
        // Freeing will be handled by the DeviceBuffer's allocator call in this simple case,
        // but we need to match the VTable.
        _ = ptr;
    }

    pub fn asAiOps(self: *Self) *const ai_ops.AiOps {
        const ops = adapters.createAiOps(Self, self);
        const ops_ptr = self.allocator.create(ai_ops.AiOps) catch unreachable;
        ops_ptr.* = ops;
        return ops_ptr;
    }
};
