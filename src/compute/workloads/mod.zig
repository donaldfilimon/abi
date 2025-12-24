//! Workload definitions
//!
//! Matrix multiplication, neural inference, and other compute kernels.

const std = @import("std");

pub const matmul = @import("matmul.zig");
pub const neural_inference = @import("neural_inference.zig");

pub const Matrix = matmul.Matrix;
pub const MatrixMultiplication = matmul.MatrixMultiplication;
pub const NeuralInference = neural_inference.NeuralInference;

pub const WorkloadVTable = struct {
    exec: *const fn (user: *anyopaque, ctx: *ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque,
    destroy: *const fn (user: *anyopaque, a: std.mem.Allocator) void,
    name: []const u8,
};

pub const WorkItem = struct {
    id: u64,
    user: *anyopaque,
    vtable: *const WorkloadVTable,
    priority: f32,
    hints: WorkloadHints,
};

pub const WorkloadHints = struct {
    cpu_affinity: ?u32,
    estimated_duration_us: ?u64,
};

pub const DEFAULT_HINTS = WorkloadHints{ .cpu_affinity = null, .estimated_duration_us = null };

pub const ExecutionContext = struct {
    worker_id: u32,
    arena: *std.heap.ArenaAllocator,
};

pub const ResultVTable = struct {
    destroy: *const fn (ptr: *anyopaque, a: std.mem.Allocator) void,
};

pub const ResultHandle = struct {
    ptr: *anyopaque,
    vtable: *const ResultVTable,

    pub fn as(self: ResultHandle, comptime T: type) *T {
        return @ptrCast(@alignCast(self.ptr));
    }

    pub fn deinit(self: *ResultHandle, a: std.mem.Allocator) void {
        self.vtable.destroy(self.ptr, a);
        self.* = undefined;
    }
};
