//! Workload ABI - work items and result handles
//!
//! Defines the vtable-based workload interface and result lifetime management.
//! Workloads implement exec/destroy functions; results have explicit ownership.

const std = @import("std");

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
    gpu_vtable: ?*const anyopaque = null,
};

pub const WorkloadHints = struct {
    cpu_affinity: ?u32,
    estimated_duration_us: ?u64,
    prefers_gpu: bool = false,
    requires_gpu: bool = false,
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
