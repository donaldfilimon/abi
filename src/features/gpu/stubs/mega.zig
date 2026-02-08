const std = @import("std");

pub const Coordinator = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const BackendInstance = struct {};

pub const WorkloadProfile = struct {
    category: WorkloadCategory = .general,
    precision: Precision = .f32,
};

pub const WorkloadCategory = enum { general, inference, training, rendering };

pub const ScheduleDecision = struct {
    backend_index: usize = 0,
    estimated_time_ms: f64 = 0,
};

pub const Precision = enum { f32, f16, bf16, int8, int4 };
