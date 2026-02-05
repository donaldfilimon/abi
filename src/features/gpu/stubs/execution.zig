const std = @import("std");
const backend_mod = @import("backend.zig");

pub const LaunchConfig = struct {};
pub const ExecutionResult = struct {
    execution_time_ns: u64 = 0,
    elements_processed: usize = 0,
    bytes_transferred: usize = 0,
    backend: backend_mod.Backend = .cpu,
    device_id: u32 = 0,
};
pub const ExecutionStats = struct {};
pub const HealthStatus = enum { healthy, degraded, unhealthy };
pub const GpuStats = struct {
    kernels_launched: usize = 0,
    buffers_created: usize = 0,
    bytes_allocated: usize = 0,
    total_execution_time_ns: u64 = 0,
};
pub const MatrixDims = struct { m: usize = 0, n: usize = 0, k: usize = 0 };
pub const MultiGpuConfig = struct {};
pub const LoadBalanceStrategy = enum { round_robin, least_loaded };

pub const ReduceResult = struct {
    value: f32 = 0.0,
    stats: ExecutionResult = .{},
};

pub const DotProductResult = struct {
    value: f32 = 0.0,
    stats: ExecutionResult = .{},
};
