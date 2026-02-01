const std = @import("std");

pub const Profiler = struct {};
pub const TimingResult = struct {};
pub const OccupancyResult = struct {};
pub const MemoryBandwidth = struct {};

pub const MetricsSummary = struct {
    total_kernel_invocations: usize = 0,
    avg_kernel_time_ns: f64 = 0.0,
    kernels_per_second: f64 = 0.0,
};

pub const KernelMetrics = struct {};
pub const MetricsCollector = struct {};
