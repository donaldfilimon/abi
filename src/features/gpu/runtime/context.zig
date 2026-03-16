const unified = @import("../unified.zig");
const std = @import("std");

pub const Gpu = unified.Gpu;
pub const GpuConfig = unified.GpuConfig;
pub const HealthStatus = unified.HealthStatus;
pub const GpuStats = unified.GpuStats;
pub const MemoryInfo = unified.MemoryInfo;
pub const MultiGpuConfig = unified.MultiGpuConfig;

test {
    std.testing.refAllDecls(@This());
}
