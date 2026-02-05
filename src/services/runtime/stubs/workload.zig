const std = @import("std");
const types = @import("types.zig");

pub const ExecutionContext = struct {};
pub const WorkloadHints = struct {
    priority: Priority = .normal,
    estimated_cycles: ?u64 = null,
};
pub const Priority = enum { low, normal, high, critical };
pub const WorkloadVTable = struct {};
pub const GPUWorkloadVTable = struct {};
pub const ResultHandle = struct {};
pub const ResultVTable = struct {};
pub const WorkItem = struct {};

pub fn runWorkItem(_: WorkItem) types.Error!void {
    return error.RuntimeDisabled;
}
