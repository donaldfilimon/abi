const std = @import("std");

pub const DeviceGroup = struct {};
pub const WorkDistribution = struct {};
pub const GroupStats = struct {};

pub const GPUCluster = struct {
    pub fn init(_: std.mem.Allocator, _: GPUClusterConfig) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};
pub const GPUClusterConfig = struct {};
pub const ReduceOp = enum { sum, max, min, avg };
pub const AllReduceAlgorithm = enum { ring, tree, butterfly };
pub const ParallelismStrategy = enum { data, model, pipeline, hybrid };
pub const ModelPartition = struct {};
pub const DeviceBarrier = struct {};
pub const GradientBucket = struct {};
pub const GradientBucketManager = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

test {
    std.testing.refAllDecls(@This());
}
