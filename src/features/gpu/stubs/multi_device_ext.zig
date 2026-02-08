const std = @import("std");

pub const GPUCluster = struct {
    pub fn init(_: std.mem.Allocator, _: GPUClusterConfig) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const GPUClusterConfig = struct {
    max_devices: usize = 8,
};

pub const ReduceOp = enum { sum, product, min, max, mean };

pub const AllReduceAlgorithm = enum { ring, tree, recursive_halving, butterfly };

pub const ParallelismStrategy = enum { data, model, pipeline, hybrid };

pub const ModelPartition = struct {
    device_id: usize = 0,
    layer_start: usize = 0,
    layer_end: usize = 0,
};

pub const DeviceBarrier = struct {
    pub fn init(_: usize) @This() {
        return .{};
    }
    pub fn wait(_: *@This()) void {}
};

pub const GradientBucket = struct {
    id: usize = 0,
    size: usize = 0,
};

pub const GradientBucketManager = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};
