//! NUMA-aware scheduling stub for platforms without thread support.

const std = @import("std");

pub const NumaNode = struct {
    id: u32,
    cpu_mask: u64,
    memory_mb: u64,
};

pub const NumaTopology = struct {
    pub fn init(_: std.mem.Allocator) !NumaTopology {
        return error.ThreadsNotSupported;
    }
    pub fn deinit(_: *NumaTopology) void {}
    pub fn getLocalNode(_: *NumaTopology) ?NumaNode {
        return null;
    }
};
