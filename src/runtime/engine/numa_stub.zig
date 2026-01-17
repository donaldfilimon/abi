//! NUMA Stub for WASM/Freestanding Targets
//!
//! Provides stub implementations when NUMA topology detection is not available.

const std = @import("std");

/// Stub CPU topology
pub const CpuTopology = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CpuTopology {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *CpuTopology) void {
        _ = self;
    }

    pub fn getNumaNodeCount(self: *const CpuTopology) usize {
        _ = self;
        return 1;
    }

    pub fn getCpuCount(self: *const CpuTopology) usize {
        _ = self;
        return 1;
    }
};

/// Stub NUMA node
pub const NumaNode = struct {
    id: usize = 0,
    cpu_count: usize = 1,
};

/// Stub CPU info
pub const CpuInfo = struct {
    id: usize = 0,
    numa_node: usize = 0,
};
