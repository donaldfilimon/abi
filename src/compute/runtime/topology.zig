//! Topology detection
//!
//! CPU core count, cache line size, and NUMA detection.

const std = @import("std");

pub const TopologyInfo = struct {
    core_count: usize,
    cache_line_size: usize,
    numa_node_count: ?usize,
};

pub fn detect() TopologyInfo {
    return .{
        .core_count = std.Thread.getCpuCount() catch 1,
        .cache_line_size = std.cache_line_size,
        .numa_node_count = null,
    };
}
