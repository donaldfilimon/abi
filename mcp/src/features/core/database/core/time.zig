//! Clocks, monotonic timestamps, logical clocks.

const std = @import("std");

pub const LogicalClock = struct {
    counter: u64,
};

pub const Timestamp = struct {
    ms: i64,
};
