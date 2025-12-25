const std = @import("std");
const observability = @import("../../shared/observability/mod.zig");

pub const MetricsCollector = observability.MetricsCollector;
pub const Counter = observability.Counter;
pub const Histogram = observability.Histogram;

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
