const std = @import("std");
const config = @import("config.zig");

pub const OptimizerType = config.OptimizerType;
pub const SchedulerType = config.SchedulerType;
pub const SchedulerConfig = config.SchedulerConfig;
pub const OptimizerConfig = config.OptimizerConfig;
pub const OptimizerOps = config.OptimizerOps;

pub const createStatelessOps = config.createStatelessOps;

pub usingnamespace config;

test {
    std.testing.refAllDecls(@This());
}
