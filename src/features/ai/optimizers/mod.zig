const std = @import("std");
const config_mod = @import("config.zig");

pub const config = config_mod;

pub const OptimizerType = config.OptimizerType;
pub const SchedulerType = config.SchedulerType;
pub const SchedulerConfig = config.SchedulerConfig;
pub const OptimizerConfig = config.OptimizerConfig;
pub const OptimizerOps = config.OptimizerOps;

pub const createStatelessOps = config.createStatelessOps;

test {
    std.testing.refAllDecls(@This());
}
