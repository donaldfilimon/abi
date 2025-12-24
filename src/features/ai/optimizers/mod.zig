const std = @import("std");
const config_mod = @import("config.zig");
const implementations_mod = @import("implementations.zig");

pub const config = config_mod;

pub const OptimizerType = config.OptimizerType;
pub const SchedulerType = config.SchedulerType;
pub const SchedulerConfig = config.SchedulerConfig;
pub const OptimizerConfig = config.OptimizerConfig;
pub const OptimizerOps = config.OptimizerOps;
pub const OptimizerHandle = config.OptimizerHandle;

pub const createStatelessHandle = config.createStatelessHandle;
pub const createStatelessOps = config.createStatelessOps;

// Re-export optimization implementations
pub const implementations = implementations_mod;

test {
    std.testing.refAllDecls(@This());
}
