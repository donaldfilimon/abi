const std = @import("std");
const training = @import("training.zig");

pub const DistributedConfig = training.DistributedConfig;
pub const ParameterServer = training.ParameterServer;

pub usingnamespace training;

test {
    std.testing.refAllDecls(@This());
}
