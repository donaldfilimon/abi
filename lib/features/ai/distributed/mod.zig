const std = @import("std");
const training_mod = @import("training.zig");

pub const training = training_mod;

pub const DistributedConfig = training.DistributedConfig;
pub const ParameterServer = training.ParameterServer;

test {
    std.testing.refAllDecls(@This());
}
