const std = @import("std");
const training = @import("training.zig");

pub const DistributedConfig = training.DistributedConfig;
pub const ParameterServer = training.ParameterServer;

// usingnamespace training;  // intentionally omitted: explicit re-exports above

test {
    std.testing.refAllDecls(@This());
}
