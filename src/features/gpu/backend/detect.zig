const backend = @import("../backend.zig");
const std = @import("std");

pub const moduleEnabled = backend.moduleEnabled;
pub const isEnabled = backend.isEnabled;
pub const backendAvailability = backend.backendAvailability;
pub const availableBackends = backend.availableBackends;

test {
    std.testing.refAllDecls(@This());
}
