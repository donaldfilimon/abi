const backend = @import("../backend.zig");
const std = @import("std");

pub const Backend = backend.Backend;
pub const backendAvailability = backend.backendAvailability;
pub const moduleEnabled = backend.moduleEnabled;

test {
    std.testing.refAllDecls(@This());
}
