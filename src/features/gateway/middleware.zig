const types = @import("types.zig");
const std = @import("std");

/// Gateway middleware type aliases are centralized here so future middleware
/// handlers can be split out without changing the public gateway module.
pub const MiddlewareType = types.MiddlewareType;

test {
    std.testing.refAllDecls(@This());
}
