pub const Request = @import("types.zig").GenerateConfig;
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
