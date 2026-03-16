pub const Response = @import("types.zig").GenerateResult;
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
