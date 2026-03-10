pub const Response = @import("types").GenerateResult;
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
