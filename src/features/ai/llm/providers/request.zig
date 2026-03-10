pub const Request = @import("types").GenerateConfig;
const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
