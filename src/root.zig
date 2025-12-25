pub const abi = @import("abi.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
