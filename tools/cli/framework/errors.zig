pub const Error = error{
    UnknownCommand,
    ForwardLoop,
};

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
