const std = @import("std");
const cli_root = @import("cli_root");

test {
    _ = cli_root;
}

test {
    std.testing.refAllDecls(@This());
}
