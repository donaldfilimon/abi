//! Focused auth integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const auth_tests = @import("integration/auth_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
