//! Focused PITR integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const pitr_tests = @import("integration/pitr_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
