//! Focused tasks integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const tasks_tests = @import("integration/tasks_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
