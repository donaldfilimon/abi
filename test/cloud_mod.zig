//! Focused cloud integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const cloud_tests = @import("integration/cloud_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
