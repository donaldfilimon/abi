//! Focused pipeline integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const pipeline_tests = @import("integration/pipeline_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
