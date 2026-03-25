//! Focused multi-agent integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const multi_agent_tests = @import("integration/multi_agent_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
