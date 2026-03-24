//! Focused agents integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const agents_tests = @import("integration/agents_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
