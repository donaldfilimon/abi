//! Focused search integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const search_tests = @import("integration/search_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
