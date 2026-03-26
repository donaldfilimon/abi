//! Focused connectors integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const connector_tests = @import("integration/connector_test.zig");
const connector_errors_tests = @import("integration/connector_errors_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
