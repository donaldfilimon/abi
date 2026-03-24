//! Focused gateway integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const gateway_tests = @import("integration/gateway_test.zig");
const gateway_runtime_tests = @import("integration/gateway_runtime_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
