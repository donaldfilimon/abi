//! Focused GPU integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const gpu_tests = @import("integration/gpu_test.zig");
const gpu_policy_tests = @import("integration/gpu_policy_contract_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
