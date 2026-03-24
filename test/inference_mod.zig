//! Focused inference integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const inference_tests = @import("integration/inference_test.zig");
const inference_async_tests = @import("integration/inference_async_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
