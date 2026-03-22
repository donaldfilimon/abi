//! Integration test root.
//!
//! This module is the entry point for integration tests that use the abi package.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Integration test modules
const persona_pipeline = @import("integration/persona_pipeline_test.zig");
const database_tests = @import("integration/database_test.zig");
const inference_tests = @import("integration/inference_test.zig");
const security_tests = @import("integration/security_test.zig");
const auth_tests = @import("integration/auth_test.zig");
const gateway_tests = @import("integration/gateway_test.zig");
const analytics_tests = @import("integration/analytics_test.zig");

test {
    std.testing.refAllDecls(@This());
}
