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
const storage_tests = @import("integration/storage_test.zig");
const search_tests = @import("integration/search_test.zig");
const cache_tests = @import("integration/cache_test.zig");

test {
    std.testing.refAllDecls(@This());
}
