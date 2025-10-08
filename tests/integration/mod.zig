//! Integration Tests Module
//!
//! This module provides integration tests for the Abi framework,
//! testing interactions between different components and features.

const std = @import("std");
const abi = @import("abi");

// Test framework initialization
test {
    std.testing.refAllDecls(@This());
}

// Import all integration test suites
const ai_pipeline_tests = @import("ai_pipeline_test.zig");
const database_ops_tests = @import("database_ops_test.zig");
const framework_lifecycle_tests = @import("framework_lifecycle_test.zig");

// Re-export test suites
test "AI Pipeline Integration" {
    std.testing.refAllDecls(ai_pipeline_tests);
}

test "Database Operations Integration" {
    std.testing.refAllDecls(database_ops_tests);
}

test "Framework Lifecycle Integration" {
    std.testing.refAllDecls(framework_lifecycle_tests);
}
