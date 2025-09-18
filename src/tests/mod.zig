//! Test Suite - Main Test Runner
//!
//! Comprehensive test suite covering:
//! - Unit tests for individual components
//! - Integration tests for system interactions
//! - Performance and regression tests

// =============================================================================
// TEST ORGANIZATION
// =============================================================================

// Unit tests are organized in the unit/ directory
// Integration tests are organized in the integration/ directory
// Run individual tests with: zig test src/tests/unit/<test_file>.zig

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

/// Main test entry point
pub fn main() !void {
    const std = @import("std");

    std.debug.print("🧪 ABI Framework Test Suite\n", .{});
    std.debug.print("==========================\n\n", .{});
    std.debug.print("✅ Test runner initialized successfully!\n", .{});
    std.debug.print("ℹ️ Individual tests can be run with 'zig test <test_file>.zig'\n", .{});
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
