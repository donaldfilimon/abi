//! Profile/Profile System Tests
//!
//! These tests verify the canonical behavior profiles system (v3).
//! Legacy profile tests were removed during the profiles→profiles migration.
//! Profile-specific tests now live in src/features/ai/profiles/mod.zig.

const std = @import("std");

test "placeholder — profiles tests are in-tree" {
    // Profile system tests are embedded in the feature module:
    //   src/features/ai/profiles/mod.zig (inline tests)
    //   build/test_discovery.zig (manifest-driven discovery)
    //
    // Run with: zig build feature-tests --summary all
    try std.testing.expect(true);
}
