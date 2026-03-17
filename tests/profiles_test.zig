//! Profile System Smoke Tests (external test root)
//!
//! Lightweight verification that profile types are accessible through the
//! public `abi` package interface.  Comprehensive profile tests live in
//! src/services/tests/e2e_profiles_test.zig (run via `zig build test`).

const std = @import("std");
const abi = @import("abi");

test "profile types are accessible via abi.ai.prompts" {
    const prompts = abi.ai.prompts;

    // Verify the ProfileType enum exists and has expected variants.
    const profile_type: prompts.ProfileType = .assistant;
    try std.testing.expect(profile_type == .assistant);

    // Verify getProfile returns a non-empty system prompt.
    const profile = prompts.getProfile(.assistant);
    try std.testing.expect(profile.system_prompt.len > 0);
    try std.testing.expectEqualStrings("assistant", profile.name);
}

test "all profiles have valid temperature ranges" {
    const prompts = abi.ai.prompts;
    const all_types = prompts.listProfiles();

    try std.testing.expect(all_types.len > 0);

    for (all_types) |profile_type| {
        const profile = prompts.getProfile(profile_type);
        try std.testing.expect(profile.suggested_temperature >= 0.0);
        try std.testing.expect(profile.suggested_temperature <= 2.0);
        try std.testing.expect(profile.name.len > 0);
    }
}
