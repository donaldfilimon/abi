//! End-to-End Multi-Profile Integration Tests
//!
//! Comprehensive tests for the multi-profile AI assistant system covering:
//! - Profile definitions and retrieval
//! - System prompt generation
//! - Profile configuration (temperature, examples)
//! - All profile types (assistant, coder, writer, etc.)
//! - Edge cases: invalid profiles, empty prompts
//! - Integration with AI module
//!
//! These tests verify the profile system works correctly without
//! requiring external AI services.

const std = @import("std");
const abi = @import("abi");
const e2e = @import("e2e/mod.zig");

// Use the prompts module which has the simple getProfile/listProfiles API
const prompts = abi.ai.prompts;

// ============================================================================
// Profile Definition Tests
// ============================================================================

// Test retrieving the assistant profile.
// Verifies default helpful assistant configuration.
test "profile: assistant definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.assistant);

    try std.testing.expectEqualStrings("assistant", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), profile.suggested_temperature, 0.001);
    try std.testing.expect(!profile.include_examples);

    // Should contain key phrases
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "helpful") != null);
}

// Test retrieving the coder profile.
// Verifies programming-focused configuration.
test "profile: coder definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.coder);

    try std.testing.expectEqualStrings("coder", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Coder should have lower temperature for more deterministic output
    try std.testing.expect(profile.suggested_temperature < 0.5);

    // Should include coding-related content
    try std.testing.expect(profile.include_examples);
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "code") != null);
}

// Test retrieving the writer profile.
// Verifies creative writing configuration.
test "profile: writer definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.writer);

    try std.testing.expectEqualStrings("writer", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Writer should have higher temperature for creativity
    try std.testing.expect(profile.suggested_temperature > 0.7);

    // Should contain writing-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "creative") != null or
        std.mem.indexOf(u8, profile.system_prompt, "writing") != null);
}

// Test retrieving the analyst profile.
// Verifies data analysis configuration.
test "profile: analyst definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.analyst);

    try std.testing.expectEqualStrings("analyst", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Analyst should have moderate temperature
    try std.testing.expect(profile.suggested_temperature >= 0.3);
    try std.testing.expect(profile.suggested_temperature <= 0.6);

    // Should contain analysis-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "analysis") != null or
        std.mem.indexOf(u8, profile.system_prompt, "data") != null);
}

// Test retrieving the companion profile.
// Verifies friendly conversational configuration.
test "profile: companion definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.companion);

    try std.testing.expectEqualStrings("companion", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Companion should have higher temperature for natural conversation
    try std.testing.expect(profile.suggested_temperature >= 0.7);

    // Should contain friendly-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "friendly") != null or
        std.mem.indexOf(u8, profile.system_prompt, "conversational") != null);
}

// Test retrieving the docs profile.
// Verifies documentation specialist configuration.
test "profile: docs definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.docs);

    try std.testing.expectEqualStrings("docs", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Docs should have lower temperature for consistency
    try std.testing.expect(profile.suggested_temperature < 0.5);

    // Should include examples for documentation
    try std.testing.expect(profile.include_examples);

    // Should contain documentation-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "documentation") != null or
        std.mem.indexOf(u8, profile.system_prompt, "technical") != null);
}

// Test retrieving the reviewer profile.
// Verifies code review specialist configuration.
test "profile: reviewer definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.reviewer);

    try std.testing.expectEqualStrings("reviewer", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Reviewer should have low temperature for precise feedback
    try std.testing.expect(profile.suggested_temperature <= 0.3);

    // Should contain review-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "review") != null or
        std.mem.indexOf(u8, profile.system_prompt, "code") != null);
}

// Test retrieving the minimal profile.
// Verifies direct response configuration.
test "profile: minimal definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.minimal);

    try std.testing.expectEqualStrings("minimal", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Minimal prompt should be short
    try std.testing.expect(profile.system_prompt.len < 200);

    // Should not include examples (minimal mode)
    try std.testing.expect(!profile.include_examples);
}

// ============================================================================
// Abbey Profile Tests
// ============================================================================

// Test retrieving the Abbey profile.
// Verifies emotionally intelligent polymath configuration.
test "profile: abbey definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.abbey);

    try std.testing.expectEqualStrings("abbey", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Abbey should have moderate temperature balancing creativity and precision
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), profile.suggested_temperature, 0.1);

    // Abbey should include examples
    try std.testing.expect(profile.include_examples);

    // Should contain Abbey-specific content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "Abbey") != null);

    // Should have emotional intelligence content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "emotional") != null or
        std.mem.indexOf(u8, profile.system_prompt, "Emotional") != null);
}

// ============================================================================
// Ralph Profile Tests
// ============================================================================

// Test retrieving the Ralph profile.
// Verifies iterative worker configuration.
test "profile: ralph definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.ralph);

    try std.testing.expectEqualStrings("ralph", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Ralph should have low temperature for precise iteration
    try std.testing.expect(profile.suggested_temperature <= 0.3);

    // Should contain Ralph-specific content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "Ralph") != null);

    // Should have iteration-related content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "ITERATE") != null or
        std.mem.indexOf(u8, profile.system_prompt, "iterate") != null);
}

// ============================================================================
// Aviva Profile Tests
// ============================================================================

// Test retrieving the Aviva profile.
// Verifies direct expert configuration.
test "profile: aviva definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.aviva);

    try std.testing.expectEqualStrings("aviva", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Aviva should have low temperature for precise, factual output
    try std.testing.expect(profile.suggested_temperature <= 0.3);

    // Should contain Aviva-specific content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "Aviva") != null);

    // Should have direct/concise content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "direct") != null or
        std.mem.indexOf(u8, profile.system_prompt, "concise") != null);
}

// ============================================================================
// Abi Profile Tests
// ============================================================================

// Test retrieving the Abi profile.
// Verifies adaptive moderator configuration.
test "profile: abi definition" {
    try e2e.skipIfAiDisabled();

    const profile = prompts.getProfile(.abi);

    try std.testing.expectEqualStrings("abi", profile.name);
    try std.testing.expect(profile.description.len > 0);
    try std.testing.expect(profile.system_prompt.len > 0);

    // Abi should have moderate temperature for adaptive routing
    try std.testing.expect(profile.suggested_temperature >= 0.3);
    try std.testing.expect(profile.suggested_temperature <= 0.7);

    // Should contain Abi-specific content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "Abi") != null);

    // Should have routing/moderation content
    try std.testing.expect(std.mem.indexOf(u8, profile.system_prompt, "route") != null or
        std.mem.indexOf(u8, profile.system_prompt, "routing") != null or
        std.mem.indexOf(u8, profile.system_prompt, "moderator") != null);
}

// ============================================================================
// Profile Listing Tests
// ============================================================================

// Test listing all available profiles.
// Verifies complete profile enumeration.
test "profile: list all" {
    try e2e.skipIfAiDisabled();

    const all_profiles = prompts.listProfiles();

    // Should have at least 8 profiles
    try std.testing.expect(all_profiles.len >= 8);

    // Verify expected profiles are present
    var found_assistant = false;
    var found_coder = false;
    var found_abbey = false;
    var found_ralph = false;

    for (all_profiles) |profile_type| {
        if (profile_type == .assistant) found_assistant = true;
        if (profile_type == .coder) found_coder = true;
        if (profile_type == .abbey) found_abbey = true;
        if (profile_type == .ralph) found_ralph = true;
    }

    try std.testing.expect(found_assistant);
    try std.testing.expect(found_coder);
    try std.testing.expect(found_abbey);
    try std.testing.expect(found_ralph);
}

// ============================================================================
// Temperature Configuration Tests
// ============================================================================

// Test temperature values are within valid range.
// All temperatures should be between 0.0 and 2.0.
test "profile: temperature ranges" {
    try e2e.skipIfAiDisabled();

    const all_types = prompts.listProfiles();

    for (all_types) |profile_type| {
        const profile = prompts.getProfile(profile_type);

        // Temperature should be in valid range
        try std.testing.expect(profile.suggested_temperature >= 0.0);
        try std.testing.expect(profile.suggested_temperature <= 2.0);
    }
}

// Test temperature ordering by profile purpose.
// Creative profiles should have higher temperatures than analytical.
test "profile: temperature ordering" {
    try e2e.skipIfAiDisabled();

    const writer = prompts.getProfile(.writer);
    const coder = prompts.getProfile(.coder);
    const reviewer = prompts.getProfile(.reviewer);

    // Writer (creative) should have higher temp than coder (precise)
    try std.testing.expect(writer.suggested_temperature > coder.suggested_temperature);

    // Coder should have higher or equal temp to reviewer (very precise)
    try std.testing.expect(coder.suggested_temperature >= reviewer.suggested_temperature);
}

// ============================================================================
// System Prompt Quality Tests
// ============================================================================

// Test system prompts have sufficient length.
// Prompts should have meaningful content, not just placeholders.
test "profile: prompt length requirements" {
    try e2e.skipIfAiDisabled();

    const all_types = prompts.listProfiles();

    for (all_types) |profile_type| {
        const profile = prompts.getProfile(profile_type);

        // Minimal profile can be short, others should be substantial
        if (profile_type == .minimal) {
            try std.testing.expect(profile.system_prompt.len >= 10);
        } else {
            try std.testing.expect(profile.system_prompt.len >= 50);
        }
    }
}

// Test system prompts contain no control characters.
// Prompts should be clean text suitable for display.
test "profile: prompt character validity" {
    try e2e.skipIfAiDisabled();

    const all_types = prompts.listProfiles();

    for (all_types) |profile_type| {
        const profile = prompts.getProfile(profile_type);

        for (profile.system_prompt) |c| {
            // Should not contain null bytes or bell characters
            try std.testing.expect(c != 0);
            try std.testing.expect(c != 7); // Bell
            try std.testing.expect(c != 8); // Backspace

            // Allowed: printable ASCII, newline, tab, carriage return
            const is_printable = (c >= 32 and c <= 126);
            const is_whitespace = (c == '\n' or c == '\r' or c == '\t');
            const is_extended = (c >= 128); // UTF-8 continuation
            const is_backslash = (c == '\\');

            try std.testing.expect(is_printable or is_whitespace or is_extended or is_backslash);
        }
    }
}

// ============================================================================
// Profile Type Enumeration Tests
// ============================================================================

// Test ProfileType enum values.
// Verifies enum is properly defined and accessible.
test "profile type: enum values" {
    try e2e.skipIfAiDisabled();

    // All profile types should be valid enum values
    const types = [_]prompts.ProfileType{
        .assistant,
        .coder,
        .writer,
        .analyst,
        .companion,
        .docs,
        .reviewer,
        .minimal,
        .abbey,
        .ralph,
        .aviva,
        .abi,
    };

    // Each type should be distinct
    for (types, 0..) |t1, i| {
        for (types[i + 1 ..]) |t2| {
            try std.testing.expect(t1 != t2);
        }
    }
}

// Test ProfileType can be converted to/from integers.
// Useful for serialization and indexing.
test "profile type: integer conversion" {
    try e2e.skipIfAiDisabled();

    // Convert to int and back
    const original = prompts.ProfileType.coder;
    const as_int = @intFromEnum(original);
    const back = @as(prompts.ProfileType, @enumFromInt(as_int));

    try std.testing.expectEqual(original, back);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// Test that profile retrieval is consistent.
// Same profile type should always return same definition.
test "edge case: retrieval consistency" {
    try e2e.skipIfAiDisabled();

    const profile1 = prompts.getProfile(.abbey);
    const profile2 = prompts.getProfile(.abbey);

    // Should return identical data
    try std.testing.expectEqualStrings(profile1.name, profile2.name);
    try std.testing.expectEqualStrings(profile1.description, profile2.description);
    try std.testing.expectEqualStrings(profile1.system_prompt, profile2.system_prompt);
    try std.testing.expectEqual(profile1.suggested_temperature, profile2.suggested_temperature);
    try std.testing.expectEqual(profile1.include_examples, profile2.include_examples);
}

// Test rapid profile switching.
// Should handle fast switches without issues.
test "edge case: rapid profile switching" {
    try e2e.skipIfAiDisabled();

    const types = prompts.listProfiles();

    // Switch between profiles rapidly
    for (0..100) |_| {
        for (types) |profile_type| {
            const profile = prompts.getProfile(profile_type);
            _ = profile.name;
            _ = profile.system_prompt;
        }
    }
}

// ============================================================================
// Prompt Builder Integration Tests
// ============================================================================

// Test creating a prompt builder with a profile.
// Verifies integration with the builder pattern.
test "prompt builder: with profile" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var builder = prompts.PromptBuilder.init(allocator, .coder);
    defer builder.deinit();

    try builder.addUserMessage("Write a hello world program");
    const prompt = try builder.build(.text);
    defer allocator.free(prompt);

    // Should contain the coder profile's system prompt
    try std.testing.expect(prompt.len > 0);
}

// Test creating a prompt builder with custom profile.
// Verifies custom profile can be provided.
test "prompt builder: custom profile" {
    try e2e.skipIfAiDisabled();

    const allocator = std.testing.allocator;

    const custom = prompts.Profile{
        .name = "custom",
        .description = "Custom test profile",
        .system_prompt = "You are a custom test assistant.",
        .suggested_temperature = 0.5,
        .include_examples = false,
    };

    var builder = prompts.createBuilderWithCustomProfile(allocator, custom);
    defer builder.deinit();

    try builder.addUserMessage("Hello");
    const prompt = try builder.build(.text);
    defer allocator.free(prompt);

    // Should contain the custom system prompt
    try std.testing.expect(std.mem.indexOf(u8, prompt, "custom test assistant") != null);
}
