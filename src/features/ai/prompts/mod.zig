//! Centralized Prompt Templates and Builder
//!
//! Provides consistent, well-documented prompt templates for all AI/LLM features.
//! All prompts are exportable for inspection and debugging via --show-prompt flag.

const std = @import("std");
pub const types = @import("types");
pub const builder = @import("builder.zig");
pub const ralph = @import("ralph.zig");
pub const profiles = @import("profiles.zig");

// Re-export main types
pub const ProfileType = types.ProfileType;
pub const Profile = profiles.Profile;
pub const PromptBuilder = builder.PromptBuilder;
pub const Message = builder.Message;
pub const Role = builder.Role;
pub const PromptFormat = builder.PromptFormat;

/// Get a profile definition by type.
/// Maps from the canonical ProfileType to the prompts-internal ProfileType
/// and returns the corresponding prompt definition.
pub fn getProfile(profile_type: ProfileType) Profile {
    const prompts_type = std.meta.stringToEnum(
        profiles.ProfileType,
        @tagName(profile_type),
    ) orelse return profiles.getProfile(.assistant);
    return profiles.getProfile(prompts_type);
}

/// List all available profiles
pub fn listProfiles() []const ProfileType {
    return &[_]ProfileType{ .assistant, .coder, .writer, .analyst };
}

/// Create a prompt builder with default assistant profile
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}

/// Create a prompt builder with a specific profile
pub fn createBuilderWithProfile(allocator: std.mem.Allocator, profile_type: ProfileType) PromptBuilder {
    return PromptBuilder.init(allocator, profile_type);
}

/// Create a prompt builder with a custom profile definition
pub fn createBuilderWithCustomProfile(allocator: std.mem.Allocator, profile: Profile) PromptBuilder {
    return PromptBuilder.initCustom(allocator, profile);
}

test "prompt module basics" {
    const allocator = std.testing.allocator;

    var b = createBuilder(allocator);
    defer b.deinit();

    try b.addUserMessage("Hello");
    const prompt = try b.build(.text);
    defer allocator.free(prompt);

    try std.testing.expect(prompt.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
