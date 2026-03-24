//! Prompts stub — active when AI feature is disabled.

const std = @import("std");

// Sub-module stubs
pub const profiles = struct {
    pub const Profile = OuterProfile;
    pub const ProfileType = OuterProfileType;
    pub const getProfile = outerGetProfile;
    pub const listProfiles = outerListProfiles;

    const OuterProfile = struct {
        name: []const u8 = "disabled",
        description: []const u8 = "",
        system_prompt: []const u8 = "",
        suggested_temperature: f32 = 0.7,
        include_examples: bool = false,
    };

    const OuterProfileType = enum {
        assistant,
        coder,
        writer,
        analyst,
        companion,
        docs,
        reviewer,
        minimal,
        abbey,
        ralph,
        aviva,
        abi,
        ava,
    };

    fn outerGetProfile(_: OuterProfileType) OuterProfile {
        return .{};
    }
    fn outerListProfiles() []const OuterProfileType {
        return &.{};
    }
};

pub const builder = struct {
    pub const PromptBuilder = OuterPromptBuilder;
    pub const Message = OuterMessage;
    pub const Role = OuterRole;
    pub const PromptFormat = OuterPromptFormat;

    const OuterRole = enum { system, user, assistant, tool };
    const OuterMessage = struct { role: OuterRole = .user, content: []const u8 = "" };
    const OuterPromptFormat = enum { text, json, chatml, llama, raw };

    const OuterPromptBuilder = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: profiles.OuterProfileType) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn initCustom(allocator: std.mem.Allocator, _: profiles.OuterProfile) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *OuterPromptBuilder) void {}
        pub fn addUserMessage(_: *OuterPromptBuilder, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn addMessage(_: *OuterPromptBuilder, _: OuterRole, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn build(_: *OuterPromptBuilder, _: OuterPromptFormat) ![]u8 {
            return error.FeatureDisabled;
        }
        pub fn exportDebug(_: *OuterPromptBuilder) ![]u8 {
            return error.FeatureDisabled;
        }
    };
};

pub const ralph = struct {
    pub const LOOP_INJECTION_TEMPLATE: []const u8 = "";
    pub const STOP_HOOK_TEMPLATE: []const u8 = "";

    pub fn formatLoopInjection(_: std.mem.Allocator, _: usize, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }
};

// Stub for the named "types" module that mod.zig imports
pub const types = struct {
    pub const ProfileType = profiles.OuterProfileType;
};

// Re-export main types
pub const Profile = profiles.OuterProfile;
pub const ProfileType = profiles.OuterProfileType;
pub const PromptBuilder = builder.OuterPromptBuilder;
pub const Message = builder.OuterMessage;
pub const Role = builder.OuterRole;
pub const PromptFormat = builder.OuterPromptFormat;

pub fn getProfile(_: ProfileType) Profile {
    return .{};
}
pub fn listProfiles() []const ProfileType {
    return &.{};
}
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}
pub fn createBuilderWithProfile(allocator: std.mem.Allocator, profile_type: ProfileType) PromptBuilder {
    return PromptBuilder.init(allocator, profile_type);
}
pub fn createBuilderWithCustomProfile(allocator: std.mem.Allocator, _: Profile) PromptBuilder {
    return PromptBuilder.initCustom(allocator, .{});
}

test {
    std.testing.refAllDecls(@This());
}
