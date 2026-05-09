//! Prompts stub — active when AI feature is disabled.

const std = @import("std");
const shared = @import("types.zig");

// Re-export canonical types
pub const ProfileType = shared.ProfileType;
pub const Profile = shared.Profile;
pub const Role = shared.Role;
pub const Message = shared.Message;
pub const PromptFormat = shared.PromptFormat;

// Sub-module stubs
pub const profiles = struct {
    pub const Profile = shared.Profile;
    pub const ProfileType = shared.ProfileType;
    pub const getProfile = outerGetProfile;
    pub const listProfiles = outerListProfiles;

    fn outerGetProfile(_: shared.ProfileType) shared.Profile {
        return .{};
    }
    fn outerListProfiles() []const shared.ProfileType {
        return &.{};
    }
};

pub const builder = struct {
    pub const PromptBuilder = OuterPromptBuilder;
    pub const Message = shared.Message;
    pub const Role = shared.Role;
    pub const PromptFormat = shared.PromptFormat;

    const OuterPromptBuilder = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: shared.ProfileType) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn initCustom(allocator: std.mem.Allocator, _: shared.Profile) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *OuterPromptBuilder) void {}
        pub fn addUserMessage(_: *OuterPromptBuilder, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn addMessage(_: *OuterPromptBuilder, _: shared.Role, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn build(_: *OuterPromptBuilder, _: shared.PromptFormat) ![]u8 {
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
    pub const ProfileType = shared.ProfileType;
};

pub const PromptBuilder = builder.OuterPromptBuilder;

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
