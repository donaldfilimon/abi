//! Agent Roles and Profiles
//!
//! Provides identity, specialization, and behavioral constraints for agents
//! in multi-agent workflows. Each agent can be assigned a role that defines
//! its expertise domain, capabilities, and interaction style.
//!
//! Roles enable:
//! - **Intelligent routing**: Coordinator assigns tasks to the best-suited agent
//! - **Behavioral constraints**: Temperature, token limits, system prompts per role
//! - **Capability matching**: Workflow steps require capabilities that roles provide
//! - **Team composition**: Build balanced teams from complementary roles

const std = @import("std");

const types = @import("roles/types.zig");
const presets_mod = @import("roles/presets.zig");
const registry_mod = @import("roles/registry.zig");

pub const Domain = types.Domain;
pub const Capability = types.Capability;
pub const InteractionStyle = types.InteractionStyle;
pub const BehaviorConstraints = types.BehaviorConstraints;
pub const Profile = types.Profile;

pub const presets = presets_mod.presets;

pub const ProfileRegistry = registry_mod.ProfileRegistry;

test "profile capability matching" {
    const p = presets.code_reviewer;
    try std.testing.expect(p.hasCapability(.code_review));
    try std.testing.expect(p.hasCapability(.critique));
    try std.testing.expect(!p.hasCapability(.code_generation));
}

test "profile match score" {
    const p = presets.implementer;
    const required = [_]Capability{ .code_generation, .refactoring, .test_writing };
    try std.testing.expectEqual(@as(u32, 3), p.matchScore(&required));
    try std.testing.expect(p.satisfiesAll(&required));

    const partial = [_]Capability{ .code_generation, .security_audit };
    try std.testing.expectEqual(@as(u32, 1), p.matchScore(&partial));
    try std.testing.expect(!p.satisfiesAll(&partial));
}

test "profile registry basics" {
    var reg = ProfileRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();
    try std.testing.expectEqual(@as(usize, 6), reg.count());

    const reviewer = reg.get("code-reviewer");
    try std.testing.expect(reviewer != null);
    try std.testing.expectEqualStrings("Code Reviewer", reviewer.?.name);
}

test "profile registry find best match" {
    var reg = ProfileRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const required = [_]Capability{ .code_review, .critique };
    const best = reg.findBestMatch(&required);
    try std.testing.expect(best != null);
    try std.testing.expectEqualStrings("code-reviewer", best.?.id);
}

test "profile registry find by capability" {
    var reg = ProfileRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const results = try reg.findByCapability(std.testing.allocator, .synthesis);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 3), results.len);
}

test "profile registry find by domain" {
    var reg = ProfileRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const coders = try reg.findByDomain(std.testing.allocator, .coding);
    defer std.testing.allocator.free(coders);

    try std.testing.expectEqual(@as(usize, 2), coders.len);
}

test "preset profile constraints" {
    const reviewer = presets.code_reviewer;
    try std.testing.expect(reviewer.constraints.can_veto);
    try std.testing.expect(!reviewer.constraints.can_delegate);
    try std.testing.expect(reviewer.constraints.show_reasoning);

    const architect = presets.architect;
    try std.testing.expect(architect.constraints.can_delegate);
    try std.testing.expect(!architect.constraints.can_veto);
}

test {
    std.testing.refAllDecls(@This());
}
