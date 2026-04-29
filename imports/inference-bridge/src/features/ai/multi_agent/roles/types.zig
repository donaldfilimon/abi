//! Agent role and profile types.

const std = @import("std");

pub const Domain = enum {
    coding,
    reasoning,
    creative,
    analysis,
    research,
    planning,
    testing,
    security,
    documentation,
    general,

    pub fn toString(self: Domain) []const u8 {
        return @tagName(self);
    }
};

pub const Capability = enum {
    code_generation,
    code_review,
    refactoring,
    test_writing,
    data_analysis,
    summarization,
    translation,
    ideation,
    problem_decomposition,
    critique,
    synthesis,
    security_audit,
    doc_writing,
    planning,
    verification,

    pub fn toString(self: Capability) []const u8 {
        return @tagName(self);
    }
};

pub const InteractionStyle = enum {
    concise,
    balanced,
    detailed,
    methodical,
    assertive,
    exploratory,
};

pub const BehaviorConstraints = struct {
    min_temperature: f32 = 0.0,
    max_temperature: f32 = 1.0,
    default_temperature: f32 = 0.7,
    max_output_tokens: u32 = 4096,
    show_reasoning: bool = false,
    can_delegate: bool = false,
    can_veto: bool = false,
    max_retries: u32 = 2,
};

pub const Profile = struct {
    id: []const u8,
    name: []const u8,
    domain: Domain,
    capabilities: []const Capability,
    style: InteractionStyle,
    constraints: BehaviorConstraints,
    system_prompt: []const u8,
    description: []const u8 = "",

    pub fn hasCapability(self: Profile, cap: Capability) bool {
        for (self.capabilities) |c| {
            if (c == cap) return true;
        }
        return false;
    }

    pub fn matchScore(self: Profile, required: []const Capability) u32 {
        var score: u32 = 0;
        for (required) |req| {
            if (self.hasCapability(req)) score += 1;
        }
        return score;
    }

    pub fn satisfiesAll(self: Profile, required: []const Capability) bool {
        return self.matchScore(required) == @as(u32, @intCast(required.len));
    }
};

test {
    std.testing.refAllDecls(@This());
}
