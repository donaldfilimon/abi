//! Canonical behavior profiles for routed AI interactions.

const std = @import("std");
const legacy_types = @import("../personas/types.zig");
const legacy_registry = @import("../personas/registry.zig");

pub const BehaviorProfile = enum {
    collaborative,
    direct,
    governance,
    iterative,

    pub fn displayName(self: BehaviorProfile) []const u8 {
        return switch (self) {
            .collaborative => "Collaborative",
            .direct => "Direct",
            .governance => "Governance",
            .iterative => "Iterative",
        };
    }
};

pub const LegacyPersonaType = legacy_types.PersonaType;
pub const ProfileRegistry = legacy_registry.PersonaRegistry;

pub fn fromLegacyPersona(persona: LegacyPersonaType) BehaviorProfile {
    return switch (persona) {
        .assistant, .companion, .docs, .abbey => .collaborative,
        .coder, .analyst, .reviewer, .minimal, .aviva => .direct,
        .abi => .governance,
        .writer, .ralph => .iterative,
    };
}

pub fn defaultLegacyPersona(profile: BehaviorProfile) LegacyPersonaType {
    return switch (profile) {
        .collaborative => .abbey,
        .direct => .aviva,
        .governance => .abi,
        .iterative => .ralph,
    };
}

test "behavior profiles normalize branded personas" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyPersona(.abbey));
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyPersona(.aviva));
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyPersona(.abi));
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyPersona(.ralph));
}
