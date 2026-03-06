//! Canonical behavior-profile surface layered over the legacy persona system.

const legacy_types = @import("../personas/types.zig");
const legacy_registry = @import("../personas/registry.zig");

pub const LegacyPersonaType = legacy_types.PersonaType;
pub const ProfileRegistry = legacy_registry.PersonaRegistry;

/// Neutral canonical behavior modes for the branded orchestration quartet.
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

/// Map a legacy branded persona onto the neutral behavior-profile contract.
pub fn fromLegacyPersona(persona: LegacyPersonaType) ?BehaviorProfile {
    return switch (persona) {
        .abbey => .collaborative,
        .aviva => .direct,
        .abi => .governance,
        .ralph => .iterative,
        else => null,
    };
}

/// Select the default legacy implementation that backs a behavior profile in wave 1.
pub fn defaultLegacyPersona(profile: BehaviorProfile) LegacyPersonaType {
    return switch (profile) {
        .collaborative => .abbey,
        .direct => .aviva,
        .governance => .abi,
        .iterative => .ralph,
    };
}

test "behavior profiles map to branded personas" {
    try std.testing.expectEqual(BehaviorProfile.collaborative, fromLegacyPersona(.abbey).?);
    try std.testing.expectEqual(BehaviorProfile.direct, fromLegacyPersona(.aviva).?);
    try std.testing.expectEqual(BehaviorProfile.governance, fromLegacyPersona(.abi).?);
    try std.testing.expectEqual(BehaviorProfile.iterative, fromLegacyPersona(.ralph).?);
}

const std = @import("std");
