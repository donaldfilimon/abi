//! Behavior-profile stub surface when AI features are disabled.

const std = @import("std");
const legacy_personas = @import("../personas/stub.zig");

pub const BehaviorProfile = enum {
    collaborative,
    direct,
    governance,
    iterative,
};

pub const LegacyPersonaType = legacy_personas.PersonaType;
pub const ProfileRegistry = legacy_personas.PersonaRegistry;

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

test {
    std.testing.refAllDecls(@This());
}
