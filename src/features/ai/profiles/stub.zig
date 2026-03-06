//! Behavior-profile stub — disabled at compile time.

pub const BehaviorProfile = enum {
    collaborative,
    direct,
    governance,
    iterative,
};

pub const LegacyPersonaType = enum {
    abbey,
    aviva,
    abi,
    ralph,
};

pub const ProfileRegistry = struct {};

pub fn fromLegacyPersona(_: LegacyPersonaType) ?BehaviorProfile {
    return null;
}

pub fn defaultLegacyPersona(profile: BehaviorProfile) LegacyPersonaType {
    return switch (profile) {
        .collaborative => .abbey,
        .direct => .aviva,
        .governance => .abi,
        .iterative => .ralph,
    };
}
