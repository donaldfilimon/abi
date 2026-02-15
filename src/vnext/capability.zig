//! vNext capability model.
//!
//! This is the forward-facing capability enum used by `abi.vnext`.
//! Values intentionally mirror core `Feature` variants for staged compatibility.

pub const Capability = enum {
    gpu,
    ai,
    llm,
    embeddings,
    agents,
    training,
    database,
    network,
    observability,
    web,
    personas,
    cloud,
    analytics,
    auth,
    messaging,
    cache,
    storage,
    search,
    mobile,
    gateway,
    pages,
    benchmarks,
    reasoning,

    pub fn name(self: Capability) []const u8 {
        return @tagName(self);
    }
};

pub fn fromFeature(feature: anytype) Capability {
    return @enumFromInt(@intFromEnum(feature));
}

pub fn toFeature(comptime FeatureType: type, capability: Capability) FeatureType {
    return @enumFromInt(@intFromEnum(capability));
}

const std = @import("std");

test "capability name returns tag string" {
    try std.testing.expectEqualStrings("gpu", Capability.gpu.name());
    try std.testing.expectEqualStrings("ai", Capability.ai.name());
    try std.testing.expectEqualStrings("reasoning", Capability.reasoning.name());
}

test "capability roundtrip preserves identity" {
    inline for (std.meta.fields(Capability)) |field| {
        const cap: Capability = @enumFromInt(field.value);
        const converted = fromFeature(cap);
        try std.testing.expectEqual(cap, converted);
    }
}

test "capability toFeature and fromFeature are inverse" {
    const original = Capability.database;
    const as_feature = toFeature(Capability, original);
    const back = fromFeature(as_feature);
    try std.testing.expectEqual(original, back);
}
