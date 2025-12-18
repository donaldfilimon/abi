//! Features Module
//!
//! High-level feature modules for the ABI framework

const std = @import("std");

/// Symbolic identifiers for the high level feature families exposed by the
/// framework module. Keeping the enum local avoids circular dependencies with
/// `framework/config.zig` while still enabling compile-time iteration.
pub const FeatureTag = enum(u3) {
    ai,
    gpu,
    database,
    web,
    monitoring,
    connectors,
    simd,
};

const feature_tags = std.enums.values(FeatureTag);
pub const feature_count = feature_tags.len;

inline fn featureIndex(tag: FeatureTag) usize {
    return @intFromEnum(tag);
}

/// Public feature modules grouped for discoverability.
pub const ai = @import("ai/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const database = @import("database/mod.zig");
pub const web = @import("web/mod.zig");
pub const monitoring = @import("monitoring/mod.zig");
pub const connectors = @import("connectors/mod.zig");
pub const simd = @import("../shared/simd.zig");

/// Feature configuration and management
pub const config = struct {
    /// Feature enablement flags
    pub const FeatureFlags = std.StaticBitSet(feature_count);

    /// Creates feature flags from enabled features
    pub fn createFlags(enabled_features: []const FeatureTag) FeatureFlags {
        var flags = FeatureFlags.initEmpty();
        for (enabled_features) |feature| {
            flags.set(featureIndex(feature));
        }
        return flags;
    }

    /// Gets the name of a feature tag
    pub fn getName(tag: FeatureTag) []const u8 {
        return switch (tag) {
            .ai => "ai",
            .gpu => "gpu",
            .database => "database",
            .web => "web",
            .monitoring => "monitoring",
            .connectors => "connectors",
            .simd => "simd",
        };
    }

    /// Gets the description of a feature tag
    pub fn getDescription(tag: FeatureTag) []const u8 {
        return switch (tag) {
            .ai => "Artificial Intelligence and Machine Learning",
            .gpu => "GPU acceleration and compute",
            .database => "Vector database and storage",
            .web => "Web services and HTTP",
            .monitoring => "Observability and metrics",
            .connectors => "External service connectors",
            .simd => "SIMD runtime and vectorized math",
        };
    }
};

/// Invoke the visitor for every feature module re-exported by this file.
pub fn forEachFeature(ctx: anytype, visitor: anytype) void {
    inline for (feature_tags) |feature| {
        const path = switch (feature) {
            .ai => "features/ai/mod.zig",
            .gpu => "features/gpu/mod.zig",
            .database => "features/database/mod.zig",
            .web => "features/web/mod.zig",
            .monitoring => "features/monitoring/mod.zig",
            .connectors => "features/connectors/mod.zig",
            .simd => "shared/simd.zig",
        };
        visitor(ctx, feature, path);
    }
}

/// Feature initialization and lifecycle management
pub const lifecycle = struct {
    /// Initializes all enabled features
    pub fn initFeatures(allocator: std.mem.Allocator, enabled_features: []const FeatureTag) !void {
        for (enabled_features) |feature| {
            switch (feature) {
                .ai => try ai.init(allocator),
                .gpu => try gpu.init(allocator),
                .database => try database.init(allocator),
                .web => try web.init(allocator),
                .monitoring => try monitoring.init(allocator),
                .connectors => try connectors.init(allocator),
                .simd => {},
            }
        }
    }

    /// Deinitializes all enabled features
    pub fn deinitFeatures(enabled_features: []const FeatureTag) void {
        for (enabled_features) |feature| {
            switch (feature) {
                .ai => ai.deinit(),
                .gpu => gpu.deinit(),
                .database => database.deinit(),
                .web => web.deinit(),
                .monitoring => monitoring.deinit(),
                .connectors => connectors.deinit(),
                .simd => {},
            }
        }
    }
};

test "feature registry exposes all modules" {
    const FeatureMask = std.bit_set.IntegerBitSet(feature_count);
    var features_seen = FeatureMask.initEmpty();
    forEachFeature(&features_seen, struct {
        fn visit(mask: *FeatureMask, kind: FeatureTag, _: []const u8) void {
            mask.set(featureIndex(kind));
        }
    }.visit);
    try std.testing.expectEqual(@as(usize, feature_count), features_seen.count());
}

test "feature configuration" {
    const enabled = [_]FeatureTag{ .ai, .database, .web, .simd };
    const flags = config.createFlags(&enabled);

    try std.testing.expect(flags.isSet(featureIndex(.ai)));
    try std.testing.expect(!flags.isSet(featureIndex(.gpu)));
    try std.testing.expect(flags.isSet(featureIndex(.database)));
    try std.testing.expect(flags.isSet(featureIndex(.web)));
    try std.testing.expect(!flags.isSet(featureIndex(.monitoring)));
    try std.testing.expect(!flags.isSet(featureIndex(.connectors)));
    try std.testing.expect(flags.isSet(featureIndex(.simd)));

    try std.testing.expectEqualStrings("ai", config.getName(.ai));
    try std.testing.expectEqualStrings("GPU acceleration and compute", config.getDescription(.gpu));
}
