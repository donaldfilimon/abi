//! Features Module
//!
//! High-level feature modules for the ABI framework

const std = @import("std");

/// Symbolic identifiers for the high level feature families exposed by the
/// framework module. Keeping the enum local avoids circular dependencies with
/// `framework/config.zig` while still enabling compile-time iteration.
pub const FeatureTag = enum { ai, gpu, database, web, monitoring, connectors, compute, simd };
pub const feature_count = std.enums.values(FeatureTag).len;

/// Public feature modules grouped for discoverability.
pub const ai = @import("ai/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const database = @import("database/mod.zig");
pub const web = @import("web/mod.zig");
pub const monitoring = @import("monitoring/mod.zig");
pub const connectors = @import("connectors/mod.zig");
pub const compute = @import("compute/mod.zig");
pub const simd = @import("../shared/simd.zig");

/// Feature configuration and management
pub const config = struct {
    pub const tag_count = feature_count;
    /// Feature enablement flags
    pub const FeatureFlags = std.StaticBitSet(tag_count);

    /// Convert a feature tag into its bitset index
    pub fn tagIndex(tag: FeatureTag) usize {
        return @intFromEnum(tag);
    }

    /// Get all declared feature tags in declaration order
    pub fn allTags() []const FeatureTag {
        return std.enums.values(FeatureTag);
    }

    /// Creates feature flags from enabled features
    pub fn createFlags(enabled_features: []const FeatureTag) FeatureFlags {
        var flags = FeatureFlags.initEmpty();
        for (enabled_features) |feature| {
            flags.set(tagIndex(feature));
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
            .compute => "compute",
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
            .compute => "CPU/GPU compute engine",
            .simd => "SIMD acceleration and vectorized math",
        };
    }
};

/// Invoke the visitor for every feature module re-exported by this file.
pub fn forEachFeature(ctx: anytype, visitor: anytype) void {
    visitor(ctx, .ai, "features/ai/mod.zig");
    visitor(ctx, .gpu, "features/gpu/mod.zig");
    visitor(ctx, .database, "features/database/mod.zig");
    visitor(ctx, .web, "features/web/mod.zig");
    visitor(ctx, .monitoring, "features/monitoring/mod.zig");
    visitor(ctx, .connectors, "features/connectors/mod.zig");
    visitor(ctx, .compute, "features/compute/mod.zig");
    visitor(ctx, .simd, "shared/simd.zig");
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
                .compute => try compute.init(allocator),
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
                .compute => compute.deinit(),
                .simd => {},
            }
        }
    }
};

test "feature registry exposes all modules" {
    const FeatureMask = std.bit_set.IntegerBitSet(config.tag_count);
    var features_seen = FeatureMask.initEmpty();
    forEachFeature(&features_seen, struct {
        fn visit(mask: *FeatureMask, kind: FeatureTag, _: []const u8) void {
            mask.set(config.tagIndex(kind));
        }
    }.visit);
    try std.testing.expectEqual(@as(usize, config.tag_count), features_seen.count());
}

test "feature configuration" {
    const enabled = [_]FeatureTag{ .ai, .database, .web, .simd };
    const flags = config.createFlags(&enabled);

    try std.testing.expect(flags.isSet(config.tagIndex(.ai)));
    try std.testing.expect(!flags.isSet(config.tagIndex(.gpu)));
    try std.testing.expect(flags.isSet(config.tagIndex(.database)));
    try std.testing.expect(flags.isSet(config.tagIndex(.web)));
    try std.testing.expect(!flags.isSet(config.tagIndex(.monitoring)));
    try std.testing.expect(!flags.isSet(config.tagIndex(.connectors)));

    try std.testing.expectEqualStrings("ai", config.getName(.ai));
    try std.testing.expectEqualStrings("GPU acceleration and compute", config.getDescription(.gpu));
}
