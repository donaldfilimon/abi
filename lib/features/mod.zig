//! Features Module
//!
//! High-level feature modules for the ABI framework

const std = @import("std");

/// Symbolic identifiers for the high level feature families exposed by the
/// framework module. Keeping the enum local avoids circular dependencies with
/// `framework/config.zig` while still enabling compile-time iteration.
pub const FeatureTag = enum { ai, gpu, database, web, monitoring, connectors };

/// Public feature modules grouped for discoverability.
pub const ai = @import("ai/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const database = @import("database/mod.zig");
pub const web = @import("web/mod.zig");
pub const monitoring = @import("monitoring/mod.zig");
pub const connectors = @import("connectors/mod.zig");

/// Feature configuration and management
pub const config = struct {
    /// Feature enablement flags
    pub const FeatureFlags = std.StaticBitSet(6);

    /// Creates feature flags from enabled features
    pub fn createFlags(enabled_features: []const FeatureTag) FeatureFlags {
        var flags = FeatureFlags.initEmpty();
        for (enabled_features) |feature| {
            const idx = switch (feature) {
                .ai => 0,
                .gpu => 1,
                .database => 2,
                .web => 3,
                .monitoring => 4,
                .connectors => 5,
            };
            flags.set(idx);
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
            }
        }
    }
};

test "feature registry exposes all modules" {
    const FeatureMask = std.bit_set.IntegerBitSet(6);
    var features_seen = FeatureMask.initEmpty();
    forEachFeature(&features_seen, struct {
        fn visit(mask: *FeatureMask, kind: FeatureTag, _: []const u8) void {
            const idx = switch (kind) {
                .ai => 0,
                .gpu => 1,
                .database => 2,
                .web => 3,
                .monitoring => 4,
                .connectors => 5,
            };
            mask.set(idx);
        }
    }.visit);
    try std.testing.expectEqual(@as(usize, 6), features_seen.count());
}

test "feature configuration" {
    const enabled = [_]FeatureTag{ .ai, .database, .web };
    const flags = config.createFlags(&enabled);

    try std.testing.expect(flags.isSet(0)); // ai
    try std.testing.expect(!flags.isSet(1)); // gpu
    try std.testing.expect(flags.isSet(2)); // database
    try std.testing.expect(flags.isSet(3)); // web
    try std.testing.expect(!flags.isSet(4)); // monitoring
    try std.testing.expect(!flags.isSet(5)); // connectors

    try std.testing.expectEqualStrings("ai", config.getName(.ai));
    try std.testing.expectEqualStrings("GPU acceleration and compute", config.getDescription(.gpu));
}
