//! Features Module
//!
//! High-level feature modules for the ABI framework.

const std = @import("std");

/// Symbolic identifiers for the high level feature families exposed by the framework.
pub const FeatureTag = enum {
    ai,
    gpu,
    database,
    web,
    monitoring,
    connectors,
    compute,
    simd,
    network,
};

pub const feature_count = std.enums.values(FeatureTag).len;

/// Public feature modules grouped for discoverability.
pub const ai = @import("ai/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const database = @import("database/mod.zig");
pub const web = @import("web/mod.zig");
pub const monitoring = @import("monitoring/mod.zig");
pub const connectors = @import("connectors/mod.zig");
pub const network = @import("network/mod.zig");
pub const compute = @import("../compute/mod.zig");
pub const simd = @import("../shared/simd.zig");

pub const config = struct {
    pub const tag_count = feature_count;
    pub const FeatureFlags = std.StaticBitSet(tag_count);

    pub fn tagIndex(tag: FeatureTag) usize {
        return @intFromEnum(tag);
    }

    pub fn allTags() []const FeatureTag {
        return std.enums.values(FeatureTag);
    }

    pub fn createFlags(enabled_features: []const FeatureTag) FeatureFlags {
        var flags = FeatureFlags.initEmpty();
        for (enabled_features) |feature| {
            flags.set(tagIndex(feature));
        }
        return flags;
    }

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
            .network => "network",
        };
    }

    pub fn getDescription(tag: FeatureTag) []const u8 {
        return switch (tag) {
            .ai => "Artificial intelligence and agents",
            .gpu => "GPU acceleration and compute",
            .database => "Vector database and storage",
            .web => "Web services and HTTP",
            .monitoring => "Observability and metrics",
            .connectors => "External service connectors",
            .compute => "Compute engine",
            .simd => "SIMD acceleration",
            .network => "Distributed network compute",
        };
    }
};

pub fn forEachFeature(ctx: anytype, visitor: anytype) void {
    visitor(ctx, .ai, "features/ai/mod.zig");
    visitor(ctx, .gpu, "features/gpu/mod.zig");
    visitor(ctx, .database, "features/database/mod.zig");
    visitor(ctx, .web, "features/web/mod.zig");
    visitor(ctx, .monitoring, "features/monitoring/mod.zig");
    visitor(ctx, .connectors, "features/connectors/mod.zig");
    visitor(ctx, .compute, "compute/mod.zig");
    visitor(ctx, .simd, "shared/simd.zig");
    visitor(ctx, .network, "features/network/mod.zig");
}

pub const lifecycle = struct {
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
                .network => try network.init(allocator),
            }
        }
    }

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
                .network => network.deinit(),
            }
        }
    }
};
