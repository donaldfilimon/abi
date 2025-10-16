const std = @import("std");
const build_options = @import("build_options");

/// Symbolic identifiers for the high level feature families exposed by the
/// framework module. Keeping the enum local avoids circular dependencies with
/// `framework/config.zig` while still enabling compile-time iteration.
pub const FeatureTag = enum { ai, gpu, database, web, monitoring, connectors };

/// Public feature modules grouped for discoverability.
pub const ai = if (!@hasDecl(build_options, "enable_ai") or build_options.enable_ai)
    @import("ai/mod.zig")
else
    struct {};

pub const gpu = if (!@hasDecl(build_options, "enable_gpu") or build_options.enable_gpu)
    @import("gpu/mod.zig")
else
    struct {};

pub const database = if (!@hasDecl(build_options, "enable_database") or build_options.enable_database)
    @import("database/mod.zig")
else
    struct {};

pub const web = if (!@hasDecl(build_options, "enable_web") or build_options.enable_web)
    @import("web/mod.zig")
else
    struct {};

pub const monitoring = if (!@hasDecl(build_options, "enable_monitoring") or build_options.enable_monitoring)
    @import("monitoring/mod.zig")
else
    struct {};

pub const connectors = if (!@hasDecl(build_options, "enable_connectors") or build_options.enable_connectors)
    @import("connectors/mod.zig")
else
    struct {};

/// Invoke the visitor for every feature module re-exported by this file.
pub fn forEachFeature(ctx: anytype, visitor: anytype) void {
    visitor(ctx, .ai, "features/ai/mod.zig");
    visitor(ctx, .gpu, "features/gpu/mod.zig");
    visitor(ctx, .database, "features/database/mod.zig");
    visitor(ctx, .web, "features/web/mod.zig");
    visitor(ctx, .monitoring, "features/monitoring/mod.zig");
    visitor(ctx, .connectors, "features/connectors/mod.zig");
}

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
