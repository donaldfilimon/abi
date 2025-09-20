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

/// Invoke the visitor for every feature module re-exported by this file.
pub fn forEachFeature(visitor: anytype) void {
    visitor(.ai, "features/ai/mod.zig");
    visitor(.gpu, "features/gpu/mod.zig");
    visitor(.database, "features/database/mod.zig");
    visitor(.web, "features/web/mod.zig");
    visitor(.monitoring, "features/monitoring/mod.zig");
    visitor(.connectors, "features/connectors/mod.zig");
}

test "feature registry exposes all modules" {
    const FeatureMask = std.bit_set.IntegerBitSet(6);
    var features_seen = FeatureMask.initEmpty();
    forEachFeature(struct {
        fn visit(kind: FeatureTag, _: []const u8) void {
            const idx = switch (kind) {
                .ai => 0,
                .gpu => 1,
                .database => 2,
                .web => 3,
                .monitoring => 4,
                .connectors => 5,
            };
            features_seen.set(idx);
        }
    }.visit);
    try std.testing.expectEqual(@as(usize, 6), features_seen.count());
}
