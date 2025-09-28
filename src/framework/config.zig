const std = @import("std");

/// Enumerates the coarse feature families that can be toggled at runtime.
pub const Feature = enum(u3) {
    ai,
    database,
    web,
    monitoring,
    gpu,
    connectors,
    simd,
};

pub const feature_count = @typeInfo(Feature).Enum.fields.len;
const FeatureMask = std.bit_set.IntegerBitSet(feature_count);

/// Bit-set backed feature selection utility used by the framework runtime.
pub const FeatureToggles = struct {
    mask: FeatureMask = FeatureMask.initEmpty(),

    pub fn enable(self: *FeatureToggles, feature: Feature) void {
        self.mask.set(@intFromEnum(feature));
    }

    pub fn disable(self: *FeatureToggles, feature: Feature) void {
        self.mask.unset(@intFromEnum(feature));
    }

    pub fn set(self: *FeatureToggles, feature: Feature, value: bool) void {
        if (value) {
            self.enable(feature);
        } else {
            self.disable(feature);
        }
    }

    pub fn enableMany(self: *FeatureToggles, features: []const Feature) void {
        for (features) |feature| {
            self.enable(feature);
        }
    }

    pub fn disableMany(self: *FeatureToggles, features: []const Feature) void {
        for (features) |feature| {
            self.disable(feature);
        }
    }

    pub fn isEnabled(self: FeatureToggles, feature: Feature) bool {
        return self.mask.isSet(@intFromEnum(feature));
    }

    pub fn count(self: FeatureToggles) usize {
        return self.mask.count();
    }

    pub fn clear(self: *FeatureToggles) void {
        self.mask = FeatureMask.initEmpty();
    }

    pub fn iterator(self: FeatureToggles) FeatureIterator {
        return .{ .mask = self.mask, .index = 0 };
    }

    pub fn toOwnedSlice(self: FeatureToggles, allocator: std.mem.Allocator) ![]Feature {
        var list = try allocator.alloc(Feature, self.count());
        var iter = self.iterator();
        var idx: usize = 0;
        while (iter.next()) |feature| : (idx += 1) {
            list[idx] = feature;
        }
        return list;
    }
};

/// Iterator used to traverse enabled features.
pub const FeatureIterator = struct {
    mask: FeatureMask,
    index: usize,

    pub fn next(self: *FeatureIterator) ?Feature {
        while (self.index < feature_count) : (self.index += 1) {
            if (self.mask.isSet(self.index)) {
                const feature = @as(Feature, @enumFromInt(self.index));
                self.index += 1;
                return feature;
            }
        }
        return null;
    }
};

/// Human readable name for a feature.
pub fn featureLabel(feature: Feature) []const u8 {
    return switch (feature) {
        .ai => "AI/Agents",
        .database => "Vector Database",
        .web => "Web Services",
        .monitoring => "Monitoring",
        .gpu => "GPU Acceleration",
        .connectors => "External Connectors",
        .simd => "SIMD Runtime",
    };
}

/// Short description describing the role of each feature for summary output.
pub fn featureDescription(feature: Feature) []const u8 {
    return switch (feature) {
        .ai => "Conversation agents, training loops, and inference helpers",
        .database => "High-performance embedding and vector persistence layer",
        .web => "HTTP servers, clients, and gateway orchestration",
        .monitoring => "Instrumentation, telemetry, and health checks",
        .gpu => "GPU kernel dispatch and compute pipelines",
        .connectors => "Third-party integrations and adapters",
        .simd => "Runtime SIMD utilities and fast math operations",
    };
}

/// Configuration supplied when bootstrapping the framework.
pub const FrameworkOptions = struct {
    /// Optional explicit feature set. When provided all boolean toggles are
    /// ignored in favour of this list.
    enabled_features: ?[]const Feature = null,
    /// Features that should be disabled even if present in `enabled_features`
    /// or enabled through the boolean convenience flags.
    disabled_features: []const Feature = &.{},

    // Convenience booleans matching the public quick-start documentation.
    enable_ai: bool = true,
    enable_database: bool = true,
    enable_web: bool = true,
    enable_monitoring: bool = true,
    enable_gpu: bool = false,
    enable_connectors: bool = false,
    enable_simd: bool = true,

    /// Plugin loader configuration.
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,
    auto_register_plugins: bool = false,
    auto_start_plugins: bool = false,
};

/// Compute the feature toggles implied by the provided options.
pub fn deriveFeatureToggles(options: FrameworkOptions) FeatureToggles {
    var toggles = FeatureToggles{};

    if (options.enabled_features) |features| {
        toggles.enableMany(features);
    } else {
        toggles.set(.ai, options.enable_ai);
        toggles.set(.database, options.enable_database);
        toggles.set(.web, options.enable_web);
        toggles.set(.monitoring, options.enable_monitoring);
        toggles.set(.gpu, options.enable_gpu);
        toggles.set(.connectors, options.enable_connectors);
        toggles.set(.simd, options.enable_simd);
    }

    toggles.disableMany(options.disabled_features);
    return toggles;
}

test "feature toggles enable and disable entries" {
    var toggles = FeatureToggles{};
    try std.testing.expectEqual(@as(usize, 0), toggles.count());

    toggles.enableMany(&.{ .ai, .database, .web });
    try std.testing.expect(toggles.isEnabled(.ai));
    try std.testing.expect(toggles.isEnabled(.database));
    try std.testing.expect(toggles.isEnabled(.web));
    try std.testing.expectEqual(@as(usize, 3), toggles.count());

    toggles.disable(.database);
    try std.testing.expect(!toggles.isEnabled(.database));
    try std.testing.expectEqual(@as(usize, 2), toggles.count());
}

test "deriveFeatureToggles respects overrides" {
    const overrides = FrameworkOptions{
        .enabled_features = &.{ .ai, .gpu },
        .disabled_features = &.{.gpu},
    };
    const toggles = deriveFeatureToggles(overrides);
    try std.testing.expect(toggles.isEnabled(.ai));
    try std.testing.expect(!toggles.isEnabled(.gpu));
    try std.testing.expect(!toggles.isEnabled(.database));
}
