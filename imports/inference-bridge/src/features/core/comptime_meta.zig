//! Comptime Feature DSL — Single Source of Truth
//!
//! Declarative feature descriptor table that procedurally drives:
//! - Framework struct field declarations
//! - Feature init in `context_init.zig`
//! - Feature deinit in `shutdown.zig`
//! - Config validation in `config/mod.zig`
//!
//! When adding a new feature, update ONLY this table and `feature_imports.zig`.
//!
//! NOTE: Zig's `@import` requires string literals, so modules must be
//! pre-resolved in `feature_imports.zig`.  This table stores metadata only;
//! the module type is looked up via `feature_imports` at the call site.

const std = @import("std");
const build_options = @import("build_options");
const config_mod = @import("config/mod.zig");
const feature_catalog = @import("feature_catalog.zig");

pub const Feature = feature_catalog.Feature;

// ============================================================================
// Feature Descriptor
// ============================================================================

/// How a feature's Context.init should be called during framework startup.
pub const InitMode = enum {
    /// `fw.<field> = try mod.Context.init(allocator, cfg_value);`
    standard,
    /// Skip auto-init (feature is initialized by custom code, e.g. HA).
    manual,
};

/// Declarative per-feature descriptor.  One entry per framework context slot.
pub const FeatureDescriptor = struct {
    /// Field name on both Framework and Config structs (e.g. "gpu").
    field_name: []const u8,
    /// Feature enum variant (e.g. .gpu).
    feature_tag: Feature,
    /// Build options field that gates this feature (e.g. "enable_gpu").
    build_flag: []const u8,
    /// How init should behave.
    init_mode: InitMode = .standard,
};

// ============================================================================
// Canonical Descriptor Table
// ============================================================================

/// The single source of truth for all framework features.
/// Order matters: init runs top-to-bottom, deinit runs bottom-to-top.
pub const descriptors = [_]FeatureDescriptor{
    // ── Top-level features ──────────────────────────────────────────────
    .{ .field_name = "gpu", .feature_tag = .gpu, .build_flag = "feat_gpu" },
    .{ .field_name = "ai", .feature_tag = .ai, .build_flag = "feat_ai" },
    .{ .field_name = "database", .feature_tag = .database, .build_flag = "feat_database" },
    .{ .field_name = "network", .feature_tag = .network, .build_flag = "feat_network" },
    .{ .field_name = "observability", .feature_tag = .observability, .build_flag = "feat_observability" },
    .{ .field_name = "web", .feature_tag = .web, .build_flag = "feat_web" },
    .{ .field_name = "cloud", .feature_tag = .cloud, .build_flag = "feat_cloud" },
    .{ .field_name = "analytics", .feature_tag = .analytics, .build_flag = "feat_analytics" },
    .{ .field_name = "auth", .feature_tag = .auth, .build_flag = "feat_auth" },
    .{ .field_name = "messaging", .feature_tag = .messaging, .build_flag = "feat_messaging" },
    .{ .field_name = "cache", .feature_tag = .cache, .build_flag = "feat_cache" },
    .{ .field_name = "storage", .feature_tag = .storage, .build_flag = "feat_storage" },
    .{ .field_name = "search", .feature_tag = .search, .build_flag = "feat_search" },
    .{ .field_name = "gateway", .feature_tag = .gateway, .build_flag = "feat_gateway" },
    .{ .field_name = "pages", .feature_tag = .pages, .build_flag = "feat_pages" },
    .{ .field_name = "benchmarks", .feature_tag = .benchmarks, .build_flag = "feat_benchmarks" },
    .{ .field_name = "mobile", .feature_tag = .mobile, .build_flag = "feat_mobile" },
    .{ .field_name = "compute", .feature_tag = .compute, .build_flag = "feat_compute", .init_mode = .manual },
    .{ .field_name = "documents", .feature_tag = .documents, .build_flag = "feat_documents", .init_mode = .manual },
    .{ .field_name = "desktop", .feature_tag = .desktop, .build_flag = "feat_desktop", .init_mode = .manual },
};

pub const descriptor_count = descriptors.len;

/// Number of top-level features (those with .standard init_mode).
pub const top_level_count = countTopLevel();

fn countTopLevel() comptime_int {
    var n: comptime_int = 0;
    for (descriptors) |d| {
        if (d.init_mode == .standard) n += 1;
    }
    return n;
}

// ============================================================================
// Descriptor Index Helpers
// ============================================================================

/// Get reversed descriptor indices for shutdown ordering.
pub fn reversedIndices() [descriptor_count]usize {
    var indices: [descriptor_count]usize = undefined;
    for (0..descriptor_count) |i| {
        indices[i] = descriptor_count - 1 - i;
    }
    return indices;
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that no compile-time-disabled feature is enabled in config.
/// Driven by the descriptor table — no manual maintenance needed.
pub fn validateTopLevel(cfg: config_mod.Config) config_mod.ConfigError!void {
    inline for (descriptors) |desc| {
        if (desc.init_mode == .standard) {
            if (@field(cfg, desc.field_name) != null and !@field(build_options, desc.build_flag)) {
                return config_mod.ConfigError.FeatureDisabled;
            }
        }
    }
    // Special case: LLM nested under AI with its own flag.
    if (cfg.ai) |ai| {
        if (ai.llm != null and !build_options.feat_llm) {
            return config_mod.ConfigError.FeatureDisabled;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "descriptor count matches expected" {
    // 20 top-level features
    try std.testing.expectEqual(@as(usize, 20), descriptor_count);
}

test "top_level_count excludes manual" {
    try std.testing.expectEqual(@as(usize, 17), top_level_count);
}

test "reversedIndices is correct" {
    const rev = comptime reversedIndices();
    try std.testing.expectEqual(@as(usize, descriptor_count - 1), rev[0]);
    try std.testing.expectEqual(@as(usize, 0), rev[descriptor_count - 1]);
}

test {
    std.testing.refAllDecls(@This());
}
