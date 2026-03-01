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
    /// AI sub-module: non-fatal catch with warning log.
    non_fatal,
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
    .{ .field_name = "gpu", .feature_tag = .gpu, .build_flag = "enable_gpu" },
    .{ .field_name = "ai", .feature_tag = .ai, .build_flag = "enable_ai" },
    .{ .field_name = "database", .feature_tag = .database, .build_flag = "enable_database" },
    .{ .field_name = "network", .feature_tag = .network, .build_flag = "enable_network" },
    .{ .field_name = "observability", .feature_tag = .observability, .build_flag = "enable_profiling" },
    .{ .field_name = "web", .feature_tag = .web, .build_flag = "enable_web" },
    .{ .field_name = "cloud", .feature_tag = .cloud, .build_flag = "enable_cloud" },
    .{ .field_name = "analytics", .feature_tag = .analytics, .build_flag = "enable_analytics" },
    .{ .field_name = "auth", .feature_tag = .auth, .build_flag = "enable_auth" },
    .{ .field_name = "messaging", .feature_tag = .messaging, .build_flag = "enable_messaging" },
    .{ .field_name = "cache", .feature_tag = .cache, .build_flag = "enable_cache" },
    .{ .field_name = "storage", .feature_tag = .storage, .build_flag = "enable_storage" },
    .{ .field_name = "search", .feature_tag = .search, .build_flag = "enable_search" },
    .{ .field_name = "gateway", .feature_tag = .gateway, .build_flag = "enable_gateway" },
    .{ .field_name = "pages", .feature_tag = .pages, .build_flag = "enable_pages" },
    .{ .field_name = "benchmarks", .feature_tag = .benchmarks, .build_flag = "enable_benchmarks" },
    .{ .field_name = "mobile", .feature_tag = .mobile, .build_flag = "enable_mobile" },
    // ── AI facade sub-modules ───────────────────────────────────────────
    .{ .field_name = "ai_core", .feature_tag = .agents, .build_flag = "enable_ai", .init_mode = .non_fatal },
    .{ .field_name = "ai_inference", .feature_tag = .llm, .build_flag = "enable_llm", .init_mode = .non_fatal },
    .{ .field_name = "ai_training", .feature_tag = .training, .build_flag = "enable_training", .init_mode = .non_fatal },
    .{ .field_name = "ai_reasoning", .feature_tag = .reasoning, .build_flag = "enable_reasoning", .init_mode = .non_fatal },
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
        if (ai.llm != null and !build_options.enable_llm) {
            return config_mod.ConfigError.FeatureDisabled;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "descriptor count matches expected" {
    // 17 top-level + 4 AI facades = 21
    try std.testing.expectEqual(@as(usize, 21), descriptor_count);
}

test "top_level_count excludes non-fatal and manual" {
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
