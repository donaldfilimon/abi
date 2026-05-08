//! Shared comptime-gated feature module imports for framework lifecycle code.
//!
//! This module centralizes the 20 comptime-gated feature imports used by
//! framework.zig, context_init.zig, and shutdown.zig to eliminate duplication.
//! When adding a new feature, update ONLY this file and the three consumers
//! will automatically pick up the change.

const build_options = @import("build_options");

// ── Feature modules (20 standard) ─────────────────────────────────────
pub const gpu_mod = if (build_options.feat_gpu) @import("../../gpu/mod.zig") else @import("../../gpu/stub.zig");
pub const ai_mod = if (build_options.feat_ai) @import("../../ai/mod.zig") else @import("../../ai/stub.zig");
pub const database_mod = if (build_options.feat_database) @import("../../database/mod.zig") else @import("../../database/stub.zig");
pub const network_mod = if (build_options.feat_network) @import("../../network/mod.zig") else @import("../../network/stub.zig");
pub const observability_mod = if (build_options.feat_observability) @import("../../observability/mod.zig") else @import("../../observability/stub.zig");
pub const web_mod = if (build_options.feat_web) @import("../../web/mod.zig") else @import("../../web/stub.zig");
pub const cloud_mod = if (build_options.feat_cloud) @import("../../cloud/mod.zig") else @import("../../cloud/stub.zig");
pub const analytics_mod = if (build_options.feat_analytics) @import("../../analytics/mod.zig") else @import("../../analytics/stub.zig");
pub const auth_mod = if (build_options.feat_auth) @import("../../auth/mod.zig") else @import("../../auth/stub.zig");
pub const messaging_mod = if (build_options.feat_messaging) @import("../../messaging/mod.zig") else @import("../../messaging/stub.zig");
pub const cache_mod = if (build_options.feat_cache) @import("../../cache/mod.zig") else @import("../../cache/stub.zig");
pub const storage_mod = if (build_options.feat_storage) @import("../../storage/mod.zig") else @import("../../storage/stub.zig");
pub const search_mod = if (build_options.feat_search) @import("../../search/mod.zig") else @import("../../search/stub.zig");
pub const gateway_mod = if (build_options.feat_gateway) @import("../../gateway/mod.zig") else @import("../../gateway/stub.zig");
pub const pages_mod = if (build_options.feat_pages) @import("../../observability/pages/mod.zig") else @import("../../observability/pages/stub.zig");
pub const benchmarks_mod = if (build_options.feat_benchmarks) @import("../../benchmarks/mod.zig") else @import("../../benchmarks/stub.zig");
pub const mobile_mod = if (build_options.feat_mobile) @import("../../mobile/mod.zig") else @import("../../mobile/stub.zig");
pub const compute_mod = if (build_options.feat_compute) @import("../../compute/mod.zig") else @import("../../compute/stub.zig");
pub const documents_mod = if (build_options.feat_documents) @import("../../documents/mod.zig") else @import("../../documents/stub.zig");
pub const desktop_mod = if (build_options.feat_desktop) @import("../../desktop/mod.zig") else @import("../../desktop/stub.zig");

const Feature = @import("../feature_catalog.zig").Feature;

/// Get the context type for a given feature.
pub fn FeatureContext(comptime feature: Feature) type {
    return switch (feature) {
        .gpu => *gpu_mod.Context,
        .ai => *ai_mod.Context,
        .database => *database_mod.Context,
        .network => *network_mod.Context,
        .observability => *observability_mod.Context,
        .web => *web_mod.Context,
        .cloud => *cloud_mod.Context,
        .analytics => *analytics_mod.Context,
        .auth => *auth_mod.Context,
        .messaging => *messaging_mod.Context,
        .cache => *cache_mod.Context,
        .storage => *storage_mod.Context,
        .search => *search_mod.Context,
        .gateway => *gateway_mod.Context,
        .pages => *pages_mod.Context,
        .benchmarks => *benchmarks_mod.Context,
        .mobile => *mobile_mod.Context,
        .compute => *compute_mod.Context,
        .documents => *documents_mod.Context,
        .desktop => *desktop_mod.Context,
        // Protocols/Connectors/etc usually don't have a Context struct in the same way,
        // or they are not yet fully integrated into the Framework struct fields.
        else => @compileError("Feature context type not defined for " ++ @tagName(feature)),
    };
}
