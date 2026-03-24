//! Shared comptime-gated feature module imports for framework lifecycle code.
//!
//! This module centralizes the 20 comptime-gated feature imports used by
//! framework.zig, context_init.zig, and shutdown.zig to eliminate duplication.
//! When adding a new feature, update ONLY this file and the three consumers
//! will automatically pick up the change.

const build_options = @import("build_options");

// ── Feature modules (20 standard) ─────────────────────────────────────
pub const gpu_mod = if (build_options.feat_gpu) @import("../../features/gpu/mod.zig") else @import("../../features/gpu/stub.zig");
pub const ai_mod = if (build_options.feat_ai) @import("../../features/ai/mod.zig") else @import("../../features/ai/stub.zig");
pub const database_mod = if (build_options.feat_database) @import("../../features/database/mod.zig") else @import("../../features/database/stub.zig");
pub const network_mod = if (build_options.feat_network) @import("../../features/network/mod.zig") else @import("../../features/network/stub.zig");
pub const observability_mod = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");
pub const web_mod = if (build_options.feat_web) @import("../../features/web/mod.zig") else @import("../../features/web/stub.zig");
pub const cloud_mod = if (build_options.feat_cloud) @import("../../features/cloud/mod.zig") else @import("../../features/cloud/stub.zig");
pub const analytics_mod = if (build_options.feat_analytics) @import("../../features/analytics/mod.zig") else @import("../../features/analytics/stub.zig");
pub const auth_mod = if (build_options.feat_auth) @import("../../features/auth/mod.zig") else @import("../../features/auth/stub.zig");
pub const messaging_mod = if (build_options.feat_messaging) @import("../../features/messaging/mod.zig") else @import("../../features/messaging/stub.zig");
pub const cache_mod = if (build_options.feat_cache) @import("../../features/cache/mod.zig") else @import("../../features/cache/stub.zig");
pub const storage_mod = if (build_options.feat_storage) @import("../../features/storage/mod.zig") else @import("../../features/storage/stub.zig");
pub const search_mod = if (build_options.feat_search) @import("../../features/search/mod.zig") else @import("../../features/search/stub.zig");
pub const gateway_mod = if (build_options.feat_gateway) @import("../../features/gateway/mod.zig") else @import("../../features/gateway/stub.zig");
pub const pages_mod = if (build_options.feat_pages) @import("../../features/observability/pages/mod.zig") else @import("../../features/observability/pages/stub.zig");
pub const benchmarks_mod = if (build_options.feat_benchmarks) @import("../../features/benchmarks/mod.zig") else @import("../../features/benchmarks/stub.zig");
pub const mobile_mod = if (build_options.feat_mobile) @import("../../features/mobile/mod.zig") else @import("../../features/mobile/stub.zig");
pub const compute_mod = if (build_options.feat_compute) @import("../../features/compute/mod.zig") else @import("../../features/compute/stub.zig");
pub const documents_mod = if (build_options.feat_documents) @import("../../features/documents/mod.zig") else @import("../../features/documents/stub.zig");
pub const desktop_mod = if (build_options.feat_desktop) @import("../../features/desktop/mod.zig") else @import("../../features/desktop/stub.zig");
