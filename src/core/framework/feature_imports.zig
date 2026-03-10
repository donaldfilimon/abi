//! Shared comptime-gated feature module imports for framework lifecycle code.
//!
//! This module centralizes the 21 comptime-gated feature imports used by
//! framework.zig, context_init.zig, and shutdown.zig to eliminate duplication.
//! When adding a new feature, update ONLY this file and the three consumers
//! will automatically pick up the change.

const build_options = @import("build_options");

// ── Feature modules (17 standard + 4 AI facades) ──────────────────────
pub const gpu_mod = if (build_options.feat_gpu) @import("../../features/gpu") else @import("../../features/gpu/stub");
pub const ai_mod = if (build_options.feat_ai) @import("../../features/ai") else @import("../../features/ai/stub");
pub const database_mod = if (build_options.feat_database) @import("../../features/database") else @import("../../features/database/stub");
pub const network_mod = if (build_options.feat_network) @import("../../features/network") else @import("../../features/network/stub");
pub const observability_mod = if (build_options.feat_profiling) @import("../../features/observability") else @import("../../features/observability/stub");
pub const web_mod = if (build_options.feat_web) @import("../../features/web") else @import("../../features/web/stub");
pub const cloud_mod = if (build_options.feat_cloud) @import("../../features/cloud") else @import("../../features/cloud/stub");
pub const analytics_mod = if (build_options.feat_analytics) @import("../../features/analytics") else @import("../../features/analytics/stub");
pub const auth_mod = if (build_options.feat_auth) @import("../../features/auth") else @import("../../features/auth/stub");
pub const messaging_mod = if (build_options.feat_messaging) @import("../../features/messaging") else @import("../../features/messaging/stub");
pub const cache_mod = if (build_options.feat_cache) @import("../../features/cache") else @import("../../features/cache/stub");
pub const storage_mod = if (build_options.feat_storage) @import("../../features/storage") else @import("../../features/storage/stub");
pub const search_mod = if (build_options.feat_search) @import("../../features/search") else @import("../../features/search/stub");
pub const gateway_mod = if (build_options.feat_gateway) @import("../../features/gateway") else @import("../../features/gateway/stub");
pub const pages_mod = if (build_options.feat_pages) @import("../../features/observability/pages") else @import("../../features/observability/pages/stub");
pub const benchmarks_mod = if (build_options.feat_benchmarks) @import("../../features/benchmarks") else @import("../../features/benchmarks/stub");
pub const mobile_mod = if (build_options.feat_mobile) @import("../../features/mobile") else @import("../../features/mobile/stub");

// ── AI facade modules ─────────────────────────────────────────────────
pub const ai_core_mod = if (build_options.feat_ai) @import("../../features/ai/facades/core") else @import("../../features/ai/facades/core_stub");
pub const ai_inference_mod = if (build_options.feat_llm) @import("../../features/ai/facades/inference") else @import("../../features/ai/facades/inference_stub");
pub const ai_training_mod = if (build_options.feat_training) @import("../../features/ai/facades/training") else @import("../../features/ai/facades/training_stub");
pub const ai_reasoning_mod = if (build_options.feat_reasoning) @import("../../features/ai/facades/reasoning") else @import("../../features/ai/facades/reasoning_stub");
