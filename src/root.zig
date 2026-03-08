//! ABI Framework v3.0 — Public Module Exports
//!
//! The Abbey-Aviva-Abi multi-persona AI framework with distributed
//! vector database, neural persona routing, and SIMD-accelerated operations.
//!
//! ## Modules
//! - `App` / `AppBuilder`: Primary runtime orchestration types
//! - `wdbx`: Vector database with HNSW indexing and SIMD distance functions
//! - `personas`: Multi-persona system with Abbey, Aviva, and Abi moderator
//! - `inference`: Token generation with paged KV-cache and priority scheduling
//! - `api`: REST API server with OpenAI-compatible endpoints

const std = @import("std");
const abi = @import("abi");

// ── Framework Runtime ───────────────────────────────────────────────────
pub const App = abi.App;
pub const AppBuilder = abi.AppBuilder;

// ── WDBX Vector Database ────────────────────────────────────────────────
pub const wdbx = @import("wdbx");
pub const simd = @import("wdbx/simd.zig");
pub const hnsw = @import("wdbx/hnsw.zig");
pub const quantize = @import("wdbx/quantize.zig");
pub const engine = @import("wdbx/engine.zig");
pub const distance = @import("wdbx/distance.zig");

// ── Persona System ──────────────────────────────────────────────────────
pub const personas = @import("personas/personas.zig");
pub const routing = @import("personas/routing.zig");
pub const safety = @import("personas/safety.zig");

// ── Inference Engine ────────────────────────────────────────────────────
pub const sampler = @import("inference/sampler.zig");
pub const kv_cache = @import("inference/kv_cache.zig");
pub const scheduler = @import("inference/scheduler.zig");
pub const inference_engine = @import("inference/engine.zig");

// ── REST API ────────────────────────────────────────────────────────────
pub const metrics = @import("api_server/metrics.zig");
pub const auth = @import("api_server/auth.zig");
pub const handlers = @import("api_server/handlers.zig");
pub const server = @import("api_server/server.zig");

// ── Convenience aliases ─────────────────────────────────────────────────
pub const Database = engine.Engine;
pub const HnswIndex = hnsw.HNSW;
pub const Distance = distance.Distance;
pub const PersonaType = personas.PersonaType;
pub const AbiModerator = routing.AbiModerator;
pub const Engine = engine.Engine;
pub const InferenceEngine = inference_engine.Engine;
pub const Server = server.Server;

/// Framework version — delegates to the canonical abi module version
/// which reads from build_options.package_version.
pub fn version() []const u8 {
    return abi.version();
}

// ── Test discovery ──────────────────────────────────────────────────────
test {
    std.testing.refAllDecls(@This());
}
