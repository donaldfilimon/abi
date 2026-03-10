//! ABI Framework v3.0 — Public Module Exports
//!
//! The Abbey-Aviva-Abi multi-persona AI framework with distributed
//! vector database, neural persona routing, and SIMD-accelerated operations.
//!
//! ## Source Layout
//!
//! ```
//! src/
//! ├── root.zig           This file — public re-exports
//! ├── abi.zig            Internal framework wiring (features + services)
//! ├── core/              Framework core (always compiled)
//! │   ├── config/        Configuration loaders
//! │   ├── database/      WDBX vector DB engine, HNSW, SIMD, quantization
//! │   ├── errors.zig     Unified error types
//! │   └── registry/      Feature & service registry
//! ├── features/          Comptime-gated modules (19 modules, each mod+stub)
//! │   ├── ai/            AI subsystem (agents, training, profiles, multi-agent)
//! │   ├── gpu/           GPU compute (Metal, CUDA, Vulkan, OpenGL, WebGPU)
//! │   ├── database/      Database feature gate (delegates to core/database/)
//! │   ├── web/           HTTP server, middleware, routes
//! │   ├── network/       Distributed coordination, Raft, load balancing
//! │   └── ...            cache, auth, cloud, storage, search, etc.
//! ├── services/          Always-on infrastructure
//! │   ├── runtime/       Task engine, scheduling, concurrency primitives
//! │   ├── connectors/    LLM provider clients (OpenAI, Anthropic, Ollama, etc.)
//! │   ├── shared/        Utilities, security, SIMD, logging
//! │   └── ...            MCP server, LSP, tasks, HA
//! ├── inference/         Standalone inference (sampler, KV-cache, scheduler)
//! ├── api_server/        REST API with OpenAI-compatible endpoints
//! └── cel/               CEL expression language parser
//! ```
//!
//! ## Key Modules
//! - `App` / `AppBuilder`: Primary runtime orchestration types
//! - `database`: Unified vector database with HNSW indexing and SIMD
//! - `profiles`: Behavior-based interaction profiles (Canonical v3)
//! - `inference`: Token generation with paged KV-cache and priority scheduling
//! - `api`: REST API server with OpenAI-compatible endpoints

const std = @import("std");
const abi = @import("abi");

// ── Framework Runtime ───────────────────────────────────────────────────
pub const App = abi.App;
pub const AppBuilder = abi.AppBuilder;
pub const services = abi.services;
pub const features = abi.features;

// ── Database & Vectors (Core) ───────────────────────────────────────────
pub const database = @import("core/database/wdbx.zig");
pub const wdbx = database; // Canonical alias for V3
pub const simd = @import("core/database/simd");
pub const hnsw = @import("core/database/hnsw");
pub const quantize = @import("core/database/quantize");
pub const engine = @import("core/database/engine");
pub const distance = @import("core/database/distance");

// ── AI Profiles & Coordination ──────────────────────────────────────────
pub const profiles = @import("features/ai/profiles");
pub const personas = profiles; // Primary compatibility alias
pub const routing = @import("features/ai/routing_logic");

// ── Inference Engine ────────────────────────────────────────────────────
pub const sampler = @import("inference/sampler");
pub const kv_cache = @import("inference/kv_cache");
pub const scheduler = @import("inference/scheduler");
pub const inference_engine = @import("inference/engine");

// ── REST API ────────────────────────────────────────────────────────────
pub const metrics = @import("api_server/metrics");
pub const auth = @import("api_server/auth");
pub const handlers = @import("api_server/handlers");
pub const server = @import("api_server/server");

// ── Convenience aliases ─────────────────────────────────────────────────
pub const Database = engine.Engine;
pub const HnswIndex = hnsw.HNSW;
pub const Distance = distance.Distance;
pub const PersonaType = profiles.LegacyPersonaType;
pub const BehaviorProfile = profiles.BehaviorProfile;
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
