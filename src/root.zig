//! Public ABI package root.
//!
//! Canonical API: `abi.<domain>` (e.g. `abi.gpu`, `abi.ai`, `abi.connectors`).

const std = @import("std");
const build_options = @import("build_options");

// ── Core ─────────────────────────────────────────────────────────────────

/// Application configuration: feature flags, platform settings, build profiles.
pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

/// Framework error types and error set definitions.
pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

/// Service and plugin registry for runtime module discovery.
pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

/// Framework lifecycle: initialization, shutdown, state management.
pub const framework = @import("core/framework.zig");

// ── Services (non-feature-gated) ─────────────────────────────────────────

/// Shared foundations: logging, security, time/SIMD primitives.
pub const foundation = @import("foundation/mod.zig");
/// Runtime services: task scheduling, event loops, resource management.
pub const runtime = @import("runtime/mod.zig");
/// Platform abstraction: OS detection, capabilities, environment.
pub const platform = @import("platform/mod.zig");
/// External service connectors: HTTP clients, API adapters.
pub const connectors = if (build_options.feat_connectors) @import("connectors/mod.zig") else @import("connectors/stub.zig");
/// Shared CLI helpers for command dispatch and serve parsing.
pub const cli = @import("cli.zig");
/// C-ABI FFI endpoints for linking as a static library (libabi.a).
pub const ffi = @import("ffi.zig");
/// Task management: async job queues, scheduling, progress tracking.
pub const tasks = if (build_options.feat_tasks) @import("tasks/mod.zig") else @import("tasks/stub.zig");
/// Model Context Protocol (MCP) server and client implementation.
pub const mcp = if (build_options.feat_mcp) @import("protocols/mcp/mod.zig") else @import("protocols/mcp/stub.zig");
/// Language Server Protocol (LSP) implementation.
pub const lsp = if (build_options.feat_lsp) @import("protocols/lsp/mod.zig") else @import("protocols/lsp/stub.zig");
/// Agent Communication Protocol (ACP) for multi-agent messaging.
pub const acp = if (build_options.feat_acp) @import("protocols/acp/mod.zig") else @import("protocols/acp/stub.zig");
/// High availability: leader election, failover, health monitoring.
pub const ha = if (build_options.feat_ha) @import("protocols/ha/mod.zig") else @import("protocols/ha/stub.zig");
/// ML inference: engine, scheduler, sampler, paged KV cache.
pub const inference = if (build_options.feat_inference) @import("inference/mod.zig") else @import("inference/stub.zig");

// ── Features (comptime-gated mod/stub) ───────────────────────────────────

/// GPU compute: Metal/Vulkan/OpenGL backends, kernel dispatch, memory pools.
pub const gpu = if (build_options.feat_gpu) @import("features/gpu/mod.zig") else @import("features/gpu/stub.zig");
/// AI services: LLM providers, embeddings, agents, tool execution, reasoning.
pub const ai = if (build_options.feat_ai) @import("features/ai/mod.zig") else @import("features/ai/stub.zig");
/// Vector database: HNSW index, WDBX storage, persistence, DiskANN.
pub const database = if (build_options.feat_database) @import("features/database/mod.zig") else @import("features/database/stub.zig");
/// Networking: Raft consensus, distributed coordination, transport.
pub const network = if (build_options.feat_network) @import("features/network/mod.zig") else @import("features/network/stub.zig");
/// Observability: metrics collection, tracing, and profiling.
pub const observability = if (build_options.feat_observability) @import("features/observability/mod.zig") else @import("features/observability/stub.zig");
/// Web framework: HTTP routing, middleware, request handling.
pub const web = if (build_options.feat_web) @import("features/web/mod.zig") else @import("features/web/stub.zig");
/// Dashboard pages: observability UI components gated independently from the core observability module.
pub const pages = if (build_options.feat_pages) @import("features/observability/pages/mod.zig") else @import("features/observability/pages/stub.zig");
/// Analytics: event tracking, aggregation, reporting.
pub const analytics = if (build_options.feat_analytics) @import("features/analytics/mod.zig") else @import("features/analytics/stub.zig");
/// Cloud integration: provider adapters, deployment, scaling.
pub const cloud = if (build_options.feat_cloud) @import("features/cloud/mod.zig") else @import("features/cloud/stub.zig");
/// Authentication and authorization: JWT, OAuth, RBAC.
pub const auth = if (build_options.feat_auth) @import("features/auth/mod.zig") else @import("features/auth/stub.zig");
/// Messaging: pub/sub, queues, event streaming.
pub const messaging = if (build_options.feat_messaging) @import("features/messaging/mod.zig") else @import("features/messaging/stub.zig");
/// Caching: in-memory and distributed cache with eviction policies.
pub const cache = if (build_options.feat_cache) @import("features/cache/mod.zig") else @import("features/cache/stub.zig");
/// Storage backends: file, object, and block storage abstractions.
pub const storage = if (build_options.feat_storage) @import("features/storage/mod.zig") else @import("features/storage/stub.zig");
/// Full-text search: BM25 inverted index with persistence.
pub const search = if (build_options.feat_search) @import("features/search/mod.zig") else @import("features/search/stub.zig");
/// Mobile: iOS/Android sensor access, notifications, permissions.
pub const mobile = if (build_options.feat_mobile) @import("features/mobile/mod.zig") else @import("features/mobile/stub.zig");
/// API gateway: routing, rate limiting, request transformation.
pub const gateway = if (build_options.feat_gateway) @import("features/gateway/mod.zig") else @import("features/gateway/stub.zig");
/// Benchmarking: SIMD, memory, concurrency, database, network suites.
pub const benchmarks = if (build_options.feat_benchmarks) @import("features/benchmarks/mod.zig") else @import("features/benchmarks/stub.zig");
/// Distributed compute: map-reduce, task distribution, work stealing.
pub const compute = if (build_options.feat_compute) @import("features/compute/mod.zig") else @import("features/compute/stub.zig");
/// Document processing: parsing, indexing, format conversion.
pub const documents = if (build_options.feat_documents) @import("features/documents/mod.zig") else @import("features/documents/stub.zig");
/// Desktop integration: native windowing, system tray, notifications.
pub const desktop = if (build_options.feat_desktop) @import("features/desktop/mod.zig") else @import("features/desktop/stub.zig");
/// Terminal user interface: raw mode, ANSI rendering, widgets, dashboard.
pub const tui = if (build_options.feat_tui) @import("features/tui/mod.zig") else @import("features/tui/stub.zig");

// ── Convenience aliases ──────────────────────────────────────────────────

pub const meta = struct {
    pub const package_version = build_options.package_version;
    pub const features = @import("core/feature_catalog.zig");

    pub fn version() []const u8 {
        return package_version;
    }
};

const FrameworkApp = framework.Framework;
const FrameworkAppBuilder = framework.FrameworkBuilder;

pub const app = struct {
    pub const App = FrameworkApp;
    pub const AppBuilder = FrameworkAppBuilder;
    pub const Error = FrameworkApp.Error;

    pub fn builder(allocator: std.mem.Allocator) FrameworkAppBuilder {
        return FrameworkApp.builder(allocator);
    }

    pub fn version() []const u8 {
        return meta.version();
    }
};

pub const App = FrameworkApp;
pub const AppBuilder = FrameworkAppBuilder;
pub const Gpu = gpu.Gpu;
pub const GpuBackend = gpu.Backend;

pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return App.builder(allocator);
}

pub fn version() []const u8 {
    return meta.version();
}

pub const feature_catalog = meta.features;

test {
    std.testing.refAllDecls(@This());
}
