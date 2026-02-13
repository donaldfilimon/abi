//! ABI Framework - Main Library Interface
//!
//! A modern Zig 0.16 framework for modular AI services, vector search,
//! and high-performance compute. This is the primary entry point for all
//! ABI functionality.
//!
//! ## Features
//!
//! - **AI Module**: Local LLM inference, embeddings, agents, training pipelines
//! - **GPU Acceleration**: Multi-backend support (CUDA, Vulkan, Metal, WebGPU)
//! - **Vector Database**: WDBX with HNSW/IVF-PQ indexing
//! - **Distributed Compute**: Raft consensus, task distribution
//! - **Observability**: Metrics, tracing, and profiling
//!
//! ## Quick Start
//!
//! ```zig
//! const std = @import("std");
//! const abi = @import("abi");
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     // Minimal initialization with defaults
//!     var fw = try abi.initDefault(allocator);
//!     defer fw.deinit();
//!
//!     // Check framework version
//!     std.debug.print("ABI v{s}\n", .{abi.version()});
//! }
//! ```
//!
//! ## Builder Pattern
//!
//! For more control over which features are enabled:
//!
//! ```zig
//! var fw = try abi.Framework.builder(allocator)
//!     .withGpu(.{ .backend = .vulkan })
//!     .withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
//!     .withDatabase(.{ .path = "./data" })
//!     .build();
//! defer fw.deinit();
//!
//! // Access enabled features
//! if (fw.isEnabled(.gpu)) {
//!     const gpu_ctx = try fw.getGpu();
//!     // Use GPU features...
//! }
//! ```
//!
//! ## Feature Modules
//!
//! Access feature modules through the namespace exports:
//! - `abi.ai` - AI capabilities (LLM, embeddings, agents, training)
//! - `abi.gpu` - GPU acceleration and compute
//! - `abi.database` - Vector database operations
//! - `abi.network` - Distributed networking
//! - `abi.observability` - Metrics and tracing
//! - `abi.web` - HTTP utilities

const std = @import("std");
const build_options = @import("build_options");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError("ABI requires Zig 0.16.0 or newer");
    }
}

// ============================================================================
// New Modular Architecture (v2)
// ============================================================================

/// Unified configuration system.
pub const config = @import("core/config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

/// Framework orchestration with builder pattern.
pub const framework = @import("core/framework.zig");
pub const Framework = framework.Framework;
pub const FrameworkBuilder = framework.FrameworkBuilder;

/// Composable error hierarchy for framework operations.
pub const errors = @import("core/errors.zig");
pub const FrameworkError = errors.FrameworkError;

/// Plugin registry for feature management.
pub const registry = @import("core/registry/mod.zig");
pub const Registry = registry.Registry;

/// Runtime infrastructure (always available).
pub const runtime = @import("services/runtime/mod.zig");

/// Platform detection and abstraction.
pub const platform = @import("services/platform/mod.zig");

/// Shared utilities.
pub const shared = @import("services/shared/mod.zig");

/// GPU acceleration.
pub const gpu = if (build_options.enable_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");

/// AI capabilities (modular sub-features).
pub const ai = if (build_options.enable_ai)
    @import("features/ai/mod.zig")
else
    @import("features/ai/stub.zig");

/// AI Core — agents, tools, prompts, memory, discovery.
pub const ai_core = if (build_options.enable_ai)
    @import("features/ai_core/mod.zig")
else
    @import("features/ai_core/stub.zig");

/// AI Inference — LLM, embeddings, vision, streaming, transformer.
pub const inference = if (build_options.enable_llm)
    @import("features/ai_inference/mod.zig")
else
    @import("features/ai_inference/stub.zig");

/// AI Training — training pipelines, federated learning, data loading.
pub const training = if (build_options.enable_training)
    @import("features/ai_training/mod.zig")
else
    @import("features/ai_training/stub.zig");

/// AI Reasoning — Abbey, RAG, eval, templates, explore, orchestration.
pub const reasoning = if (build_options.enable_reasoning)
    @import("features/ai_reasoning/mod.zig")
else
    @import("features/ai_reasoning/stub.zig");

/// Vector database.
pub const database = if (build_options.enable_database)
    @import("features/database/mod.zig")
else
    @import("features/database/stub.zig");

/// Distributed network.
pub const network = if (build_options.enable_network)
    @import("features/network/mod.zig")
else
    @import("features/network/stub.zig");

/// Observability (metrics, tracing, profiling).
pub const observability = if (build_options.enable_profiling)
    @import("features/observability/mod.zig")
else
    @import("features/observability/stub.zig");

/// Web utilities.
pub const web = if (build_options.enable_web)
    @import("features/web/mod.zig")
else
    @import("features/web/stub.zig");

/// Analytics event tracking.
pub const analytics = if (build_options.enable_analytics)
    @import("features/analytics/mod.zig")
else
    @import("features/analytics/stub.zig");

/// Cloud function adapters.
pub const cloud = if (build_options.enable_cloud)
    @import("features/cloud/mod.zig")
else
    @import("features/cloud/stub.zig");

/// Authentication and security.
pub const auth = if (build_options.enable_auth)
    @import("features/auth/mod.zig")
else
    @import("features/auth/stub.zig");

/// Event bus and messaging.
pub const messaging = if (build_options.enable_messaging)
    @import("features/messaging/mod.zig")
else
    @import("features/messaging/stub.zig");

/// In-memory caching.
pub const cache = if (build_options.enable_cache)
    @import("features/cache/mod.zig")
else
    @import("features/cache/stub.zig");

/// Unified file/object storage.
pub const storage = if (build_options.enable_storage)
    @import("features/storage/mod.zig")
else
    @import("features/storage/stub.zig");

/// Full-text search.
pub const search = if (build_options.enable_search)
    @import("features/search/mod.zig")
else
    @import("features/search/stub.zig");

/// High availability (replication, backup, PITR).
pub const ha = @import("services/ha/mod.zig");

/// Task management system.
pub const tasks = @import("services/tasks/mod.zig");

// ============================================================================
// Service Modules (always available)
// ============================================================================

/// External service connectors (OpenAI, Anthropic, Ollama, etc.).
pub const connectors = @import("services/connectors/mod.zig");

// Legacy framework types (still used in tests/CLI — scheduled for removal)
pub const FrameworkOptions = framework.FrameworkOptions;

// Shared utilities (direct imports from shared/)
pub const logging = @import("services/shared/logging.zig");
pub const plugins = @import("services/shared/plugins.zig");
pub const simd = @import("services/shared/simd.zig");
pub const utils = @import("services/shared/utils.zig");
pub const os = @import("services/shared/os.zig");

// Convenience re-exports. For new code, prefer the namespaced versions:
//   abi.simd.vectorAdd, abi.gpu.Gpu, abi.ai.DiscordTools, etc.
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const hasSimdSupport = simd.hasSimdSupport;
pub const Gpu = gpu.Gpu;
pub const GpuBackend = gpu.Backend;
pub const discord = connectors.discord;
pub const DiscordTools = ai.DiscordTools;

// ============================================================================
// Primary API
// ============================================================================

/// Initialize the ABI framework with the given configuration.
///
/// This is a flexible initialization function that accepts multiple configuration types
/// for backward compatibility. For new code, prefer using `initDefault` or `initWithConfig`.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for framework resources. The framework stores this
///   allocator and uses it for all internal allocations. The caller retains ownership
///   and must ensure the allocator outlives the framework.
/// - `config_or_options`: Configuration for the framework. Accepts:
///   - `Config`: The new unified configuration struct
///   - `FrameworkOptions`: Legacy options struct (deprecated)
///   - `void`: Uses default configuration
///   - Struct literal: Coerced to `Config`
///
/// ## Returns
///
/// A fully initialized `Framework` instance, or an error if initialization fails.
///
/// ## Errors
///
/// - `ConfigError.FeatureDisabled`: A feature is enabled in config but disabled at compile time
/// - `error.OutOfMemory`: Memory allocation failed
/// - `error.FeatureInitFailed`: A feature module failed to initialize
///
/// ## Example
///
/// ```zig
/// // With explicit Config
/// var fw = try abi.init(allocator, abi.Config.defaults());
/// defer fw.deinit();
///
/// // With struct literal
/// var fw = try abi.init(allocator, .{
///     .gpu = .{ .backend = .vulkan },
/// });
/// defer fw.deinit();
/// ```
pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework {
    const T = @TypeOf(config_or_options);

    if (T == Config) {
        return Framework.init(allocator, config_or_options);
    } else if (T == FrameworkOptions) {
        return Framework.init(allocator, config_or_options.toConfig());
    } else if (T == void) {
        return Framework.initDefault(allocator);
    } else {
        // Assume it's a struct literal that can be coerced to Config
        const config_val: Config = config_or_options;
        return Framework.init(allocator, config_val);
    }
}

/// Initialize the ABI framework with default configuration.
///
/// This is the simplest way to initialize the framework. It enables all features
/// that are available at compile time with their default settings.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for framework resources
///
/// ## Returns
///
/// A fully initialized `Framework` instance with default configuration.
///
/// ## Example
///
/// ```zig
/// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
/// defer _ = gpa.deinit();
///
/// var fw = try abi.initDefault(gpa.allocator());
/// defer fw.deinit();
///
/// // Framework is ready to use with all default features
/// std.debug.print("Version: {s}\n", .{abi.version()});
/// ```
pub fn initDefault(allocator: std.mem.Allocator) !Framework {
    return Framework.initDefault(allocator);
}

/// Initialize the ABI framework with custom configuration.
///
/// Use this function when you need fine-grained control over which features
/// are enabled and their configuration. Accepts multiple configuration formats
/// for backward compatibility.
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for framework resources
/// - `cfg`: Configuration, one of:
///   - `Config`: New unified configuration struct (recommended)
///   - `FrameworkOptions`: Legacy options (deprecated)
///   - Struct literal: Coerced to `Config`
///   - Empty struct `{}`: Uses default configuration
///
/// ## Returns
///
/// A fully initialized `Framework` instance with the specified configuration.
///
/// ## Example
///
/// ```zig
/// // Enable only specific features
/// var fw = try abi.initWithConfig(allocator, .{
///     .gpu = .{ .backend = .cuda },
///     .ai = .{
///         .llm = .{ .model_path = "./models/llama-7b.gguf" },
///     },
///     .database = .{ .path = "./vector_db" },
/// });
/// defer fw.deinit();
/// ```
pub fn initWithConfig(allocator: std.mem.Allocator, cfg: anytype) !Framework {
    const T = @TypeOf(cfg);

    if (T == Config) {
        return Framework.init(allocator, cfg);
    } else if (T == FrameworkOptions) {
        return Framework.init(allocator, cfg.toConfig());
    } else if (T == @TypeOf(.{})) {
        // Empty struct literal - use defaults
        return Framework.initDefault(allocator);
    } else {
        // Assume it's a struct literal that can be coerced to Config
        const config_val: Config = cfg;
        return Framework.init(allocator, config_val);
    }
}

/// Shutdown and clean up the framework.
///
/// This is a convenience wrapper around `Framework.deinit()`. It releases all
/// resources held by the framework, including feature contexts, the registry,
/// and internal state.
///
/// After calling this function, the framework instance should not be used.
///
/// ## Parameters
///
/// - `fw`: Pointer to the framework instance to shut down
///
/// ## Example
///
/// ```zig
/// var fw = try abi.initDefault(allocator);
/// // ... use the framework ...
/// abi.shutdown(&fw);  // Clean up resources
/// ```
///
/// ## Note
///
/// Using `defer fw.deinit()` directly is equivalent and often preferred:
/// ```zig
/// var fw = try abi.initDefault(allocator);
/// defer fw.deinit();  // Automatically clean up on scope exit
/// ```
pub fn shutdown(fw: *Framework) void {
    fw.deinit();
}

/// Get the ABI framework version string.
///
/// Returns the semantic version of the ABI framework as defined at build time.
/// This can be used for logging, compatibility checks, or displaying version
/// information to users.
///
/// ## Returns
///
/// A compile-time constant string containing the version (e.g., "0.1.0").
///
/// ## Example
///
/// ```zig
/// std.debug.print("Running ABI Framework v{s}\n", .{abi.version()});
/// ```
pub fn version() []const u8 {
    return build_options.package_version;
}

// ============================================================================
// Tests
// ============================================================================

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.4.0", version());
}

test "framework initialization with defaults" {
    // Note: Full framework init requires feature modules to be properly set up
    // This test verifies the API compiles correctly
    const cfg = Config.defaults();
    try std.testing.expect(cfg.gpu != null or !build_options.enable_gpu);
}

test "config builder pattern" {
    var builder = config.Builder.init(std.testing.allocator);
    const cfg = builder
        .withGpuDefaults()
        .withAiDefaults()
        .build();

    if (build_options.enable_gpu) {
        try std.testing.expect(cfg.gpu != null);
    }
    if (build_options.enable_ai) {
        try std.testing.expect(cfg.ai != null);
    }
}
