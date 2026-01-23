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
pub const config = @import("config/mod.zig");
pub const Config = config.Config;
pub const Feature = config.Feature;

/// Framework orchestration with builder pattern.
pub const framework = @import("framework.zig");
pub const Framework = framework.Framework;
pub const FrameworkBuilder = framework.FrameworkBuilder;

/// Plugin registry for feature management.
pub const registry = @import("registry/mod.zig");
pub const Registry = registry.Registry;

/// Runtime infrastructure (always available).
pub const runtime = @import("runtime/mod.zig");

/// GPU acceleration.
pub const gpu = if (build_options.enable_gpu)
    @import("gpu/mod.zig")
else
    @import("gpu/stub.zig");

/// AI capabilities (modular sub-features).
pub const ai = if (build_options.enable_ai)
    @import("ai/mod.zig")
else
    @import("ai/stub.zig");

/// Vector database.
pub const database = if (build_options.enable_database)
    @import("database/mod.zig")
else
    @import("database/stub.zig");

/// Distributed network.
pub const network = if (build_options.enable_network)
    @import("network/mod.zig")
else
    @import("network/stub.zig");

/// Observability (metrics, tracing, profiling).
pub const observability = if (build_options.enable_profiling)
    @import("observability/mod.zig")
else
    @import("observability/stub.zig");

/// Convenience alias for system information utilities.
pub const systemInfo = observability.SystemInfo;

/// Web utilities.
pub const web = if (build_options.enable_web)
    @import("web/mod.zig")
else
    @import("web/stub.zig");

/// Cloud function adapters.
pub const cloud = if (build_options.enable_web)
    @import("cloud/mod.zig")
else
    @import("cloud/stub.zig");

/// High availability (replication, backup, PITR).
pub const ha = @import("ha/mod.zig");

/// Task management system.
pub const tasks = @import("tasks/mod.zig");

// ============================================================================
// Legacy Compatibility Layer
// ============================================================================

/// Core utilities (legacy).
pub const core = @import("shared/legacy/mod.zig");

/// Connectors (legacy).
pub const connectors = @import("connectors/mod.zig");

/// Monitoring (legacy - use observability).
pub const monitoring = observability;

// Legacy framework types
pub const FrameworkOptions = framework.FrameworkOptions;
pub const FrameworkConfiguration = framework.FrameworkConfiguration;
pub const RuntimeConfig = framework.RuntimeConfig;
pub const runtimeConfigFromOptions = framework.runtimeConfigFromOptions;

// Shared utilities (direct imports from shared/)
pub const logging = @import("shared/logging.zig");
pub const plugins = @import("shared/plugins.zig");
pub const platform = @import("shared/platform.zig");
pub const simd = @import("shared/simd.zig");
pub const utils = @import("shared/utils_combined.zig");
pub const os = @import("shared/os.zig");

// SIMD functions exported directly
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const vectorL2Norm = simd.vectorL2Norm;
pub const cosineSimilarity = simd.cosineSimilarity;
pub const hasSimdSupport = simd.hasSimdSupport;

// GPU type aliases (legacy - prefer abi.gpu.* namespace instead)
// Core GPU types
pub const Gpu = gpu.Gpu;
pub const GpuConfig = gpu.GpuConfig;
pub const GpuBackend = gpu.Backend;
// Kernel DSL (commonly used for custom kernels)
pub const KernelBuilder = gpu.KernelBuilder;
pub const KernelIR = gpu.KernelIR;
pub const PortableKernelSource = gpu.PortableKernelSource;

// Network type aliases (legacy)
pub const NetworkConfig = network.NetworkConfig;
pub const NetworkState = network.NetworkState;

// AI type aliases (legacy)
pub const TransformerConfig = ai.TransformerConfig;
pub const TransformerModel = ai.TransformerModel;
pub const StreamingGenerator = ai.StreamingGenerator;
pub const StreamToken = ai.StreamToken;
pub const StreamState = ai.StreamState;
pub const GenerationConfig = ai.GenerationConfig;

// Discord connector (legacy)
pub const discord = connectors.discord;
pub const DiscordClient = discord.Client;
pub const DiscordConfig = discord.Config;
pub const DiscordTools = ai.DiscordTools;

/// WDBX compatibility namespace.
pub const wdbx = if (build_options.enable_database) struct {
    const db = @import("database/mod.zig");
    pub const database_mod = db.database;
    pub const helpers = db.db_helpers;
    pub const cli = db.cli;
    pub const http = db.http;

    pub const createDatabase = db.wdbx.createDatabase;
    pub const connectDatabase = db.wdbx.connectDatabase;
    pub const closeDatabase = db.wdbx.closeDatabase;
    pub const insertVector = db.wdbx.insertVector;
    pub const searchVectors = db.wdbx.searchVectors;
    pub const deleteVector = db.wdbx.deleteVector;
    pub const updateVector = db.wdbx.updateVector;
    pub const getVector = db.wdbx.getVector;
    pub const listVectors = db.wdbx.listVectors;
    pub const getStats = db.wdbx.getStats;
    pub const optimize = db.wdbx.optimize;
    pub const backup = db.wdbx.backup;
    pub const restore = db.wdbx.restore;
} else struct {};

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

/// Create a framework with default configuration (legacy compatibility).
pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework {
    return initDefault(allocator);
}

/// Create a framework with custom configuration (legacy compatibility).
pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework {
    return initWithConfig(allocator, config_or_options);
}

// ============================================================================
// Tests
// ============================================================================

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.1.0", version());
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
