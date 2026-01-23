//! ABI Framework - Main Library Interface
//!
//! A modern Zig 0.16 framework for modular AI services, vector search,
//! and high-performance compute.
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
//!     // Minimal initialization
//!     var fw = try abi.init(allocator);
//!     defer fw.deinit();
//!
//!     // Or use the builder pattern
//!     var fw2 = try abi.Framework.builder(allocator)
//!         .withGpu(.{ .backend = .vulkan })
//!         .withAi(.{ .llm = .{} })
//!         .build();
//!     defer fw2.deinit();
//! }
//! ```

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
pub const config = @import("config.zig");
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

/// High availability (replication, backup, PITR).
pub const ha = @import("ha/mod.zig");

/// Task management system.
pub const tasks = @import("tasks.zig");

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

/// Initialize the ABI framework.
/// When called with just an allocator, uses default configuration.
/// When called with allocator and config, uses the provided configuration.
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
/// Convenience function for simple initialization.
pub fn initDefault(allocator: std.mem.Allocator) !Framework {
    return Framework.initDefault(allocator);
}

/// Initialize the ABI framework with custom configuration.
/// Accepts Config, FrameworkOptions (legacy), or struct literal.
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

/// Shutdown the framework (convenience wrapper).
pub fn shutdown(fw: *Framework) void {
    fw.deinit();
}

/// Get framework version.
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
