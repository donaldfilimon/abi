//! ABI Framework - Main Library Interface
//!
//! High level entrypoints and re-exports for the modernized runtime.

const std = @import("std");
const build_options = @import("build_options");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError("ABI requires Zig 0.16.0 or newer");
    }
}

/// Core utilities and fundamental types
pub const core = @import("core/mod.zig");
/// Feature modules grouped for discoverability
pub const features = @import("features/mod.zig");
/// Individual feature namespaces re-exported at the root for ergonomic imports.
pub const ai = features.ai;
pub const gpu = features.gpu;
pub const database = features.database;
pub const web = features.web;
pub const monitoring = features.monitoring;
pub const connectors = features.connectors;
pub const network = features.network;
pub const compute = @import("compute/mod.zig");
/// Framework orchestration layer that coordinates features and plugins.
pub const framework = @import("framework/mod.zig");
pub const Feature = framework.Feature;
pub const Framework = framework.Framework;
pub const FrameworkOptions = framework.FrameworkOptions;
pub const FrameworkConfiguration = framework.FrameworkConfiguration;
pub const RuntimeConfig = framework.RuntimeConfig;
pub const runtimeConfigFromOptions = framework.runtimeConfigFromOptions;
pub const logging = @import("shared/logging/mod.zig");
pub const plugins = @import("shared/plugins/mod.zig");
pub const observability = @import("shared/observability/mod.zig");
pub const platform = @import("shared/platform/mod.zig");
pub const simd = @import("shared/simd.zig");
pub const config = utils.config;
// SIMD functions exported directly

// SIMD functions
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const vectorL2Norm = simd.vectorL2Norm;
pub const cosineSimilarity = simd.cosineSimilarity;
pub const hasSimdSupport = simd.hasSimdSupport;
pub const GpuBackend = gpu.Backend;
pub const GpuBackendInfo = gpu.BackendInfo;
pub const GpuBackendAvailability = gpu.BackendAvailability;
pub const GpuBackendDetectionLevel = gpu.DetectionLevel;
pub const GpuDeviceInfo = gpu.DeviceInfo;
pub const GpuDeviceCapability = gpu.DeviceCapability;
pub const GpuBuffer = gpu.Buffer;
pub const GpuBufferFlags = gpu.BufferFlags;
pub const GpuMemoryPool = gpu.MemoryPool;
pub const GpuMemoryStats = gpu.MemoryStats;
pub const GpuMemoryError = gpu.MemoryError;
pub const NetworkConfig = network.NetworkConfig;
pub const NetworkState = network.NetworkState;
pub const TransformerConfig = ai.transformer.TransformerConfig;
pub const TransformerModel = ai.transformer.TransformerModel;
pub const StreamingGenerator = ai.streaming.StreamingGenerator;
pub const StreamToken = ai.streaming.StreamToken;
pub const StreamState = ai.streaming.StreamState;
pub const GenerationConfig = ai.streaming.GenerationConfig;
pub const utils = @import("shared/utils/mod.zig");

// Discord connector convenience exports
pub const discord = connectors.discord;
pub const DiscordClient = discord.Client;
pub const DiscordConfig = discord.Config;
pub const DiscordTools = ai.DiscordTools;

/// Compatibility namespace for the WDBX tooling.
/// Compatibility namespace for the WDBX tooling.
pub const wdbx = if (build_options.enable_database) struct {
    pub const database = features.database.database;
    pub const helpers = features.database.db_helpers;
    pub const cli = features.database.cli;
    pub const http = features.database.http;

    pub const createDatabase = features.database.wdbx.createDatabase;
    pub const connectDatabase = features.database.wdbx.connectDatabase;
    pub const closeDatabase = features.database.wdbx.closeDatabase;
    pub const insertVector = features.database.wdbx.insertVector;
    pub const searchVectors = features.database.wdbx.searchVectors;
    pub const deleteVector = features.database.wdbx.deleteVector;
    pub const updateVector = features.database.wdbx.updateVector;
    pub const getVector = features.database.wdbx.getVector;
    pub const listVectors = features.database.wdbx.listVectors;
    pub const getStats = features.database.wdbx.getStats;
    pub const optimize = features.database.wdbx.optimize;
    pub const backup = features.database.wdbx.backup;
    pub const restore = features.database.wdbx.restore;
} else struct {};

/// Initialise the ABI framework and return the orchestration handle.
pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework {
    const runtime_config = try resolveRuntimeConfig(allocator, config_or_options);
    return try framework.Framework.init(allocator, runtime_config);
}

/// Convenience wrapper around `Framework.deinit`.
pub fn shutdown(instance: *Framework) void {
    instance.deinit();
}

/// Get framework version information.
pub fn version() []const u8 {
    return build_options.package_version;
}

/// Create a framework with default configuration.
pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework {
    return try init(allocator, FrameworkOptions{});
}

/// Create a framework with custom configuration.
pub fn createFramework(allocator: std.mem.Allocator, config_or_options: anytype) !Framework {
    const runtime_config = try resolveRuntimeConfig(allocator, config_or_options);
    return try framework.createFramework(allocator, runtime_config);
}

fn resolveRuntimeConfig(allocator: std.mem.Allocator, config_or_options: anytype) !RuntimeConfig {
    return switch (@TypeOf(config_or_options)) {
        RuntimeConfig => config_or_options,
        FrameworkOptions => try runtimeConfigFromOptions(allocator, config_or_options),
        FrameworkConfiguration => try config_or_options.toRuntimeConfig(allocator),
        else => @compileError(
            "Unsupported configuration type for abi.init. Supported types: " ++
                "RuntimeConfig, FrameworkOptions, FrameworkConfiguration",
        ),
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.1.0", version());
}

test "framework initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_instance = try createDefaultFramework(gpa.allocator());
    defer framework_instance.deinit();

    try std.testing.expect(framework_instance.isRunning());
    try std.testing.expect(framework_instance.isFeatureEnabled(.ai));
    try std.testing.expect(framework_instance.isFeatureEnabled(.database));
}
