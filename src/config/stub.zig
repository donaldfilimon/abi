const std = @import("std");
const types = @import("stubs/types.zig");
const loader_stub = @import("stubs/loader.zig");
const gpu_config = @import("stubs/gpu_config.zig");
const ai_config = @import("stubs/ai_config.zig");
const database_config = @import("stubs/database_config.zig");
const network_config = @import("stubs/network_config.zig");
const observability_config = @import("stubs/observability_config.zig");
const web_config = @import("stubs/web_config.zig");
const plugin_config = @import("stubs/plugin_config.zig");

// Re-exports from submodules
pub const GpuConfig = gpu_config.GpuConfig;
pub const RecoveryConfig = gpu_config.RecoveryConfig;

pub const AiConfig = ai_config.AiConfig;
pub const LlmConfig = ai_config.LlmConfig;
pub const EmbeddingsConfig = ai_config.EmbeddingsConfig;
pub const AgentsConfig = ai_config.AgentsConfig;
pub const TrainingConfig = ai_config.TrainingConfig;
pub const PersonasConfig = ai_config.PersonasConfig;

pub const DatabaseConfig = database_config.DatabaseConfig;

pub const NetworkConfig = network_config.NetworkConfig;
pub const UnifiedMemoryConfig = network_config.UnifiedMemoryConfig;
pub const LinkingConfig = network_config.LinkingConfig;

pub const ObservabilityConfig = observability_config.ObservabilityConfig;

pub const WebConfig = web_config.WebConfig;

pub const PluginConfig = plugin_config.PluginConfig;

pub const ConfigLoader = loader_stub.ConfigLoader;
pub const LoadError = types.LoadError;
pub const ConfigError = types.ConfigError;
pub const Feature = types.Feature;

// ============================================================================
// Unified Config Stub
// ============================================================================

pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},

    pub fn defaults() Config {
        return .{};
    }

    pub fn minimal() Config {
        return .{};
    }

    pub fn isEnabled(self: Config, feature: Feature) bool {
        _ = self;
        _ = feature;
        return false;
    }

    pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature {
        _ = self;
        return allocator.alloc(Feature, 0);
    }
};

// ============================================================================
// Builder Pattern Stub
// ============================================================================

pub const Builder = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .config = Config.minimal(),
        };
    }

    pub fn withDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withGpu(self: *Builder, cfg: GpuConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withGpuDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withAi(self: *Builder, cfg: AiConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withAiDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withLlm(self: *Builder, cfg: LlmConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withDatabase(self: *Builder, cfg: DatabaseConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withDatabaseDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withNetwork(self: *Builder, cfg: NetworkConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withNetworkDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withObservability(self: *Builder, cfg: ObservabilityConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withObservabilityDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withWeb(self: *Builder, cfg: WebConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn withWebDefaults(self: *Builder) *Builder {
        return self;
    }

    pub fn withPlugins(self: *Builder, cfg: PluginConfig) *Builder {
        _ = cfg;
        return self;
    }

    pub fn build(self: *Builder) Config {
        return self.config;
    }
};

pub fn validate(config: Config) ConfigError!void {
    _ = config;
    return error.ConfigDisabled;
}
