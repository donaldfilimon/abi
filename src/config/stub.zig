const std = @import("std");
const types = @import("types.zig");
const loader_stub = @import("loader.zig");

// Re-exports from submodules
pub const GpuConfig = @import("gpu_config.zig").GpuConfig;
pub const RecoveryConfig = @import("gpu_config.zig").RecoveryConfig;

pub const AiConfig = @import("ai_config.zig").AiConfig;
pub const LlmConfig = @import("ai_config.zig").LlmConfig;
pub const EmbeddingsConfig = @import("ai_config.zig").EmbeddingsConfig;
pub const AgentsConfig = @import("ai_config.zig").AgentsConfig;
pub const TrainingConfig = @import("ai_config.zig").TrainingConfig;
pub const PersonasConfig = @import("ai_config.zig").PersonasConfig;

pub const DatabaseConfig = @import("database_config.zig").DatabaseConfig;

pub const NetworkConfig = @import("network_config.zig").NetworkConfig;
pub const UnifiedMemoryConfig = @import("network_config.zig").UnifiedMemoryConfig;
pub const LinkingConfig = @import("network_config.zig").LinkingConfig;

pub const ObservabilityConfig = @import("observability_config.zig").ObservabilityConfig;

pub const WebConfig = @import("web_config.zig").WebConfig;

pub const PluginConfig = @import("plugin_config.zig").PluginConfig;

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
