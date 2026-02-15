//! Builder logic split out of the framework façade.

const std = @import("std");
const config_module = @import("../config/mod.zig");

pub fn init(comptime Builder: type, allocator: std.mem.Allocator) Builder {
    return .{
        .allocator = allocator,
        .config_builder = config_module.Builder.init(allocator),
        .io = null,
    };
}

pub fn withDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withDefaults();
    return self;
}

pub fn withGpu(comptime Builder: type, self: *Builder, gpu_cfg: config_module.GpuConfig) *Builder {
    _ = self.config_builder.withGpu(gpu_cfg);
    return self;
}

pub fn withGpuDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withGpuDefaults();
    return self;
}

pub fn withIo(comptime Builder: type, self: *Builder, io: std.Io) *Builder {
    self.io = io;
    return self;
}

pub fn withAi(comptime Builder: type, self: *Builder, ai_cfg: config_module.AiConfig) *Builder {
    _ = self.config_builder.withAi(ai_cfg);
    return self;
}

pub fn withAiDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withAiDefaults();
    return self;
}

pub fn withLlm(comptime Builder: type, self: *Builder, llm_cfg: config_module.LlmConfig) *Builder {
    _ = self.config_builder.withLlm(llm_cfg);
    return self;
}

pub fn withDatabase(comptime Builder: type, self: *Builder, db_cfg: config_module.DatabaseConfig) *Builder {
    _ = self.config_builder.withDatabase(db_cfg);
    return self;
}

pub fn withDatabaseDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withDatabaseDefaults();
    return self;
}

pub fn withNetwork(comptime Builder: type, self: *Builder, net_cfg: config_module.NetworkConfig) *Builder {
    _ = self.config_builder.withNetwork(net_cfg);
    return self;
}

pub fn withNetworkDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withNetworkDefaults();
    return self;
}

pub fn withObservability(comptime Builder: type, self: *Builder, obs_cfg: config_module.ObservabilityConfig) *Builder {
    _ = self.config_builder.withObservability(obs_cfg);
    return self;
}

pub fn withObservabilityDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withObservabilityDefaults();
    return self;
}

pub fn withWeb(comptime Builder: type, self: *Builder, web_cfg: config_module.WebConfig) *Builder {
    _ = self.config_builder.withWeb(web_cfg);
    return self;
}

pub fn withWebDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withWebDefaults();
    return self;
}

pub fn withAnalytics(comptime Builder: type, self: *Builder, analytics_cfg: config_module.AnalyticsConfig) *Builder {
    _ = self.config_builder.withAnalytics(analytics_cfg);
    return self;
}

pub fn withAnalyticsDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withAnalyticsDefaults();
    return self;
}

pub fn withCloud(comptime Builder: type, self: *Builder, cloud_cfg: config_module.CloudConfig) *Builder {
    _ = self.config_builder.withCloud(cloud_cfg);
    return self;
}

pub fn withCloudDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withCloudDefaults();
    return self;
}

pub fn withAuth(comptime Builder: type, self: *Builder, auth_cfg: config_module.AuthConfig) *Builder {
    _ = self.config_builder.withAuth(auth_cfg);
    return self;
}

pub fn withAuthDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withAuthDefaults();
    return self;
}

pub fn withMessaging(comptime Builder: type, self: *Builder, msg_cfg: config_module.MessagingConfig) *Builder {
    _ = self.config_builder.withMessaging(msg_cfg);
    return self;
}

pub fn withMessagingDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withMessagingDefaults();
    return self;
}

pub fn withCache(comptime Builder: type, self: *Builder, cache_cfg: config_module.CacheConfig) *Builder {
    _ = self.config_builder.withCache(cache_cfg);
    return self;
}

pub fn withCacheDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withCacheDefaults();
    return self;
}

pub fn withStorage(comptime Builder: type, self: *Builder, storage_cfg: config_module.StorageConfig) *Builder {
    _ = self.config_builder.withStorage(storage_cfg);
    return self;
}

pub fn withStorageDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withStorageDefaults();
    return self;
}

pub fn withSearch(comptime Builder: type, self: *Builder, search_cfg: config_module.SearchConfig) *Builder {
    _ = self.config_builder.withSearch(search_cfg);
    return self;
}

pub fn withSearchDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withSearchDefaults();
    return self;
}

pub fn withGateway(comptime Builder: type, self: *Builder, gateway_cfg: config_module.GatewayConfig) *Builder {
    _ = self.config_builder.withGateway(gateway_cfg);
    return self;
}

pub fn withGatewayDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withGatewayDefaults();
    return self;
}

pub fn withPages(comptime Builder: type, self: *Builder, pages_cfg: config_module.PagesConfig) *Builder {
    _ = self.config_builder.withPages(pages_cfg);
    return self;
}

pub fn withPagesDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withPagesDefaults();
    return self;
}

pub fn withBenchmarks(comptime Builder: type, self: *Builder, benchmarks_cfg: config_module.BenchmarksConfig) *Builder {
    _ = self.config_builder.withBenchmarks(benchmarks_cfg);
    return self;
}

pub fn withBenchmarksDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withBenchmarksDefaults();
    return self;
}

pub fn withMobile(comptime Builder: type, self: *Builder, mobile_cfg: config_module.MobileConfig) *Builder {
    _ = self.config_builder.withMobile(mobile_cfg);
    return self;
}

pub fn withMobileDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withMobileDefaults();
    return self;
}

pub fn withPlugins(comptime Builder: type, self: *Builder, plugin_cfg: config_module.PluginConfig) *Builder {
    _ = self.config_builder.withPlugins(plugin_cfg);
    return self;
}

pub fn build(comptime Framework: type, comptime Builder: type, self: *Builder) Framework.Error!Framework {
    const config = self.config_builder.build();
    if (self.io) |io| {
        return Framework.initWithIo(self.allocator, config, io);
    }
    return Framework.init(self.allocator, config);
}
