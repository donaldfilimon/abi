const std = @import("std");

pub const LogLevel = enum(u8) {
    debug,
    info,
    warn,
    err,
};

pub const GpuBackend = enum(u8) {
    stdgpu,
    metal,
    cuda,
    vulkan,
    webgpu,
    opengl,
    webgl2,
    fpga,
    tpu,
    simulated,
};

pub const FeatureFlags = struct {
    feat_ai: bool = true,
    feat_gpu: bool = true,
    feat_tui: bool = false,
    feat_accelerator: bool = true,
    feat_shader: bool = true,
    feat_mlir: bool = true,
    feat_mobile: bool = false,
    feat_wdbx: bool = true,
    feat_os_control: bool = true,
};

pub const PathConfig = struct {
    data_dir: []const u8 = "/tmp/abi/data",
    cache_dir: []const u8 = "/tmp/abi/cache",
    log_dir: []const u8 = "/tmp/abi/logs",
    config_dir: []const u8 = "/tmp/abi/config",
    plugin_dir: []const u8 = "/tmp/abi/plugins",
};

pub const LimitConfig = struct {
    max_streams: u32 = 10,
    max_memory_mb: u32 = 1024,
    max_cpu_percent: u8 = 50,
    max_concurrent_tasks: u32 = 100,
    max_vector_dim: u32 = 4096,
    max_block_size_kb: u32 = 64,
    request_timeout_ms: u32 = 30000,
    max_log_entries: usize = 10000,
};

pub const AiConfig = struct {
    default_profile: []const u8 = "abi",
    max_tokens: u32 = 4096,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    enable_constitution: bool = true,
    max_retries: u8 = 3,
};

pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: u32 = 100,
    enable_cors: bool = true,
    tls_enabled: bool = false,
};

pub const Config = struct {
    allocator: std.mem.Allocator,
    features: FeatureFlags,
    paths: PathConfig,
    limits: LimitConfig,
    ai: AiConfig,
    server: ServerConfig,
    gpu_backend: GpuBackend,
    log_level: LogLevel,
    version: []const u8,

    pub fn init(allocator: std.mem.Allocator) Config {
        return .{
            .allocator = allocator,
            .features = FeatureFlags{},
            .paths = PathConfig{},
            .limits = LimitConfig{},
            .ai = AiConfig{},
            .server = ServerConfig{},
            .gpu_backend = .stdgpu,
            .log_level = .info,
            .version = "0.1.0-dev",
        };
    }

    pub fn deinit(self: *Config) void {
        _ = self;
    }

    pub fn isFeatureEnabled(self: *const Config, feature: []const u8) bool {
        return if (std.mem.eql(u8, feature, "ai")) self.features.feat_ai else if (std.mem.eql(u8, feature, "gpu")) self.features.feat_gpu else if (std.mem.eql(u8, feature, "tui")) self.features.feat_tui else if (std.mem.eql(u8, feature, "accelerator")) self.features.feat_accelerator else if (std.mem.eql(u8, feature, "shader")) self.features.feat_shader else if (std.mem.eql(u8, feature, "mlir")) self.features.feat_mlir else if (std.mem.eql(u8, feature, "mobile")) self.features.feat_mobile else if (std.mem.eql(u8, feature, "wdbx")) self.features.feat_wdbx else if (std.mem.eql(u8, feature, "os_control")) self.features.feat_os_control else false;
    }

    pub fn setFeature(self: *Config, feature: []const u8, enabled: bool) void {
        if (std.mem.eql(u8, feature, "ai")) self.features.feat_ai = enabled else if (std.mem.eql(u8, feature, "gpu")) self.features.feat_gpu = enabled else if (std.mem.eql(u8, feature, "tui")) self.features.feat_tui = enabled else if (std.mem.eql(u8, feature, "accelerator")) self.features.feat_accelerator = enabled else if (std.mem.eql(u8, feature, "shader")) self.features.feat_shader = enabled else if (std.mem.eql(u8, feature, "mlir")) self.features.feat_mlir = enabled else if (std.mem.eql(u8, feature, "mobile")) self.features.feat_mobile = enabled else if (std.mem.eql(u8, feature, "wdbx")) self.features.feat_wdbx = enabled else if (std.mem.eql(u8, feature, "os_control")) self.features.feat_os_control = enabled;
    }

    pub fn getEnabledFeatures(self: *const Config) std.ArrayListUnmanaged([]const u8) {
        var list = std.ArrayListUnmanaged([]const u8).empty;
        const allocator = self.allocator;

        inline for (.{
            .{ "ai", @field(self.features, "feat_ai") },
            .{ "gpu", @field(self.features, "feat_gpu") },
            .{ "tui", @field(self.features, "feat_tui") },
            .{ "accelerator", @field(self.features, "feat_accelerator") },
            .{ "shader", @field(self.features, "feat_shader") },
            .{ "mlir", @field(self.features, "feat_mlir") },
            .{ "mobile", @field(self.features, "feat_mobile") },
            .{ "wdbx", @field(self.features, "feat_wdbx") },
            .{ "os_control", @field(self.features, "feat_os_control") },
        }) |entry| {
            if (entry[1]) {
                list.append(allocator, entry[0]) catch |err| {
                    std.log.warn("failed to append enabled feature '{s}': {s}", .{ entry[0], @errorName(err) });
                };
            }
        }

        return list;
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "Config init" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expect(config.features.feat_ai);
    try std.testing.expect(config.features.feat_gpu);
    try std.testing.expect(!config.features.feat_tui);
    try std.testing.expectEqual(.stdgpu, config.gpu_backend);
    try std.testing.expectEqual(.info, config.log_level);
}

test "Config feature toggle" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expect(config.isFeatureEnabled("ai"));
    config.setFeature("ai", false);
    try std.testing.expect(!config.isFeatureEnabled("ai"));
}

test "Config enabled features list" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    config.setFeature("mobile", true);

    var features = config.getEnabledFeatures();
    defer features.deinit(std.testing.allocator);

    try std.testing.expect(features.items.len >= 6);
}

test "Config limits" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expectEqual(@as(u32, 10), config.limits.max_streams);
    try std.testing.expectEqual(@as(u32, 1024), config.limits.max_memory_mb);
    try std.testing.expectEqual(@as(u8, 50), config.limits.max_cpu_percent);
}

test "Config server" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expectEqualStrings("127.0.0.1", config.server.host);
    try std.testing.expectEqual(@as(u16, 8080), config.server.port);
}
