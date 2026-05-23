//! Framework Configuration Management
const std = @import("std");

pub const LogLevel = enum(u8) {
    debug,
    info,
    warn,
    err,
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
    paths: PathConfig,
    limits: LimitConfig,
    ai: AiConfig,
    server: ServerConfig,
    log_level: LogLevel,
    version: []const u8,

    pub fn init(allocator: std.mem.Allocator) Config {
        return .{
            .allocator = allocator,
            .paths = PathConfig{},
            .limits = LimitConfig{},
            .ai = AiConfig{},
            .server = ServerConfig{},
            .log_level = .info,
            .version = "0.1.0-dev",
        };
    }

    pub fn deinit(self: *Config) void {
        _ = self;
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "Config init" {
    var config = Config.init(std.testing.allocator);
    defer config.deinit();

    try std.testing.expectEqual(.info, config.log_level);
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
