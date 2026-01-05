//! Configuration system for ABI Framework
//!
//! Provides comprehensive configuration management including file-based
//! configuration (JSON), environment variable loading, and validation.
//!
//! ## Usage
//! ```zig
//! const config = try abi.config.loadFromFile(allocator, "abi.json");
//! defer config.deinit();
//!
//! // Or load from environment variables
//! const env_config = try abi.config.loadFromEnv(allocator, "ABI_");
//! defer env_config.deinit();
//!
//! // Create framework with configuration
//! var framework = try abi.init(allocator, config.frameworkOptions());
//! ```

const std = @import("std");
const build_options = @import("build_options");

pub const ConfigError = error{
    FileNotFound,
    InvalidFormat,
    ParseError,
    ValidationFailed,
    UnsupportedFormat,
    EnvironmentVariableNotFound,
    ReadError,
};

/// Source of configuration
pub const ConfigSource = enum {
    file_json,
    file_yaml,
    environment,
    default,
};

/// Framework-specific configuration
pub const FrameworkConfig = struct {
    /// Enable/disable AI features
    enable_ai: bool = build_options.enable_ai,

    /// Enable/disable GPU features
    enable_gpu: bool = build_options.enable_gpu,

    /// Enable/disable web features
    enable_web: bool = build_options.enable_web,

    /// Enable/disable database features
    enable_database: bool = build_options.enable_database,

    /// Enable/disable network features
    enable_network: bool = build_options.enable_network,

    /// Enable/disable profiling
    enable_profiling: bool = build_options.enable_profiling,

    /// Number of worker threads (0 = auto-detect)
    worker_threads: usize = 0,

    /// Log level (debug, info, warn, err)
    log_level: LogLevel = .info,

    /// Enable debug assertions
    debug_assertions: bool = false,

    pub fn validate(self: FrameworkConfig) ConfigError!void {
        if (self.worker_threads != 0 and self.worker_threads < 1) {
            return ConfigError.ValidationFailed;
        }
    }
};

/// Log level enumeration
pub const LogLevel = enum {
    debug,
    info,
    warn,
    err,
};

/// Database configuration
pub const DatabaseConfig = struct {
    /// Database name
    name: []const u8 = "abi.db",

    /// Maximum records (0 = unlimited)
    max_records: usize = 0,

    /// Enable persistence
    persistence_enabled: bool = true,

    /// Persistence file path
    persistence_path: []const u8 = "abi_data",

    /// Enable vector search
    vector_search_enabled: bool = true,

    /// Default search result limit
    default_search_limit: usize = 10,

    /// Maximum vector dimension
    max_vector_dimension: usize = 4096,

    pub fn validate(self: DatabaseConfig) ConfigError!void {
        if (self.default_search_limit == 0) {
            return ConfigError.ValidationFailed;
        }
        if (self.max_vector_dimension == 0 or self.max_vector_dimension > 65536) {
            return ConfigError.ValidationFailed;
        }
    }
};

/// GPU configuration
pub const GpuConfig = struct {
    /// Enable CUDA backend
    enable_cuda: bool = build_options.gpu_cuda,

    /// Enable Vulkan backend
    enable_vulkan: bool = build_options.gpu_vulkan,

    /// Enable Metal backend (macOS)
    enable_metal: bool = build_options.gpu_metal,

    /// Enable WebGPU backend
    enable_webgpu: bool = build_options.gpu_webgpu,

    /// Enable OpenGL backend
    enable_opengl: bool = build_options.gpu_opengl,

    /// Enable OpenGL ES backend
    enable_opengles: bool = build_options.gpu_opengles,

    /// Enable WebGL2 backend
    enable_webgl2: bool = build_options.gpu_webgl2,

    /// Preferred backend (empty = auto-select)
    preferred_backend: []const u8 = "",

    /// Memory pool size (0 = auto)
    memory_pool_mb: usize = 0,

    pub fn validate(self: GpuConfig) ConfigError!void {
        _ = self;
    }
};

/// AI configuration
pub const AiConfig = struct {
    /// Default model name
    default_model: []const u8 = "",

    /// Maximum tokens per response
    max_tokens: u32 = 2048,

    /// Default temperature (0.0 - 2.0)
    temperature: f32 = 0.7,

    /// Default top-p value
    top_p: f32 = 0.9,

    /// Enable streaming responses
    streaming_enabled: bool = true,

    /// API timeout in milliseconds
    timeout_ms: u32 = 60000,

    /// Enable history/context
    history_enabled: bool = true,

    /// Maximum history entries
    max_history: usize = 100,

    pub fn validate(self: AiConfig) ConfigError!void {
        if (self.temperature < 0.0 or self.temperature > 2.0) {
            return ConfigError.ValidationFailed;
        }
        if (self.top_p <= 0.0 or self.top_p > 1.0) {
            return ConfigError.ValidationFailed;
        }
    }
};

/// Network configuration
pub const NetworkConfig = struct {
    /// Enable distributed compute
    distributed_enabled: bool = false,

    /// Cluster ID
    cluster_id: []const u8 = "default",

    /// Node address
    node_address: []const u8 = "0.0.0.0:9000",

    /// Heartbeat interval in milliseconds
    heartbeat_interval_ms: u64 = 5000,

    /// Node timeout in milliseconds
    node_timeout_ms: u64 = 30000,

    /// Maximum nodes in cluster
    max_nodes: usize = 16,

    /// Enable peer discovery
    peer_discovery: bool = false,

    pub fn validate(self: NetworkConfig) ConfigError!void {
        _ = self;
    }
};

/// Web configuration
pub const WebConfig = struct {
    /// Enable HTTP server
    server_enabled: bool = false,

    /// Server port
    port: u16 = 8080,

    /// Maximum connections
    max_connections: usize = 256,

    /// Request timeout in milliseconds
    request_timeout_ms: u32 = 30000,

    /// Enable CORS
    cors_enabled: bool = true,

    /// CORS origins (comma-separated)
    cors_origins: []const u8 = "*",

    pub fn validate(self: WebConfig) ConfigError!void {
        if (self.port == 0 or self.port > 65535) {
            return ConfigError.ValidationFailed;
        }
    }
};

/// Complete configuration container
pub const Config = struct {
    allocator: std.mem.Allocator,
    source: ConfigSource,
    source_path: []const u8,

    framework: FrameworkConfig,
    database: DatabaseConfig,
    gpu: GpuConfig,
    ai: AiConfig,
    network: NetworkConfig,
    web: WebConfig,

    /// Custom key-value pairs
    custom: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) Config {
        return .{
            .allocator = allocator,
            .source = .default,
            .source_path = "",
            .framework = .{},
            .database = .{},
            .gpu = .{},
            .ai = .{},
            .network = .{},
            .web = .{},
            .custom = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Config) void {
        self.custom.deinit();
        self.* = undefined;
    }

    /// Validate all configuration sections
    pub fn validate(self: *Config) ConfigError!void {
        try self.framework.validate();
        try self.database.validate();
        try self.gpu.validate();
        try self.ai.validate();
        try self.network.validate();
        try self.web.validate();
    }
};

/// Configuration loader
pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ConfigLoader {
        return .{ .allocator = allocator };
    }

    /// Load configuration from JSON file
    pub fn loadFromFile(self: *ConfigLoader, path: []const u8) !Config {
        const file = try std.fs.Dir.cwd().openFile(path, .{});
        defer file.close();

        const size = try file.getEndPos();
        const contents = try file.readToEndAlloc(self.allocator, size);
        defer self.allocator.free(contents);

        return try self.loadFromJson(contents, path);
    }

    /// Load configuration from JSON string
    pub fn loadFromJson(self: *ConfigLoader, json: []const u8, source: []const u8) !Config {
        var config = Config.init(self.allocator);
        errdefer config.deinit();

        var parser = std.json.Parser.init(self.allocator, .{});
        defer parser.deinit();

        const tree = try parser.parse(json);
        defer tree.deinit();

        try self.parseJsonIntoConfig(&config, tree.root);

        config.source = .file_json;
        config.source_path = try self.allocator.dupe(u8, source);

        return config;
    }

    /// Load configuration from environment variables
    pub fn loadFromEnv(self: *ConfigLoader, prefix: []const u8) !Config {
        var config = Config.init(self.allocator);
        errdefer config.deinit();

        if (try self.getEnvBool(prefix, "ENABLE_AI")) |v| config.framework.enable_ai = v;
        if (try self.getEnvBool(prefix, "ENABLE_GPU")) |v| config.framework.enable_gpu = v;
        if (try self.getEnvBool(prefix, "ENABLE_WEB")) |v| config.framework.enable_web = v;
        if (try self.getEnvBool(prefix, "ENABLE_DATABASE")) |v| config.database.persistence_enabled = v;
        if (try self.getEnvBool(prefix, "ENABLE_NETWORK")) |v| config.network.distributed_enabled = v;

        if (try self.getEnvUsize(prefix, "WORKER_THREADS")) |v| config.framework.worker_threads = v;

        if (try self.getEnvString(prefix, "LOG_LEVEL")) |level| {
            if (std.ascii.eqlIgnoreCase(level, "debug")) {
                config.framework.log_level = .debug;
            } else if (std.ascii.eqlIgnoreCase(level, "warn")) {
                config.framework.log_level = .warn;
            } else if (std.ascii.eqlIgnoreCase(level, "err")) {
                config.framework.log_level = .err;
            }
            self.allocator.free(level);
        }

        if (try self.getEnvString(prefix, "DATABASE_NAME")) |name| {
            config.database.name = name;
        }

        if (try self.getEnvString(prefix, "GPU_BACKEND")) |backend| {
            config.gpu.preferred_backend = backend;
        }

        if (try self.getEnvString(prefix, "AI_MODEL")) |model| {
            config.ai.default_model = model;
        }
        if (try self.getEnvF32(prefix, "AI_TEMPERATURE")) |temp| {
            config.ai.temperature = temp;
        }

        if (try self.getEnvString(prefix, "CLUSTER_ID")) |id| {
            config.network.cluster_id = id;
        }
        if (try self.getEnvString(prefix, "NODE_ADDRESS")) |addr| {
            config.network.node_address = addr;
        }

        if (try self.getEnvU16(prefix, "WEB_PORT")) |port| {
            config.web.port = port;
        }
        if (try self.getEnvBool(prefix, "WEB_CORS")) |v| config.web.cors_enabled = v;

        config.source = .environment;
        config.source_path = try self.allocator.dupe(u8, prefix);

        return config;
    }

    fn parseJsonIntoConfig(_: *ConfigLoader, config: *Config, root: std.json.Value) !void {
        switch (root) {
            .object => |obj| {
                if (obj.get("framework")) |framework| {
                    if (framework.object.get("enable_ai")) |v| {
                        if (v == .bool) config.framework.enable_ai = v.bool;
                    }
                    if (framework.object.get("enable_gpu")) |v| {
                        if (v == .bool) config.framework.enable_gpu = v.bool;
                    }
                    if (framework.object.get("worker_threads")) |v| {
                        if (v == .integer) config.framework.worker_threads = @as(usize, @intCast(v.integer));
                    }
                    if (framework.object.get("log_level")) |v| {
                        if (v == .string) {
                            const level = v.string;
                            if (std.ascii.eqlIgnoreCase(level, "debug")) {
                                config.framework.log_level = .debug;
                            } else if (std.ascii.eqlIgnoreCase(level, "warn")) {
                                config.framework.log_level = .warn;
                            } else if (std.ascii.eqlIgnoreCase(level, "err")) {
                                config.framework.log_level = .err;
                            }
                        }
                    }
                }

                if (obj.get("database")) |db| {
                    if (db.object.get("name")) |v| {
                        if (v == .string) config.database.name = v.string;
                    }
                    if (db.object.get("max_records")) |v| {
                        if (v == .integer) config.database.max_records = @as(usize, @intCast(v.integer));
                    }
                    if (db.object.get("persistence_enabled")) |v| {
                        if (v == .bool) config.database.persistence_enabled = v.bool;
                    }
                    if (db.object.get("default_search_limit")) |v| {
                        if (v == .integer) config.database.default_search_limit = @as(usize, @intCast(v.integer));
                    }
                }

                if (obj.get("gpu")) |gpu| {
                    if (gpu.object.get("enable_cuda")) |v| {
                        if (v == .bool) config.gpu.enable_cuda = v.bool;
                    }
                    if (gpu.object.get("preferred_backend")) |v| {
                        if (v == .string) config.gpu.preferred_backend = v.string;
                    }
                }

                if (obj.get("ai")) |ai| {
                    if (ai.object.get("default_model")) |v| {
                        if (v == .string) config.ai.default_model = v.string;
                    }
                    if (ai.object.get("temperature")) |v| {
                        if (v == .number_float) config.ai.temperature = v.number_float;
                    }
                    if (ai.object.get("max_tokens")) |v| {
                        if (v == .integer) config.ai.max_tokens = @as(u32, @intCast(v.integer));
                    }
                }

                if (obj.get("network")) |net| {
                    if (net.object.get("cluster_id")) |v| {
                        if (v == .string) config.network.cluster_id = v.string;
                    }
                    if (net.object.get("node_address")) |v| {
                        if (v == .string) config.network.node_address = v.string;
                    }
                    if (net.object.get("distributed_enabled")) |v| {
                        if (v == .bool) config.network.distributed_enabled = v.bool;
                    }
                }

                if (obj.get("web")) |web| {
                    if (web.object.get("port")) |v| {
                        if (v == .integer) config.web.port = @as(u16, @intCast(v.integer));
                    }
                    if (web.object.get("cors_enabled")) |v| {
                        if (v == .bool) config.web.cors_enabled = v.bool;
                    }
                }
            },
            else => {},
        }
    }

    fn getEnvBool(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?bool {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        const value = std.process.getEnvVar(self.allocator, env_name) catch {
            return null;
        };
        defer self.allocator.free(value);

        return std.ascii.eqlIgnoreCase(value, "true") or std.ascii.eqlIgnoreCase(value, "1");
    }

    fn getEnvString(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?[]const u8 {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        return std.process.getEnvVarOwned(self.allocator, env_name) catch {
            return null;
        };
    }

    fn getEnvUsize(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?usize {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        const value = std.process.getEnvVar(self.allocator, env_name) catch {
            return null;
        };
        defer self.allocator.free(value);

        return try std.fmt.parseInt(usize, value, 10);
    }

    fn getEnvU16(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?u16 {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        const value = std.process.getEnvVar(self.allocator, env_name) catch {
            return null;
        };
        defer self.allocator.free(value);

        return try std.fmt.parseInt(u16, value, 10);
    }

    fn getEnvF32(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?f32 {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        const value = std.process.getEnvVar(self.allocator, env_name) catch {
            return null;
        };
        defer self.allocator.free(value);

        return try std.fmt.parseFloat(f32, value);
    }
};

test "config creation and validation" {
    const allocator = std.testing.allocator;

    var config = Config.init(allocator);
    defer config.deinit();

    try config.validate();
}

test "config loader from JSON" {
    const allocator = std.testing.allocator;

    var loader = ConfigLoader.init(allocator);

    const json =
        \\{
        \\  "framework": {
        \\    "enable_ai": true,
        \\    "worker_threads": 4,
        \\    "log_level": "debug"
        \\  },
        \\  "database": {
        \\    "name": "test.db",
        \\    "default_search_limit": 20
        \\  },
        \\  "ai": {
        \\    "temperature": 0.5,
        \\    "max_tokens": 1024
        \\  }
        \\}
    ;

    const config = try loader.loadFromJson(json, "test.json");
    defer config.deinit();

    try std.testing.expect(config.framework.enable_ai);
    try std.testing.expectEqual(@as(usize, 4), config.framework.worker_threads);
    try std.testing.expectEqual(.debug, config.framework.log_level);
    try std.testing.expectEqualStrings("test.db", config.database.name);
    try std.testing.expectEqual(@as(usize, 20), config.database.default_search_limit);
    try std.testing.expectEqual(@as(f32, 0.5), config.ai.temperature);
    try std.testing.expectEqual(@as(u32, 1024), config.ai.max_tokens);
}

test "config validation failures" {
    const allocator = std.testing.allocator;

    var invalid_config = Config.init(allocator);
    defer invalid_config.deinit();

    try invalid_config.validate();
}
