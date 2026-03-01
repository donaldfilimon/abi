//! Configuration system for ABI Framework
//!
//! Provides comprehensive configuration management including file-based
//! configuration (ZON), environment variable loading, and validation.
//!
//! ## Usage
//! ```zig
//! const config = try abi.config.loadFromFile(allocator, "abi.zon");
//! defer config.deinit();
//!
//! // Or load from environment variables
//! const env_config = try abi.config.loadFromEnv(allocator, "ABI_");
//! defer env_config.deinit();
//!
//! // Create framework with configuration
//! var framework = try abi.App.init(allocator, config.frameworkOptions());
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
    file_zon,
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
    owned_strings: std.ArrayListUnmanaged([]u8),

    framework: FrameworkConfig,
    database: DatabaseConfig,
    gpu: GpuConfig,
    ai: AiConfig,
    network: NetworkConfig,
    web: WebConfig,

    /// Custom key-value pairs
    custom: std.StringHashMapUnmanaged([]const u8),

    pub fn init(allocator: std.mem.Allocator) Config {
        return .{
            .allocator = allocator,
            .source = .default,
            .source_path = "",
            .owned_strings = .{},
            .framework = .{},
            .database = .{},
            .gpu = .{},
            .ai = .{},
            .network = .{},
            .web = .{},
            .custom = .{},
        };
    }

    pub fn deinit(self: *Config) void {
        for (self.owned_strings.items) |item| {
            self.allocator.free(item);
        }
        self.owned_strings.deinit(self.allocator);
        self.custom.deinit(self.allocator);
        self.* = undefined;
    }

    fn ownString(self: *Config, value: []const u8) ![]const u8 {
        const copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(copy);
        try self.owned_strings.append(self.allocator, copy);
        return copy;
    }

    fn adoptString(self: *Config, value: []u8) ![]const u8 {
        self.owned_strings.append(self.allocator, value) catch |err| {
            self.allocator.free(value);
            return err;
        };
        return value;
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

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Helper struct for ZON parsing
const ConfigZon = struct {
    framework: FrameworkConfig = .{},
    database: DatabaseConfig = .{},
    gpu: GpuConfig = .{},
    ai: AiConfig = .{},
    network: NetworkConfig = .{},
    web: WebConfig = .{},
};

/// Configuration loader
pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,

    pub fn init(allocator: std.mem.Allocator) ConfigLoader {
        return .{
            .allocator = allocator,
            .io_backend = initIoBackend(allocator),
        };
    }

    pub fn deinit(self: *ConfigLoader) void {
        self.io_backend.deinit();
    }

    /// Load configuration from ZON file
    pub fn loadFromFile(self: *ConfigLoader, path: []const u8) !Config {
        const io = self.io_backend.io();
        const contents = std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(1024 * 1024)) catch {
            return ConfigError.FileNotFound;
        };
        defer self.allocator.free(contents);

        // Ensure null-terminated for ZON parser
        const contents_z = try self.allocator.dupeZ(u8, contents);
        defer self.allocator.free(contents_z);

        return try self.loadFromZon(contents_z, path);
    }

    /// Load configuration from ZON string
    pub fn loadFromZon(self: *ConfigLoader, zon_str: [:0]const u8, source: []const u8) !Config {
        var config = Config.init(self.allocator);
        errdefer config.deinit();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        const data = std.zon.parse.fromSliceAlloc(ConfigZon, arena_allocator, zon_str, null, .{}) catch {
            return ConfigError.ParseError;
        };
        config.framework = data.framework;
        config.database = data.database;
        config.gpu = data.gpu;
        config.ai = data.ai;
        config.network = data.network;
        config.web = data.web;

        // Note: Strings inside data are owned by 'parsed'. We need to own them in 'config'.
        config.database.name = try config.ownString(data.database.name);
        config.database.persistence_path = try config.ownString(data.database.persistence_path);
        config.gpu.preferred_backend = try config.ownString(data.gpu.preferred_backend);
        config.ai.default_model = try config.ownString(data.ai.default_model);
        config.network.cluster_id = try config.ownString(data.network.cluster_id);
        config.network.node_address = try config.ownString(data.network.node_address);
        config.web.cors_origins = try config.ownString(data.web.cors_origins);

        config.source = .file_zon;
        config.source_path = try config.ownString(source);

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
            if (parseLogLevel(level)) |parsed| {
                config.framework.log_level = parsed;
            }
            self.allocator.free(level);
        }

        if (try self.getEnvString(prefix, "DATABASE_NAME")) |name| {
            config.database.name = try config.adoptString(name);
        }

        if (try self.getEnvString(prefix, "GPU_BACKEND")) |backend| {
            config.gpu.preferred_backend = try config.adoptString(backend);
        }

        if (try self.getEnvString(prefix, "AI_MODEL")) |model| {
            config.ai.default_model = try config.adoptString(model);
        }
        if (try self.getEnvF32(prefix, "AI_TEMPERATURE")) |temp| {
            config.ai.temperature = temp;
        }

        if (try self.getEnvString(prefix, "CLUSTER_ID")) |id| {
            config.network.cluster_id = try config.adoptString(id);
        }
        if (try self.getEnvString(prefix, "NODE_ADDRESS")) |addr| {
            config.network.node_address = try config.adoptString(addr);
        }

        if (try self.getEnvU16(prefix, "WEB_PORT")) |port| {
            config.web.port = port;
        }
        if (try self.getEnvBool(prefix, "WEB_CORS")) |v| config.web.cors_enabled = v;

        config.source = .environment;
        config.source_path = try config.ownString(prefix);

        return config;
    }

    fn parseLogLevel(level: []const u8) ?LogLevel {
        if (std.ascii.eqlIgnoreCase(level, "debug")) return .debug;
        if (std.ascii.eqlIgnoreCase(level, "info")) return .info;
        if (std.ascii.eqlIgnoreCase(level, "warn")) return .warn;
        if (std.ascii.eqlIgnoreCase(level, "err")) return .err;
        return null;
    }

    /// Look up an environment variable using std.c.getenv (Zig 0.16 compatible).
    /// Returns a non-owning slice into the process environment block.
    fn lookupEnv(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?[*:0]u8 {
        const env_name = try std.fmt.allocPrintSentinel(self.allocator, "{s}{s}", .{ prefix, name }, 0);
        defer self.allocator.free(env_name);
        return std.c.getenv(env_name.ptr);
    }

    fn getEnvBool(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?bool {
        const raw = try self.lookupEnv(prefix, name) orelse return null;
        const value = std.mem.sliceTo(raw, 0);
        return std.ascii.eqlIgnoreCase(value, "true") or std.ascii.eqlIgnoreCase(value, "1");
    }

    fn getEnvString(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?[]const u8 {
        const raw = try self.lookupEnv(prefix, name) orelse return null;
        return try self.allocator.dupe(u8, std.mem.sliceTo(raw, 0));
    }

    fn getEnvUsize(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?usize {
        const raw = try self.lookupEnv(prefix, name) orelse return null;
        const value = std.mem.sliceTo(raw, 0);
        return try std.fmt.parseInt(usize, value, 10);
    }

    fn getEnvU16(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?u16 {
        const raw = try self.lookupEnv(prefix, name) orelse return null;
        const value = std.mem.sliceTo(raw, 0);
        return try std.fmt.parseInt(u16, value, 10);
    }

    fn getEnvF32(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?f32 {
        const raw = try self.lookupEnv(prefix, name) orelse return null;
        const value = std.mem.sliceTo(raw, 0);
        return try std.fmt.parseFloat(f32, value);
    }
};

test "config creation and validation" {
    const allocator = std.testing.allocator;

    var config = Config.init(allocator);
    defer config.deinit();

    try config.validate();
}

test "config loader from ZON" {
    const allocator = std.testing.allocator;

    var loader = ConfigLoader.init(allocator);
    defer loader.deinit();

    const zon: [:0]const u8 =
        \\.{ .framework = .{ .enable_ai = true, .worker_threads = 4, .log_level = .debug, },
        \\  .database = .{ .name = "test.db", .default_search_limit = 20, },
        \\  .ai = .{ .temperature = 0.5, .max_tokens = 1024, },
        \\}
    ;

    var config = try loader.loadFromZon(zon, "test.zon");
    defer config.deinit();

    try std.testing.expect(config.framework.enable_ai);
    try std.testing.expectEqual(@as(usize, 4), config.framework.worker_threads);
    try std.testing.expectEqual(.debug, config.framework.log_level);
    try std.testing.expectEqualStrings("test.db", config.database.name);
    try std.testing.expectEqual(@as(usize, 20), config.database.default_search_limit);
    try std.testing.expectEqual(@as(f32, 0.5), config.ai.temperature);
    try std.testing.expectEqual(@as(u32, 1024), config.ai.max_tokens);
}

test "config log level parsing" {
    try std.testing.expectEqual(@as(?LogLevel, .info), ConfigLoader.parseLogLevel("info"));
    try std.testing.expectEqual(@as(?LogLevel, .info), ConfigLoader.parseLogLevel("INFO"));
    try std.testing.expectEqual(@as(?LogLevel, .warn), ConfigLoader.parseLogLevel("Warn"));
    try std.testing.expectEqual(@as(?LogLevel, null), ConfigLoader.parseLogLevel("invalid"));
}

test {
    std.testing.refAllDecls(@This());
}
