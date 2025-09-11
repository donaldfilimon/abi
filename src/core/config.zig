//! Configuration Management - Framework configuration and validation
//!
//! This module provides configuration management for the Abi AI Framework,
//! including configuration validation, loading, and management.

const std = @import("std");
const builtin = @import("builtin");

const errors = @import("errors.zig");

const FrameworkError = errors.FrameworkError;

/// Main framework configuration
pub const FrameworkConfig = struct {
    // Core features
    enable_gpu: bool = true,
    enable_simd: bool = true,
    enable_memory_tracking: bool = true,
    enable_performance_profiling: bool = true,

    // Concurrency settings
    max_concurrent_agents: u32 = 10,
    max_concurrent_requests: u32 = 1000,
    thread_pool_size: u32 = 4,

    // Plugin system
    plugin_directory: []const u8 = "plugins/",
    enable_plugin_hot_reload: bool = true,
    max_plugins: u32 = 50,

    // Logging
    log_level: std.log.Level = .info,
    log_file: ?[]const u8 = null,
    enable_structured_logging: bool = true,

    // Performance
    enable_compression: bool = true,
    enable_caching: bool = true,
    cache_size_mb: u32 = 256,

    // Security
    enable_authentication: bool = false,
    enable_encryption: bool = false,
    max_request_size_mb: u32 = 10,

    // Database
    database_path: []const u8 = "data/",
    enable_database_compression: bool = true,
    max_database_size_gb: u32 = 10,

    // Web server
    web_server_port: u16 = 8080,
    web_server_host: []const u8 = "0.0.0.0",
    enable_websocket: bool = true,
    enable_cors: bool = true,

    // Monitoring
    enable_metrics: bool = true,
    metrics_port: u16 = 9090,
    enable_health_checks: bool = true,

    /// Validate the configuration
    pub fn validate(self: FrameworkConfig) FrameworkError!void {
        // Validate concurrency settings
        if (self.max_concurrent_agents == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.max_concurrent_requests == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.thread_pool_size == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate plugin settings
        if (self.max_plugins == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate performance settings
        if (self.cache_size_mb == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate security settings
        if (self.max_request_size_mb == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate database settings
        if (self.max_database_size_gb == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate web server settings
        if (self.web_server_port == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        // Validate monitoring settings
        if (self.enable_metrics and self.metrics_port == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }

    /// Create a default configuration
    pub fn default() FrameworkConfig {
        return FrameworkConfig{};
    }

    /// Create a minimal configuration for testing
    pub fn minimal() FrameworkConfig {
        return FrameworkConfig{
            .enable_gpu = false,
            .enable_simd = false,
            .enable_memory_tracking = false,
            .enable_performance_profiling = false,
            .max_concurrent_agents = 1,
            .max_concurrent_requests = 10,
            .thread_pool_size = 1,
            .plugin_directory = "test_plugins/",
            .enable_plugin_hot_reload = false,
            .max_plugins = 5,
            .log_level = .debug,
            .enable_structured_logging = false,
            .enable_compression = false,
            .enable_caching = false,
            .cache_size_mb = 1,
            .enable_authentication = false,
            .enable_encryption = false,
            .max_request_size_mb = 1,
            .database_path = "test_data/",
            .enable_database_compression = false,
            .max_database_size_gb = 1,
            .web_server_port = 8081,
            .web_server_host = "127.0.0.1",
            .enable_websocket = false,
            .enable_cors = false,
            .enable_metrics = false,
            .metrics_port = 9091,
            .enable_health_checks = false,
        };
    }

    /// Create a production configuration
    pub fn production() FrameworkConfig {
        return FrameworkConfig{
            .enable_gpu = true,
            .enable_simd = true,
            .enable_memory_tracking = true,
            .enable_performance_profiling = true,
            .max_concurrent_agents = 100,
            .max_concurrent_requests = 10000,
            .thread_pool_size = 16,
            .plugin_directory = "/opt/abi/plugins/",
            .enable_plugin_hot_reload = false,
            .max_plugins = 100,
            .log_level = .info,
            .log_file = "/var/log/abi/framework.log",
            .enable_structured_logging = true,
            .enable_compression = true,
            .enable_caching = true,
            .cache_size_mb = 1024,
            .enable_authentication = true,
            .enable_encryption = true,
            .max_request_size_mb = 100,
            .database_path = "/opt/abi/data/",
            .enable_database_compression = true,
            .max_database_size_gb = 100,
            .web_server_port = 8080,
            .web_server_host = "0.0.0.0",
            .enable_websocket = true,
            .enable_cors = true,
            .enable_metrics = true,
            .metrics_port = 9090,
            .enable_health_checks = true,
        };
    }
};

/// Agent configuration
pub const AgentConfig = struct {
    name: []const u8,
    default_persona: PersonaType = .adaptive,
    max_context_length: usize = 4096,
    enable_history: bool = true,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    capabilities: AgentCapabilities = .{},
    memory_size: usize = 1024 * 1024,
    enable_logging: bool = true,
    log_level: std.log.Level = .info,
    use_custom_allocator: bool = false,
    enable_simd: bool = true,
    max_concurrent_requests: u32 = 10,
    enable_persona_routing: bool = true,

    /// Validate agent configuration
    pub fn validate(self: AgentConfig) FrameworkError!void {
        if (self.name.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.temperature < 0.0 or self.temperature > 2.0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.top_p < 0.0 or self.top_p > 1.0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.max_context_length == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.memory_size == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.max_concurrent_requests == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }
};

/// Persona types for agents
pub const PersonaType = enum {
    empathetic,
    direct,
    adaptive,
    creative,
    technical,
    solver,
    educator,
    counselor,
    analytical,
    supportive,
};

/// Agent capabilities
pub const AgentCapabilities = packed struct(u64) {
    text_generation: bool = false,
    code_generation: bool = false,
    image_analysis: bool = false,
    audio_processing: bool = false,
    memory_management: bool = false,
    learning: bool = false,
    reasoning: bool = false,
    planning: bool = false,
    vector_search: bool = false,
    function_calling: bool = false,
    multimodal: bool = false,
    streaming: bool = false,
    real_time: bool = false,
    batch_processing: bool = false,
    custom_operations: bool = false,
    _reserved: u49 = 0,

    /// Validate capability dependencies
    pub fn validate(self: AgentCapabilities) bool {
        if (self.vector_search and !self.memory_management) return false;
        if (self.multimodal and !(self.text_generation or self.image_analysis)) return false;
        return true;
    }
};

/// Web server configuration
pub const WebServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "0.0.0.0",
    enable_websocket: bool = true,
    enable_cors: bool = true,
    max_connections: u32 = 1000,
    request_timeout_ms: u32 = 30000,
    enable_compression: bool = true,
    enable_ssl: bool = false,
    ssl_cert_path: ?[]const u8 = null,
    ssl_key_path: ?[]const u8 = null,

    /// Validate web server configuration
    pub fn validate(self: WebServerConfig) FrameworkError!void {
        if (self.port == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.max_connections == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.request_timeout_ms == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.enable_ssl) {
            if (self.ssl_cert_path == null or self.ssl_key_path == null) {
                return FrameworkError.InvalidConfiguration;
            }
        }
    }
};

/// Database configuration
pub const DatabaseConfig = struct {
    path: []const u8 = "data/",
    enable_compression: bool = true,
    max_size_gb: u32 = 10,
    enable_indexing: bool = true,
    index_algorithm: IndexAlgorithm = .hnsw,
    enable_replication: bool = false,
    replication_factor: u32 = 3,

    /// Validate database configuration
    pub fn validate(self: DatabaseConfig) FrameworkError!void {
        if (self.path.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.max_size_gb == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.enable_replication and self.replication_factor == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }
};

/// Index algorithms for vector database
pub const IndexAlgorithm = enum {
    flat,
    hnsw,
    ivf,
    lsh,
};

/// Plugin configuration
pub const PluginConfig = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    enabled: bool = true,
    auto_load: bool = true,
    dependencies: []const []const u8 = &[_][]const u8{},
    settings: std.StringHashMap([]const u8),

    /// Validate plugin configuration
    pub fn validate(self: PluginConfig) FrameworkError!void {
        if (self.name.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.version.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }

        if (self.description.len == 0) {
            return FrameworkError.InvalidConfiguration;
        }
    }
};

/// Configuration loader for loading configurations from files
pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Load framework configuration from JSON file
    pub fn loadFrameworkConfig(self: *Self, path: []const u8) !FrameworkConfig {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);

        // Parse JSON and create configuration
        // This is a simplified implementation
        return FrameworkConfig.default();
    }
    /// Save framework configuration to JSON file
    pub fn saveFrameworkConfig(self: *Self, config: FrameworkConfig, path: []const u8) !void {
        _ = self; // Mark as used

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Convert configuration to JSON and write
        var buffer: [4096]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const writer = stream.writer();

        try writer.print("{{", .{});
        try writer.print("\"enable_gpu\":{},", .{config.enable_gpu});
        try writer.print("\"enable_simd\":{},", .{config.enable_simd});
        try writer.print("\"enable_memory_tracking\":{},", .{config.enable_memory_tracking});
        try writer.print("\"enable_performance_profiling\":{},", .{config.enable_performance_profiling});
        try writer.print("\"log_level\":\"{s}\"", .{@tagName(config.log_level)});
        try writer.print("}}", .{});

        const json_content = stream.getWritten();
        try file.writeAll(json_content);
    }

    /// Load agent configuration from JSON file
    pub fn loadAgentConfig(self: *Self, path: []const u8) !AgentConfig {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);

        // Parse JSON and create configuration
        // This is a simplified implementation
        return AgentConfig{
            .name = "default_agent",
        };
    }

    /// Save agent configuration to JSON file
    pub fn saveAgentConfig(self: *Self, config: AgentConfig, path: []const u8) !void {
        _ = self; // Mark as used

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Convert configuration to JSON and write
        var buffer: [4096]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const writer = stream.writer();

        try writer.print("{{", .{});
        try writer.print("\"name\":\"{s}\",", .{config.name});
        try writer.print("\"temperature\":{d},", .{config.temperature});
        try writer.print("\"top_p\":{d},", .{config.top_p});
        try writer.print("\"max_context_length\":{d},", .{config.max_context_length});
        try writer.print("\"memory_size\":{d},", .{config.memory_size});
        try writer.print("\"max_concurrent_requests\":{d}", .{config.max_concurrent_requests});
        try writer.print("}}", .{});

        const json_content = stream.getWritten();
        try file.writeAll(json_content);
    }
};

test "framework configuration validation" {
    const testing = std.testing;

    // Test valid configuration
    const valid_config = FrameworkConfig.default();
    try testing.expectError(error.InvalidConfiguration, valid_config.validate());

    // Test minimal configuration
    const minimal_config = FrameworkConfig.minimal();
    try minimal_config.validate();

    // Test production configuration
    const production_config = FrameworkConfig.production();
    try production_config.validate();
}

test "agent configuration validation" {
    const testing = std.testing;

    // Test valid agent configuration
    const valid_agent_config = AgentConfig{
        .name = "test_agent",
        .temperature = 0.7,
        .top_p = 0.9,
        .max_context_length = 4096,
        .memory_size = 1024 * 1024,
        .max_concurrent_requests = 10,
    };
    try valid_agent_config.validate();

    // Test invalid agent configuration
    const invalid_agent_config = AgentConfig{
        .name = "",
        .temperature = 3.0,
        .top_p = 1.5,
        .max_context_length = 0,
        .memory_size = 0,
        .max_concurrent_requests = 0,
    };
    try testing.expectError(error.InvalidConfiguration, invalid_agent_config.validate());
}

test "web server configuration validation" {
    const testing = std.testing;

    // Test valid web server configuration
    const valid_web_config = WebServerConfig{
        .port = 8080,
        .max_connections = 1000,
        .request_timeout_ms = 30000,
    };
    try valid_web_config.validate();

    // Test invalid web server configuration
    const invalid_web_config = WebServerConfig{
        .port = 0,
        .max_connections = 0,
        .request_timeout_ms = 0,
    };
    try testing.expectError(error.InvalidConfiguration, invalid_web_config.validate());
}

test "database configuration validation" {
    const testing = std.testing;

    // Test valid database configuration
    const valid_db_config = DatabaseConfig{
        .path = "data/",
        .max_size_gb = 10,
    };
    try valid_db_config.validate();

    // Test invalid database configuration
    const invalid_db_config = DatabaseConfig{
        .path = "",
        .max_size_gb = 0,
    };
    try testing.expectError(error.InvalidConfiguration, invalid_db_config.validate());
}

test "plugin configuration validation" {
    const testing = std.testing;

    // Test valid plugin configuration
    const valid_plugin_config = PluginConfig{
        .name = "test_plugin",
        .version = "1.0.0",
        .description = "Test plugin",
        .settings = std.StringHashMap([]const u8).init(testing.allocator),
    };
    try valid_plugin_config.validate();

    // Test invalid plugin configuration
    const invalid_plugin_config = PluginConfig{
        .name = "",
        .version = "",
        .description = "",
        .settings = std.StringHashMap([]const u8).init(testing.allocator),
    };
    try testing.expectError(error.InvalidConfiguration, invalid_plugin_config.validate());
}
