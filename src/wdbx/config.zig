//! WDBX Configuration Module
//!
//! Provides comprehensive configuration management for the WDBX system including:
//! - .wdbx-config file parsing
//! - Environment variable overrides
//! - Default configuration values
//! - Configuration validation
//! - Hot reloading support

const std = @import("std");
const core = @import("core.zig");

/// Configuration file format (JSON-based)
pub const WdbxConfig = struct {
    // Database configuration
    database: DatabaseConfig = .{},

    // Server configuration
    server: ServerConfig = .{},

    // Performance configuration
    performance: PerformanceConfig = .{},

    // Monitoring configuration
    monitoring: MonitoringConfig = .{},

    // Security configuration
    security: SecurityConfig = .{},

    // Logging configuration
    logging: LoggingConfig = .{},

    pub const DatabaseConfig = struct {
        path: []const u8 = "./wdbx.db",
        dimensions: u32 = 8,

        // HNSW configuration
        hnsw_m: u32 = 16,
        hnsw_ef_construction: u32 = 200,
        hnsw_ef_search: u32 = 50,
        hnsw_max_level: u32 = 16,

        // Storage configuration
        memory_map_size: usize = 1024 * 1024 * 1024, // 1GB
        wal_enabled: bool = true,
        wal_sync_interval_ms: u32 = 1000,
        checkpoint_interval_ms: u32 = 10000,

        // Sharding configuration
        enable_sharding: bool = false,
        shard_count: u32 = 4,
        shard_size_limit: usize = 512 * 1024 * 1024, // 512MB

        // Compression configuration
        enable_compression: bool = true,
        compression_algorithm: []const u8 = "lz4",
        compression_level: u8 = 1,
        compression_block_size: usize = 64 * 1024, // 64KB
    };

    pub const ServerConfig = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 8080,
        max_connections: u32 = 1000,
        max_request_size: usize = 1024 * 1024, // 1MB
        request_timeout_ms: u32 = 30000,
        enable_cors: bool = true,
        enable_auth: bool = true,
        jwt_secret: []const u8 = "",

        // Batch operations
        max_batch_size: u32 = 100,
        batch_timeout_ms: u32 = 5000,

        // Pagination
        default_page_size: u32 = 20,
        max_page_size: u32 = 1000,

        // Windows networking optimizations
        enable_windows_optimizations: bool = true,
        socket_keepalive: bool = true,
        tcp_nodelay: bool = true,
        socket_buffer_size: u32 = 8192,
    };

    pub const PerformanceConfig = struct {
        // CPU and Memory settings
        worker_threads: u32 = 0, // 0 = auto-detect
        memory_limit_mb: u32 = 0, // 0 = unlimited
        gc_interval_ms: u32 = 60000, // 1 minute

        // SIMD optimization
        enable_simd: bool = true,
        simd_alignment: u32 = 32,

        // Performance monitoring
        enable_profiling: bool = false,
        profile_interval_ms: u32 = 5000,

        // Cache settings
        query_cache_size: u32 = 1000,
        result_cache_ttl_ms: u32 = 300000, // 5 minutes
    };

    pub const MonitoringConfig = struct {
        // Prometheus metrics
        enable_prometheus: bool = false,
        prometheus_port: u16 = 9090,
        prometheus_path: []const u8 = "/metrics",

        // Health checks
        enable_health_checks: bool = true,
        health_check_interval_ms: u32 = 10000,

        // Performance sampling
        enable_cpu_sampling: bool = false,
        enable_memory_sampling: bool = false,
        sampling_interval_ms: u32 = 1000,

        // Alerts
        enable_alerts: bool = false,
        alert_webhook_url: []const u8 = "",
        memory_threshold_percent: f32 = 80.0,
        cpu_threshold_percent: f32 = 80.0,
    };

    pub const SecurityConfig = struct {
        // Rate limiting
        rate_limit_enabled: bool = true,
        rate_limit_requests_per_minute: u32 = 1000,
        rate_limit_burst: u32 = 100,

        // API keys
        require_api_key: bool = false,
        api_keys: [][]const u8 = &[_][]const u8{},

        // TLS/SSL
        enable_tls: bool = false,
        tls_cert_path: []const u8 = "",
        tls_key_path: []const u8 = "",

        // CORS
        cors_origins: [][]const u8 = &[_][]const u8{"*"},
        cors_methods: [][]const u8 = &[_][]const u8{ "GET", "POST", "PUT", "DELETE", "OPTIONS" },
        cors_headers: [][]const u8 = &[_][]const u8{ "Content-Type", "Authorization" },
    };

    pub const LoggingConfig = struct {
        level: []const u8 = "info", // debug, info, warn, error
        output: []const u8 = "stdout", // stdout, stderr, file
        file_path: []const u8 = "./wdbx.log",
        max_file_size_mb: u32 = 100,
        max_files: u32 = 5,
        enable_json_format: bool = false,
        enable_timestamps: bool = true,
        enable_request_logging: bool = true,
    };
};

/// Configuration manager
pub const ConfigManager = struct {
    allocator: std.mem.Allocator,
    config: WdbxConfig,
    config_path: []const u8,
    last_modified: i128,

    const Self = @This();

    /// Default configuration file name
    pub const DEFAULT_CONFIG_FILE = ".wdbx-config";

    /// Initialize configuration manager
    pub fn init(allocator: std.mem.Allocator, config_path: ?[]const u8) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const path = config_path orelse DEFAULT_CONFIG_FILE;

        self.* = .{
            .allocator = allocator,
            .config = .{},
            .config_path = try allocator.dupe(u8, path),
            .last_modified = 0,
        };

        // Try to load configuration file
        self.loadFromFile() catch |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("Configuration file not found, using defaults: {s}\n", .{path});
                try self.createDefaultConfigFile();
            },
            else => return err,
        };

        // Apply environment variable overrides
        try self.applyEnvironmentOverrides();

        // Validate configuration
        try self.validate();

        return self;
    }

    /// Deinitialize configuration manager
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.config_path);
        self.allocator.destroy(self);
    }

    /// Get current configuration
    pub fn getConfig(self: *Self) *const WdbxConfig {
        return &self.config;
    }

    /// Load configuration from file
    pub fn loadFromFile(self: *Self) !void {
        const file = try std.fs.cwd().openFile(self.config_path, .{});
        defer file.close();

        const stat = try file.stat();
        self.last_modified = stat.mtime;

        const contents = try file.readToEndAlloc(self.allocator, 1024 * 1024); // 1MB max
        defer self.allocator.free(contents);

        const parsed = try std.json.parseFromSlice(WdbxConfig, self.allocator, contents, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        self.config = parsed.value;

        std.debug.print("Configuration loaded from: {s}\n", .{self.config_path});
    }

    /// Create default configuration file
    pub fn createDefaultConfigFile(self: *Self) !void {
        const file = try std.fs.cwd().createFile(self.config_path, .{});
        defer file.close();

        const json_string = try std.json.stringifyAlloc(self.allocator, self.config, .{
            .whitespace = .indent_2,
        });
        defer self.allocator.free(json_string);

        try file.writeAll(json_string);

        std.debug.print("Default configuration file created: {s}\n", .{self.config_path});
    }

    /// Apply environment variable overrides
    pub fn applyEnvironmentOverrides(self: *Self) !void {
        // Database overrides
        if (std.process.getEnvVarOwned(self.allocator, "WDBX_DB_PATH")) |path| {
            defer self.allocator.free(path);
            self.config.database.path = try self.allocator.dupe(u8, path);
        } else |_| {}

        if (std.process.getEnvVarOwned(self.allocator, "WDBX_DB_DIMENSIONS")) |dim_str| {
            defer self.allocator.free(dim_str);
            self.config.database.dimensions = std.fmt.parseInt(u32, dim_str, 10) catch self.config.database.dimensions;
        } else |_| {}

        // Server overrides
        if (std.process.getEnvVarOwned(self.allocator, "WDBX_SERVER_HOST")) |host| {
            defer self.allocator.free(host);
            self.config.server.host = try self.allocator.dupe(u8, host);
        } else |_| {}

        if (std.process.getEnvVarOwned(self.allocator, "WDBX_SERVER_PORT")) |port_str| {
            defer self.allocator.free(port_str);
            self.config.server.port = std.fmt.parseInt(u16, port_str, 10) catch self.config.server.port;
        } else |_| {}

        // Performance overrides
        if (std.process.getEnvVarOwned(self.allocator, "WDBX_WORKER_THREADS")) |threads_str| {
            defer self.allocator.free(threads_str);
            self.config.performance.worker_threads = std.fmt.parseInt(u32, threads_str, 10) catch self.config.performance.worker_threads;
        } else |_| {}

        // Enable/disable features
        if (std.process.getEnvVarOwned(self.allocator, "WDBX_ENABLE_PROMETHEUS")) |enable_str| {
            defer self.allocator.free(enable_str);
            self.config.monitoring.enable_prometheus = std.mem.eql(u8, enable_str, "true");
        } else |_| {}

        std.debug.print("Environment variable overrides applied\n");
    }

    /// Check if configuration file has been modified and reload if necessary
    pub fn checkAndReload(self: *Self) !bool {
        const file = std.fs.cwd().openFile(self.config_path, .{}) catch return false;
        defer file.close();

        const stat = try file.stat();
        if (stat.mtime > self.last_modified) {
            try self.loadFromFile();
            try self.applyEnvironmentOverrides();
            try self.validate();
            std.debug.print("Configuration reloaded due to file changes\n");
            return true;
        }

        return false;
    }

    /// Validate configuration values
    pub fn validate(self: *Self) !void {
        // Validate database configuration
        if (self.config.database.dimensions == 0) {
            return core.WdbxError.InvalidConfiguration;
        }

        if (self.config.database.hnsw_m == 0 or self.config.database.hnsw_m > 128) {
            return core.WdbxError.InvalidConfiguration;
        }

        // Validate server configuration
        if (self.config.server.port == 0) {
            return core.WdbxError.InvalidConfiguration;
        }

        if (self.config.server.max_batch_size == 0 or self.config.server.max_batch_size > 10000) {
            return core.WdbxError.InvalidConfiguration;
        }

        // Validate performance configuration
        if (self.config.performance.worker_threads == 0) {
            self.config.performance.worker_threads = @intCast(std.Thread.getCpuCount() catch 4);
        }

        // Validate monitoring configuration
        if (self.config.monitoring.enable_prometheus and self.config.monitoring.prometheus_port == 0) {
            return core.WdbxError.InvalidConfiguration;
        }

        std.debug.print("Configuration validation passed\n");
    }

    /// Save current configuration to file
    pub fn save(self: *Self) !void {
        const file = try std.fs.cwd().createFile(self.config_path, .{});
        defer file.close();

        const json_string = try std.json.stringifyAlloc(self.allocator, self.config, .{
            .whitespace = .indent_2,
        });
        defer self.allocator.free(json_string);

        try file.writeAll(json_string);

        const stat = try file.stat();
        self.last_modified = stat.mtime;

        std.debug.print("Configuration saved to: {s}\n", .{self.config_path});
    }
};

/// Configuration utilities
pub const ConfigUtils = struct {
    /// Get configuration value by path (e.g., "database.hnsw_m")
    pub fn getValue(config: *const WdbxConfig, path: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
        var parts = std.mem.split(u8, path, ".");

        const section = parts.next() orelse return null;
        const key = parts.next() orelse return null;

        if (std.mem.eql(u8, section, "database")) {
            return getFieldValue(@TypeOf(config.database), &config.database, key, allocator);
        } else if (std.mem.eql(u8, section, "server")) {
            return getFieldValue(@TypeOf(config.server), &config.server, key, allocator);
        } else if (std.mem.eql(u8, section, "performance")) {
            return getFieldValue(@TypeOf(config.performance), &config.performance, key, allocator);
        } else if (std.mem.eql(u8, section, "monitoring")) {
            return getFieldValue(@TypeOf(config.monitoring), &config.monitoring, key, allocator);
        } else if (std.mem.eql(u8, section, "security")) {
            return getFieldValue(@TypeOf(config.security), &config.security, key, allocator);
        } else if (std.mem.eql(u8, section, "logging")) {
            return getFieldValue(@TypeOf(config.logging), &config.logging, key, allocator);
        }

        return null;
    }

    /// Helper function to get field value by name
    fn getFieldValue(comptime T: type, obj: *const T, field_name: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
        inline for (std.meta.fields(T)) |field| {
            if (std.mem.eql(u8, field.name, field_name)) {
                const value = @field(obj, field.name);
                return switch (@TypeOf(value)) {
                    []const u8 => try allocator.dupe(u8, value),
                    u32, u16, u8 => try std.fmt.allocPrint(allocator, "{}", .{value}),
                    f32 => try std.fmt.allocPrint(allocator, "{d:.2}", .{value}),
                    bool => try allocator.dupe(u8, if (value) "true" else "false"),
                    else => null,
                };
            }
        }
        return null;
    }

    /// Print configuration summary
    pub fn printSummary(config: *const WdbxConfig) void {
        std.debug.print("\n🔧 WDBX Configuration Summary\n");
        std.debug.print("═══════════════════════════════\n");
        std.debug.print("Database:\n");
        std.debug.print("  Path: {s}\n", .{config.database.path});
        std.debug.print("  Dimensions: {}\n", .{config.database.dimensions});
        std.debug.print("  HNSW M: {}\n", .{config.database.hnsw_m});
        std.debug.print("  Compression: {s}\n", .{if (config.database.enable_compression) "enabled" else "disabled"});

        std.debug.print("\nServer:\n");
        std.debug.print("  Address: {s}:{}\n", .{ config.server.host, config.server.port });
        std.debug.print("  Max Connections: {}\n", .{config.server.max_connections});
        std.debug.print("  Batch Size: {}\n", .{config.server.max_batch_size});
        std.debug.print("  Page Size: {}\n", .{config.server.default_page_size});

        std.debug.print("\nPerformance:\n");
        std.debug.print("  Worker Threads: {}\n", .{config.performance.worker_threads});
        std.debug.print("  SIMD: {s}\n", .{if (config.performance.enable_simd) "enabled" else "disabled"});
        std.debug.print("  Profiling: {s}\n", .{if (config.performance.enable_profiling) "enabled" else "disabled"});

        std.debug.print("\nMonitoring:\n");
        std.debug.print("  Prometheus: {s}\n", .{if (config.monitoring.enable_prometheus) "enabled" else "disabled"});
        std.debug.print("  CPU Sampling: {s}\n", .{if (config.monitoring.enable_cpu_sampling) "enabled" else "disabled"});
        std.debug.print("  Memory Sampling: {s}\n", .{if (config.monitoring.enable_memory_sampling) "enabled" else "disabled"});
        std.debug.print("═══════════════════════════════\n\n");
    }
};
