//! WDBX Configuration Module
//!
//! Provides comprehensive configuration management for the WDBX system including:
//! - .wdbx-config file parsing
//! - Environment variable overrides
//! - Default configuration values
//! - Configuration validation with schema checking
//! - Hot reloading support
//! - Error code standardization

const std = @import("std");
const core = @import("core.zig");

/// Configuration validation error codes
pub const ConfigValidationError = error{
    // Schema validation errors
    InvalidDatabasePath,
    InvalidDimensions,
    InvalidHnswParameters,
    InvalidMemorySettings,
    InvalidCompressionSettings,
    InvalidShardingConfig,

    // Server validation errors
    InvalidHostAddress,
    InvalidPortNumber,
    InvalidConnectionLimits,
    InvalidTimeoutSettings,
    InvalidBatchSettings,
    InvalidPaginationSettings,

    // Performance validation errors
    InvalidThreadCount,
    InvalidMemoryLimits,
    InvalidCacheSettings,
    InvalidProfileSettings,

    // Monitoring validation errors
    InvalidPrometheusConfig,
    InvalidHealthCheckConfig,
    InvalidSamplingConfig,
    InvalidAlertConfig,

    // Security validation errors
    InvalidRateLimitConfig,
    InvalidTlsConfig,
    InvalidCorsConfig,
    InvalidAuthConfig,

    // Logging validation errors
    InvalidLogLevel,
    InvalidLogOutput,
    InvalidLogSettings,

    // Cross-section validation errors
    ConfigConflict,
    IncompatibleSettings,
    ResourceOverallocation,
};

/// Configuration schema validator
pub const ConfigValidator = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Validate entire configuration with comprehensive checks
    pub fn validateConfig(self: *Self, config: *const WdbxConfig) ConfigValidationError!void {
        try self.validateDatabase(&config.database);
        try self.validateServer(&config.server);
        try self.validatePerformance(&config.performance);
        try self.validateMonitoring(&config.monitoring);
        try self.validateSecurity(&config.security);
        try self.validateLogging(&config.logging);
        try self.validateCrossSection(config);
    }

    /// Validate database configuration
    fn validateDatabase(self: *Self, db_config: *const WdbxConfig.DatabaseConfig) ConfigValidationError!void {
        _ = self;

        // Path validation
        if (db_config.path.len == 0) {
            return ConfigValidationError.InvalidDatabasePath;
        }

        // Dimensions validation
        if (db_config.dimensions == 0 or db_config.dimensions > 4096) {
            return ConfigValidationError.InvalidDimensions;
        }

        // HNSW parameters validation
        if (db_config.hnsw_m == 0 or db_config.hnsw_m > 128) {
            return ConfigValidationError.InvalidHnswParameters;
        }
        if (db_config.hnsw_ef_construction < db_config.hnsw_ef_search) {
            return ConfigValidationError.InvalidHnswParameters;
        }
        if (db_config.hnsw_max_level > 32) {
            return ConfigValidationError.InvalidHnswParameters;
        }

        // Memory settings validation
        if (db_config.memory_map_size < 1024 * 1024) { // Minimum 1MB
            return ConfigValidationError.InvalidMemorySettings;
        }

        // WAL settings validation
        if (db_config.wal_enabled and db_config.wal_sync_interval_ms == 0) {
            return ConfigValidationError.InvalidMemorySettings;
        }

        // Sharding validation
        if (db_config.enable_sharding) {
            if (db_config.shard_count == 0 or db_config.shard_count > 1024) {
                return ConfigValidationError.InvalidShardingConfig;
            }
            if (db_config.shard_size_limit < 1024 * 1024) { // Minimum 1MB per shard
                return ConfigValidationError.InvalidShardingConfig;
            }
        }

        // Compression validation
        if (db_config.enable_compression) {
            const valid_algorithms = [_][]const u8{ "lz4", "zstd", "gzip" };
            var valid = false;
            for (valid_algorithms) |algo| {
                if (std.mem.eql(u8, db_config.compression_algorithm, algo)) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                return ConfigValidationError.InvalidCompressionSettings;
            }

            if (db_config.compression_level > 9) {
                return ConfigValidationError.InvalidCompressionSettings;
            }
            if (db_config.compression_block_size < 1024) { // Minimum 1KB blocks
                return ConfigValidationError.InvalidCompressionSettings;
            }
        }
    }

    /// Validate server configuration
    fn validateServer(self: *Self, server_config: *const WdbxConfig.ServerConfig) ConfigValidationError!void {
        _ = self;

        // Host validation
        if (server_config.host.len == 0) {
            return ConfigValidationError.InvalidHostAddress;
        }

        // Port validation
        if (server_config.port == 0 or server_config.port > 65535) {
            return ConfigValidationError.InvalidPortNumber;
        }

        // Connection limits validation
        if (server_config.max_connections == 0 or server_config.max_connections > 100000) {
            return ConfigValidationError.InvalidConnectionLimits;
        }

        // Request size validation
        if (server_config.max_request_size == 0 or server_config.max_request_size > 100 * 1024 * 1024) {
            return ConfigValidationError.InvalidConnectionLimits;
        }

        // Timeout validation
        if (server_config.request_timeout_ms == 0 or server_config.request_timeout_ms > 300000) {
            return ConfigValidationError.InvalidTimeoutSettings;
        }

        // Batch settings validation
        if (server_config.max_batch_size == 0 or server_config.max_batch_size > 10000) {
            return ConfigValidationError.InvalidBatchSettings;
        }
        if (server_config.batch_timeout_ms == 0 or server_config.batch_timeout_ms > 60000) {
            return ConfigValidationError.InvalidBatchSettings;
        }

        // Pagination validation
        if (server_config.default_page_size == 0 or server_config.default_page_size > server_config.max_page_size) {
            return ConfigValidationError.InvalidPaginationSettings;
        }
        if (server_config.max_page_size > 10000) {
            return ConfigValidationError.InvalidPaginationSettings;
        }

        // JWT secret validation when auth is enabled and JWT secret is explicitly set
        if (server_config.enable_auth and server_config.jwt_secret != null and server_config.jwt_secret.?.len < 32) {
            return ConfigValidationError.InvalidAuthConfig;
        }
    }

    /// Validate performance configuration
    fn validatePerformance(self: *Self, perf_config: *const WdbxConfig.PerformanceConfig) ConfigValidationError!void {
        _ = self;

        // Thread count validation
        if (perf_config.worker_threads > 256) {
            return ConfigValidationError.InvalidThreadCount;
        }

        // Memory limit validation
        if (perf_config.memory_limit_mb > 0 and perf_config.memory_limit_mb < 100) {
            return ConfigValidationError.InvalidMemoryLimits;
        }

        // GC interval validation
        if (perf_config.gc_interval_ms < 1000) { // Minimum 1 second
            return ConfigValidationError.InvalidMemoryLimits;
        }

        // SIMD alignment validation
        if (perf_config.enable_simd) {
            const valid_alignments = [_]u32{ 16, 32, 64 };
            var valid = false;
            for (valid_alignments) |alignment| {
                if (perf_config.simd_alignment == alignment) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                return ConfigValidationError.InvalidMemorySettings;
            }
        }

        // Profiling validation
        if (perf_config.enable_profiling and perf_config.profile_interval_ms < 100) {
            return ConfigValidationError.InvalidProfileSettings;
        }

        // Cache validation
        if (perf_config.query_cache_size == 0) {
            return ConfigValidationError.InvalidCacheSettings;
        }
        if (perf_config.result_cache_ttl_ms < 1000) { // Minimum 1 second TTL
            return ConfigValidationError.InvalidCacheSettings;
        }
    }

    /// Validate monitoring configuration
    fn validateMonitoring(self: *Self, mon_config: *const WdbxConfig.MonitoringConfig) ConfigValidationError!void {
        _ = self;

        // Prometheus validation
        if (mon_config.enable_prometheus) {
            if (mon_config.prometheus_port == 0 or mon_config.prometheus_port > 65535) {
                return ConfigValidationError.InvalidPrometheusConfig;
            }
            if (mon_config.prometheus_path.len == 0) {
                return ConfigValidationError.InvalidPrometheusConfig;
            }
        }

        // Health check validation
        if (mon_config.enable_health_checks and mon_config.health_check_interval_ms < 1000) {
            return ConfigValidationError.InvalidHealthCheckConfig;
        }

        // Sampling validation
        if (mon_config.enable_cpu_sampling or mon_config.enable_memory_sampling) {
            if (mon_config.sampling_interval_ms < 100 or mon_config.sampling_interval_ms > 60000) {
                return ConfigValidationError.InvalidSamplingConfig;
            }
        }

        // Alert validation
        if (mon_config.enable_alerts) {
            if (mon_config.alert_webhook_url.len == 0) {
                return ConfigValidationError.InvalidAlertConfig;
            }
            if (mon_config.memory_threshold_percent <= 0 or mon_config.memory_threshold_percent > 100) {
                return ConfigValidationError.InvalidAlertConfig;
            }
            if (mon_config.cpu_threshold_percent <= 0 or mon_config.cpu_threshold_percent > 100) {
                return ConfigValidationError.InvalidAlertConfig;
            }
        }
    }

    /// Validate security configuration
    fn validateSecurity(self: *Self, sec_config: *const WdbxConfig.SecurityConfig) ConfigValidationError!void {
        _ = self;

        // Rate limiting validation
        if (sec_config.rate_limit_enabled) {
            if (sec_config.rate_limit_requests_per_minute == 0) {
                return ConfigValidationError.InvalidRateLimitConfig;
            }
            if (sec_config.rate_limit_burst == 0 or sec_config.rate_limit_burst > sec_config.rate_limit_requests_per_minute) {
                return ConfigValidationError.InvalidRateLimitConfig;
            }
        }

        // API key validation
        if (sec_config.require_api_key and sec_config.api_keys.len == 0) {
            return ConfigValidationError.InvalidAuthConfig;
        }

        // TLS validation
        if (sec_config.enable_tls) {
            if (sec_config.tls_cert_path.len == 0 or sec_config.tls_key_path.len == 0) {
                return ConfigValidationError.InvalidTlsConfig;
            }
        }

        // CORS validation
        if (sec_config.cors_origins.len == 0) {
            return ConfigValidationError.InvalidCorsConfig;
        }
    }

    /// Validate logging configuration
    fn validateLogging(self: *Self, log_config: *const WdbxConfig.LoggingConfig) ConfigValidationError!void {
        _ = self;

        // Log level validation
        const valid_levels = [_][]const u8{ "trace", "debug", "info", "warn", "error", "fatal" };
        var valid = false;
        for (valid_levels) |level| {
            if (std.mem.eql(u8, log_config.level, level)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return ConfigValidationError.InvalidLogLevel;
        }

        // Output validation
        const valid_outputs = [_][]const u8{ "stdout", "stderr", "file" };
        valid = false;
        for (valid_outputs) |output| {
            if (std.mem.eql(u8, log_config.output, output)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return ConfigValidationError.InvalidLogOutput;
        }

        // File settings validation
        if (std.mem.eql(u8, log_config.output, "file")) {
            if (log_config.file_path.len == 0) {
                return ConfigValidationError.InvalidLogSettings;
            }
            if (log_config.max_file_size_mb == 0 or log_config.max_file_size_mb > 1000) {
                return ConfigValidationError.InvalidLogSettings;
            }
            if (log_config.max_files == 0 or log_config.max_files > 100) {
                return ConfigValidationError.InvalidLogSettings;
            }
        }
    }

    /// Validate cross-section configuration consistency
    fn validateCrossSection(self: *Self, config: *const WdbxConfig) ConfigValidationError!void {
        _ = self;

        // Check if Prometheus port conflicts with server port
        if (config.monitoring.enable_prometheus and
            config.monitoring.prometheus_port == config.server.port)
        {
            return ConfigValidationError.ConfigConflict;
        }

        // Check memory allocation conflicts
        if (config.performance.memory_limit_mb > 0) {
            const estimated_db_memory = config.database.memory_map_size / (1024 * 1024);
            if (estimated_db_memory > config.performance.memory_limit_mb / 2) {
                return ConfigValidationError.ResourceOverallocation;
            }
        }

        // Check thread count vs CPU cores
        if (config.performance.worker_threads > 0) {
            const cpu_count = std.Thread.getCpuCount() catch 4;
            if (config.performance.worker_threads > cpu_count * 2) {
                return ConfigValidationError.ResourceOverallocation;
            }
        }

        // Check sharding compatibility with compression
        if (config.database.enable_sharding and config.database.enable_compression) {
            if (config.database.compression_block_size > config.database.shard_size_limit / 100) {
                return ConfigValidationError.IncompatibleSettings;
            }
        }

        // Check TLS and authentication compatibility
        if (config.security.enable_tls and !config.server.enable_auth) {
            // Warning: TLS without auth might not be intended, but not an error
        }
    }

    /// Generate validation report
    pub fn generateValidationReport(self: *Self, config: *const WdbxConfig) ![]const u8 {
        var report = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        defer report.deinit(self.allocator);

        try report.appendSlice(self.allocator, "🔍 Configuration Validation Report\n");
        try report.appendSlice(self.allocator, "══════════════════════════════════\n\n");

        // Database validation details
        try report.appendSlice(self.allocator, "📊 Database Configuration:\n");
        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Path: {s}\n", .{config.database.path});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Dimensions: {d}\n", .{config.database.dimensions});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ HNSW M: {d}\n", .{config.database.hnsw_m});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Memory Map: {d} MB\n", .{config.database.memory_map_size / (1024 * 1024)});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        if (config.database.enable_sharding) {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Sharding: {d} shards\n", .{config.database.shard_count});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        if (config.database.enable_compression) {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Compression: {s} (level {d})\n", .{ config.database.compression_algorithm, config.database.compression_level });
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        // Server validation details
        try report.appendSlice(self.allocator, "\n🌐 Server Configuration:\n");
        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Address: {s}:{d}\n", .{ config.server.host, config.server.port });
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Max Connections: {d}\n", .{config.server.max_connections});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Batch Size: {d}\n", .{config.server.max_batch_size});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Page Size: {d}\n", .{config.server.default_page_size});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        // Performance validation details
        try report.appendSlice(self.allocator, "\n⚡ Performance Configuration:\n");
        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Worker Threads: {d}\n", .{config.performance.worker_threads});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Cache Size: {d}\n", .{config.performance.query_cache_size});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ SIMD: {s}\n", .{if (config.performance.enable_simd) "enabled" else "disabled"});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        // Security validation details
        try report.appendSlice(self.allocator, "\n🔒 Security Configuration:\n");
        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Authentication: {s}\n", .{if (config.server.enable_auth) "enabled" else "disabled"});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ Rate Limiting: {s}\n", .{if (config.security.rate_limit_enabled) "enabled" else "disabled"});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        {
            const line = try std.fmt.allocPrint(self.allocator, "  ✓ TLS: {s}\n", .{if (config.security.enable_tls) "enabled" else "disabled"});
            defer self.allocator.free(line);
            try report.appendSlice(self.allocator, line);
        }

        try report.appendSlice(self.allocator, "\n✅ Configuration validation passed!\n");

        return try self.allocator.dupe(u8, report.items);
    }
};

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
        jwt_secret: ?[]const u8 = null,

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
        api_keys: []const []const u8 = &[_][]const u8{},

        // TLS/SSL
        enable_tls: bool = false,
        tls_cert_path: []const u8 = "",
        tls_key_path: []const u8 = "",

        // CORS
        cors_origins: []const []const u8 = &[_][]const u8{"*"},
        cors_methods: []const []const u8 = &[_][]const u8{ "GET", "POST", "PUT", "DELETE", "OPTIONS" },
        cors_headers: []const []const u8 = &[_][]const u8{ "Content-Type", "Authorization" },
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
        const config_path_allocated = try allocator.dupe(u8, path);
        errdefer allocator.free(config_path_allocated);

        self.* = .{
            .allocator = allocator,
            .config = .{},
            .config_path = config_path_allocated,
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

        // Read file contents (max 1MB) using buffered reader (Zig 0.16 I/O)
        var read_buf: [64 * 1024]u8 = undefined;
        var reader = file.reader(&read_buf);
        var contents_list = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        defer contents_list.deinit(self.allocator);

        var tmp: [4096]u8 = undefined;
        while (true) {
            const n = reader.read(&tmp) catch 0;
            if (n == 0) break;
            try contents_list.appendSlice(self.allocator, tmp[0..n]);
            if (contents_list.items.len > 1024 * 1024) break;
        }

        const contents = contents_list.items;

        const parsed = try std.json.parseFromSlice(WdbxConfig, self.allocator, contents, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        self.config = parsed.value;

        std.debug.print("Configuration loaded from: {s}\n", .{self.config_path});
    }

    /// Create default configuration file (Zig 0.16 compatible)
    pub fn createDefaultConfigFile(self: *Self) !void {
        const file = try std.fs.cwd().createFile(self.config_path, .{});
        defer file.close();

        // Minimal valid JSON to bootstrap; full save implemented separately
        const json_string = "{\n  \"database\": {},\n  \"server\": {}\n}\n";
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

        std.debug.print("Environment variable overrides applied\n", .{});
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
            std.debug.print("Configuration reloaded due to file changes\n", .{});
            return true;
        }

        return false;
    }

    /// Validate configuration values using comprehensive schema validation
    pub fn validate(self: *Self) !void {
        var validator = ConfigValidator.init(self.allocator);

        // Perform comprehensive validation
        validator.validateConfig(&self.config) catch |err| {
            std.debug.print("⚠️ Configuration validation failed: {s}\n", .{@errorName(err)});
            switch (err) {
                ConfigValidationError.InvalidDatabasePath => {
                    std.debug.print("❌ Database path cannot be empty\n", .{});
                },
                ConfigValidationError.InvalidDimensions => {
                    std.debug.print("❌ Vector dimensions must be between 1 and 4096\n", .{});
                },
                ConfigValidationError.InvalidHnswParameters => {
                    std.debug.print("❌ HNSW parameters are invalid (M: 1-128, ef_construction >= ef_search)\n", .{});
                },
                ConfigValidationError.InvalidPortNumber => {
                    std.debug.print("❌ Port number must be between 1 and 65535\n", .{});
                },
                ConfigValidationError.ConfigConflict => {
                    std.debug.print("❌ Configuration conflict detected (e.g., port conflicts)\n", .{});
                },
                ConfigValidationError.ResourceOverallocation => {
                    std.debug.print("❌ Resource overallocation detected (memory or CPU)\n", .{});
                },
                else => {
                    std.debug.print("❌ Validation error: {s}\n", .{@errorName(err)});
                },
            }
            return core.WdbxError.ConfigurationValidationFailed;
        };

        // Auto-configure worker threads if not set
        if (self.config.performance.worker_threads == 0) {
            self.config.performance.worker_threads = @intCast(std.Thread.getCpuCount() catch 4);
        }

        // Generate and optionally display validation report
        if (std.process.getEnvVarOwned(self.allocator, "WDBX_VERBOSE_CONFIG")) |_| {
            const report = validator.generateValidationReport(&self.config) catch |err| {
                std.debug.print("Failed to generate validation report: {s}\n", .{@errorName(err)});
                return;
            };
            defer self.allocator.free(report);
            std.debug.print("{s}\n", .{report});
        } else |_| {
            std.debug.print("✅ Configuration validation passed\n", .{});
        }
    }

    /// Get configuration value by key path (e.g., "database.dimensions")
    pub fn getValue(self: *Self, key: []const u8) !?[]const u8 {
        return ConfigUtils.getValue(&self.config, key, self.allocator);
    }

    /// Set configuration value by key path (e.g., "database.dimensions=128")
    pub fn setValue(self: *Self, key: []const u8, value: []const u8) !void {
        var parts = std.mem.splitScalar(u8, key, '.');
        const section = parts.next() orelse return core.WdbxError.InvalidConfiguration;
        const field = parts.next() orelse return core.WdbxError.InvalidConfiguration;

        if (std.mem.eql(u8, section, "database")) {
            try self.setFieldValue(@TypeOf(self.config.database), &self.config.database, field, value);
        } else if (std.mem.eql(u8, section, "server")) {
            try self.setFieldValue(@TypeOf(self.config.server), &self.config.server, field, value);
        } else if (std.mem.eql(u8, section, "performance")) {
            try self.setFieldValue(@TypeOf(self.config.performance), &self.config.performance, field, value);
        } else if (std.mem.eql(u8, section, "monitoring")) {
            try self.setFieldValue(@TypeOf(self.config.monitoring), &self.config.monitoring, field, value);
        } else if (std.mem.eql(u8, section, "security")) {
            try self.setFieldValue(@TypeOf(self.config.security), &self.config.security, field, value);
        } else if (std.mem.eql(u8, section, "logging")) {
            try self.setFieldValue(@TypeOf(self.config.logging), &self.config.logging, field, value);
        } else {
            return core.WdbxError.InvalidConfiguration;
        }
    }

    /// Helper function to set field value by name
    fn setFieldValue(self: *Self, comptime T: type, obj: *T, field_name: []const u8, value_str: []const u8) !void {
        inline for (std.meta.fields(T)) |field| {
            if (std.mem.eql(u8, field.name, field_name)) {
                const value = switch (field.type) {
                    []const u8 => try self.allocator.dupe(u8, value_str),
                    u32 => std.fmt.parseInt(u32, value_str, 10) catch return core.WdbxError.InvalidParameter,
                    u16 => std.fmt.parseInt(u16, value_str, 10) catch return core.WdbxError.InvalidParameter,
                    u8 => std.fmt.parseInt(u8, value_str, 10) catch return core.WdbxError.InvalidParameter,
                    f32 => std.fmt.parseFloat(f32, value_str) catch return core.WdbxError.InvalidParameter,
                    bool => std.mem.eql(u8, value_str, "true") or std.mem.eql(u8, value_str, "1"),
                    else => return core.WdbxError.InvalidParameter,
                };

                // Free old string value if it exists
                if (field.type == []const u8) {
                    const current_value = @field(obj, field.name);
                    if (current_value.len > 0) {
                        self.allocator.free(current_value);
                    }
                }

                @field(obj, field.name) = value;
                return;
            }
        }
        return core.WdbxError.InvalidConfiguration;
    }

    /// List all configuration values
    pub fn listAll(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 0);
        defer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, "Configuration Values:\n");
        try buffer.appendSlice(allocator, "====================\n");

        // Database section
        try buffer.appendSlice(allocator, "\n[Database]\n");
        inline for (std.meta.fields(@TypeOf(self.config.database))) |field| {
            const value = @field(self.config.database, field.name);
            const line = try std.fmt.allocPrint(allocator, "{s} = ", .{field.name});
            defer allocator.free(line);
            try buffer.appendSlice(allocator, line);

            switch (@TypeOf(value)) {
                []const u8 => try buffer.appendSlice(allocator, value),
                u32, u16, u8 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                f32 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{d:.2}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                bool => try buffer.appendSlice(allocator, if (value) "true" else "false"),
                else => try buffer.appendSlice(allocator, "<complex>"),
            }
            try buffer.appendSlice(allocator, "\n");
        }

        // Server section
        try buffer.appendSlice(allocator, "\n[Server]\n");
        inline for (std.meta.fields(@TypeOf(self.config.server))) |field| {
            const value = @field(self.config.server, field.name);
            const line = try std.fmt.allocPrint(allocator, "{s} = ", .{field.name});
            defer allocator.free(line);
            try buffer.appendSlice(allocator, line);

            switch (@TypeOf(value)) {
                []const u8 => try buffer.appendSlice(allocator, value),
                u32, u16, u8 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                f32 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{d:.2}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                bool => try buffer.appendSlice(allocator, if (value) "true" else "false"),
                else => try buffer.appendSlice(allocator, "<complex>"),
            }
            try buffer.appendSlice(allocator, "\n");
        }

        // Performance section
        try buffer.appendSlice(allocator, "\n[Performance]\n");
        inline for (std.meta.fields(@TypeOf(self.config.performance))) |field| {
            const value = @field(self.config.performance, field.name);
            const line = try std.fmt.allocPrint(allocator, "{s} = ", .{field.name});
            defer allocator.free(line);
            try buffer.appendSlice(allocator, line);

            switch (@TypeOf(value)) {
                []const u8 => try buffer.appendSlice(allocator, value),
                u32, u16, u8 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                f32 => {
                    const val_str = try std.fmt.allocPrint(allocator, "{d:.2}", .{value});
                    defer allocator.free(val_str);
                    try buffer.appendSlice(allocator, val_str);
                },
                bool => try buffer.appendSlice(allocator, if (value) "true" else "false"),
                else => try buffer.appendSlice(allocator, "<complex>"),
            }
            try buffer.appendSlice(allocator, "\n");
        }

        return try buffer.toOwnedSlice(allocator);
    }

    /// Save current configuration to file
    pub fn save(self: *Self) !void {
        const file = try std.fs.cwd().createFile(self.config_path, .{});
        defer file.close();

        // Serialize a minimal subset for Zig 0.16 compatibility
        var buffer = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        defer buffer.deinit(self.allocator);
        try buffer.appendSlice(self.allocator, "{\n");
        try buffer.appendSlice(self.allocator, "  \"server\": {\n");
        const line = try std.fmt.allocPrint(self.allocator, "    \"port\": {d}\n", .{self.config.server.port});
        defer self.allocator.free(line);
        try buffer.appendSlice(self.allocator, line);
        try buffer.appendSlice(self.allocator, "  }\n");
        try buffer.appendSlice(self.allocator, "}\n");

        try file.writeAll(buffer.items);

        const stat = try file.stat();
        self.last_modified = stat.mtime;

        std.debug.print("Configuration saved to: {s}\n", .{self.config_path});
    }
};

/// Configuration utilities
pub const ConfigUtils = struct {
    /// Get configuration value by path (e.g., "database.hnsw_m")
    pub fn getValue(config: *const WdbxConfig, path: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
        var parts = std.mem.splitScalar(u8, path, '.');

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
        std.debug.print("\n🔧 WDBX Configuration Summary\n", .{});
        std.debug.print("═══════════════════════════════\n", .{});
        std.debug.print("Database:\n", .{});
        std.debug.print("  Path: {s}\n", .{config.database.path});
        std.debug.print("  Dimensions: {}\n", .{config.database.dimensions});
        std.debug.print("  HNSW M: {}\n", .{config.database.hnsw_m});
        std.debug.print("  Compression: {s}\n", .{if (config.database.enable_compression) "enabled" else "disabled"});

        std.debug.print("\nServer:\n", .{});
        std.debug.print("  Address: {s}:{}\n", .{ config.server.host, config.server.port });
        std.debug.print("  Max Connections: {}\n", .{config.server.max_connections});
        std.debug.print("  Batch Size: {}\n", .{config.server.max_batch_size});
        std.debug.print("  Page Size: {}\n", .{config.server.default_page_size});

        std.debug.print("\nPerformance:\n", .{});
        std.debug.print("  Worker Threads: {}\n", .{config.performance.worker_threads});
        std.debug.print("  SIMD: {s}\n", .{if (config.performance.enable_simd) "enabled" else "disabled"});
        std.debug.print("  Profiling: {s}\n", .{if (config.performance.enable_profiling) "enabled" else "disabled"});

        std.debug.print("\nMonitoring:\n", .{});
        std.debug.print("  Prometheus: {s}\n", .{if (config.monitoring.enable_prometheus) "enabled" else "disabled"});
        std.debug.print("  CPU Sampling: {s}\n", .{if (config.monitoring.enable_cpu_sampling) "enabled" else "disabled"});
        std.debug.print("  Memory Sampling: {s}\n", .{if (config.monitoring.enable_memory_sampling) "enabled" else "disabled"});
        std.debug.print("═══════════════════════════════\n\n", .{});
    }
};
