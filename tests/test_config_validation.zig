//! Configuration Validation Tests
//!
//! Comprehensive tests for the enhanced configuration validation system.

const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const expectError = testing.expectError;

const wdbx = @import("abi").wdbx;
const config = wdbx.config;
const core = wdbx.core;

test "ConfigValidator - valid configuration passes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    // Create a valid configuration
    var valid_config = config.WdbxConfig{};
    valid_config.database.path = "test.db";
    valid_config.database.dimensions = 128;
    valid_config.database.hnsw_m = 16;
    valid_config.database.hnsw_ef_construction = 200;
    valid_config.database.hnsw_ef_search = 50;

    // Should pass validation
    try validator.validateConfig(&valid_config);
}

test "ConfigValidator - invalid database path fails" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var invalid_config = config.WdbxConfig{};
    invalid_config.database.path = ""; // Invalid empty path

    try expectError(config.ConfigValidationError.InvalidDatabasePath, validator.validateConfig(&invalid_config));
}

test "ConfigValidator - invalid dimensions fail" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    // Test zero dimensions
    var invalid_config = config.WdbxConfig{};
    invalid_config.database.path = "test.db";
    invalid_config.database.dimensions = 0;

    try expectError(config.ConfigValidationError.InvalidDimensions, validator.validateConfig(&invalid_config));

    // Test excessive dimensions
    invalid_config.database.dimensions = 5000;
    try expectError(config.ConfigValidationError.InvalidDimensions, validator.validateConfig(&invalid_config));
}

test "ConfigValidator - invalid HNSW parameters fail" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var invalid_config = config.WdbxConfig{};
    invalid_config.database.path = "test.db";
    invalid_config.database.dimensions = 128;

    // Test invalid M parameter
    invalid_config.database.hnsw_m = 0;
    try expectError(config.ConfigValidationError.InvalidHnswParameters, validator.validateConfig(&invalid_config));

    invalid_config.database.hnsw_m = 200; // Too high
    try expectError(config.ConfigValidationError.InvalidHnswParameters, validator.validateConfig(&invalid_config));

    // Test ef_construction < ef_search
    invalid_config.database.hnsw_m = 16;
    invalid_config.database.hnsw_ef_construction = 50;
    invalid_config.database.hnsw_ef_search = 100;
    try expectError(config.ConfigValidationError.InvalidHnswParameters, validator.validateConfig(&invalid_config));
}

test "ConfigValidator - invalid server configuration fails" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var invalid_config = config.WdbxConfig{};
    invalid_config.database.path = "test.db";
    invalid_config.database.dimensions = 128;

    // Test invalid port
    invalid_config.server.port = 0;
    try expectError(config.ConfigValidationError.InvalidPortNumber, validator.validateConfig(&invalid_config));

    // Valid max port (ensure no error)
    invalid_config.server.port = 65535;
    try validator.validateConfig(&invalid_config);

    // Test invalid host
    invalid_config.server.port = 8080;
    invalid_config.server.host = "";
    try expectError(config.ConfigValidationError.InvalidHostAddress, validator.validateConfig(&invalid_config));
}

test "ConfigValidator - cross-section validation detects conflicts" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var conflicting_config = config.WdbxConfig{};
    conflicting_config.database.path = "test.db";
    conflicting_config.database.dimensions = 128;
    conflicting_config.server.port = 8080;
    conflicting_config.monitoring.enable_prometheus = true;
    conflicting_config.monitoring.prometheus_port = 8080; // Same as server port

    try expectError(config.ConfigValidationError.ConfigConflict, validator.validateConfig(&conflicting_config));
}

test "ConfigValidator - resource overallocation detection" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var overallocated_config = config.WdbxConfig{};
    overallocated_config.database.path = "test.db";
    overallocated_config.database.dimensions = 128;
    overallocated_config.database.memory_map_size = 512 * 1024 * 1024; // 512MB
    overallocated_config.performance.memory_limit_mb = 100; // Too small for database

    try expectError(config.ConfigValidationError.ResourceOverallocation, validator.validateConfig(&overallocated_config));
}

test "ConfigValidator - compression validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var config_with_compression = config.WdbxConfig{};
    config_with_compression.database.path = "test.db";
    config_with_compression.database.dimensions = 128;
    config_with_compression.database.enable_compression = true;
    config_with_compression.database.compression_algorithm = "invalid_algo";

    try expectError(config.ConfigValidationError.InvalidCompressionSettings, validator.validateConfig(&config_with_compression));

    // Test valid compression algorithms
    config_with_compression.database.compression_algorithm = "lz4";
    try validator.validateConfig(&config_with_compression);

    config_with_compression.database.compression_algorithm = "zstd";
    try validator.validateConfig(&config_with_compression);

    config_with_compression.database.compression_algorithm = "gzip";
    try validator.validateConfig(&config_with_compression);
}

test "ConfigValidator - logging validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var config_with_logging = config.WdbxConfig{};
    config_with_logging.database.path = "test.db";
    config_with_logging.database.dimensions = 128;

    // Test invalid log level
    config_with_logging.logging.level = "invalid_level";
    try expectError(config.ConfigValidationError.InvalidLogLevel, validator.validateConfig(&config_with_logging));

    // Test invalid output
    config_with_logging.logging.level = "info";
    config_with_logging.logging.output = "invalid_output";
    try expectError(config.ConfigValidationError.InvalidLogOutput, validator.validateConfig(&config_with_logging));

    // Test file logging without path
    config_with_logging.logging.output = "file";
    config_with_logging.logging.file_path = "";
    try expectError(config.ConfigValidationError.InvalidLogSettings, validator.validateConfig(&config_with_logging));
}

test "ConfigValidator - validation report generation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var valid_config = config.WdbxConfig{};
    valid_config.database.path = "test.db";
    valid_config.database.dimensions = 128;
    valid_config.database.enable_compression = true;
    valid_config.database.compression_algorithm = "lz4";
    valid_config.database.enable_sharding = true;
    valid_config.database.shard_count = 4;

    const report = try validator.generateValidationReport(&valid_config);
    defer allocator.free(report);

    // Check that report contains expected sections
    try expect(std.mem.indexOf(u8, report, "Configuration Validation Report") != null);
    try expect(std.mem.indexOf(u8, report, "Database Configuration") != null);
    try expect(std.mem.indexOf(u8, report, "Server Configuration") != null);
    try expect(std.mem.indexOf(u8, report, "Performance Configuration") != null);
    try expect(std.mem.indexOf(u8, report, "Security Configuration") != null);
    try expect(std.mem.indexOf(u8, report, "Configuration validation passed") != null);
}

test "ConfigManager - enhanced validation integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a temporary config file
    const config_path = "test_config.json";
    defer std.fs.cwd().deleteFile(config_path) catch {};

    const config_content =
        \\{
        \\  "database": {
        \\    "path": "test.db",
        \\    "dimensions": 128,
        \\    "hnsw_m": 16
        \\  },
        \\  "server": {
        \\    "host": "127.0.0.1",
        \\    "port": 8080
        \\  }
        \\}
    ;

    const file = try std.fs.cwd().createFile(config_path, .{});
    defer file.close();
    try file.writeAll(config_content);

    // Test config manager with enhanced validation
    var manager = try config.ConfigManager.init(allocator, config_path);
    defer manager.deinit();

    // Validation should pass
    try manager.validate();

    const cfg = manager.getConfig();
    try expect(cfg.database.dimensions == 128);
    try expect(cfg.server.port == 8080);
}

test "ConfigManager - validation failure handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a temporary config file with invalid data
    const config_path = "test_invalid_config.json";
    defer std.fs.cwd().deleteFile(config_path) catch {};

    const invalid_config_content =
        \\{
        \\  "database": {
        \\    "path": "",
        \\    "dimensions": 0
        \\  },
        \\  "server": {
        \\    "port": 0
        \\  }
        \\}
    ;

    const file = try std.fs.cwd().createFile(config_path, .{});
    defer file.close();
    try file.writeAll(invalid_config_content);

    // Test config manager with invalid configuration - init should fail
    try expectError(core.WdbxError.ConfigurationValidationFailed, config.ConfigManager.init(allocator, config_path));
}

test "ConfigValidator - security validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var security_config = config.WdbxConfig{};
    security_config.database.path = "test.db";
    security_config.database.dimensions = 128;

    // Test rate limiting validation
    security_config.security.rate_limit_enabled = true;
    security_config.security.rate_limit_requests_per_minute = 0; // Invalid
    try expectError(config.ConfigValidationError.InvalidRateLimitConfig, validator.validateConfig(&security_config));

    // Test TLS validation
    security_config.security.rate_limit_requests_per_minute = 1000;
    security_config.security.enable_tls = true;
    security_config.security.tls_cert_path = ""; // Invalid empty path
    try expectError(config.ConfigValidationError.InvalidTlsConfig, validator.validateConfig(&security_config));

    // Test API key validation
    security_config.security.enable_tls = false;
    security_config.server.enable_auth = true;
    security_config.server.jwt_secret = "short"; // Too short
    try expectError(config.ConfigValidationError.InvalidAuthConfig, validator.validateConfig(&security_config));
}

test "ConfigValidator - monitoring validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var monitoring_config = config.WdbxConfig{};
    monitoring_config.database.path = "test.db";
    monitoring_config.database.dimensions = 128;

    // Test Prometheus validation
    monitoring_config.monitoring.enable_prometheus = true;
    monitoring_config.monitoring.prometheus_port = 0; // Invalid
    try expectError(config.ConfigValidationError.InvalidPrometheusConfig, validator.validateConfig(&monitoring_config));

    monitoring_config.monitoring.prometheus_port = 9090;
    monitoring_config.monitoring.prometheus_path = ""; // Invalid empty path
    try expectError(config.ConfigValidationError.InvalidPrometheusConfig, validator.validateConfig(&monitoring_config));

    // Test sampling validation
    monitoring_config.monitoring.prometheus_path = "/metrics";
    monitoring_config.monitoring.enable_cpu_sampling = true;
    monitoring_config.monitoring.sampling_interval_ms = 50; // Too low
    try expectError(config.ConfigValidationError.InvalidSamplingConfig, validator.validateConfig(&monitoring_config));
}

test "ConfigValidator - performance validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var validator = config.ConfigValidator.init(allocator);

    var perf_config = config.WdbxConfig{};
    perf_config.database.path = "test.db";
    perf_config.database.dimensions = 128;

    // Test thread count validation
    perf_config.performance.worker_threads = 300; // Too many
    try expectError(config.ConfigValidationError.InvalidThreadCount, validator.validateConfig(&perf_config));

    // Test memory limit validation
    perf_config.performance.worker_threads = 8;
    perf_config.performance.memory_limit_mb = 50; // Too low
    try expectError(config.ConfigValidationError.InvalidMemoryLimits, validator.validateConfig(&perf_config));

    // Test SIMD alignment validation
    perf_config.performance.memory_limit_mb = 1024;
    perf_config.performance.enable_simd = true;
    perf_config.performance.simd_alignment = 48; // Invalid alignment
    try expectError(config.ConfigValidationError.InvalidMemorySettings, validator.validateConfig(&perf_config));

    // Test cache validation
    perf_config.performance.simd_alignment = 32;
    perf_config.performance.query_cache_size = 0; // Invalid
    try expectError(config.ConfigValidationError.InvalidCacheSettings, validator.validateConfig(&perf_config));
}
