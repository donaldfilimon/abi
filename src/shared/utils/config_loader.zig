//! Configuration file loader supporting JSON format.
const std = @import("std");

pub const ConfigError = error{
    FileNotFound,
    InvalidFormat,
    ParseError,
    ValidationFailed,
    UnsupportedFormat,
    IOError,
};

pub const ConfigFormat = enum {
    json,
    yaml,
};

pub const AppConfig = struct {
    framework: FrameworkConfig,
    database: DatabaseConfig,
    network: NetworkConfig,
    monitoring: MonitoringConfig,
    security: SecurityConfig,
};

pub const FrameworkConfig = struct {
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_web: bool = true,
    enable_database: bool = true,
    enable_network: bool = false,
    enable_profiling: bool = false,
    worker_threads: usize = 0,
    log_level: []const u8 = "info",
};

pub const DatabaseConfig = struct {
    name: []const u8 = "abi.db",
    max_records: usize = 0,
    persistence_enabled: bool = true,
    persistence_path: []const u8 = "abi_data",
    vector_search_enabled: bool = true,
    shard_count: usize = 1,
    replica_count: usize = 1,
    enable_auto_reindex: bool = true,
    reindex_interval_ms: u64 = 30000,
};

pub const NetworkConfig = struct {
    listen_address: []const u8 = "0.0.0.0",
    port: u16 = 8080,
    max_connections: usize = 1000,
    timeout_seconds: u32 = 30,
    tls_enabled: bool = false,
    tls_cert_path: ?[]const u8 = null,
    tls_key_path: ?[]const u8 = null,
};

pub const MonitoringConfig = struct {
    enable_prometheus: bool = true,
    prometheus_port: u16 = 9090,
    prometheus_path: []const u8 = "/metrics",
    enable_otel: bool = false,
    otel_endpoint: []const u8 = "http://localhost:4318",
    enable_tracing: bool = true,
};

pub const SecurityConfig = struct {
    api_key_required: bool = true,
    enable_rbac: bool = true,
    default_role: []const u8 = "user",
    max_keys_per_user: usize = 10,
    key_rotation_days: u64 = 90,
    tls_enabled: bool = false,
    mtls_enabled: bool = false,
};

pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,
    config: AppConfig,

    pub fn init(allocator: std.mem.Allocator) ConfigLoader {
        return .{
            .allocator = allocator,
            .config = .{
                .framework = .{},
                .database = .{},
                .network = .{},
                .monitoring = .{},
                .security = .{},
            },
        };
    }

    pub fn deinit(self: *ConfigLoader) void {
        self.freeConfigStrings();
        self.* = undefined;
    }

    pub fn loadFromFile(self: *ConfigLoader, path: []const u8) !void {
        const ext = self.getExtension(path);
        const format: ConfigFormat = if (std.mem.eql(u8, ext, ".json"))
            .json
        else if (std.mem.eql(u8, ext, ".yaml") or std.mem.eql(u8, ext, ".yml"))
            .yaml
        else
            return error.UnsupportedFormat;

        var io_backend = std.Io.Threaded.init(self.allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(1024 * 1024));
        defer self.allocator.free(data);

        switch (format) {
            .json => try self.loadJson(data),
            .yaml => return error.UnsupportedFormat,
        }
    }

    pub fn loadFromEnv(self: *ConfigLoader, prefix: []const u8) !void {
        const env_ai = try self.getEnv(prefix, "ENABLE_AI");
        if (env_ai) |val| self.config.framework.enable_ai = val;

        const env_gpu = try self.getEnv(prefix, "ENABLE_GPU");
        if (env_gpu) |val| self.config.framework.enable_gpu = val;

        const env_network = try self.getEnv(prefix, "ENABLE_NETWORK");
        if (env_network) |val| self.config.framework.enable_network = val;

        const env_threads = try self.getEnv(prefix, "WORKER_THREADS");
        if (env_threads) |val| self.config.framework.worker_threads = val;

        const env_port = try self.getEnv(prefix, "PORT");
        if (env_port) |val| self.config.network.port = val;

        const env_log_level = try self.getEnv(prefix, "LOG_LEVEL");
        if (env_log_level) |val| self.config.framework.log_level = try self.allocator.dupe(u8, val);
    }

    pub fn loadJson(self: *ConfigLoader, data: []const u8) !void {
        const parsed = try std.json.parseFromSlice(AppConfig, self.allocator, data);

        self.freeConfigStrings();

        self.config = parsed;
    }

    pub fn saveToFile(self: *ConfigLoader, path: []const u8) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        const stringified = try std.json.stringifyAlloc(self.allocator, self.config, .{ .whitespace = .indent_2 });
        defer self.allocator.free(stringified);

        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, stringified);
    }

    pub fn validate(self: *ConfigLoader) !void {
        if (self.config.framework.worker_threads != 0 and self.config.framework.worker_threads < 1) {
            return ConfigError.ValidationFailed;
        }

        if (self.config.network.port == 0) {
            return ConfigError.ValidationFailed;
        }

        if (self.config.database.shard_count < 1) {
            return ConfigError.ValidationFailed;
        }
    }

    pub fn getFrameworkOptions(self: *const ConfigLoader) struct {
        enable_ai: bool,
        enable_gpu: bool,
        enable_web: bool,
        enable_database: bool,
        enable_network: bool,
        enable_profiling: bool,
    } {
        return .{
            .enable_ai = self.config.framework.enable_ai,
            .enable_gpu = self.config.framework.enable_gpu,
            .enable_web = self.config.framework.enable_web,
            .enable_database = self.config.framework.enable_database,
            .enable_network = self.config.framework.enable_network,
            .enable_profiling = self.config.framework.enable_profiling,
        };
    }

    pub fn getConfig(self: *const ConfigLoader) *const AppConfig {
        return &self.config;
    }

    fn getExtension(self: *ConfigLoader, path: []const u8) []const u8 {
        const last_dot = std.mem.lastIndexOfScalar(u8, path, '.');
        if (last_dot == null) return "";
        return path[last_dot.?..];
    }

    fn getEnv(self: *ConfigLoader, prefix: []const u8, name: []const u8) !?bool {
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, name });
        defer self.allocator.free(env_name);

        const value = std.process.getEnvVar(self.allocator, env_name) catch {
            return null;
        };
        defer self.allocator.free(value);

        return std.ascii.eqlIgnoreCase(value, "true") or std.ascii.eqlIgnoreCase(value, "1");
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

    fn freeConfigStrings(self: *ConfigLoader) void {
        if (self.config.framework.log_level.ptr != @constCast(&"info").ptr) {
            self.allocator.free(self.config.framework.log_level);
        }
        if (self.config.database.name.ptr != @constCast(&"abi.db").ptr) {
            self.allocator.free(self.config.database.name);
        }
        if (self.config.database.persistence_path.ptr != @constCast(&"abi_data").ptr) {
            self.allocator.free(self.config.database.persistence_path);
        }
        if (self.config.network.listen_address.ptr != @constCast(&"0.0.0.0").ptr) {
            self.allocator.free(self.config.network.listen_address);
        }
        if (self.config.monitoring.prometheus_path.ptr != @constCast(&"/metrics").ptr) {
            self.allocator.free(self.config.monitoring.prometheus_path);
        }
        if (self.config.monitoring.otel_endpoint.ptr != @constCast(&"http://localhost:4318").ptr) {
            self.allocator.free(self.config.monitoring.otel_endpoint);
        }
        if (self.config.security.default_role.ptr != @constCast(&"user").ptr) {
            self.allocator.free(self.config.security.default_role);
        }
    }
};

test "config loader init" {
    const allocator = std.testing.allocator;
    var loader = ConfigLoader.init(allocator);
    defer loader.deinit();

    try std.testing.expect(loader.config.framework.enable_ai);
    try std.testing.expectEqual(@as(usize, 0), loader.config.framework.worker_threads);
}

test "config load json" {
    const allocator = std.testing.allocator;
    var loader = ConfigLoader.init(allocator);
    defer loader.deinit();

    const json =
        \\{
        \\  "framework": {
        \\    "enable_ai": false,
        \\    "worker_threads": 4
        \\  },
        \\  "network": {
        \\    "port": 9000
        \\  }
        \\}
    ;

    try loader.loadJson(json);
    try std.testing.expect(!loader.config.framework.enable_ai);
    try std.testing.expectEqual(@as(usize, 4), loader.config.framework.worker_threads);
    try std.testing.expectEqual(@as(u16, 9000), loader.config.network.port);
}

test "config validate" {
    const allocator = std.testing.allocator;
    var loader = ConfigLoader.init(allocator);
    defer loader.deinit();

    loader.config.network.port = 8080;
    try loader.validate();

    loader.config.network.port = 0;
    try std.testing.expectError(ConfigError.ValidationFailed, loader.validate());
}
