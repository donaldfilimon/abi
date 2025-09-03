//! Configuration loader with environment variable support

const std = @import("std");
const core = @import("core/mod.zig");

/// Load configuration from multiple sources
pub fn loadConfig(allocator: std.mem.Allocator) !*core.config.Config {
    var config = try allocator.create(core.config.Config);
    errdefer allocator.destroy(config);
    
    config.* = core.config.Config.init(allocator);
    errdefer config.deinit();
    
    // 1. Load defaults
    try loadDefaults(config);
    
    // 2. Load from default config file if exists
    if (std.fs.cwd().openFile("config.toml", .{})) |file| {
        file.close();
        const loaded = try core.config.Config.load(allocator, "config.toml");
        try mergeConfig(config, &loaded);
        loaded.deinit();
    } else |_| {
        // Try loading from config directory
        if (std.fs.cwd().openFile("config/default.toml", .{})) |file| {
            file.close();
            const loaded = try core.config.Config.load(allocator, "config/default.toml");
            try mergeConfig(config, &loaded);
            loaded.deinit();
        } else |_| {}
    }
    
    // 3. Override with environment variables
    try loadEnvironmentVariables(config);
    
    // 4. Validate configuration
    try validateConfig(config);
    
    return config;
}

/// Load default configuration values
fn loadDefaults(config: *core.config.Config) !void {
    // Database defaults
    try config.setString("database.path", "wdbx.db");
    try config.setInt("database.max_file_size", 1024 * 1024 * 1024);
    try config.setInt("database.page_size", 4096);
    try config.setInt("database.cache_size", 100 * 1024 * 1024);
    try config.setBool("database.enable_compression", false);
    try config.setInt("database.compression_level", 6);
    try config.setBool("database.auto_checkpoint", true);
    try config.setInt("database.checkpoint_interval", 1000);
    
    // Server defaults
    try config.setString("server.host", "127.0.0.1");
    try config.setInt("server.port", 8080);
    try config.setInt("server.max_connections", 100);
    try config.setInt("server.request_timeout", 30);
    try config.setBool("server.enable_cors", true);
    
    // AI defaults
    try config.setString("ai.model_path", "models/");
    try config.setInt("ai.embedding_dimensions", 768);
    try config.setInt("ai.batch_size", 32);
    try config.setFloat("ai.learning_rate", 0.001);
    
    // Performance defaults
    try config.setInt("performance.thread_count", 0);
    try config.setBool("performance.use_simd", true);
    try config.setInt("performance.batch_size", 1000);
    try config.setInt("performance.memory_pool_size", 256 * 1024 * 1024);
    
    // Logging defaults
    try config.setString("logging.level", "info");
    try config.setString("logging.file", "wdbx.log");
    try config.setBool("logging.console", true);
    try config.setBool("logging.use_color", true);
    
    // Security defaults
    try config.setBool("security.enable_auth", false);
    try config.setString("security.auth_method", "token");
    try config.setInt("security.token_expiration", 3600);
    
    // Monitoring defaults
    try config.setBool("monitoring.enable_metrics", true);
    try config.setInt("monitoring.metrics_interval", 60);
    try config.setString("monitoring.metrics_format", "prometheus");
    
    // Plugin defaults
    try config.setBool("plugins.enable_plugins", true);
    try config.setString("plugins.plugin_dir", "plugins/");
    try config.setBool("plugins.auto_load", true);
}

/// Load configuration from environment variables
fn loadEnvironmentVariables(config: *core.config.Config) !void {
    const env_map = std.process.getEnvMap(config.allocator) catch return;
    defer env_map.deinit();
    
    var it = env_map.iterator();
    while (it.next()) |entry| {
        if (std.mem.startsWith(u8, entry.key_ptr.*, "WDBX_")) {
            const key = entry.key_ptr.*[5..]; // Skip "WDBX_" prefix
            const value = entry.value_ptr.*;
            
            // Convert UPPER_CASE to lower.case
            const config_key = try convertEnvKey(config.allocator, key);
            defer config.allocator.free(config_key);
            
            // Try to parse value as different types
            if (std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "false")) {
                try config.setBool(config_key, std.mem.eql(u8, value, "true"));
            } else if (std.fmt.parseInt(i64, value, 10)) |int_val| {
                try config.setInt(config_key, int_val);
            } else if (std.fmt.parseFloat(f64, value)) |float_val| {
                try config.setFloat(config_key, float_val);
            } else {
                try config.setString(config_key, value);
            }
        }
    }
}

/// Convert environment variable key to config key format
fn convertEnvKey(allocator: std.mem.Allocator, env_key: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();
    
    for (env_key) |char| {
        if (char == '_') {
            try result.append('.');
        } else if (char >= 'A' and char <= 'Z') {
            try result.append(char + 32); // Convert to lowercase
        } else {
            try result.append(char);
        }
    }
    
    return result.toOwnedSlice();
}

/// Merge two configurations
fn mergeConfig(dest: *core.config.Config, src: *const core.config.Config) !void {
    var it = src.values.iterator();
    while (it.next()) |entry| {
        const value_copy = try entry.value_ptr.clone(dest.allocator);
        try dest.set(entry.key_ptr.*, value_copy);
    }
}

/// Validate configuration values
fn validateConfig(config: *core.config.Config) !void {
    // Validate database settings
    if (config.getInt("database.cache_size")) |cache_size| {
        if (cache_size < 1024 * 1024) { // Minimum 1MB
            return error.InvalidCacheSize;
        }
    }
    
    // Validate server settings
    if (config.getInt("server.port")) |port| {
        if (port < 1 or port > 65535) {
            return error.InvalidPort;
        }
    }
    
    // Validate performance settings
    if (config.getInt("performance.thread_count")) |threads| {
        if (threads < 0) {
            return error.InvalidThreadCount;
        }
    }
    
    // Validate AI settings
    if (config.getInt("ai.embedding_dimensions")) |dims| {
        if (dims < 1 or dims > 4096) {
            return error.InvalidEmbeddingDimensions;
        }
    }
}

/// Configuration watcher for hot reloading
pub const ConfigWatcher = struct {
    allocator: std.mem.Allocator,
    config: *core.config.Config,
    path: []const u8,
    last_modified: i128,
    thread: ?std.Thread,
    running: std.atomic.Value(bool),
    
    pub fn init(allocator: std.mem.Allocator, config: *core.config.Config, path: []const u8) !ConfigWatcher {
        const stat = try std.fs.cwd().statFile(path);
        
        return .{
            .allocator = allocator,
            .config = config,
            .path = path,
            .last_modified = stat.mtime,
            .thread = null,
            .running = std.atomic.Value(bool).init(false),
        };
    }
    
    pub fn start(self: *ConfigWatcher) !void {
        self.running.store(true, .release);
        self.thread = try std.Thread.spawn(.{}, watchLoop, .{self});
    }
    
    pub fn stop(self: *ConfigWatcher) void {
        self.running.store(false, .release);
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }
    }
    
    fn watchLoop(self: *ConfigWatcher) void {
        while (self.running.load(.acquire)) {
            std.time.sleep(1 * std.time.ns_per_s); // Check every second
            
            const stat = std.fs.cwd().statFile(self.path) catch continue;
            
            if (stat.mtime > self.last_modified) {
                self.last_modified = stat.mtime;
                
                // Reload configuration
                const new_config = core.config.Config.load(self.allocator, self.path) catch {
                    std.log.err("Failed to reload configuration", .{});
                    continue;
                };
                defer new_config.deinit();
                
                // Merge with existing config
                mergeConfig(self.config, &new_config) catch {
                    std.log.err("Failed to merge configuration", .{});
                    continue;
                };
                
                std.log.info("Configuration reloaded from {s}", .{self.path});
            }
        }
    }
};

test "config loader" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Test environment key conversion
    const converted = try convertEnvKey(allocator, "DATABASE_CACHE_SIZE");
    defer allocator.free(converted);
    try testing.expectEqualStrings("database.cache.size", converted);
    
    // Test config loading
    const config = try loadConfig(allocator);
    defer {
        config.deinit();
        allocator.destroy(config);
    }
    
    // Verify defaults are loaded
    try testing.expectEqualStrings("wdbx.db", config.getString("database.path").?);
    try testing.expectEqual(@as(i64, 8080), config.getInt("server.port").?);
}

test "config validation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var config = core.config.Config.init(allocator);
    defer config.deinit();
    
    // Test invalid port
    try config.setInt("server.port", 70000);
    try testing.expectError(error.InvalidPort, validateConfig(&config));
    
    // Test valid configuration
    try config.setInt("server.port", 8080);
    try config.setInt("database.cache_size", 10 * 1024 * 1024);
    try validateConfig(&config);
}