//! Configuration Loader
//!
//! Provides configuration loading from multiple sources with priority:
//! 1. Explicit code configuration (highest priority)
//! 2. Environment variables (ABI_* prefix)
//! 3. Default values (lowest priority)
//!
//! ## Environment Variables
//!
//! | Variable | Description | Default |
//! |----------|-------------|---------|
//! | ABI_LOG_LEVEL | Logging level (debug, info, warn, error) | info |
//! | ABI_GPU_BACKEND | GPU backend (auto, cuda, vulkan, metal) | auto |
//! | ABI_LLM_MODEL_PATH | Path to LLM model file | none |
//! | ABI_DB_PATH | Database file path | abi.db |
//! | ABI_OPENAI_API_KEY | OpenAI API key | none |
//! | ABI_ANTHROPIC_API_KEY | Anthropic API key | none |
//!
//! ## Usage
//!
//! ```zig
//! var loader = ConfigLoader.init(allocator);
//! const config = try loader.load();
//! defer loader.deinit();
//! ```

const std = @import("std");
const mod = @import("mod.zig");
const Config = mod.Config;
const GpuConfig = mod.GpuConfig;
const AiConfig = mod.AiConfig;
const DatabaseConfig = mod.DatabaseConfig;
const build_options = @import("build_options");

/// Configuration loading errors.
pub const LoadError = error{
    InvalidValue,
    MissingRequired,
    ParseError,
    OutOfMemory,
};

/// Configuration loader with environment variable support.
pub const ConfigLoader = struct {
    allocator: std.mem.Allocator,
    allocated_strings: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .allocated_strings = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.allocated_strings.items) |s| {
            self.allocator.free(s);
        }
        self.allocated_strings.deinit(self.allocator);
    }

    /// Load configuration from environment variables merged with defaults.
    pub fn load(self: *Self) LoadError!Config {
        var config = Config.defaults();

        // Apply environment variable overrides
        try self.applyEnvOverrides(&config);

        return config;
    }

    /// Load configuration, merging with an existing base config.
    pub fn loadWithBase(self: *Self, base: Config) LoadError!Config {
        var config = base;
        try self.applyEnvOverrides(&config);
        return config;
    }

    fn applyEnvOverrides(self: *Self, config: *Config) LoadError!void {
        // GPU configuration
        if (build_options.enable_gpu) {
            if (config.gpu == null) config.gpu = GpuConfig.defaults();
            if (self.getEnv("ABI_GPU_BACKEND")) |backend_str| {
                if (parseGpuBackend(backend_str)) |backend| {
                    config.gpu.?.preferred_backend = backend;
                }
            }
        }

        // AI configuration
        if (build_options.enable_ai) {
            if (config.ai == null) config.ai = AiConfig.defaults();

            // LLM model path
            if (self.getEnv("ABI_LLM_MODEL_PATH")) |path| {
                if (config.ai.?.llm == null) {
                    config.ai.?.llm = .{};
                }
                const path_copy = self.allocator.dupe(u8, path) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, path_copy) catch return error.OutOfMemory;
                config.ai.?.llm.?.model_path = path_copy;
            }

            // Temperature
            if (self.getEnv("ABI_LLM_TEMPERATURE")) |temp_str| {
                if (std.fmt.parseFloat(f32, temp_str)) |temp| {
                    config.ai.?.temperature = temp;
                } else |_| {}
            }

            // Max tokens
            if (self.getEnv("ABI_LLM_MAX_TOKENS")) |tokens_str| {
                if (std.fmt.parseInt(u32, tokens_str, 10)) |tokens| {
                    config.ai.?.max_tokens = tokens;
                } else |_| {}
            }
        }

        // Database configuration
        if (build_options.enable_database) {
            if (config.database == null) config.database = DatabaseConfig.defaults();

            if (self.getEnv("ABI_DB_PATH")) |path| {
                const path_copy = self.allocator.dupe(u8, path) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, path_copy) catch return error.OutOfMemory;
                config.database.?.path = path_copy;
            }
        }
    }

    fn getEnv(self: *Self, name: []const u8) ?[]const u8 {
        _ = self;
        // Use std.posix.getenv for build-time safe env access
        return std.posix.getenv(name);
    }

    fn parseGpuBackend(s: []const u8) ?GpuConfig.Backend {
        const map = std.StaticStringMap(GpuConfig.Backend).initComptime(.{
            .{ "auto", .auto },
            .{ "cuda", .cuda },
            .{ "vulkan", .vulkan },
            .{ "metal", .metal },
            .{ "webgpu", .webgpu },
            .{ "opengl", .opengl },
            .{ "none", .none },
        });
        return map.get(s);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ConfigLoader basic usage" {
    var loader = ConfigLoader.init(std.testing.allocator);
    defer loader.deinit();

    const config = try loader.load();
    try mod.validate(config);
}

test "ConfigLoader with base config" {
    var loader = ConfigLoader.init(std.testing.allocator);
    defer loader.deinit();

    const base = Config.minimal();
    const config = try loader.loadWithBase(base);

    // Minimal config should still have no features enabled
    // unless env vars override
    _ = config;
}

test "parseGpuBackend" {
    const loader = ConfigLoader.init(std.testing.allocator);
    _ = loader;

    try std.testing.expectEqual(GpuConfig.Backend.cuda, ConfigLoader.parseGpuBackend("cuda").?);
    try std.testing.expectEqual(GpuConfig.Backend.vulkan, ConfigLoader.parseGpuBackend("vulkan").?);
    try std.testing.expect(ConfigLoader.parseGpuBackend("invalid") == null);
}
