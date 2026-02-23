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
//! | ABI_GPU_BACKEND | GPU backend (auto, cuda, vulkan, metal, webgpu, tpu, none) | auto |
//! | ABI_LLM_MODEL_PATH | Path to LLM model file | none |
//! | ABI_DB_PATH | Database file path | abi.db |
//! | ABI_OPENAI_API_KEY | OpenAI API key | none |
//! | ABI_ANTHROPIC_API_KEY | Anthropic API key | none |
//! | ABI_LSP_ZLS_PATH | ZLS binary path | zls |
//! | ABI_LSP_ZIG_EXE_PATH | Zig compiler path for ZLS | none |
//! | ABI_LSP_WORKSPACE_ROOT | LSP workspace root | none |
//! | ABI_LSP_LOG_LEVEL | ZLS log level (info, warn, error, debug) | info |
//! | ABI_LSP_ENABLE_SNIPPETS | Enable completion snippets (true/false) | true |
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
const LspConfig = mod.LspConfig;
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
            // Securely wipe strings that may contain sensitive data (paths, tokens)
            // before freeing to prevent memory forensics
            std.crypto.secureZero(u8, @constCast(s));
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
                    config.gpu.?.backend = backend;
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

            // Temperature and max_tokens are runtime InferenceConfig settings,
            // not static AiConfig/LlmConfig fields. Set via llm.Engine.init().
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

        // LSP configuration (ZLS)
        if (self.getEnv("ABI_LSP_ZLS_PATH") != null or
            self.getEnv("ABI_LSP_ZIG_EXE_PATH") != null or
            self.getEnv("ABI_LSP_WORKSPACE_ROOT") != null or
            self.getEnv("ABI_LSP_LOG_LEVEL") != null or
            self.getEnv("ABI_LSP_ENABLE_SNIPPETS") != null)
        {
            if (config.lsp == null) config.lsp = LspConfig.defaults();

            if (self.getEnv("ABI_LSP_ZLS_PATH")) |path| {
                const path_copy = self.allocator.dupe(u8, path) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, path_copy) catch return error.OutOfMemory;
                config.lsp.?.zls_path = path_copy;
            }

            if (self.getEnv("ABI_LSP_ZIG_EXE_PATH")) |path| {
                const path_copy = self.allocator.dupe(u8, path) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, path_copy) catch return error.OutOfMemory;
                config.lsp.?.zig_exe_path = path_copy;
            }

            if (self.getEnv("ABI_LSP_WORKSPACE_ROOT")) |path| {
                const path_copy = self.allocator.dupe(u8, path) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, path_copy) catch return error.OutOfMemory;
                config.lsp.?.workspace_root = path_copy;
            }

            if (self.getEnv("ABI_LSP_LOG_LEVEL")) |level| {
                const level_copy = self.allocator.dupe(u8, level) catch return error.OutOfMemory;
                self.allocated_strings.append(self.allocator, level_copy) catch return error.OutOfMemory;
                config.lsp.?.log_level = level_copy;
            }

            if (self.getEnv("ABI_LSP_ENABLE_SNIPPETS")) |flag| {
                if (parseBool(flag)) |value| {
                    config.lsp.?.enable_snippets = value;
                }
            }
        }
    }

    fn getEnv(self: *Self, name: []const u8) ?[]const u8 {
        _ = self;
        var key_buf: [256]u8 = undefined;
        const key_len = @min(name.len, 255);
        @memcpy(key_buf[0..key_len], name[0..key_len]);
        key_buf[key_len] = 0;
        const key_z: [*:0]const u8 = @ptrCast(&key_buf);
        const ptr = std.c.getenv(key_z) orelse return null;
        return std.mem.span(ptr);
    }

    fn parseGpuBackend(s: []const u8) ?GpuConfig.Backend {
        const map = std.StaticStringMap(GpuConfig.Backend).initComptime(.{
            .{ "auto", .auto },
            .{ "cuda", .cuda },
            .{ "vulkan", .vulkan },
            .{ "metal", .metal },
            .{ "webgpu", .webgpu },
            .{ "opengl", .opengl },
            .{ "tpu", .tpu },
            .{ "none", .cpu }, // no GPU → CPU fallback
        });
        return map.get(s);
    }

    fn parseBool(s: []const u8) ?bool {
        if (std.ascii.eqlIgnoreCase(s, "true") or
            std.ascii.eqlIgnoreCase(s, "yes") or
            std.ascii.eqlIgnoreCase(s, "1"))
        {
            return true;
        }
        if (std.ascii.eqlIgnoreCase(s, "false") or
            std.ascii.eqlIgnoreCase(s, "no") or
            std.ascii.eqlIgnoreCase(s, "0"))
        {
            return false;
        }
        return null;
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
    try std.testing.expectEqual(GpuConfig.Backend.tpu, ConfigLoader.parseGpuBackend("tpu").?);
    try std.testing.expectEqual(GpuConfig.Backend.cpu, ConfigLoader.parseGpuBackend("none").?);
    try std.testing.expect(ConfigLoader.parseGpuBackend("invalid") == null);
}

test "parseGpuBackend handles all valid backends" {
    const backends = [_][]const u8{ "auto", "cuda", "vulkan", "metal", "webgpu", "opengl", "tpu", "none" };
    for (backends) |backend| {
        const result = ConfigLoader.parseGpuBackend(backend);
        try std.testing.expect(result != null);
    }
}

test "ConfigLoader handles missing env vars gracefully" {
    var loader = ConfigLoader.init(std.testing.allocator);
    defer loader.deinit();

    const config = try loader.load();
    // Should use defaults when env vars are not set
    if (build_options.enable_gpu) {
        try std.testing.expect(config.gpu != null);
    }
}

test "ConfigLoader defaults are valid" {
    var loader = ConfigLoader.init(std.testing.allocator);
    defer loader.deinit();

    const config = try loader.load();
    try mod.validate(config);
}
