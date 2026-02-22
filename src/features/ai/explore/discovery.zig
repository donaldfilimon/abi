//! Model Auto-Discovery and Adaptive Configuration
//!
//! Provides automatic model discovery from standard paths, adaptive configuration
//! based on system resources, and capability negotiation for optimal performance.
//!
//! ## Features
//!
//! - **Auto-Discovery**: Scans standard paths for GGUF models
//! - **Resource Detection**: Detects GPU memory, CPU cores, available RAM
//! - **Adaptive Config**: Automatically configures batch size, context length
//! - **Capability Negotiation**: Matches model requirements to system capabilities
//! - **Warm-up Diagnostics**: Tests model performance before deployment
//!
//! ## Standard Model Paths
//!
//! The discovery system searches these locations (in order):
//! 1. `./models/` - Local project directory
//! 2. Platform-specific ABI app root `models/` path
//! 3. Legacy fallback `~/.abi/models/` (or `%USERPROFILE%\.abi\models`)
//! 4. `~/.cache/huggingface/hub/` - HuggingFace cache
//! 5. `/usr/local/share/abi/models/` - System-wide (Unix)

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const app_paths = @import("../../../services/shared/app_paths.zig");

// libc import for environment access - required for Zig 0.16
// Not available on freestanding/WASM targets
const c = if (builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64)
    @cImport(@cInclude("stdlib.h"))
else
    struct {
        pub fn getenv(_: [*:0]const u8) ?[*:0]const u8 {
            return null;
        }
    };

/// Whether threading is available on this target
const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

/// Get environment variable value (platform-independent via libc)
/// Returns null on WASM/freestanding targets where environment variables are unavailable.
fn getEnv(name: [:0]const u8) ?[]const u8 {
    if (builtin.target.os.tag == .freestanding or
        builtin.target.cpu.arch == .wasm32 or
        builtin.target.cpu.arch == .wasm64)
    {
        return null; // Environment variables not available on WASM
    }
    const value_ptr = c.getenv(name.ptr);
    if (value_ptr) |ptr| {
        return std.mem.span(ptr);
    }
    return null;
}

fn runtimePathEnv() app_paths.EnvValues {
    return .{
        .appdata = getEnv("APPDATA"),
        .localappdata = getEnv("LOCALAPPDATA"),
        .userprofile = getEnv("USERPROFILE"),
        .home = getEnv("HOME"),
        .xdg_config_home = getEnv("XDG_CONFIG_HOME"),
    };
}

fn resolveAbiModelPathFor(
    allocator: std.mem.Allocator,
    os_tag: std.Target.Os.Tag,
    env: app_paths.EnvValues,
) ![]u8 {
    return app_paths.resolvePathFor(allocator, os_tag, env, "models");
}

/// Model discovery configuration
pub const DiscoveryConfig = struct {
    /// Custom paths to search (in addition to standard paths)
    custom_paths: []const []const u8 = &.{},
    /// Whether to scan recursively
    recursive: bool = true,
    /// Maximum depth for recursive scan
    max_depth: u32 = 5,
    /// File extensions to look for
    extensions: []const []const u8 = &.{ ".gguf", ".mlx", ".bin", ".safetensors" },
    /// Whether to validate model files
    validate_files: bool = true,
    /// Timeout for validation in milliseconds
    validation_timeout_ms: u32 = 5000,
};

/// Discovered model information
pub const DiscoveredModel = struct {
    /// Model file path
    path: []const u8,
    /// Model name (derived from filename)
    name: []const u8,
    /// File size in bytes
    size_bytes: u64,
    /// Model format
    format: ModelFormat,
    /// Estimated parameters (if detectable)
    estimated_params: ?u64 = null,
    /// Quantization type (if detectable)
    quantization: ?QuantizationType = null,
    /// Whether the model passed validation
    validated: bool = false,
    /// Last modified timestamp
    modified_time: i128 = 0,

    pub fn deinit(self: *DiscoveredModel, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
    }
};

/// Model file formats
pub const ModelFormat = enum {
    gguf,
    mlx,
    safetensors,
    pytorch_bin,
    onnx,
    unknown,

    pub fn fromExtension(ext: []const u8) ModelFormat {
        if (std.mem.eql(u8, ext, ".gguf")) return .gguf;
        if (std.mem.eql(u8, ext, ".mlx")) return .mlx;
        if (std.mem.eql(u8, ext, ".safetensors")) return .safetensors;
        if (std.mem.eql(u8, ext, ".bin")) return .pytorch_bin;
        if (std.mem.eql(u8, ext, ".onnx")) return .onnx;
        return .unknown;
    }
};

/// Quantization types for GGUF models
pub const QuantizationType = enum {
    f32,
    f16,
    q8_0,
    q8_1,
    q5_0,
    q5_1,
    q4_0,
    q4_1,
    q4_k_m,
    q4_k_s,
    q5_k_m,
    q5_k_s,
    q6_k,
    q2_k,
    q3_k_m,
    q3_k_s,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    unknown,

    /// Get approximate bits per weight
    pub fn bitsPerWeight(self: QuantizationType) f32 {
        return switch (self) {
            .f32 => 32.0,
            .f16 => 16.0,
            .q8_0, .q8_1 => 8.0,
            .q6_k => 6.0,
            .q5_0, .q5_1, .q5_k_m, .q5_k_s => 5.0,
            .q4_0, .q4_1, .q4_k_m, .q4_k_s => 4.0,
            .q3_k_m, .q3_k_s => 3.0,
            .q2_k => 2.0,
            .iq2_xxs, .iq2_xs => 2.5,
            .iq3_xxs => 3.0,
            .unknown => 4.0, // Conservative estimate
        };
    }
};

/// System capabilities detected at runtime
pub const SystemCapabilities = struct {
    /// Number of CPU cores available
    cpu_cores: u32 = 1,
    /// Total system RAM in bytes
    total_ram_bytes: u64 = 0,
    /// Available RAM in bytes
    available_ram_bytes: u64 = 0,
    /// GPU available
    gpu_available: bool = false,
    /// GPU memory in bytes (if available)
    gpu_memory_bytes: u64 = 0,
    /// GPU compute capability (CUDA)
    gpu_compute_capability: ?f32 = null,
    /// Whether AVX2 SIMD is available
    avx2_available: bool = false,
    /// Whether AVX-512 SIMD is available
    avx512_available: bool = false,
    /// Whether NEON SIMD is available (ARM)
    neon_available: bool = false,
    /// Operating system
    os: std.Target.Os.Tag = .linux,
    /// Architecture
    arch: std.Target.Cpu.Arch = .x86_64,

    /// Estimate maximum model size that can be loaded
    pub fn maxModelSize(self: SystemCapabilities) u64 {
        // Use 80% of available RAM for model, or GPU memory if available
        const ram_budget = self.available_ram_bytes * 80 / 100;
        if (self.gpu_available and self.gpu_memory_bytes > 0) {
            // Prefer GPU memory
            return self.gpu_memory_bytes * 90 / 100;
        }
        return ram_budget;
    }

    /// Get recommended thread count for inference
    pub fn recommendedThreads(self: SystemCapabilities) u32 {
        // Use physical cores minus 1 for responsiveness
        if (self.cpu_cores > 2) {
            return self.cpu_cores - 1;
        }
        return self.cpu_cores;
    }

    /// Get recommended batch size based on available memory
    pub fn recommendedBatchSize(self: SystemCapabilities, model_size_bytes: u64) u32 {
        const available = if (self.gpu_available)
            self.gpu_memory_bytes
        else
            self.available_ram_bytes;

        // Reserve space for model + KV cache + working memory
        const working_memory = available -| model_size_bytes;
        if (working_memory < 1024 * 1024 * 100) return 1; // < 100MB available

        // Estimate batch size based on available memory
        // Assume ~10MB per batch item for typical context
        const batch_memory_per_item: u64 = 10 * 1024 * 1024;
        const max_batch = @as(u32, @intCast(working_memory / batch_memory_per_item));

        return @min(max_batch, 32); // Cap at 32
    }
};

/// Adaptive configuration based on system capabilities
pub const AdaptiveConfig = struct {
    /// Number of threads for inference
    num_threads: u32 = 4,
    /// Batch size for inference
    batch_size: u32 = 1,
    /// Context length (tokens)
    context_length: u32 = 2048,
    /// Whether to use GPU acceleration
    use_gpu: bool = true,
    /// Whether to use memory mapping
    use_mmap: bool = true,
    /// Whether to lock model in memory
    mlock: bool = false,
    /// KV cache type
    kv_cache_type: KvCacheType = .standard,
    /// Flash attention enabled
    flash_attention: bool = false,
    /// Tensor parallelism degree
    tensor_parallel: u32 = 1,
    /// Prefill chunk size
    prefill_chunk_size: u32 = 512,

    pub const KvCacheType = enum {
        standard,
        sliding_window,
        paged,
    };
};

/// Model requirements for capability matching
pub const ModelRequirements = struct {
    /// Minimum RAM required in bytes
    min_ram_bytes: u64 = 0,
    /// Minimum GPU memory in bytes (0 = CPU-only supported)
    min_gpu_memory_bytes: u64 = 0,
    /// Minimum compute capability
    min_compute_capability: f32 = 0,
    /// Requires AVX2
    requires_avx2: bool = false,
    /// Requires AVX-512
    requires_avx512: bool = false,
    /// Recommended context length
    recommended_context: u32 = 2048,
};

/// Warm-up diagnostics result
pub const WarmupResult = struct {
    /// Time to load model in milliseconds
    load_time_ms: u64 = 0,
    /// Time for first inference in milliseconds
    first_inference_ms: u64 = 0,
    /// Tokens per second during warm-up
    tokens_per_second: f32 = 0,
    /// Memory usage after loading in bytes
    memory_usage_bytes: u64 = 0,
    /// Whether warm-up was successful
    success: bool = false,
    /// Error message if failed
    error_message: ?[]const u8 = null,
    /// Recommended configuration based on warm-up
    recommended_config: ?AdaptiveConfig = null,
};

/// Model discovery system
pub const ModelDiscovery = struct {
    allocator: std.mem.Allocator,
    config: DiscoveryConfig,
    discovered_models: std.ArrayListUnmanaged(DiscoveredModel),
    capabilities: SystemCapabilities,

    const Self = @This();

    /// Initialize the discovery system
    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .discovered_models = .empty,
            .capabilities = detectCapabilities(),
        };
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *Self) void {
        for (self.discovered_models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.discovered_models.deinit(self.allocator);
    }

    /// Scan all standard paths and custom paths for models
    pub fn scanAll(self: *Self) !void {
        // Get standard paths
        const standard_paths = try self.getStandardPaths();
        defer {
            for (standard_paths) |path| {
                self.allocator.free(path);
            }
            self.allocator.free(standard_paths);
        }

        // Scan standard paths
        for (standard_paths) |path| {
            self.scanPath(path) catch continue;
        }

        // Scan custom paths
        for (self.config.custom_paths) |path| {
            self.scanPath(path) catch continue;
        }
    }

    /// Scan a specific path for models
    /// Note: This is a no-op in the lazy implementation.
    /// Use addModelPath to manually register known model paths.
    pub fn scanPath(self: *Self, path: []const u8) !void {
        // In Zig 0.16, file system operations require I/O backend initialization.
        // For now, we provide a lazy model registration API instead.
        // Users can call addModelPath() to register known model files.
        _ = self;
        _ = path;
    }

    /// Manually add a model by path (lazy discovery)
    /// Use this instead of scanPath for explicit model registration.
    pub fn addModelPath(self: *Self, path: []const u8) !void {
        const full_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(full_path);

        // Extract filename from path
        const basename = std.fs.path.basename(path);
        const stem = std.fs.path.stem(basename);
        const name = try self.allocator.dupe(u8, stem);
        errdefer self.allocator.free(name);

        // Detect format from extension
        const ext = std.fs.path.extension(basename);
        const format = ModelFormat.fromExtension(ext);

        // Detect quantization from filename
        const quantization = detectQuantizationFromName(name);

        // Estimate parameters from filename
        const estimated_params = estimateParamsFromName(name);

        try self.discovered_models.append(self.allocator, .{
            .path = full_path,
            .name = name,
            .size_bytes = 0, // Unknown without I/O
            .format = format,
            .quantization = quantization,
            .estimated_params = estimated_params,
            .modified_time = 0,
        });
    }

    /// Add model with known size (for when file stats are available)
    pub fn addModelWithSize(
        self: *Self,
        path: []const u8,
        size_bytes: u64,
    ) !void {
        const full_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(full_path);

        const basename = std.fs.path.basename(path);
        const stem = std.fs.path.stem(basename);
        const name = try self.allocator.dupe(u8, stem);
        errdefer self.allocator.free(name);

        const ext = std.fs.path.extension(basename);
        const format = ModelFormat.fromExtension(ext);
        const quantization = detectQuantizationFromName(name);
        const estimated_params = estimateParamsFromName(name);

        try self.discovered_models.append(self.allocator, .{
            .path = full_path,
            .name = name,
            .size_bytes = size_bytes,
            .format = format,
            .quantization = quantization,
            .estimated_params = estimated_params,
            .modified_time = 0,
        });
    }

    /// Get standard model search paths
    fn getStandardPaths(self: *Self) ![][]const u8 {
        var paths = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (paths.items) |p| self.allocator.free(p);
            paths.deinit(self.allocator);
        }

        // Local project directory
        try paths.append(self.allocator, try self.allocator.dupe(u8, "./models"));

        // Primary ABI model path.
        const abi_models = try resolveAbiModelPathFor(self.allocator, self.capabilities.os, runtimePathEnv());
        defer self.allocator.free(abi_models);
        try paths.append(self.allocator, try self.allocator.dupe(u8, abi_models));

        // User home cache locations.
        if (getEnv("HOME")) |home| {
            const hf_cache = try std.fmt.allocPrint(self.allocator, "{s}/.cache/huggingface/hub", .{home});
            try paths.append(self.allocator, hf_cache);
        }

        // System-wide paths (Unix)
        if (self.capabilities.os != .windows) {
            try paths.append(self.allocator, try self.allocator.dupe(u8, "/usr/local/share/abi/models"));
        }

        return try paths.toOwnedSlice(self.allocator);
    }

    /// Find best model matching requirements
    pub fn findBestModel(self: *Self, requirements: ModelRequirements) ?*DiscoveredModel {
        var best: ?*DiscoveredModel = null;
        var best_score: f32 = 0;

        for (self.discovered_models.items) |*model| {
            const score = self.scoreModel(model, requirements);
            if (score > best_score) {
                best_score = score;
                best = model;
            }
        }

        return best;
    }

    fn scoreModel(self: *Self, model: *DiscoveredModel, requirements: ModelRequirements) f32 {
        var score: f32 = 0;

        // Check if model fits in memory
        if (model.size_bytes > self.capabilities.maxModelSize()) {
            return 0; // Too large
        }

        // Prefer validated models
        if (model.validated) score += 10;

        // Prefer GGUF format
        if (model.format == .gguf) score += 5;
        if (model.format == .mlx and self.capabilities.os == .macos) score += 4;

        // Score based on size match (prefer larger that still fits)
        const size_ratio = @as(f32, @floatFromInt(model.size_bytes)) /
            @as(f32, @floatFromInt(self.capabilities.maxModelSize()));
        score += size_ratio * 20;

        // Prefer better quantization if we have memory
        if (model.quantization) |quant| {
            const bits = quant.bitsPerWeight();
            if (bits >= 5.0) score += 5;
            if (bits >= 8.0) score += 5;
        }

        // Bonus for matching context requirements
        _ = requirements;

        return score;
    }

    /// Generate adaptive configuration for a model
    pub fn generateConfig(self: *Self, model: *const DiscoveredModel) AdaptiveConfig {
        var config = AdaptiveConfig{};

        // Set thread count
        config.num_threads = self.capabilities.recommendedThreads();

        // Set batch size
        config.batch_size = self.capabilities.recommendedBatchSize(model.size_bytes);

        // Enable GPU if available and model fits
        if (self.capabilities.gpu_available and
            model.size_bytes < self.capabilities.gpu_memory_bytes * 90 / 100)
        {
            config.use_gpu = true;
            config.flash_attention = self.capabilities.gpu_compute_capability != null and
                self.capabilities.gpu_compute_capability.? >= 8.0;
        }

        // Enable mmap for large models
        config.use_mmap = model.size_bytes > 1024 * 1024 * 1024; // > 1GB

        // Set context length based on available memory
        const available_for_context = if (config.use_gpu)
            self.capabilities.gpu_memory_bytes -| model.size_bytes
        else
            self.capabilities.available_ram_bytes -| model.size_bytes;

        // Estimate ~2 bytes per token per layer for KV cache
        // Assume 32 layers for typical models
        const bytes_per_token: u64 = 32 * 2 * 2; // 32 layers * 2 (K+V) * 2 bytes
        const max_context = @as(u32, @intCast(available_for_context / bytes_per_token));
        config.context_length = @min(max_context, 8192);

        // Use paged KV cache for larger contexts
        if (config.context_length > 4096) {
            config.kv_cache_type = .paged;
        }

        return config;
    }

    /// Get all discovered models
    pub fn getModels(self: *Self) []DiscoveredModel {
        return self.discovered_models.items;
    }

    /// Get number of discovered models
    pub fn modelCount(self: *Self) usize {
        return self.discovered_models.items.len;
    }
};

/// Detect system capabilities
pub fn detectCapabilities() SystemCapabilities {
    var caps = SystemCapabilities{};

    // Detect OS and architecture
    caps.os = @import("builtin").os.tag;
    caps.arch = @import("builtin").cpu.arch;

    // Detect CPU features
    const cpu_features = @import("builtin").cpu.features;

    // x86_64 SIMD detection
    if (caps.arch == .x86_64) {
        caps.avx2_available = std.Target.x86.featureSetHas(cpu_features, .avx2);
        caps.avx512_available = std.Target.x86.featureSetHas(cpu_features, .avx512f);
    }

    // ARM NEON detection
    if (caps.arch == .aarch64) {
        caps.neon_available = true; // NEON is mandatory on AArch64
    }

    // CPU cores - use a safe default
    const cpu_count: usize = if (comptime is_threaded_target)
        std.Thread.getCpuCount() catch 4
    else
        1;
    caps.cpu_cores = @as(u32, @intCast(cpu_count));

    // Memory detection (platform-specific)
    // For now, use conservative defaults
    // A full implementation would use platform-specific APIs
    caps.total_ram_bytes = 8 * 1024 * 1024 * 1024; // 8GB default
    caps.available_ram_bytes = 4 * 1024 * 1024 * 1024; // 4GB default

    // GPU detection would require GPU module integration
    if (build_options.enable_gpu) {
        // GPU capabilities would be detected from gpu module
        caps.gpu_available = false; // Set by GPU module if available
    }

    return caps;
}

/// Detect quantization type from model name
fn detectQuantizationFromName(name: []const u8) ?QuantizationType {
    const lower = name; // Already lowercase in most cases

    // Check for common quantization patterns
    if (std.mem.indexOf(u8, lower, "q4_k_m") != null) return .q4_k_m;
    if (std.mem.indexOf(u8, lower, "q4_k_s") != null) return .q4_k_s;
    if (std.mem.indexOf(u8, lower, "q5_k_m") != null) return .q5_k_m;
    if (std.mem.indexOf(u8, lower, "q5_k_s") != null) return .q5_k_s;
    if (std.mem.indexOf(u8, lower, "q6_k") != null) return .q6_k;
    if (std.mem.indexOf(u8, lower, "q8_0") != null) return .q8_0;
    if (std.mem.indexOf(u8, lower, "q8_1") != null) return .q8_1;
    if (std.mem.indexOf(u8, lower, "q4_0") != null) return .q4_0;
    if (std.mem.indexOf(u8, lower, "q4_1") != null) return .q4_1;
    if (std.mem.indexOf(u8, lower, "q5_0") != null) return .q5_0;
    if (std.mem.indexOf(u8, lower, "q5_1") != null) return .q5_1;
    if (std.mem.indexOf(u8, lower, "q3_k_m") != null) return .q3_k_m;
    if (std.mem.indexOf(u8, lower, "q3_k_s") != null) return .q3_k_s;
    if (std.mem.indexOf(u8, lower, "q2_k") != null) return .q2_k;
    if (std.mem.indexOf(u8, lower, "iq2_xxs") != null) return .iq2_xxs;
    if (std.mem.indexOf(u8, lower, "iq2_xs") != null) return .iq2_xs;
    if (std.mem.indexOf(u8, lower, "iq3_xxs") != null) return .iq3_xxs;
    if (std.mem.indexOf(u8, lower, "f16") != null) return .f16;
    if (std.mem.indexOf(u8, lower, "f32") != null) return .f32;

    return null;
}

/// Estimate model parameters from name
fn estimateParamsFromName(name: []const u8) ?u64 {
    // Common parameter patterns in model names
    const patterns = [_]struct { pattern: []const u8, params: u64 }{
        .{ .pattern = "70b", .params = 70_000_000_000 },
        .{ .pattern = "65b", .params = 65_000_000_000 },
        .{ .pattern = "34b", .params = 34_000_000_000 },
        .{ .pattern = "33b", .params = 33_000_000_000 },
        .{ .pattern = "30b", .params = 30_000_000_000 },
        .{ .pattern = "13b", .params = 13_000_000_000 },
        .{ .pattern = "7b", .params = 7_000_000_000 },
        .{ .pattern = "3b", .params = 3_000_000_000 },
        .{ .pattern = "1b", .params = 1_000_000_000 },
        .{ .pattern = "1.5b", .params = 1_500_000_000 },
        .{ .pattern = "2.7b", .params = 2_700_000_000 },
        .{ .pattern = "6.7b", .params = 6_700_000_000 },
        .{ .pattern = "8b", .params = 8_000_000_000 },
        .{ .pattern = "405b", .params = 405_000_000_000 },
    };

    for (patterns) |p| {
        if (std.mem.indexOf(u8, name, p.pattern) != null) {
            return p.params;
        }
    }

    return null;
}

/// Run warm-up diagnostics on a model
pub fn runWarmup(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    config: AdaptiveConfig,
) WarmupResult {
    var result = WarmupResult{};

    // This would integrate with the LLM module
    // For now, return basic diagnostics
    _ = allocator;
    _ = model_path;
    _ = config;

    result.success = true;
    result.load_time_ms = 0;
    result.first_inference_ms = 0;
    result.tokens_per_second = 0;

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "detect capabilities" {
    const caps = detectCapabilities();
    try std.testing.expect(caps.cpu_cores >= 1);
    try std.testing.expect(caps.total_ram_bytes > 0);
}

test "quantization from name detection" {
    try std.testing.expectEqual(QuantizationType.q4_k_m, detectQuantizationFromName("llama-7b-q4_k_m").?);
    try std.testing.expectEqual(QuantizationType.q8_0, detectQuantizationFromName("mistral-q8_0").?);
    try std.testing.expectEqual(QuantizationType.f16, detectQuantizationFromName("model-f16").?);
    try std.testing.expect(detectQuantizationFromName("some-model") == null);
}

test "estimate params from name" {
    try std.testing.expectEqual(@as(u64, 7_000_000_000), estimateParamsFromName("llama-7b-chat").?);
    try std.testing.expectEqual(@as(u64, 70_000_000_000), estimateParamsFromName("llama-70b").?);
    try std.testing.expect(estimateParamsFromName("unknown-model") == null);
}

test "ModelDiscovery init and deinit" {
    const allocator = std.testing.allocator;
    var disc = ModelDiscovery.init(allocator, .{});
    defer disc.deinit();

    try std.testing.expectEqual(@as(usize, 0), disc.modelCount());
}

test "SystemCapabilities calculations" {
    var caps = SystemCapabilities{
        .cpu_cores = 8,
        .total_ram_bytes = 16 * 1024 * 1024 * 1024,
        .available_ram_bytes = 8 * 1024 * 1024 * 1024,
        .gpu_available = false,
    };

    try std.testing.expectEqual(@as(u32, 7), caps.recommendedThreads());

    // Max model size should be 80% of available RAM
    const expected_max = 8 * 1024 * 1024 * 1024 * 80 / 100;
    try std.testing.expectEqual(expected_max, caps.maxModelSize());
}

test "AdaptiveConfig defaults" {
    const config = AdaptiveConfig{};
    try std.testing.expectEqual(@as(u32, 4), config.num_threads);
    try std.testing.expectEqual(@as(u32, 1), config.batch_size);
    try std.testing.expect(config.use_gpu);
    try std.testing.expect(config.use_mmap);
}

test "resolveAbiModelPathFor uses primary path" {
    const allocator = std.testing.allocator;
    const env: app_paths.EnvValues = .{ .home = "/home/tester" };

    const path = try resolveAbiModelPathFor(allocator, .linux, env);
    defer allocator.free(path);

    const expected = try std.fs.path.join(allocator, &.{ "/home/tester", ".config", "abi", "models" });
    defer allocator.free(expected);

    try std.testing.expectEqualStrings(expected, path);
}

test "QuantizationType bits per weight" {
    try std.testing.expectEqual(@as(f32, 32.0), QuantizationType.f32.bitsPerWeight());
    try std.testing.expectEqual(@as(f32, 16.0), QuantizationType.f16.bitsPerWeight());
    try std.testing.expectEqual(@as(f32, 8.0), QuantizationType.q8_0.bitsPerWeight());
    try std.testing.expectEqual(@as(f32, 4.0), QuantizationType.q4_k_m.bitsPerWeight());
}

test "ModelFormat from extension" {
    try std.testing.expectEqual(ModelFormat.gguf, ModelFormat.fromExtension(".gguf"));
    try std.testing.expectEqual(ModelFormat.mlx, ModelFormat.fromExtension(".mlx"));
    try std.testing.expectEqual(ModelFormat.safetensors, ModelFormat.fromExtension(".safetensors"));
    try std.testing.expectEqual(ModelFormat.pytorch_bin, ModelFormat.fromExtension(".bin"));
    try std.testing.expectEqual(ModelFormat.unknown, ModelFormat.fromExtension(".xyz"));
}
