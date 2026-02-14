const std = @import("std");

pub const DiscoveryConfig = struct {
    custom_paths: []const []const u8 = &.{},
    recursive: bool = true,
    max_depth: u32 = 5,
    extensions: []const []const u8 = &.{ ".gguf", ".mlx", ".bin", ".safetensors" },
    validate_files: bool = true,
    validation_timeout_ms: u32 = 5000,
};

pub const ModelFormat = enum {
    gguf,
    mlx,
    safetensors,
    pytorch_bin,
    onnx,
    unknown,

    pub fn fromExtension(_: []const u8) ModelFormat {
        return .unknown;
    }
};

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
            .unknown => 4.0,
        };
    }
};

pub const DiscoveredModel = struct {
    path: []const u8 = "",
    name: []const u8 = "",
    size_bytes: u64 = 0,
    format: ModelFormat = .unknown,
    estimated_params: ?u64 = null,
    quantization: ?QuantizationType = null,
    validated: bool = false,
    modified_time: i128 = 0,

    pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
};

pub const SystemCapabilities = struct {
    cpu_cores: u32 = 1,
    total_ram_bytes: u64 = 0,
    available_ram_bytes: u64 = 0,
    gpu_available: bool = false,
    gpu_memory_bytes: u64 = 0,
    gpu_compute_capability: ?f32 = null,
    avx2_available: bool = false,
    avx512_available: bool = false,
    neon_available: bool = false,
    os: std.Target.Os.Tag = .linux,
    arch: std.Target.Cpu.Arch = .x86_64,

    pub fn maxModelSize(self: @This()) u64 {
        return self.available_ram_bytes * 80 / 100;
    }

    pub fn recommendedThreads(self: @This()) u32 {
        if (self.cpu_cores > 2) return self.cpu_cores - 1;
        return self.cpu_cores;
    }

    pub fn recommendedBatchSize(_: @This(), _: u64) u32 {
        return 1;
    }
};

pub const AdaptiveConfig = struct {
    num_threads: u32 = 4,
    batch_size: u32 = 1,
    context_length: u32 = 2048,
    use_gpu: bool = true,
    use_mmap: bool = true,
    mlock: bool = false,
    kv_cache_type: KvCacheType = .standard,
    flash_attention: bool = false,
    tensor_parallel: u32 = 1,
    prefill_chunk_size: u32 = 512,

    pub const KvCacheType = enum {
        standard,
        sliding_window,
        paged,
    };
};

pub const ModelRequirements = struct {
    min_ram_bytes: u64 = 0,
    min_gpu_memory_bytes: u64 = 0,
    min_compute_capability: f32 = 0,
    requires_avx2: bool = false,
    requires_avx512: bool = false,
    recommended_context: u32 = 2048,
};

pub const WarmupResult = struct {
    load_time_ms: u64 = 0,
    first_inference_ms: u64 = 0,
    tokens_per_second: f32 = 0,
    memory_usage_bytes: u64 = 0,
    success: bool = false,
    error_message: ?[]const u8 = null,
    recommended_config: ?AdaptiveConfig = null,
};

pub const ModelDiscovery = struct {
    allocator: std.mem.Allocator,
    config: DiscoveryConfig,
    discovered_models: std.ArrayListUnmanaged(DiscoveredModel) = .empty,
    capabilities: SystemCapabilities = .{},

    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) @This() {
        return .{
            .allocator = allocator,
            .config = config,
            .discovered_models = .empty,
            .capabilities = detectCapabilities(),
        };
    }

    pub fn deinit(self: *@This()) void {
        for (self.discovered_models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.discovered_models.deinit(self.allocator);
    }

    pub fn scanAll(_: *@This()) !void {
        return error.AiDisabled;
    }

    pub fn scanPath(_: *@This(), _: []const u8) !void {
        return error.AiDisabled;
    }

    pub fn addModelPath(_: *@This(), _: []const u8) !void {
        return error.AiDisabled;
    }

    pub fn addModelWithSize(_: *@This(), _: []const u8, _: u64) !void {
        return error.AiDisabled;
    }

    pub fn getModels(self: *@This()) []DiscoveredModel {
        return self.discovered_models.items;
    }

    pub fn modelCount(self: *@This()) usize {
        return self.discovered_models.items.len;
    }

    pub fn findBestModel(_: *@This(), _: ModelRequirements) ?*DiscoveredModel {
        return null;
    }

    pub fn generateConfig(_: *@This(), _: *const DiscoveredModel) AdaptiveConfig {
        return .{};
    }
};

pub fn detectCapabilities() SystemCapabilities {
    return .{};
}

pub fn runWarmup(_: []const u8, _: AdaptiveConfig) WarmupResult {
    return .{};
}
