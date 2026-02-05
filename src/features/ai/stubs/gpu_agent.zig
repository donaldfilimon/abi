const std = @import("std");

pub const WorkloadType = enum {
    inference,
    training,
    embedding,
    fine_tuning,
    batch_inference,

    pub fn gpuIntensive(self: @This()) bool {
        return switch (self) {
            .training, .fine_tuning => true,
            .inference, .embedding, .batch_inference => false,
        };
    }

    pub fn memoryIntensive(self: @This()) bool {
        return switch (self) {
            .training, .fine_tuning, .batch_inference => true,
            .inference, .embedding => false,
        };
    }

    pub fn name(self: @This()) []const u8 {
        return switch (self) {
            .inference => "Inference",
            .training => "Training",
            .embedding => "Embedding",
            .fine_tuning => "FineTuning",
            .batch_inference => "BatchInference",
        };
    }
};

pub const Priority = enum {
    low,
    normal,
    high,
    critical,

    pub fn weight(self: @This()) f32 {
        return switch (self) {
            .low => 0.25,
            .normal => 1.0,
            .high => 2.0,
            .critical => 4.0,
        };
    }

    pub fn name(self: @This()) []const u8 {
        return switch (self) {
            .low => "Low",
            .normal => "Normal",
            .high => "High",
            .critical => "Critical",
        };
    }
};

pub const GpuAwareRequest = struct {
    prompt: []const u8,
    workload_type: WorkloadType,
    priority: Priority = .normal,
    max_tokens: u32 = 1024,
    temperature: f32 = 0.7,
    memory_hint_mb: ?u64 = null,
    preferred_backend: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
    stream: bool = false,
    timeout_ms: u64 = 0,
};

pub const GpuAwareResponse = struct {
    content: []const u8,
    tokens_generated: u32,
    latency_ms: u64,
    gpu_backend_used: []const u8,
    gpu_memory_used_mb: u64,
    scheduling_confidence: f32,
    energy_estimate_wh: ?f32 = null,
    device_id: u32 = 0,
    truncated: bool = false,
    error_message: ?[]const u8 = null,
};

pub const AgentStats = struct {
    total_requests: u64 = 0,
    gpu_accelerated: u64 = 0,
    cpu_fallback: u64 = 0,
    total_tokens: u64 = 0,
    total_latency_ms: u64 = 0,
    learning_episodes: u64 = 0,
    avg_scheduling_confidence: f32 = 0,
    avg_latency_ms: f32 = 0,
    failed_requests: u64 = 0,
    total_gpu_memory_mb: u64 = 0,

    pub fn updateConfidence(self: *AgentStats, new_confidence: f32) void {
        if (self.gpu_accelerated == 0) {
            self.avg_scheduling_confidence = new_confidence;
        } else {
            const n = @as(f32, @floatFromInt(self.gpu_accelerated));
            self.avg_scheduling_confidence =
                (self.avg_scheduling_confidence * (n - 1) + new_confidence) / n;
        }
    }

    pub fn updateLatency(self: *AgentStats, latency: u64) void {
        if (self.total_requests == 0) {
            self.avg_latency_ms = @floatFromInt(latency);
        } else {
            const n = @as(f32, @floatFromInt(self.total_requests));
            self.avg_latency_ms =
                (self.avg_latency_ms * (n - 1) + @as(f32, @floatFromInt(latency))) / n;
        }
    }

    pub fn successRate(self: AgentStats) f32 {
        if (self.total_requests == 0) return 1.0;
        const successful = self.total_requests - self.failed_requests;
        return @as(f32, @floatFromInt(successful)) / @as(f32, @floatFromInt(self.total_requests));
    }

    pub fn gpuUtilizationRate(self: AgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.gpu_accelerated)) /
            @as(f32, @floatFromInt(self.total_requests));
    }

    pub fn avgTokensPerRequest(self: AgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_tokens)) /
            @as(f32, @floatFromInt(self.total_requests));
    }
};

pub const BackendInfo = struct {
    name: []const u8 = "cpu",
    device_count: u32 = 0,
    total_memory_mb: u64 = 0,
    available_memory_mb: u64 = 0,
    is_healthy: bool = false,
};

pub const LearningStatsInfo = struct {
    episodes: usize = 0,
    avg_episode_reward: f32 = 0,
    exploration_rate: f32 = 0,
    replay_buffer_size: usize = 0,
};

pub const GpuAgent = struct {
    allocator: std.mem.Allocator,
    stats: AgentStats = .{},

    pub fn init(_: std.mem.Allocator) !*@This() {
        return error.AiDisabled;
    }

    pub fn initWithConfig(
        _: std.mem.Allocator,
        _: struct {
            default_timeout_ms: u64 = 30000,
            enable_learning: bool = true,
        },
    ) !*@This() {
        return error.AiDisabled;
    }

    pub fn deinit(_: *@This()) void {}

    pub fn process(_: *@This(), _: GpuAwareRequest) !GpuAwareResponse {
        return error.AiDisabled;
    }

    pub fn getStats(self: *const @This()) AgentStats {
        return self.stats;
    }

    pub fn isGpuEnabled(_: *const @This()) bool {
        return false;
    }

    pub fn isLearningEnabled(_: *const @This()) bool {
        return false;
    }

    pub fn endEpisode(_: *@This()) void {}

    pub fn resetStats(self: *@This()) void {
        self.stats = .{};
    }

    pub fn getBackendsSummary(_: *@This(), _: std.mem.Allocator) ![]const BackendInfo {
        return error.AiDisabled;
    }

    pub fn getLearningStats(_: *@This()) ?LearningStatsInfo {
        return null;
    }
};
