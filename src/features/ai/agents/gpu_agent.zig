//! GPU-Aware Self-Learning Agent
//!
//! Integrates AI agent capabilities with GPU scheduling,
//! enabling the agent to learn optimal GPU resource allocation
//! through reinforcement learning principles.
//!
//! ## Overview
//!
//! The GPU-Aware Agent combines AI orchestration with GPU scheduling to:
//!
//! - Route workloads to optimal GPU backends based on characteristics
//! - Learn from execution outcomes to improve future scheduling
//! - Track detailed statistics for monitoring and debugging
//! - Support multiple workload types (inference, training, embedding, etc.)
//!
//! ## Quick Start
//!
//! ```zig
//! const ai = @import("ai/mod.zig");
//!
//! // Initialize GPU-aware agent
//! var agent = try ai.GpuAgent.init(allocator);
//! defer agent.deinit();
//!
//! // Create a request
//! const request = ai.GpuAwareRequest{
//!     .prompt = "Analyze this code...",
//!     .workload_type = .inference,
//!     .priority = .normal,
//! };
//!
//! // Process with GPU scheduling
//! const response = try agent.process(request);
//! defer allocator.free(response.content);
//!
//! // Check statistics
//! const stats = agent.getStats();
//! std.debug.print("GPU accelerated: {d}\n", .{stats.gpu_accelerated});
//! ```
//!
//! ## Architecture
//!
//! The agent integrates with:
//! - `src/features/gpu/mega/coordinator.zig` - Cross-backend GPU coordination
//! - `src/features/gpu/mega/scheduler.zig` - Learning-based scheduling
//! - `src/features/ai/orchestration/mod.zig` - Multi-model AI orchestration
//!
//! ## Thread Safety
//!
//! The GpuAgent is not thread-safe. Use external synchronization
//! for concurrent access from multiple threads.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const build_options = @import("build_options");

// GPU integration (conditional)
const gpu_available = build_options.enable_gpu;
const gpu_mod = if (gpu_available) @import("../../gpu/mod.zig") else struct {
    pub const mega = struct {
        pub const Coordinator = void;
        pub const LearningScheduler = void;
        pub const WorkloadProfile = StubWorkloadProfile;
        pub const WorkloadCategory = StubWorkloadCategory;
        pub const ScheduleDecision = StubScheduleDecision;
        pub const BackendInstance = StubBackendInstance;
    };
    pub const Backend = enum {
        cuda,
        vulkan,
        metal,
        webgpu,
        opengl,
        opengles,
        webgl2,
        fpga,
        stdgpu,
    };
};

// Stub types for when GPU is disabled
const StubWorkloadProfile = struct {
    compute_intensity: f32 = 0.5,
    memory_requirement_mb: u64 = 0,
    is_training: bool = false,
    category: StubWorkloadCategory = .general,
};

const StubWorkloadCategory = enum {
    general,
    matrix_multiply,
    convolution,
    attention,
    embedding,
    elementwise,
    reduction,
    fft,
    sorting,
};

const StubScheduleDecision = struct {
    backend_type: BackendType = .none,
    device_id: u32 = 0,
    estimated_time_ms: u64 = 100,
    confidence: f32 = 0.0,
    reason: []const u8 = "GPU disabled",
    decision_id: u64 = 0,

    pub const BackendType = enum {
        none,
        cuda,
        vulkan,
        metal,
        webgpu,
        opengl,
        fpga,
        stdgpu,
    };
};

const StubBackendInstance = struct {
    backend_type: StubScheduleDecision.BackendType = .none,
    device_count: u32 = 0,
    total_memory_mb: u64 = 0,
    available_memory_mb: u64 = 0,
    is_healthy: bool = false,
};

// ============================================================================
// Types
// ============================================================================

/// Workload type for classification.
///
/// Different workload types have distinct resource requirements and
/// optimal scheduling strategies.
pub const WorkloadType = enum {
    /// Standard inference - single request, low latency
    inference,
    /// Model training - high GPU utilization, long running
    training,
    /// Text/vector embedding - batch processing
    embedding,
    /// Fine-tuning - GPU intensive, medium duration
    fine_tuning,
    /// Batch inference - multiple requests, throughput optimized
    batch_inference,

    /// Returns true if this workload type is GPU-intensive.
    pub fn gpuIntensive(self: WorkloadType) bool {
        return switch (self) {
            .training, .fine_tuning => true,
            .inference, .embedding, .batch_inference => false,
        };
    }

    /// Returns true if this workload type is memory-intensive.
    pub fn memoryIntensive(self: WorkloadType) bool {
        return switch (self) {
            .training, .fine_tuning, .batch_inference => true,
            .inference, .embedding => false,
        };
    }

    /// Returns a human-readable name for the workload type.
    pub fn name(self: WorkloadType) []const u8 {
        return switch (self) {
            .inference => "Inference",
            .training => "Training",
            .embedding => "Embedding",
            .fine_tuning => "FineTuning",
            .batch_inference => "BatchInference",
        };
    }

    /// Convert to GPU workload category for scheduling.
    pub fn toGpuCategory(self: WorkloadType) StubWorkloadCategory {
        return switch (self) {
            .inference => .general,
            .training => .matrix_multiply,
            .embedding => .embedding,
            .fine_tuning => .attention,
            .batch_inference => .general,
        };
    }
};

/// Priority levels for request scheduling.
///
/// Higher priority requests are processed first and may preempt
/// lower priority work on congested backends.
pub const Priority = enum {
    /// Background processing, can be delayed
    low,
    /// Default priority for most requests
    normal,
    /// User-facing requests requiring fast response
    high,
    /// Time-critical operations (real-time, SLA-bound)
    critical,

    /// Returns a numeric weight for scheduling decisions.
    pub fn weight(self: Priority) f32 {
        return switch (self) {
            .low => 0.25,
            .normal => 1.0,
            .high => 2.0,
            .critical => 4.0,
        };
    }

    /// Returns a human-readable name for the priority level.
    pub fn name(self: Priority) []const u8 {
        return switch (self) {
            .low => "Low",
            .normal => "Normal",
            .high => "High",
            .critical => "Critical",
        };
    }
};

/// Request for GPU-aware processing.
///
/// Contains the prompt/input along with hints about workload
/// characteristics to help the scheduler make optimal decisions.
pub const GpuAwareRequest = struct {
    /// The text prompt or input to process
    prompt: []const u8,
    /// Type of workload (inference, training, etc.)
    workload_type: WorkloadType,
    /// Request priority for scheduling
    priority: Priority = .normal,
    /// Maximum tokens to generate
    max_tokens: u32 = 1024,
    /// Sampling temperature (0.0 = deterministic, 1.0+ = creative)
    temperature: f32 = 0.7,
    /// Hint about memory requirements in MB (null = auto-detect)
    memory_hint_mb: ?u64 = null,
    /// Preferred GPU backend name (null = auto-select)
    preferred_backend: ?[]const u8 = null,
    /// Model ID for multi-model routing
    model_id: ?[]const u8 = null,
    /// Whether to enable streaming responses
    stream: bool = false,
    /// Request-specific timeout in milliseconds (0 = use default)
    timeout_ms: u64 = 0,
};

/// Response with GPU scheduling information.
///
/// Contains the generated content along with metadata about
/// how the request was processed.
pub const GpuAwareResponse = struct {
    /// The generated content/response
    content: []const u8,
    /// Number of tokens generated
    tokens_generated: u32,
    /// Total processing latency in milliseconds
    latency_ms: u64,
    /// GPU backend that was used (or "cpu" for fallback)
    gpu_backend_used: []const u8,
    /// GPU memory used in MB
    gpu_memory_used_mb: u64,
    /// Scheduler confidence in the routing decision (0.0-1.0)
    scheduling_confidence: f32,
    /// Estimated energy consumption in watt-hours (optional)
    energy_estimate_wh: ?f32 = null,
    /// Device ID that processed the request
    device_id: u32 = 0,
    /// Whether the response was truncated
    truncated: bool = false,
    /// Error message if processing failed (null on success)
    error_message: ?[]const u8 = null,
};

/// Statistics tracked by the GPU-aware agent.
pub const AgentStats = struct {
    /// Total number of requests processed
    total_requests: u64 = 0,
    /// Number of requests that used GPU acceleration
    gpu_accelerated: u64 = 0,
    /// Number of requests that fell back to CPU
    cpu_fallback: u64 = 0,
    /// Total tokens generated across all requests
    total_tokens: u64 = 0,
    /// Total latency across all requests in milliseconds
    total_latency_ms: u64 = 0,
    /// Number of reinforcement learning episodes completed
    learning_episodes: u64 = 0,
    /// Running average of scheduling confidence
    avg_scheduling_confidence: f32 = 0,
    /// Running average latency in milliseconds
    avg_latency_ms: f32 = 0,
    /// Number of failed requests
    failed_requests: u64 = 0,
    /// Total GPU memory used across all requests (MB)
    total_gpu_memory_mb: u64 = 0,

    /// Update the running average confidence with a new sample.
    pub fn updateConfidence(self: *AgentStats, new_confidence: f32) void {
        if (self.gpu_accelerated == 0) {
            self.avg_scheduling_confidence = new_confidence;
        } else {
            const n = @as(f32, @floatFromInt(self.gpu_accelerated));
            self.avg_scheduling_confidence =
                (self.avg_scheduling_confidence * (n - 1) + new_confidence) / n;
        }
    }

    /// Update the running average latency with a new sample.
    pub fn updateLatency(self: *AgentStats, latency: u64) void {
        if (self.total_requests == 0) {
            self.avg_latency_ms = @floatFromInt(latency);
        } else {
            const n = @as(f32, @floatFromInt(self.total_requests));
            self.avg_latency_ms =
                (self.avg_latency_ms * (n - 1) + @as(f32, @floatFromInt(latency))) / n;
        }
    }

    /// Calculate the success rate (0.0 to 1.0).
    pub fn successRate(self: AgentStats) f32 {
        if (self.total_requests == 0) return 1.0;
        const successful = self.total_requests - self.failed_requests;
        return @as(f32, @floatFromInt(successful)) / @as(f32, @floatFromInt(self.total_requests));
    }

    /// Calculate the GPU utilization rate (0.0 to 1.0).
    pub fn gpuUtilizationRate(self: AgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.gpu_accelerated)) /
            @as(f32, @floatFromInt(self.total_requests));
    }

    /// Calculate average tokens per request.
    pub fn avgTokensPerRequest(self: AgentStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.total_tokens)) /
            @as(f32, @floatFromInt(self.total_requests));
    }
};

/// Backend selection result for internal use.
const BackendSelection = struct {
    backend_name: []const u8,
    device_id: u32,
    memory_mb: u64,
    confidence: f32,
    is_gpu: bool,
};

// ============================================================================
// GPU-Aware Agent
// ============================================================================

/// GPU-Aware Self-Learning Agent.
///
/// Integrates AI agent capabilities with GPU scheduling using
/// reinforcement learning to optimize resource allocation over time.
pub const GpuAgent = struct {
    allocator: std.mem.Allocator,
    stats: AgentStats,

    // GPU components (conditionally typed)
    gpu_enabled: bool,
    gpu_coordinator: if (gpu_available) ?*gpu_mod.mega.Coordinator else ?*anyopaque,
    learning_scheduler: if (gpu_available) ?*gpu_mod.mega.LearningScheduler else ?*anyopaque,

    // Response buffer for building responses
    response_buffer: std.ArrayListUnmanaged(u8),

    // Configuration
    default_timeout_ms: u64,
    enable_learning: bool,

    /// Initialize a new GPU-aware agent.
    ///
    /// If GPU is enabled at compile time and available at runtime,
    /// initializes the GPU coordinator and learning scheduler.
    pub fn init(allocator: std.mem.Allocator) !*GpuAgent {
        const self = try allocator.create(GpuAgent);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .stats = .{},
            .gpu_enabled = gpu_available,
            .gpu_coordinator = null,
            .learning_scheduler = null,
            .response_buffer = .{},
            .default_timeout_ms = 30000,
            .enable_learning = true,
        };

        // Initialize GPU components if available
        if (gpu_available) {
            self.gpu_coordinator = gpu_mod.mega.Coordinator.init(allocator) catch null;
            if (self.gpu_coordinator) |coord| {
                self.learning_scheduler = gpu_mod.mega.LearningScheduler.init(allocator, coord) catch null;
            }
        }

        return self;
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        config: struct {
            default_timeout_ms: u64 = 30000,
            enable_learning: bool = true,
        },
    ) !*GpuAgent {
        const self = try init(allocator);
        self.default_timeout_ms = config.default_timeout_ms;
        self.enable_learning = config.enable_learning;
        return self;
    }

    /// Deinitialize the agent and free all resources.
    pub fn deinit(self: *GpuAgent) void {
        // Cleanup GPU components
        if (gpu_available) {
            if (self.learning_scheduler) |sched| {
                sched.deinit();
            }
            if (self.gpu_coordinator) |coord| {
                coord.deinit();
            }
        }

        self.response_buffer.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Process a request with GPU-aware scheduling.
    ///
    /// Routes the workload to the optimal backend based on
    /// workload characteristics and learned performance data.
    pub fn process(self: *GpuAgent, request: GpuAwareRequest) !GpuAwareResponse {
        const start_time = time.nowMs();
        self.stats.total_requests += 1;

        // Determine GPU scheduling
        const selection = self.selectBackend(request);

        // Generate response
        self.response_buffer.clearRetainingCapacity();
        try self.generateResponse(request);

        const content = try self.allocator.dupe(u8, self.response_buffer.items);
        errdefer self.allocator.free(content);

        const tokens: u32 = @intCast(@max(1, content.len / 4));

        const end_time = time.nowMs();
        const latency = @as(u64, @intCast(@max(0, end_time - start_time)));

        // Update statistics
        self.stats.total_tokens += tokens;
        self.stats.total_latency_ms += latency;
        self.stats.total_gpu_memory_mb += selection.memory_mb;
        self.stats.updateLatency(latency);

        if (selection.is_gpu) {
            self.stats.gpu_accelerated += 1;
            self.stats.updateConfidence(selection.confidence);
        } else {
            self.stats.cpu_fallback += 1;
        }

        // Record learning episode if enabled
        if (self.enable_learning and gpu_available) {
            self.recordLearningOutcome(selection, latency, true);
        }

        return .{
            .content = content,
            .tokens_generated = tokens,
            .latency_ms = latency,
            .gpu_backend_used = selection.backend_name,
            .gpu_memory_used_mb = selection.memory_mb,
            .scheduling_confidence = selection.confidence,
            .device_id = selection.device_id,
        };
    }

    /// Select the optimal backend for the given request.
    fn selectBackend(self: *GpuAgent, request: GpuAwareRequest) BackendSelection {
        if (!self.gpu_enabled or self.gpu_coordinator == null) {
            return .{
                .backend_name = "cpu",
                .device_id = 0,
                .memory_mb = 0,
                .confidence = 0.0,
                .is_gpu = false,
            };
        }

        if (gpu_available) {
            // Create workload profile for scheduling
            const profile = gpu_mod.mega.WorkloadProfile{
                .compute_intensity = if (request.workload_type.gpuIntensive()) 0.9 else 0.3,
                .memory_requirement_mb = request.memory_hint_mb orelse self.estimateMemory(request),
                .is_training = request.workload_type == .training or
                    request.workload_type == .fine_tuning,
                .category = self.mapWorkloadCategory(request.workload_type),
            };

            // Use learning scheduler if available, otherwise use coordinator
            if (self.learning_scheduler) |sched| {
                const decision = sched.schedule(profile);
                return .{
                    .backend_name = @tagName(decision.backend_type),
                    .device_id = decision.device_id,
                    .memory_mb = profile.memory_requirement_mb,
                    .confidence = decision.confidence,
                    .is_gpu = decision.backend_type != .stdgpu,
                };
            } else if (self.gpu_coordinator) |coord| {
                const decision = coord.schedule(profile);
                return .{
                    .backend_name = @tagName(decision.backend_type),
                    .device_id = decision.device_id,
                    .memory_mb = profile.memory_requirement_mb,
                    .confidence = decision.confidence,
                    .is_gpu = decision.backend_type != .stdgpu,
                };
            }
        }

        // Fallback: use heuristic-based selection
        return self.heuristicBackendSelection(request);
    }

    /// Map WorkloadType to GPU workload category.
    fn mapWorkloadCategory(self: *GpuAgent, workload_type: WorkloadType) gpu_mod.mega.WorkloadCategory {
        _ = self;
        return switch (workload_type) {
            .inference => .general,
            .training => .matrix_multiply,
            .embedding => .embedding,
            .fine_tuning => .attention,
            .batch_inference => .general,
        };
    }

    /// Estimate memory requirements based on request characteristics.
    fn estimateMemory(self: *GpuAgent, request: GpuAwareRequest) u64 {
        _ = self;
        const base_memory: u64 = 512; // Base overhead in MB
        const per_token_memory: u64 = 2; // MB per 1k tokens

        const token_memory = (request.max_tokens * per_token_memory) / 1000;

        const workload_multiplier: u64 = switch (request.workload_type) {
            .training => 4,
            .fine_tuning => 3,
            .batch_inference => 2,
            .embedding => 1,
            .inference => 1,
        };

        return (base_memory + token_memory) * workload_multiplier;
    }

    /// Heuristic-based backend selection when GPU scheduler is unavailable.
    fn heuristicBackendSelection(self: *GpuAgent, request: GpuAwareRequest) BackendSelection {
        _ = self;
        // Simple heuristic: GPU for intensive workloads, CPU otherwise
        if (request.workload_type.gpuIntensive()) {
            return .{
                .backend_name = "cuda",
                .device_id = 0,
                .memory_mb = request.memory_hint_mb orelse 2048,
                .confidence = 0.7,
                .is_gpu = true,
            };
        } else {
            return .{
                .backend_name = "vulkan",
                .device_id = 0,
                .memory_mb = request.memory_hint_mb orelse 512,
                .confidence = 0.6,
                .is_gpu = true,
            };
        }
    }

    /// Generate response content (placeholder implementation).
    fn generateResponse(self: *GpuAgent, request: GpuAwareRequest) !void {
        // Build a response indicating the workload was processed
        try self.response_buffer.appendSlice(self.allocator, "[");
        try self.response_buffer.appendSlice(self.allocator, request.workload_type.name());
        try self.response_buffer.appendSlice(self.allocator, "] Response for: ");

        const max_preview = @min(50, request.prompt.len);
        try self.response_buffer.appendSlice(self.allocator, request.prompt[0..max_preview]);

        if (request.prompt.len > 50) {
            try self.response_buffer.appendSlice(self.allocator, "...");
        }
    }

    /// Record learning outcome for reinforcement learning.
    fn recordLearningOutcome(
        self: *GpuAgent,
        selection: BackendSelection,
        latency: u64,
        success: bool,
    ) void {
        _ = selection;
        _ = latency;
        _ = success;
        // Would integrate with LearningScheduler.recordAndLearn in full implementation
        self.stats.learning_episodes += 1;
    }

    /// Get current agent statistics.
    pub fn getStats(self: *const GpuAgent) AgentStats {
        return self.stats;
    }

    /// Check if GPU acceleration is enabled and available.
    pub fn isGpuEnabled(self: *const GpuAgent) bool {
        return self.gpu_enabled and self.gpu_coordinator != null;
    }

    /// Check if learning scheduler is active.
    pub fn isLearningEnabled(self: *const GpuAgent) bool {
        return self.enable_learning and self.learning_scheduler != null;
    }

    /// End the current learning episode (for RL training).
    pub fn endEpisode(self: *GpuAgent) void {
        if (gpu_available) {
            if (self.learning_scheduler) |sched| {
                sched.endEpisode() catch |err| {
                    std.log.debug("Failed to end learning episode: {t}", .{err});
                };
            }
        }
        self.stats.learning_episodes += 1;
    }

    /// Reset all statistics.
    pub fn resetStats(self: *GpuAgent) void {
        self.stats = .{};
    }

    /// Get available backends summary.
    pub fn getBackendsSummary(self: *GpuAgent, allocator: std.mem.Allocator) ![]const BackendInfo {
        if (!gpu_available or self.gpu_coordinator == null) {
            const infos = try allocator.alloc(BackendInfo, 1);
            infos[0] = .{
                .name = "cpu",
                .device_count = 1,
                .total_memory_mb = 0,
                .available_memory_mb = 0,
                .is_healthy = true,
            };
            return infos;
        }

        if (gpu_available) {
            if (self.gpu_coordinator) |coord| {
                const backends = coord.getBackendsSummary();
                const infos = try allocator.alloc(BackendInfo, backends.len);

                for (backends, 0..) |backend, i| {
                    infos[i] = .{
                        .name = @tagName(backend.backend_type),
                        .device_count = backend.device_count,
                        .total_memory_mb = backend.total_memory_mb,
                        .available_memory_mb = backend.available_memory_mb,
                        .is_healthy = backend.is_healthy,
                    };
                }
                return infos;
            }
        }

        return try allocator.alloc(BackendInfo, 0);
    }

    /// Backend information for external reporting.
    pub const BackendInfo = struct {
        name: []const u8,
        device_count: u32,
        total_memory_mb: u64,
        available_memory_mb: u64,
        is_healthy: bool,
    };

    /// Get learning statistics if available.
    pub fn getLearningStats(self: *GpuAgent) ?LearningStatsInfo {
        if (gpu_available) {
            if (self.learning_scheduler) |sched| {
                const stats = sched.getStats();
                return .{
                    .episodes = stats.episodes,
                    .avg_episode_reward = stats.avg_episode_reward,
                    .exploration_rate = stats.exploration_rate,
                    .replay_buffer_size = stats.replay_buffer_size,
                };
            }
        }
        return null;
    }

    /// Learning statistics information.
    pub const LearningStatsInfo = struct {
        episodes: usize,
        avg_episode_reward: f32,
        exploration_rate: f32,
        replay_buffer_size: usize,
    };
};

// ============================================================================
// Tests
// ============================================================================

test "workload type properties" {
    try std.testing.expect(WorkloadType.training.gpuIntensive());
    try std.testing.expect(WorkloadType.fine_tuning.gpuIntensive());
    try std.testing.expect(!WorkloadType.inference.gpuIntensive());
    try std.testing.expect(!WorkloadType.embedding.gpuIntensive());

    try std.testing.expect(WorkloadType.training.memoryIntensive());
    try std.testing.expect(WorkloadType.batch_inference.memoryIntensive());
    try std.testing.expect(!WorkloadType.inference.memoryIntensive());
}

test "priority weights" {
    try std.testing.expect(Priority.low.weight() < Priority.normal.weight());
    try std.testing.expect(Priority.normal.weight() < Priority.high.weight());
    try std.testing.expect(Priority.high.weight() < Priority.critical.weight());
}

test "gpu agent init and deinit" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    try std.testing.expectEqual(@as(u64, 0), agent.stats.total_requests);
    try std.testing.expectEqual(@as(u64, 0), agent.stats.gpu_accelerated);
}

test "gpu agent process request" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    const request = GpuAwareRequest{
        .prompt = "Test prompt for inference",
        .workload_type = .inference,
        .priority = .normal,
    };

    const response = try agent.process(request);
    defer allocator.free(response.content);

    try std.testing.expectEqual(@as(u64, 1), agent.stats.total_requests);
    try std.testing.expect(response.tokens_generated > 0);
    try std.testing.expect(response.content.len > 0);
}

test "gpu agent process training request" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    const request = GpuAwareRequest{
        .prompt = "Fine-tune model on dataset",
        .workload_type = .training,
        .priority = .high,
        .memory_hint_mb = 4096,
    };

    const response = try agent.process(request);
    defer allocator.free(response.content);

    try std.testing.expectEqual(@as(u64, 1), agent.stats.total_requests);
    try std.testing.expect(response.latency_ms >= 0);
}

test "agent stats update" {
    var stats = AgentStats{};

    stats.total_requests = 1;
    stats.gpu_accelerated = 1;
    stats.updateConfidence(0.8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), stats.avg_scheduling_confidence, 0.01);

    stats.gpu_accelerated = 2;
    stats.updateConfidence(0.9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), stats.avg_scheduling_confidence, 0.01);
}

test "agent stats latency update" {
    var stats = AgentStats{};

    stats.total_requests = 1;
    stats.updateLatency(100);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), stats.avg_latency_ms, 0.01);

    stats.total_requests = 2;
    stats.updateLatency(200);
    try std.testing.expectApproxEqAbs(@as(f32, 150.0), stats.avg_latency_ms, 0.01);
}

test "agent stats derived metrics" {
    var stats = AgentStats{};
    stats.total_requests = 10;
    stats.failed_requests = 2;
    stats.gpu_accelerated = 6;
    stats.total_tokens = 1000;

    try std.testing.expectApproxEqAbs(@as(f32, 0.8), stats.successRate(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), stats.gpuUtilizationRate(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), stats.avgTokensPerRequest(), 0.01);
}

test "workload type names" {
    try std.testing.expectEqualStrings("Inference", WorkloadType.inference.name());
    try std.testing.expectEqualStrings("Training", WorkloadType.training.name());
    try std.testing.expectEqualStrings("Embedding", WorkloadType.embedding.name());
    try std.testing.expectEqualStrings("FineTuning", WorkloadType.fine_tuning.name());
    try std.testing.expectEqualStrings("BatchInference", WorkloadType.batch_inference.name());
}

test "priority names" {
    try std.testing.expectEqualStrings("Low", Priority.low.name());
    try std.testing.expectEqualStrings("Normal", Priority.normal.name());
    try std.testing.expectEqualStrings("High", Priority.high.name());
    try std.testing.expectEqualStrings("Critical", Priority.critical.name());
}

test "gpu agent with config" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.initWithConfig(allocator, .{
        .default_timeout_ms = 60000,
        .enable_learning = false,
    });
    defer agent.deinit();

    try std.testing.expectEqual(@as(u64, 60000), agent.default_timeout_ms);
    try std.testing.expect(!agent.enable_learning);
}

test "gpu agent reset stats" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    // Process a request
    const request = GpuAwareRequest{
        .prompt = "Test",
        .workload_type = .inference,
    };
    const response = try agent.process(request);
    allocator.free(response.content);

    // Verify stats updated
    try std.testing.expectEqual(@as(u64, 1), agent.stats.total_requests);

    // Reset and verify
    agent.resetStats();
    try std.testing.expectEqual(@as(u64, 0), agent.stats.total_requests);
}

test "gpu agent end episode" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    const initial_episodes = agent.stats.learning_episodes;
    agent.endEpisode();
    try std.testing.expect(agent.stats.learning_episodes > initial_episodes);
}

test "gpu agent backends summary" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    const backends = try agent.getBackendsSummary(allocator);
    defer allocator.free(backends);

    // Should have at least one backend (cpu fallback)
    try std.testing.expect(backends.len >= 1);
}

test {
    std.testing.refAllDecls(@This());
}
