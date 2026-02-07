//! Hybrid GPU-FPGA Coordinator
//!
//! Provides intelligent workload routing between GPU and FPGA backends
//! based on workload characteristics, latency requirements, and resource availability.
//!
//! ## Architecture
//!
//! ```
//!                    +-------------------+
//!                    | Hybrid Coordinator|
//!                    +--------+----------+
//!                             |
//!         +-------------------+-------------------+
//!         |                                       |
//!         v                                       v
//! +---------------+                       +---------------+
//! | GPU Backend   |                       | FPGA Backend  |
//! | (CUDA/Vulkan) |                       | (Distance,    |
//! | - Training    |                       |  MatMul, KV)  |
//! | - Inference   |                       | - Low latency |
//! +---------------+                       +---------------+
//! ```
//!
//! ## Workload Routing Rules
//!
//! 1. **FPGA-preferred workloads:**
//!    - Vector distance computations (HNSW, IVF-PQ)
//!    - Quantized matrix multiplications (Q4/Q8)
//!    - Streaming attention with KV-cache
//!    - Low-latency single-query inference
//!
//! 2. **GPU-preferred workloads:**
//!    - Training (backpropagation)
//!    - Large batch inference
//!    - General-purpose tensor operations
//!    - FFT and convolutions
//!
//! 3. **Hybrid workloads:**
//!    - Pipelined inference: GPU for MatMul, FPGA for attention
//!    - Speculative decoding: GPU for draft, FPGA for verification

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const build_options = @import("build_options");
const backend_mod = @import("../backend.zig");
const multi_device = @import("../multi_device.zig");
const coordinator = @import("coordinator.zig");
const fpga_kernels = @import("../backends/fpga/kernels.zig");

/// Device type for hybrid routing
pub const HybridDeviceType = enum {
    gpu, // General GPU (CUDA, Vulkan, Metal)
    fpga, // FPGA accelerator
    cpu, // CPU fallback
    any, // No preference

    pub fn priority(self: HybridDeviceType) u8 {
        return switch (self) {
            .fpga => 3, // Highest for specialized ops
            .gpu => 2,
            .cpu => 1,
            .any => 0,
        };
    }
};

/// Workload type for routing decisions
pub const HybridWorkloadType = enum {
    /// Vector distance (L2, cosine, dot product)
    vector_distance,
    /// Matrix multiplication
    matmul,
    /// Quantized matmul (Q4/Q8)
    quantized_matmul,
    /// Attention mechanism
    attention,
    /// KV-cache operations
    kv_cache,
    /// Training/backprop
    training,
    /// General tensor ops
    tensor_ops,
    /// Embedding lookup
    embedding,
    /// Convolution
    convolution,
    /// FFT
    fft,
    /// Unknown/general
    general,

    /// Get preferred device type for this workload
    pub fn preferredDevice(self: HybridWorkloadType) HybridDeviceType {
        return switch (self) {
            // FPGA-optimal workloads
            .vector_distance => .fpga,
            .quantized_matmul => .fpga,
            .kv_cache => .fpga,
            // GPU-optimal workloads
            .training => .gpu,
            .convolution => .gpu,
            .fft => .gpu,
            // Can go either way
            .attention => .any, // Depends on sequence length
            .matmul => .any, // Depends on size
            .tensor_ops => .any,
            .embedding => .any,
            .general => .any,
        };
    }
};

/// Configuration for hybrid routing
pub const HybridRoutingConfig = struct {
    /// Minimum latency (ns) to prefer FPGA
    fpga_latency_threshold_ns: u64 = 1000, // 1us
    /// Minimum batch size to prefer GPU
    gpu_batch_threshold: u32 = 16,
    /// Memory threshold (MB) to consider offloading to FPGA
    fpga_memory_threshold_mb: u64 = 256,
    /// Enable speculative routing based on history
    enable_speculation: bool = true,
    /// Enable pipelined execution between GPU and FPGA
    enable_pipeline: bool = true,
    /// Weight for latency in routing score (0-1)
    latency_weight: f32 = 0.3,
    /// Weight for throughput in routing score (0-1)
    throughput_weight: f32 = 0.4,
    /// Weight for energy efficiency in routing score (0-1)
    energy_weight: f32 = 0.3,
};

/// Workload descriptor for routing
pub const HybridWorkload = struct {
    /// Type of workload
    workload_type: HybridWorkloadType = .general,
    /// Input size in elements
    input_size: usize = 0,
    /// Output size in elements
    output_size: usize = 0,
    /// Batch size (1 for single query)
    batch_size: u32 = 1,
    /// Sequence length (for attention/KV)
    seq_len: u32 = 0,
    /// Head dimension (for attention)
    head_dim: u32 = 64,
    /// Number of heads (for attention)
    num_heads: u32 = 8,
    /// Weight precision
    weight_precision: WeightPrecision = .fp32,
    /// Latency requirement (ns, 0 = no requirement)
    latency_requirement_ns: u64 = 0,
    /// Is this part of a training loop?
    is_training: bool = false,
    /// Is this a real-time/streaming workload?
    is_streaming: bool = false,
    /// Memory requirement in MB
    memory_requirement_mb: u64 = 0,

    pub const WeightPrecision = enum {
        fp32,
        fp16,
        bf16,
        int8,
        int4,

        pub fn isFpgaOptimal(self: WeightPrecision) bool {
            return self == .int8 or self == .int4;
        }
    };

    /// Calculate estimated compute FLOPs
    pub fn estimatedFlops(self: HybridWorkload) u64 {
        return switch (self.workload_type) {
            .vector_distance => @as(u64, self.input_size) * 2, // Compare ops
            .matmul, .quantized_matmul => blk: {
                // M*N*K * 2 (multiply + add)
                const ops = @as(u64, self.batch_size) * self.input_size * self.output_size * 2;
                break :blk ops;
            },
            .attention => blk: {
                // QK^T + softmax + V matmul
                const qk_ops = @as(u64, self.seq_len) * self.seq_len * self.head_dim * 2;
                const v_ops = @as(u64, self.seq_len) * self.seq_len * self.head_dim * 2;
                break :blk (qk_ops + v_ops) * self.num_heads * self.batch_size;
            },
            .kv_cache => @as(u64, self.seq_len) * self.head_dim * self.num_heads * 4,
            else => @as(u64, self.input_size) * 10, // Generic estimate
        };
    }

    /// Estimate memory bandwidth requirement in bytes
    pub fn estimatedBandwidth(self: HybridWorkload) u64 {
        const bytes_per_elem: u64 = switch (self.weight_precision) {
            .fp32 => 4,
            .fp16, .bf16 => 2,
            .int8 => 1,
            .int4 => 1, // Rounded up
        };
        return (self.input_size + self.output_size) * bytes_per_elem * self.batch_size;
    }
};

/// Routing decision from the hybrid coordinator
pub const HybridRoutingDecision = struct {
    /// Selected device type
    device_type: HybridDeviceType,
    /// Selected backend (for GPU)
    backend: backend_mod.Backend,
    /// Device ID within backend
    device_id: multi_device.DeviceId,
    /// Estimated latency in nanoseconds
    estimated_latency_ns: u64,
    /// Estimated throughput in GFLOPS
    estimated_throughput_gflops: f32,
    /// Confidence in this decision (0-1)
    confidence: f32,
    /// Reason for the decision
    reason: []const u8,
    /// Whether to use pipelined execution
    use_pipeline: bool,
    /// If pipelined, which stages go where
    pipeline_stages: ?[]const PipelineStage,

    pub const PipelineStage = struct {
        name: []const u8,
        device_type: HybridDeviceType,
        order: u32,
    };
};

/// Hybrid GPU-FPGA Coordinator
pub const HybridCoordinator = struct {
    allocator: std.mem.Allocator,
    config: HybridRoutingConfig,

    // Backend coordinators
    mega_coordinator: *coordinator.Coordinator,

    // Device state
    gpu_available: bool,
    fpga_available: bool,
    gpu_memory_available_mb: u64,
    fpga_memory_available_mb: u64,

    // Performance history for learning
    routing_history: std.ArrayListUnmanaged(RoutingRecord),
    stats: HybridStats,
    mutex: sync.Mutex,

    const RoutingRecord = struct {
        workload: HybridWorkload,
        decision: HybridRoutingDecision,
        actual_latency_ns: u64,
        success: bool,
        timestamp: i64,
    };

    pub const HybridStats = struct {
        total_routings: u64 = 0,
        gpu_routings: u64 = 0,
        fpga_routings: u64 = 0,
        pipeline_routings: u64 = 0,
        successful_routings: u64 = 0,
        total_latency_ns: u64 = 0,
        avg_confidence: f32 = 0,
    };

    /// Initialize the hybrid coordinator
    pub fn init(allocator: std.mem.Allocator, config: HybridRoutingConfig) !*HybridCoordinator {
        const self = try allocator.create(HybridCoordinator);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .mega_coordinator = try coordinator.Coordinator.init(allocator),
            .gpu_available = detectGpuAvailable(),
            .fpga_available = detectFpgaAvailable(),
            .gpu_memory_available_mb = if (detectGpuAvailable()) 8192 else 0,
            .fpga_memory_available_mb = if (detectFpgaAvailable()) 4096 else 0,
            .routing_history = .{},
            .stats = .{},
            .mutex = .{},
        };

        return self;
    }

    /// Deinitialize the coordinator
    pub fn deinit(self: *HybridCoordinator) void {
        self.mega_coordinator.deinit();
        self.routing_history.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Route a workload to the optimal device
    pub fn route(self: *HybridCoordinator, workload: HybridWorkload) HybridRoutingDecision {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Score each device type
        const gpu_score = self.scoreDevice(.gpu, workload);
        const fpga_score = self.scoreDevice(.fpga, workload);
        const cpu_score = self.scoreDevice(.cpu, workload);

        // Select best device
        var best_score = cpu_score;
        var best_device: HybridDeviceType = .cpu;

        if (self.gpu_available and gpu_score > best_score) {
            best_score = gpu_score;
            best_device = .gpu;
        }

        if (self.fpga_available and fpga_score > best_score) {
            best_score = fpga_score;
            best_device = .fpga;
        }

        // Check if pipelining is beneficial
        const use_pipeline = self.shouldUsePipeline(workload, gpu_score, fpga_score);
        const pipeline_stages = if (use_pipeline) self.createPipelineStages(workload) else null;

        // Get backend for GPU routing
        const backend = if (best_device == .gpu)
            self.selectGpuBackend(workload)
        else if (best_device == .fpga)
            backend_mod.Backend.fpga
        else
            backend_mod.Backend.stdgpu;

        // Update stats
        self.stats.total_routings += 1;
        switch (best_device) {
            .gpu => self.stats.gpu_routings += 1,
            .fpga => self.stats.fpga_routings += 1,
            else => {},
        }
        if (use_pipeline) self.stats.pipeline_routings += 1;

        return .{
            .device_type = best_device,
            .backend = backend,
            .device_id = 0,
            .estimated_latency_ns = self.estimateLatency(best_device, workload),
            .estimated_throughput_gflops = self.estimateThroughput(best_device, workload),
            .confidence = @min(best_score / 10.0, 1.0),
            .reason = self.selectReason(best_device, workload),
            .use_pipeline = use_pipeline,
            .pipeline_stages = pipeline_stages,
        };
    }

    /// Record the outcome of a routing decision
    pub fn recordOutcome(
        self: *HybridCoordinator,
        workload: HybridWorkload,
        decision: HybridRoutingDecision,
        actual_latency_ns: u64,
        success: bool,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.routing_history.append(self.allocator, .{
            .workload = workload,
            .decision = decision,
            .actual_latency_ns = actual_latency_ns,
            .success = success,
            .timestamp = std.time.milliTimestamp(),
        });

        if (success) {
            self.stats.successful_routings += 1;
        }
        self.stats.total_latency_ns += actual_latency_ns;

        // Update running average confidence
        const n = @as(f32, @floatFromInt(self.stats.total_routings));
        self.stats.avg_confidence = (self.stats.avg_confidence * (n - 1) + decision.confidence) / n;

        // Trim history
        const max_history = 5000;
        if (self.routing_history.items.len > max_history) {
            const to_remove = self.routing_history.items.len - max_history;
            const items = self.routing_history.items;
            std.mem.copyForwards(RoutingRecord, items[0..max_history], items[to_remove..]);
            self.routing_history.shrinkRetainingCapacity(max_history);
        }
    }

    /// Score a device for a given workload
    fn scoreDevice(self: *HybridCoordinator, device: HybridDeviceType, workload: HybridWorkload) f32 {
        var score: f32 = 0.0;

        // Base preference from workload type
        const preferred = workload.workload_type.preferredDevice();
        if (preferred == device) {
            score += 5.0;
        } else if (preferred == .any) {
            score += 2.5;
        }

        // Device availability
        switch (device) {
            .gpu => if (!self.gpu_available) return 0.0,
            .fpga => if (!self.fpga_available) return 0.0,
            .cpu => score += 0.5, // Always available but low priority
            .any => {},
        }

        // Memory constraints
        const memory_available = switch (device) {
            .gpu => self.gpu_memory_available_mb,
            .fpga => self.fpga_memory_available_mb,
            .cpu => 16384, // Assume plenty of RAM
            .any => 8192,
        };
        if (workload.memory_requirement_mb > memory_available) {
            score *= 0.1; // Heavy penalty
        }

        // Batch size preference
        if (workload.batch_size >= self.config.gpu_batch_threshold) {
            if (device == .gpu) score += 3.0;
        } else {
            if (device == .fpga) score += 2.0;
        }

        // Quantization preference
        if (workload.weight_precision.isFpgaOptimal()) {
            if (device == .fpga) score += 4.0;
        }

        // Training preference
        if (workload.is_training) {
            if (device == .gpu) score += 5.0;
            if (device == .fpga) score -= 3.0; // FPGA not ideal for training
        }

        // Streaming/low-latency preference
        if (workload.is_streaming or workload.latency_requirement_ns > 0) {
            if (device == .fpga) score += 3.0; // FPGA has predictable latency
        }

        // Sequence length affects attention routing
        if (workload.workload_type == .attention) {
            if (workload.seq_len <= 512 and device == .fpga) {
                score += 2.0; // FPGA good for short sequences
            } else if (workload.seq_len > 2048 and device == .gpu) {
                score += 2.0; // GPU better for long sequences
            }
        }

        return score;
    }

    /// Check if pipelining is beneficial
    fn shouldUsePipeline(self: *HybridCoordinator, workload: HybridWorkload, gpu_score: f32, fpga_score: f32) bool {
        if (!self.config.enable_pipeline) return false;
        if (!self.gpu_available or !self.fpga_available) return false;

        // Pipelining beneficial when both devices have similar scores
        // and workload is complex enough to benefit
        const score_diff = @abs(gpu_score - fpga_score);
        if (score_diff > 2.0) return false;

        // Attention workloads can benefit from pipelining
        if (workload.workload_type == .attention and workload.seq_len > 256) {
            return true;
        }

        // Large matmul with KV-cache can benefit
        if (workload.workload_type == .matmul and workload.batch_size >= 8) {
            return true;
        }

        return false;
    }

    /// Create pipeline stages for hybrid execution
    fn createPipelineStages(self: *HybridCoordinator, workload: HybridWorkload) ?[]const HybridRoutingDecision.PipelineStage {
        _ = self;

        // Static pipeline configurations
        const attention_pipeline = [_]HybridRoutingDecision.PipelineStage{
            .{ .name = "QK^T MatMul", .device_type = .gpu, .order = 0 },
            .{ .name = "Softmax", .device_type = .fpga, .order = 1 },
            .{ .name = "V MatMul", .device_type = .gpu, .order = 2 },
        };

        const matmul_kv_pipeline = [_]HybridRoutingDecision.PipelineStage{
            .{ .name = "MatMul", .device_type = .gpu, .order = 0 },
            .{ .name = "KV Update", .device_type = .fpga, .order = 1 },
        };

        return switch (workload.workload_type) {
            .attention => &attention_pipeline,
            .matmul, .quantized_matmul => &matmul_kv_pipeline,
            else => null,
        };
    }

    /// Select best GPU backend for workload
    fn selectGpuBackend(self: *HybridCoordinator, workload: HybridWorkload) backend_mod.Backend {
        _ = workload;

        // Use mega coordinator's backend selection
        const backends = self.mega_coordinator.getBackendsSummary();
        for (backends) |backend| {
            if (backend.is_healthy and !backend.is_emulated) {
                return backend.backend_type;
            }
        }

        return .stdgpu; // Fallback
    }

    /// Estimate latency for device and workload
    fn estimateLatency(self: *HybridCoordinator, device: HybridDeviceType, workload: HybridWorkload) u64 {
        _ = self;

        const flops = workload.estimatedFlops();

        // Estimated GFLOPS by device
        const gflops: f32 = switch (device) {
            .gpu => 10000.0, // 10 TFLOPS
            .fpga => 1000.0, // 1 TFLOPS but lower latency
            .cpu => 100.0, // 100 GFLOPS
            .any => 1000.0,
        };

        // Base latency (ns per GFLOP)
        const base_latency: f32 = @as(f32, @floatFromInt(flops)) / (gflops * 1e9) * 1e9;

        // Add device-specific overhead
        const overhead: u64 = switch (device) {
            .gpu => 50000, // 50us GPU launch overhead
            .fpga => 1000, // 1us FPGA overhead
            .cpu => 100, // Minimal overhead
            .any => 10000,
        };

        return @as(u64, @intFromFloat(base_latency)) + overhead;
    }

    /// Estimate throughput for device and workload
    fn estimateThroughput(self: *HybridCoordinator, device: HybridDeviceType, workload: HybridWorkload) f32 {
        _ = self;
        _ = workload;

        return switch (device) {
            .gpu => 10000.0, // 10 TFLOPS
            .fpga => 1000.0, // 1 TFLOPS
            .cpu => 100.0, // 100 GFLOPS
            .any => 1000.0,
        };
    }

    /// Select human-readable reason
    fn selectReason(self: *HybridCoordinator, device: HybridDeviceType, workload: HybridWorkload) []const u8 {
        _ = self;

        return switch (device) {
            .gpu => switch (workload.workload_type) {
                .training => "GPU selected for training workload",
                .convolution => "GPU selected for convolution",
                .fft => "GPU selected for FFT",
                .matmul => if (workload.batch_size >= 16)
                    "GPU selected for large batch matmul"
                else
                    "GPU selected for general matmul",
                .attention => if (workload.seq_len > 2048)
                    "GPU selected for long-sequence attention"
                else
                    "GPU selected for attention",
                else => "GPU selected based on workload profile",
            },
            .fpga => switch (workload.workload_type) {
                .vector_distance => "FPGA selected for low-latency distance",
                .quantized_matmul => "FPGA selected for quantized matmul",
                .kv_cache => "FPGA selected for KV-cache ops",
                .attention => if (workload.seq_len <= 512)
                    "FPGA selected for short-sequence attention"
                else
                    "FPGA selected for streaming attention",
                else => "FPGA selected for specialized op",
            },
            .cpu => "CPU fallback (no accelerator available)",
            .any => "No specific device preference",
        };
    }

    /// Get coordinator statistics
    pub fn getStats(self: *HybridCoordinator) HybridStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Check if FPGA is available
    pub fn hasFpga(self: *HybridCoordinator) bool {
        return self.fpga_available;
    }

    /// Check if GPU is available
    pub fn hasGpu(self: *HybridCoordinator) bool {
        return self.gpu_available;
    }

    /// Get routing summary
    pub fn getRoutingSummary(self: *HybridCoordinator) RoutingSummary {
        self.mutex.lock();
        defer self.mutex.unlock();

        const total = self.stats.total_routings;
        return .{
            .total_routings = total,
            .gpu_percentage = if (total > 0)
                @as(f32, @floatFromInt(self.stats.gpu_routings)) / @as(f32, @floatFromInt(total)) * 100
            else
                0,
            .fpga_percentage = if (total > 0)
                @as(f32, @floatFromInt(self.stats.fpga_routings)) / @as(f32, @floatFromInt(total)) * 100
            else
                0,
            .pipeline_percentage = if (total > 0)
                @as(f32, @floatFromInt(self.stats.pipeline_routings)) / @as(f32, @floatFromInt(total)) * 100
            else
                0,
            .success_rate = if (total > 0)
                @as(f32, @floatFromInt(self.stats.successful_routings)) / @as(f32, @floatFromInt(total)) * 100
            else
                100,
            .avg_latency_ns = if (total > 0)
                self.stats.total_latency_ns / total
            else
                0,
        };
    }

    pub const RoutingSummary = struct {
        total_routings: u64,
        gpu_percentage: f32,
        fpga_percentage: f32,
        pipeline_percentage: f32,
        success_rate: f32,
        avg_latency_ns: u64,
    };
};

/// Detect if GPU is available at runtime
fn detectGpuAvailable() bool {
    if (comptime build_options.gpu_cuda) return true;
    if (comptime build_options.gpu_vulkan) return true;
    if (comptime build_options.gpu_metal) return true;
    if (comptime build_options.gpu_webgpu) return true;
    return false;
}

/// Detect if FPGA is available at runtime
fn detectFpgaAvailable() bool {
    return comptime build_options.gpu_fpga;
}

// ============================================================================
// Tests
// ============================================================================

test "hybrid coordinator initialization" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    // Should have stats initialized
    const stats = coord.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_routings);
}

test "workload routing" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    // Test vector distance routing
    const distance_workload = HybridWorkload{
        .workload_type = .vector_distance,
        .input_size = 1024 * 128,
        .batch_size = 1,
    };
    const decision1 = coord.route(distance_workload);
    try std.testing.expect(decision1.confidence >= 0.0);

    // Test training routing
    const training_workload = HybridWorkload{
        .workload_type = .training,
        .input_size = 1024 * 1024,
        .batch_size = 32,
        .is_training = true,
    };
    const decision2 = coord.route(training_workload);
    try std.testing.expect(decision2.confidence >= 0.0);
}

test "quantized matmul prefers FPGA" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    const workload = HybridWorkload{
        .workload_type = .quantized_matmul,
        .input_size = 4096,
        .output_size = 4096,
        .batch_size = 1,
        .weight_precision = .int4,
    };

    const decision = coord.route(workload);

    // Should prefer FPGA for quantized ops if available
    if (comptime build_options.gpu_fpga) {
        try std.testing.expectEqual(HybridDeviceType.fpga, decision.device_type);
    }
}

test "large batch prefers GPU" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    const workload = HybridWorkload{
        .workload_type = .matmul,
        .input_size = 4096,
        .output_size = 4096,
        .batch_size = 64,
    };

    const decision = coord.route(workload);

    // With large batch, GPU should be preferred
    if (detectGpuAvailable()) {
        try std.testing.expectEqual(HybridDeviceType.gpu, decision.device_type);
    }
}

test "outcome recording" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    const workload = HybridWorkload{
        .workload_type = .vector_distance,
        .input_size = 1024,
    };

    const decision = coord.route(workload);
    try coord.recordOutcome(workload, decision, 50000, true);

    const stats = coord.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_routings);
    try std.testing.expectEqual(@as(u64, 1), stats.successful_routings);
}

test "routing summary" {
    const allocator = std.testing.allocator;
    const coord = try HybridCoordinator.init(allocator, .{});
    defer coord.deinit();

    // Route several workloads
    for (0..5) |_| {
        const workload = HybridWorkload{ .workload_type = .general };
        _ = coord.route(workload);
    }

    const summary = coord.getRoutingSummary();
    try std.testing.expectEqual(@as(u64, 5), summary.total_routings);
}

test "estimated flops calculation" {
    const workload = HybridWorkload{
        .workload_type = .attention,
        .seq_len = 512,
        .head_dim = 64,
        .num_heads = 8,
        .batch_size = 1,
    };

    const flops = workload.estimatedFlops();
    try std.testing.expect(flops > 0);
}

test "estimated bandwidth calculation" {
    const workload = HybridWorkload{
        .input_size = 1024,
        .output_size = 1024,
        .batch_size = 4,
        .weight_precision = .fp32,
    };

    const bandwidth = workload.estimatedBandwidth();
    try std.testing.expectEqual(@as(u64, (1024 + 1024) * 4 * 4), bandwidth);
}
