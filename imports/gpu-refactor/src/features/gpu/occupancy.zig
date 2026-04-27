//! GPU Occupancy Calculator
//!
//! Provides sophisticated occupancy calculation and optimal launch configuration
//! selection for maximizing GPU utilization. Supports CUDA, Vulkan, and Metal
//! device characteristics.
//!
//! ## Features
//!
//! - Multi-factor occupancy calculation (threads, warps, registers, shared memory)
//! - Automatic optimal block size selection
//! - Hardware-specific tuning parameters
//! - Performance prediction based on roofline model

const std = @import("std");

/// Device compute capabilities for occupancy calculation.
pub const DeviceCapabilities = struct {
    /// Maximum threads per multiprocessor/compute unit.
    max_threads_per_sm: u32 = 2048,
    /// Maximum warps per multiprocessor.
    max_warps_per_sm: u32 = 64,
    /// Maximum blocks per multiprocessor.
    max_blocks_per_sm: u32 = 32,
    /// Warp/wavefront size.
    warp_size: u32 = 32,
    /// Total shared memory per multiprocessor (bytes).
    shared_memory_per_sm: u32 = 49152,
    /// Total registers per multiprocessor.
    registers_per_sm: u32 = 65536,
    /// Maximum registers per thread.
    max_registers_per_thread: u32 = 255,
    /// Register allocation granularity.
    register_alloc_granularity: u32 = 256,
    /// Shared memory allocation granularity (bytes).
    shared_mem_alloc_granularity: u32 = 256,
    /// Number of multiprocessors/compute units.
    num_sms: u32 = 1,
    /// Memory bandwidth (GB/s).
    memory_bandwidth_gbps: f64 = 100.0,
    /// Peak compute throughput (GFLOPS).
    peak_gflops: f64 = 1000.0,
    /// L2 cache size (bytes).
    l2_cache_size: usize = 1024 * 1024,

    /// Create capabilities for CUDA compute capability.
    pub fn forCudaComputeCapability(major: u32, minor: u32) DeviceCapabilities {
        return switch (major) {
            // Ampere (SM 8.x)
            8 => .{
                .max_threads_per_sm = 2048,
                .max_warps_per_sm = 64,
                .max_blocks_per_sm = if (minor >= 6) 32 else 16,
                .warp_size = 32,
                .shared_memory_per_sm = if (minor >= 6) 100 * 1024 else 164 * 1024,
                .registers_per_sm = 65536,
                .max_registers_per_thread = 255,
                .register_alloc_granularity = 256,
                .shared_mem_alloc_granularity = 128,
            },
            // Turing (SM 7.5) / Volta (SM 7.0)
            7 => .{
                .max_threads_per_sm = 2048,
                .max_warps_per_sm = 64,
                .max_blocks_per_sm = 16,
                .warp_size = 32,
                .shared_memory_per_sm = if (minor >= 5) 64 * 1024 else 96 * 1024,
                .registers_per_sm = 65536,
                .max_registers_per_thread = 255,
                .register_alloc_granularity = 256,
                .shared_mem_alloc_granularity = 256,
            },
            // Pascal (SM 6.x)
            6 => .{
                .max_threads_per_sm = 2048,
                .max_warps_per_sm = 64,
                .max_blocks_per_sm = 32,
                .warp_size = 32,
                .shared_memory_per_sm = 49152,
                .registers_per_sm = 65536,
                .max_registers_per_thread = 255,
                .register_alloc_granularity = 256,
                .shared_mem_alloc_granularity = 256,
            },
            // Maxwell (SM 5.x)
            5 => .{
                .max_threads_per_sm = 2048,
                .max_warps_per_sm = 64,
                .max_blocks_per_sm = 32,
                .warp_size = 32,
                .shared_memory_per_sm = 49152,
                .registers_per_sm = 65536,
                .max_registers_per_thread = 255,
                .register_alloc_granularity = 256,
                .shared_mem_alloc_granularity = 256,
            },
            // Default conservative values
            else => .{},
        };
    }

    /// Create capabilities for Vulkan subgroup size.
    pub fn forVulkan(subgroup_size: u32, max_workgroup_size: u32, shared_memory: u32) DeviceCapabilities {
        return .{
            .max_threads_per_sm = max_workgroup_size * 4, // Estimate
            .max_warps_per_sm = 64,
            .max_blocks_per_sm = 16,
            .warp_size = subgroup_size,
            .shared_memory_per_sm = shared_memory,
            .registers_per_sm = 65536,
            .max_registers_per_thread = 128,
            .register_alloc_granularity = 256,
            .shared_mem_alloc_granularity = 256,
        };
    }

    /// Create capabilities for Metal SIMD width.
    pub fn forMetal(simd_width: u32, max_threads_per_threadgroup: u32, threadgroup_memory: u32) DeviceCapabilities {
        return .{
            .max_threads_per_sm = max_threads_per_threadgroup * 4,
            .max_warps_per_sm = 64,
            .max_blocks_per_sm = 16,
            .warp_size = simd_width,
            .shared_memory_per_sm = threadgroup_memory,
            .registers_per_sm = 32768,
            .max_registers_per_thread = 128,
            .register_alloc_granularity = 256,
            .shared_mem_alloc_granularity = 16,
        };
    }
};

/// Kernel resource requirements.
pub const KernelRequirements = struct {
    /// Registers per thread.
    registers_per_thread: u32 = 32,
    /// Static shared memory per block (bytes).
    static_shared_memory: u32 = 0,
    /// Dynamic shared memory per block (bytes).
    dynamic_shared_memory: u32 = 0,
    /// Requested block size (0 = auto-select).
    requested_block_size: u32 = 0,
    /// Minimum blocks per SM for occupancy bound.
    min_blocks_per_sm: u32 = 1,

    /// Total shared memory per block.
    pub fn totalSharedMemory(self: KernelRequirements) u32 {
        return self.static_shared_memory + self.dynamic_shared_memory;
    }
};

/// Occupancy calculation result.
pub const OccupancyResult = struct {
    /// Achieved occupancy (0.0 - 1.0).
    occupancy: f32,
    /// Active warps per SM.
    active_warps_per_sm: u32,
    /// Active blocks per SM.
    active_blocks_per_sm: u32,
    /// Active threads per SM.
    active_threads_per_sm: u32,
    /// Limiting factor for occupancy.
    limiting_factor: LimitingFactor,
    /// Maximum theoretical occupancy.
    max_theoretical_occupancy: f32,

    /// Occupancy limiting factors.
    pub const LimitingFactor = enum {
        threads,
        warps,
        blocks,
        registers,
        shared_memory,
        none,
    };

    /// Check if occupancy is good (>= 50%).
    pub fn isGood(self: OccupancyResult) bool {
        return self.occupancy >= 0.5;
    }

    /// Check if occupancy is excellent (>= 75%).
    pub fn isExcellent(self: OccupancyResult) bool {
        return self.occupancy >= 0.75;
    }

    /// Get improvement suggestions based on limiting factor.
    pub fn getImprovementSuggestion(self: OccupancyResult) []const u8 {
        return switch (self.limiting_factor) {
            .threads => "Increase block size to utilize more threads per SM",
            .warps => "Block size already near optimal for warp utilization",
            .blocks => "Reduce shared memory or registers to allow more blocks",
            .registers => "Reduce register usage or use -maxrregcount compiler flag",
            .shared_memory => "Reduce shared memory usage per block",
            .none => "Occupancy is at maximum achievable level",
        };
    }
};

/// Optimal launch configuration.
pub const OptimalConfig = struct {
    /// Recommended block size (threads per block).
    block_size: u32,
    /// Grid size for the given problem.
    grid_size: u32,
    /// Expected occupancy with this configuration.
    occupancy: f32,
    /// Active blocks per SM.
    active_blocks_per_sm: u32,
    /// Efficiency score (0.0 - 1.0) considering multiple factors.
    efficiency_score: f32,
};

/// Calculate occupancy for a given configuration.
pub fn calculateOccupancy(
    caps: DeviceCapabilities,
    reqs: KernelRequirements,
    block_size: u32,
) OccupancyResult {
    // Validate block size
    if (block_size == 0 or block_size > caps.max_threads_per_sm) {
        return .{
            .occupancy = 0,
            .active_warps_per_sm = 0,
            .active_blocks_per_sm = 0,
            .active_threads_per_sm = 0,
            .limiting_factor = .threads,
            .max_theoretical_occupancy = 0,
        };
    }

    // Calculate warps per block
    const warps_per_block = (block_size + caps.warp_size - 1) / caps.warp_size;

    // Calculate blocks limited by different resources
    const blocks_by_threads = caps.max_threads_per_sm / block_size;
    const blocks_by_warps = caps.max_warps_per_sm / warps_per_block;
    const blocks_by_max = caps.max_blocks_per_sm;

    // Register-limited blocks
    var blocks_by_registers: u32 = caps.max_blocks_per_sm;
    if (reqs.registers_per_thread > 0) {
        const registers_per_warp = ((reqs.registers_per_thread * caps.warp_size +
            caps.register_alloc_granularity - 1) /
            caps.register_alloc_granularity) * caps.register_alloc_granularity;
        const registers_per_block = registers_per_warp * warps_per_block;
        if (registers_per_block > 0) {
            blocks_by_registers = caps.registers_per_sm / registers_per_block;
        }
    }

    // Shared memory-limited blocks
    var blocks_by_shared: u32 = caps.max_blocks_per_sm;
    const total_shared = reqs.totalSharedMemory();
    if (total_shared > 0) {
        const aligned_shared = ((total_shared + caps.shared_mem_alloc_granularity - 1) /
            caps.shared_mem_alloc_granularity) * caps.shared_mem_alloc_granularity;
        if (aligned_shared > 0) {
            blocks_by_shared = caps.shared_memory_per_sm / aligned_shared;
        }
    }

    // Find limiting factor and actual blocks
    var active_blocks: u32 = blocks_by_threads;
    var limiting_factor: OccupancyResult.LimitingFactor = .threads;

    if (blocks_by_warps < active_blocks) {
        active_blocks = blocks_by_warps;
        limiting_factor = .warps;
    }
    if (blocks_by_max < active_blocks) {
        active_blocks = blocks_by_max;
        limiting_factor = .blocks;
    }
    if (blocks_by_registers < active_blocks) {
        active_blocks = blocks_by_registers;
        limiting_factor = .registers;
    }
    if (blocks_by_shared < active_blocks) {
        active_blocks = blocks_by_shared;
        limiting_factor = .shared_memory;
    }

    // Ensure minimum blocks constraint
    active_blocks = @max(active_blocks, reqs.min_blocks_per_sm);
    active_blocks = @min(active_blocks, caps.max_blocks_per_sm);

    // Calculate final metrics
    const active_warps = active_blocks * warps_per_block;
    const active_threads = active_blocks * block_size;
    const occupancy = @as(f32, @floatFromInt(active_warps)) /
        @as(f32, @floatFromInt(caps.max_warps_per_sm));

    // Maximum theoretical occupancy
    const max_occupancy = @min(1.0, @as(f32, @floatFromInt(caps.max_threads_per_sm)) /
        @as(f32, @floatFromInt(caps.max_warps_per_sm * caps.warp_size)));

    if (occupancy >= max_occupancy - 0.001) {
        limiting_factor = .none;
    }

    return .{
        .occupancy = @min(occupancy, 1.0),
        .active_warps_per_sm = @min(active_warps, caps.max_warps_per_sm),
        .active_blocks_per_sm = active_blocks,
        .active_threads_per_sm = @min(active_threads, caps.max_threads_per_sm),
        .limiting_factor = limiting_factor,
        .max_theoretical_occupancy = max_occupancy,
    };
}

/// Find optimal block size for maximum occupancy.
pub fn findOptimalBlockSize(
    caps: DeviceCapabilities,
    reqs: KernelRequirements,
    problem_size: usize,
) OptimalConfig {
    // If requested block size is specified, use it
    if (reqs.requested_block_size > 0) {
        const occ = calculateOccupancy(caps, reqs, reqs.requested_block_size);
        const grid_size = @as(u32, @intCast((problem_size + reqs.requested_block_size - 1) / reqs.requested_block_size));
        return .{
            .block_size = reqs.requested_block_size,
            .grid_size = grid_size,
            .occupancy = occ.occupancy,
            .active_blocks_per_sm = occ.active_blocks_per_sm,
            .efficiency_score = occ.occupancy,
        };
    }

    // Try different block sizes (multiples of warp size)
    var best_config = OptimalConfig{
        .block_size = caps.warp_size,
        .grid_size = 1,
        .occupancy = 0,
        .active_blocks_per_sm = 0,
        .efficiency_score = 0,
    };

    // Common block sizes to try (optimized for different scenarios)
    const block_sizes = [_]u32{
        32,   64,  96,  128, 160, 192, 224, 256,
        288,  320, 384, 448, 512, 640, 768, 896,
        1024,
    };

    for (block_sizes) |block_size| {
        if (block_size > caps.max_threads_per_sm) continue;

        const occ = calculateOccupancy(caps, reqs, block_size);

        // Calculate efficiency score considering multiple factors
        const efficiency = calculateEfficiencyScore(caps, occ, block_size, problem_size);

        if (efficiency > best_config.efficiency_score) {
            const grid_size = @as(u32, @intCast((problem_size + block_size - 1) / block_size));
            best_config = .{
                .block_size = block_size,
                .grid_size = grid_size,
                .occupancy = occ.occupancy,
                .active_blocks_per_sm = occ.active_blocks_per_sm,
                .efficiency_score = efficiency,
            };
        }
    }

    return best_config;
}

/// Calculate efficiency score combining occupancy, parallelism, and memory access patterns.
fn calculateEfficiencyScore(
    caps: DeviceCapabilities,
    occ: OccupancyResult,
    block_size: u32,
    problem_size: usize,
) f32 {
    // Base score from occupancy
    var score: f32 = occ.occupancy;

    // Penalty for very small problems (not enough work to hide latency)
    const grid_size = (problem_size + block_size - 1) / block_size;
    const total_blocks = grid_size;
    const min_blocks_for_full_gpu = caps.num_sms * 4; // Want at least 4 blocks per SM

    if (total_blocks < min_blocks_for_full_gpu) {
        const parallelism_factor = @as(f32, @floatFromInt(total_blocks)) /
            @as(f32, @floatFromInt(min_blocks_for_full_gpu));
        score *= @max(0.5, parallelism_factor);
    }

    // Bonus for block sizes that are powers of 2 (better memory coalescing)
    if ((block_size & (block_size - 1)) == 0) {
        score *= 1.05;
    }

    // Slight preference for larger blocks (reduces launch overhead)
    const size_bonus = @min(0.1, @as(f32, @floatFromInt(block_size)) / 1024.0 * 0.1);
    score += size_bonus;

    // Penalty if block size doesn't divide problem evenly (wasted threads)
    const remainder = @as(u32, @intCast(problem_size % block_size));
    if (remainder > 0) {
        const waste_factor = 1.0 - (@as(f32, @floatFromInt(remainder)) /
            @as(f32, @floatFromInt(block_size))) * 0.1;
        score *= waste_factor;
    }

    return @min(score, 1.0);
}

/// Estimate kernel performance based on roofline model.
pub const PerformanceEstimate = struct {
    /// Expected execution time (microseconds).
    expected_time_us: f64,
    /// Theoretical peak performance (GFLOPS).
    peak_gflops: f64,
    /// Achieved performance (GFLOPS).
    achieved_gflops: f64,
    /// Arithmetic intensity (FLOPs/byte).
    arithmetic_intensity: f64,
    /// Whether kernel is compute-bound or memory-bound.
    is_compute_bound: bool,
    /// Efficiency relative to roofline (0.0 - 1.0).
    roofline_efficiency: f64,
};

/// Estimate kernel performance using roofline model.
pub fn estimatePerformance(
    caps: DeviceCapabilities,
    occ: OccupancyResult,
    flops_per_thread: u64,
    bytes_per_thread: u64,
    problem_size: usize,
    block_size: u32,
) PerformanceEstimate {
    const total_flops = @as(f64, @floatFromInt(flops_per_thread)) *
        @as(f64, @floatFromInt(problem_size));
    const total_bytes = @as(f64, @floatFromInt(bytes_per_thread)) *
        @as(f64, @floatFromInt(problem_size));

    // Arithmetic intensity (FLOPs per byte)
    const arithmetic_intensity = if (total_bytes > 0) total_flops / total_bytes else 0;

    // Roofline model: performance = min(peak_compute, bandwidth * arithmetic_intensity)
    const compute_bound_perf = caps.peak_gflops;
    const memory_bound_perf = caps.memory_bandwidth_gbps * arithmetic_intensity;

    const is_compute_bound = compute_bound_perf < memory_bound_perf;
    const theoretical_peak = @min(compute_bound_perf, memory_bound_perf);

    // Adjust for occupancy
    const achieved_peak = theoretical_peak * occ.occupancy;

    // Calculate expected execution time
    const expected_time_us = if (achieved_peak > 0)
        total_flops / (achieved_peak * 1000.0) // Convert GFLOPS to MFLOPS/us
    else
        0;

    // Calculate roofline efficiency
    const grid_size = (problem_size + block_size - 1) / block_size;
    const parallelism_factor = @min(1.0, @as(f64, @floatFromInt(grid_size)) /
        @as(f64, @floatFromInt(caps.num_sms * 4)));
    const roofline_efficiency = occ.occupancy * parallelism_factor;

    return .{
        .expected_time_us = expected_time_us,
        .peak_gflops = theoretical_peak,
        .achieved_gflops = achieved_peak,
        .arithmetic_intensity = arithmetic_intensity,
        .is_compute_bound = is_compute_bound,
        .roofline_efficiency = roofline_efficiency,
    };
}

/// Multi-kernel occupancy optimizer for kernel fusion decisions.
pub const FusionAnalysis = struct {
    /// Whether fusion is beneficial.
    should_fuse: bool,
    /// Expected speedup from fusion.
    expected_speedup: f32,
    /// Combined kernel occupancy.
    fused_occupancy: f32,
    /// Memory traffic reduction factor.
    memory_reduction: f32,
    /// Reason for fusion decision.
    reason: []const u8,
};

/// Analyze whether two kernels should be fused.
pub fn analyzeFusion(
    caps: DeviceCapabilities,
    kernel1_reqs: KernelRequirements,
    kernel2_reqs: KernelRequirements,
    intermediate_bytes: usize,
) FusionAnalysis {
    // Combined requirements for fused kernel
    const fused_reqs = KernelRequirements{
        .registers_per_thread = kernel1_reqs.registers_per_thread + kernel2_reqs.registers_per_thread,
        .static_shared_memory = kernel1_reqs.static_shared_memory + kernel2_reqs.static_shared_memory,
        .dynamic_shared_memory = @max(kernel1_reqs.dynamic_shared_memory, kernel2_reqs.dynamic_shared_memory),
    };

    // Calculate occupancy for separate vs fused execution
    const block_size: u32 = 256;
    const occ1 = calculateOccupancy(caps, kernel1_reqs, block_size);
    const occ2 = calculateOccupancy(caps, kernel2_reqs, block_size);
    const occ_fused = calculateOccupancy(caps, fused_reqs, block_size);

    // Calculate average separate occupancy
    const avg_separate_occ = (occ1.occupancy + occ2.occupancy) / 2.0;

    // Memory traffic reduction from eliminating intermediate buffer
    const memory_bandwidth_time = @as(f32, @floatFromInt(intermediate_bytes * 2)) /
        @as(f32, @floatCast(caps.memory_bandwidth_gbps * 1e9));

    // Launch overhead reduction (approximate)
    const launch_overhead_us: f32 = 5.0; // ~5us per kernel launch

    // Decision factors
    const occupancy_ratio = if (avg_separate_occ > 0.001)
        occ_fused.occupancy / avg_separate_occ
    else
        1.0;

    const memory_reduction = if (intermediate_bytes > 0)
        @as(f32, @floatFromInt(intermediate_bytes)) / 1e6 // MB saved
    else
        0;

    // Fusion is beneficial if:
    // 1. Fused occupancy is at least 70% of average separate occupancy
    // 2. OR significant memory traffic is eliminated
    const should_fuse = (occupancy_ratio >= 0.7 and memory_reduction > 0.1) or
        (memory_reduction > 1.0); // >1MB saved

    // Estimate speedup
    var expected_speedup: f32 = 1.0;
    if (should_fuse) {
        // Speedup from reduced memory traffic
        expected_speedup += memory_bandwidth_time * 1e6 / 100.0; // Normalize
        // Speedup from reduced launch overhead
        expected_speedup += launch_overhead_us / 100.0;
        // Adjust for occupancy change
        expected_speedup *= occupancy_ratio;
    }

    const reason = if (should_fuse)
        (if (memory_reduction > 1.0)
            "Significant memory traffic reduction"
        else
            "Good occupancy with memory savings")
    else
        (if (occupancy_ratio < 0.7)
            "Fused kernel occupancy too low"
        else
            "Insufficient memory traffic to benefit");

    return .{
        .should_fuse = should_fuse,
        .expected_speedup = @max(expected_speedup, 1.0),
        .fused_occupancy = occ_fused.occupancy,
        .memory_reduction = memory_reduction,
        .reason = reason,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "occupancy calculation basic" {
    const caps = DeviceCapabilities{};
    const reqs = KernelRequirements{};

    const occ = calculateOccupancy(caps, reqs, 256);

    try std.testing.expect(occ.occupancy > 0);
    try std.testing.expect(occ.occupancy <= 1.0);
    try std.testing.expect(occ.active_blocks_per_sm > 0);
}

test "occupancy with high register usage" {
    const caps = DeviceCapabilities{};
    const reqs = KernelRequirements{
        .registers_per_thread = 128, // High register usage
    };

    const occ = calculateOccupancy(caps, reqs, 256);

    // Should be register-limited
    try std.testing.expect(occ.occupancy < 1.0);
}

test "occupancy with shared memory" {
    const caps = DeviceCapabilities{};
    const reqs = KernelRequirements{
        .static_shared_memory = 16 * 1024, // 16KB per block
    };

    const occ = calculateOccupancy(caps, reqs, 256);

    // Should be shared memory-limited
    try std.testing.expect(occ.active_blocks_per_sm <= 3);
}

test "find optimal block size" {
    const caps = DeviceCapabilities{};
    const reqs = KernelRequirements{};

    const optimal = findOptimalBlockSize(caps, reqs, 100000);

    try std.testing.expect(optimal.block_size >= 32);
    try std.testing.expect(optimal.block_size <= 1024);
    try std.testing.expect(optimal.occupancy > 0);
    try std.testing.expect(optimal.grid_size > 0);
}

test "performance estimate" {
    const caps = DeviceCapabilities{
        .peak_gflops = 10000,
        .memory_bandwidth_gbps = 500,
    };
    const reqs = KernelRequirements{};

    const occ = calculateOccupancy(caps, reqs, 256);
    const perf = estimatePerformance(caps, occ, 100, 8, 1000000, 256);

    try std.testing.expect(perf.expected_time_us >= 0);
    try std.testing.expect(perf.arithmetic_intensity > 0);
}

test "fusion analysis" {
    const caps = DeviceCapabilities{};
    const kernel1 = KernelRequirements{
        .registers_per_thread = 32,
    };
    const kernel2 = KernelRequirements{
        .registers_per_thread = 24,
    };

    const analysis = analyzeFusion(caps, kernel1, kernel2, 4 * 1024 * 1024); // 4MB intermediate

    try std.testing.expect(analysis.fused_occupancy > 0);
    try std.testing.expect(analysis.expected_speedup >= 1.0);
}

test "cuda compute capability presets" {
    const ampere = DeviceCapabilities.forCudaComputeCapability(8, 6);
    try std.testing.expectEqual(@as(u32, 2048), ampere.max_threads_per_sm);
    try std.testing.expectEqual(@as(u32, 32), ampere.max_blocks_per_sm);

    const turing = DeviceCapabilities.forCudaComputeCapability(7, 5);
    try std.testing.expectEqual(@as(u32, 2048), turing.max_threads_per_sm);
    try std.testing.expectEqual(@as(u32, 16), turing.max_blocks_per_sm);
}

test {
    std.testing.refAllDecls(@This());
}
