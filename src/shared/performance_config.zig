//! Performance Configuration
//!
//! This module provides compile-time and runtime performance configurations
//! for optimizing the ABI framework based on target platform and use case.

const std = @import("std");
const builtin = @import("builtin");

/// Performance optimization levels
pub const OptimizationLevel = enum {
    /// Minimal optimizations for debugging
    debug,
    /// Balanced optimizations for development
    development,
    /// Aggressive optimizations for production
    production,
    /// Maximum optimizations for benchmarks
    maximum,
};

/// Platform-specific optimizations
pub const PlatformOptimizations = struct {
    /// Enable SIMD optimizations
    enable_simd: bool,
    /// Enable vectorization hints
    enable_vectorization: bool,
    /// Enable memory pooling
    enable_memory_pooling: bool,
    /// Enable batch operations
    enable_batch_ops: bool,
    /// Enable cache-friendly algorithms
    enable_cache_optimization: bool,
    /// Enable parallel processing
    enable_parallel: bool,
    /// Enable JIT compilation
    enable_jit: bool,
};

/// Get platform-specific optimizations
pub fn getPlatformOptimizations(level: OptimizationLevel) PlatformOptimizations {
    return switch (level) {
        .debug => PlatformOptimizations{
            .enable_simd = false,
            .enable_vectorization = false,
            .enable_memory_pooling = false,
            .enable_batch_ops = false,
            .enable_cache_optimization = false,
            .enable_parallel = false,
            .enable_jit = false,
        },
        .development => PlatformOptimizations{
            .enable_simd = true,
            .enable_vectorization = true,
            .enable_memory_pooling = true,
            .enable_batch_ops = false,
            .enable_cache_optimization = true,
            .enable_parallel = false,
            .enable_jit = false,
        },
        .production => PlatformOptimizations{
            .enable_simd = true,
            .enable_vectorization = true,
            .enable_memory_pooling = true,
            .enable_batch_ops = true,
            .enable_cache_optimization = true,
            .enable_parallel = true,
            .enable_jit = false,
        },
        .maximum => PlatformOptimizations{
            .enable_simd = true,
            .enable_vectorization = true,
            .enable_memory_pooling = true,
            .enable_batch_ops = true,
            .enable_cache_optimization = true,
            .enable_parallel = true,
            .enable_jit = true,
        },
    };
}

/// Performance tuning parameters
pub const PerformanceTuning = struct {
    /// Vector search batch size
    vector_search_batch_size: usize,
    /// Memory pool initial capacity
    memory_pool_initial_capacity: usize,
    /// Cache line size for alignment
    cache_line_size: usize,
    /// Maximum parallel workers
    max_parallel_workers: usize,
    /// SIMD vector width
    simd_vector_width: usize,
    /// JIT compilation threshold
    jit_threshold: usize,
};

/// Get performance tuning parameters for platform
pub fn getPerformanceTuning(platform: PlatformOptimizations) PerformanceTuning {
    return PerformanceTuning{
        .vector_search_batch_size = if (platform.enable_batch_ops) 64 else 16,
        .memory_pool_initial_capacity = if (platform.enable_memory_pooling) 1024 else 64,
        .cache_line_size = getCacheLineSize(),
        .max_parallel_workers = if (platform.enable_parallel) getMaxWorkers() else 1,
        .simd_vector_width = if (platform.enable_simd) getSimdWidth() else 1,
        .jit_threshold = if (platform.enable_jit) 1000 else std.math.maxInt(usize),
    };
}

/// Get cache line size for the target platform
fn getCacheLineSize() usize {
    return switch (builtin.target.cpu.arch) {
        .x86_64 => 64,
        .aarch64 => 64,
        .arm => 32,
        .riscv64 => 64,
        else => 64, // Conservative default
    };
}

/// Get maximum number of workers for parallel processing
fn getMaxWorkers() usize {
    return @max(1, std.Thread.getCpuCount() catch 1);
}

/// Get SIMD vector width for the target platform
fn getSimdWidth() usize {
    return switch (builtin.target.cpu.arch) {
        .x86_64 => 8, // AVX2/AVX-512
        .aarch64 => 4, // NEON
        .arm => 2, // NEON
        .riscv64 => 4, // RVV
        else => 4, // Conservative default
    };
}

/// Compile-time performance configuration
pub const CompileTimeConfig = struct {
    /// Enable debug assertions
    debug_assertions: bool,
    /// Enable bounds checking
    bounds_checking: bool,
    /// Enable overflow checking
    overflow_checking: bool,
    /// Enable runtime safety
    runtime_safety: bool,
    /// Optimization level
    optimization: builtin.OptimizeMode,
};

/// Get compile-time configuration
pub fn getCompileTimeConfig() CompileTimeConfig {
    return CompileTimeConfig{
        .debug_assertions = builtin.mode == .Debug,
        .bounds_checking = builtin.mode == .Debug,
        .overflow_checking = builtin.mode == .Debug,
        .runtime_safety = builtin.mode == .Debug,
        .optimization = builtin.optimize,
    };
}

/// Runtime performance monitoring
pub const PerformanceMonitor = struct {
    /// Allocation count
    allocation_count: std.atomic.Value(usize),
    /// Total allocated bytes
    total_allocated: std.atomic.Value(usize),
    /// Cache hit rate
    cache_hit_rate: std.atomic.Value(f32),
    /// Average operation time
    avg_operation_time: std.atomic.Value(f64),
    
    pub fn init() PerformanceMonitor {
        return PerformanceMonitor{
            .allocation_count = std.atomic.Value(usize).init(0),
            .total_allocated = std.atomic.Value(usize).init(0),
            .cache_hit_rate = std.atomic.Value(f32).init(0.0),
            .avg_operation_time = std.atomic.Value(f64).init(0.0),
        };
    }
    
    pub fn recordAllocation(self: *PerformanceMonitor, size: usize) void {
        _ = self.allocation_count.fetchAdd(1, .monotonic);
        _ = self.total_allocated.fetchAdd(size, .monotonic);
    }
    
    pub fn updateCacheHitRate(self: *PerformanceMonitor, hit_rate: f32) void {
        _ = self.cache_hit_rate.store(hit_rate, .monotonic);
    }
    
    pub fn updateOperationTime(self: *PerformanceMonitor, time_ns: u64) void {
        const current_avg = self.avg_operation_time.load(.monotonic);
        const new_avg = (current_avg + @as(f64, @floatFromInt(time_ns))) / 2.0;
        _ = self.avg_operation_time.store(new_avg, .monotonic);
    }
    
    pub fn getStats(self: *const PerformanceMonitor) struct {
        allocation_count: usize,
        total_allocated: usize,
        cache_hit_rate: f32,
        avg_operation_time: f64,
    } {
        return .{
            .allocation_count = self.allocation_count.load(.monotonic),
            .total_allocated = self.total_allocated.load(.monotonic),
            .cache_hit_rate = self.cache_hit_rate.load(.monotonic),
            .avg_operation_time = self.avg_operation_time.load(.monotonic),
        };
    }
};