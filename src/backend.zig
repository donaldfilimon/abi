//! @Definitions
//!
//! **GpuBackendConfig:**
//!   Configures the GPU backend. Fields:
//!     - `max_batch_size`: Maximum vectors per batch (default: 1024).
//!     - `memory_limit`: Maximum GPU memory in bytes (default: 512MB).
//!     - `debug_validation`: Enable/disable GPU debug/validation layers.
//!     - `power_preference`: Preferred GPU power profile (e.g., high_performance, low_power).
//!
//! **GpuBackend:**
//!   Main context for GPU-accelerated vector search. Fields:
//!     - `config`: GpuBackendConfig instance.
//!     - `allocator`: CPU-side memory allocator.
//!     - `gpu_available`: Whether a suitable GPU is available.
//!     - `memory_used`: Current GPU memory usage.
//!     - `Error`: Error set for all GPU backend operations.
//!   Methods for initialization, cleanup, and vector search (with CPU fallback).
//!
//! **GpuBackend.Error:**
//!   Error set for GPU backend operations, including:
//!     - `GpuNotAvailable`, `GpuInitializationFailed`, `AdapterNotFound`, `DeviceCreationFailed`,
//!       `PipelineCreationFailed`, `BufferCreationFailed`, `OperationTimeout`, `InsufficientGpuMemory`,
//!       `ShaderCompilationFailed`, `InvalidBufferSize`, `GpuOperationFailed`,
//!       and all `std.mem.Allocator.Error` errors.
//!
//! **GpuBackend.init(allocator, config):**
//!   Initializes a GpuBackend instance with the given allocator and configuration.
//!   Returns a pointer to the backend or an error if initialization fails.
//!
//! **GpuBackend.deinit():**
//!   Cleans up and releases all resources held by the backend.
//!
//! **GpuBackend.searchSimilar(db, query, top_k):**
//!   Performs a vector similarity search for the given query vector against the database.
//!   Returns the top_k closest results. Falls back to CPU if GPU is unavailable.
//!
//! **GpuBackend.batchSearch(db, queries, top_k):**
//!   Batch version of searchSimilar, processes multiple queries and returns results for each.
//!
//! **BatchConfig:**
//!   Configuration for batch processing. Fields:
//!     - `parallel_queries`: Number of queries to process in parallel (default: 4).
//!     - `max_batch_size`: Maximum batch size for each batch (default: 1024).
//!     - `report_progress`: Whether to log progress information.
//!
//! **BatchProcessor:**
//!   Utility for batch processing of vector search queries. Fields:
//!     - `backend`: Pointer to the GpuBackend instance.
//!     - `config`: BatchConfig instance.
//!   Methods for batch processing with or without progress reporting and callbacks.
//!
//! **GpuStats:**
//!   Tracks performance statistics for GPU operations. Fields:
//!     - `total_operations`: Total number of operations performed.
//!     - `total_gpu_time`: Total time spent in GPU operations (in microseconds).
//!     - `peak_memory_usage`: Highest observed memory usage.
//!     - `cpu_fallback_count`: Number of times CPU fallback was used.
//!   Methods to record operations, compute averages, and print statistics.

const std = @import("std");
const gpu = std.gpu;

const database = @import("database.zig");
const root = @import("root.zig");

/// Configuration for the GPU backend.
pub const GpuBackendConfig = struct {
    max_batch_size: u32 = 1024,
    memory_limit: u64 = 512 * 1024 * 1024, // 512MB
    debug_validation: bool = false,
    power_preference: gpu.PowerPreference = .high_performance,
};

/// Main context for GPU-accelerated operations.
pub const GpuBackend = struct {
    config: GpuBackendConfig,
    allocator: std.mem.Allocator,
    gpu_available: bool = false,
    memory_used: u64 = 0,

    /// Error set for all GPU backend operations.
    pub const Error = error{
        GpuNotAvailable,
        GpuInitializationFailed,
        AdapterNotFound,
        DeviceCreationFailed,
        PipelineCreationFailed,
        BufferCreationFailed,
        OperationTimeout,
        InsufficientGpuMemory,
        ShaderCompilationFailed,
        InvalidBufferSize,
        GpuOperationFailed,
    } || std.mem.Allocator.Error;

    /// Initialize the GPU backend with the given configuration.
    pub fn init(allocator: std.mem.Allocator, config: GpuBackendConfig) Error!*GpuBackend {
        const self = try allocator.create(GpuBackend);
        errdefer allocator.destroy(self);

        self.* = GpuBackend{
            .allocator = allocator,
            .config = config,
            .gpu_available = false,
            .memory_used = 0,
        };

        self.gpu_available = checkGpuAvailability();

        if (!self.gpu_available) {
            std.log.warn("GPU not available, using CPU fallback for compute operations", .{});
        }

        return self;
    }

    /// Clean up and release all resources held by the backend.
    pub fn deinit(self: *GpuBackend) void {
        self.allocator.destroy(self);
    }

    /// Check if a suitable GPU is available for compute.
    fn checkGpuAvailability() bool {
        // In a real implementation, this would check for WebGPU/Vulkan/Metal/DX12 support.
        // For now, return false to use CPU fallback.
        return false;
    }

    /// Perform a vector similarity search for the given query vector against the database.
    /// Returns the top_k closest results. Falls back to CPU if GPU is unavailable.
    pub fn searchSimilar(self: *GpuBackend, db: *database.Db, query: []const f32, top_k: usize) Error![]database.Db.Result {
        if (query.len == 0 or query.len != db.header.dim)
            return Error.InvalidBufferSize;

        if (!self.gpu_available) {
            return self.searchSimilarCpu(db, query, top_k);
        }

        // GPU implementation would go here (see gpu_examples.zig for reference).
        return Error.GpuNotAvailable;
    }

    /// CPU fallback implementation for vector similarity search.
    fn searchSimilarCpu(self: *GpuBackend, db: *database.Db, query: []const f32, top_k: usize) Error![]database.Db.Result {
        const vector_count = db.header.row_count;
        const dimension = db.header.dim;

        if (vector_count == 0)
            return self.allocator.alloc(database.Db.Result, 0);

        var results = try self.allocator.alloc(database.Db.Result, vector_count);
        defer self.allocator.free(results);

        const vector_buffer = try self.allocator.alloc(f32, dimension);
        defer self.allocator.free(vector_buffer);

        const record_size = @as(u64, dimension) * @sizeOf(f32);

        for (0..vector_count) |i| {
            const offset = db.header.records_off + @as(u64, i) * record_size;
            try db.file.seekTo(offset);

            const vector_bytes = std.mem.sliceAsBytes(vector_buffer);
            try db.file.reader().readNoEof(vector_bytes);

            var distance: f32 = 0.0;
            for (vector_buffer, 0..) |val, j| {
                const diff = val - query[j];
                distance += diff * diff;
            }

            results[i] = .{
                .index = @intCast(i),
                .score = distance,
            };
        }

        std.mem.sort(database.Db.Result, results, {}, comptime database.Db.Result.lessThanAsc);

        const result_count = @min(top_k, results.len);
        const final_results = try self.allocator.alloc(database.Db.Result, result_count);
        @memcpy(final_results, results[0..result_count]);

        return final_results;
    }

    /// Returns true if there is enough GPU memory for an operation.
    pub fn hasMemoryFor(self: *const GpuBackend, bytes: u64) bool {
        return self.memory_used + bytes <= self.config.memory_limit;
    }

    /// Batch version of searchSimilar, processes multiple queries and returns results for each.
    pub fn batchSearch(self: *GpuBackend, db: *database.Db, queries: []const []const f32, top_k: usize) Error![][]database.Db.Result {
        const results = try self.allocator.alloc([]database.Db.Result, queries.len);
        errdefer self.allocator.free(results);

        var completed: usize = 0;
        errdefer {
            for (results[0..completed]) |result| {
                self.allocator.free(result);
            }
        }

        for (queries, 0..) |query, i| {
            results[i] = try self.searchSimilar(db, query, top_k);
            completed += 1;
        }

        return results;
    }
};

/// Configuration for batch processing.
pub const BatchConfig = struct {
    parallel_queries: u32 = 4,
    max_batch_size: u32 = 1024,
    report_progress: bool = false,
};

/// Utility for batch processing of vector search queries.
pub const BatchProcessor = struct {
    backend: *GpuBackend,
    config: BatchConfig,

    pub fn init(backend: *GpuBackend, config: BatchConfig) BatchProcessor {
        return .{
            .backend = backend,
            .config = config,
        };
    }

    /// Process a batch of queries with optional progress reporting.
    pub fn processBatch(self: *BatchProcessor, db: *database.Db, queries: []const []const f32, top_k: usize) ![][]database.Db.Result {
        const batch_count = (queries.len + self.config.max_batch_size - 1) / self.config.max_batch_size;

        if (self.config.report_progress) {
            std.log.info("Processing {} queries in {} batches", .{ queries.len, batch_count });
        }

        return self.backend.batchSearch(db, queries, top_k);
    }

    /// Process queries with a callback for each result.
    pub fn processBatchWithCallback(
        self: *BatchProcessor,
        db: *database.Db,
        queries: []const []const f32,
        top_k: usize,
        callback: *const fn (index: usize, results: []database.Db.Result) void,
    ) !void {
        for (queries, 0..) |query, i| {
            const results = try self.backend.searchSimilar(db, query, top_k);
            defer self.backend.allocator.free(results);
            callback(i, results);
        }
    }
};

/// Tracks performance statistics for GPU operations.
pub const GpuStats = struct {
    total_operations: u64 = 0,
    total_gpu_time: u64 = 0,
    peak_memory_usage: u64 = 0,
    cpu_fallback_count: u64 = 0,

    pub fn recordOperation(self: *GpuStats, duration_us: u64, memory_used: u64, used_cpu: bool) void {
        self.total_operations += 1;
        self.total_gpu_time += duration_us;
        self.peak_memory_usage = @max(self.peak_memory_usage, memory_used);
        if (used_cpu) self.cpu_fallback_count += 1;
    }

    pub fn getAverageOperationTime(self: *const GpuStats) u64 {
        if (self.total_operations == 0) return 0;
        return self.total_gpu_time / self.total_operations;
    }

    pub fn print(self: *const GpuStats) void {
        std.log.info("GPU Backend Statistics:", .{});
        std.log.info("  Total operations: {}", .{self.total_operations});
        std.log.info("  Average operation time: {}μs", .{self.getAverageOperationTime()});
        std.log.info("  Peak memory usage: {} MB", .{self.peak_memory_usage / (1024 * 1024)});
        std.log.info("  CPU fallback rate: {d:.2}%", .{if (self.total_operations > 0)
            @as(f64, @floatFromInt(self.cpu_fallback_count)) * 100.0 / @as(f64, @floatFromInt(self.total_operations))
        else
            0.0});
    }
};
