//! Enhanced Performance Benchmark Suite
//!
//! This benchmark suite provides detailed performance analysis for:
//! - SIMD vector operations vs scalar implementations
//! - Vector database search performance across different scales
//! - Memory usage and allocation patterns
//! - Lock-free data structures performance
//! - Text processing and tokenization speeds
//! - Cross-platform performance characteristics
//! - Statistical analysis with confidence intervals
//! - Export capabilities for CI/CD integration

const std = @import("std");
const framework = @import("benchmark_framework.zig");
const utils = @import("abi").utils;

/// Enhanced performance benchmark configuration
pub const PerformanceBenchmarkConfig = struct {
    framework_config: framework.BenchmarkConfig = .{
        .warmup_iterations = 100,
        .measurement_iterations = 1000,
        .samples = 10,
        .enable_memory_tracking = true,
        .enable_detailed_stats = true,
        .output_format = .console,
    },
    vector_sizes: []const usize = &[_]usize{ 64, 128, 256, 512, 1024 },
    database_sizes: []const usize = &[_]usize{ 100, 1000, 10000 },
    text_sizes: []const usize = &[_]usize{ 1024, 4096, 16384, 65536 },
    alloc_sizes: []const usize = &[_]usize{ 64, 256, 1024, 4096, 16384 },
};

/// Enhanced performance benchmark suite
pub const EnhancedPerformanceBenchmarkSuite = struct {
    framework_suite: *framework.BenchmarkSuite,
    config: PerformanceBenchmarkConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: PerformanceBenchmarkConfig) !*EnhancedPerformanceBenchmarkSuite {
        const framework_suite = try framework.BenchmarkSuite.init(allocator, config.framework_config);
        const self = try allocator.create(EnhancedPerformanceBenchmarkSuite);
        self.* = .{
            .framework_suite = framework_suite,
            .config = config,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *EnhancedPerformanceBenchmarkSuite) void {
        self.framework_suite.deinit();
        self.allocator.destroy(self);
    }

    pub fn runAllBenchmarks(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("🔬 Running Enhanced Performance Benchmark Suite", .{});
        std.log.info("================================================", .{});

        // SIMD Operations
        try self.benchmarkSIMDOperations();

        // Vector Database Operations
        try self.benchmarkVectorDatabase();

        // Lock-free Operations
        try self.benchmarkLockFreeOperations();

        // Text Processing
        try self.benchmarkTextProcessing();

        // Memory Operations
        try self.benchmarkMemoryOperations();

        // Print comprehensive report
        try self.framework_suite.printReport();
    }

    fn benchmarkSIMDOperations(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("🚀 Benchmarking SIMD Vector Operations", .{});

        for (self.config.vector_sizes) |size| {
            const test_vectors = try framework.BenchmarkUtils.createTestVectors(self.allocator, size);
            defer {
                self.allocator.free(test_vectors.a);
                self.allocator.free(test_vectors.b);
                self.allocator.free(test_vectors.result);
            }

            // SIMD dot product
            const simd_context = struct {
                fn simdDot(context: @This()) !f32 {
                    return dotProductSIMD(context.a, context.b);
                }
                fn scalarDot(context: @This()) !f32 {
                    return dotProductScalar(context.a, context.b);
                }
                a: []f32,
                b: []f32,
            }{
                .a = test_vectors.a,
                .b = test_vectors.b,
            };

            // Create wrapper functions for benchmark framework
            const simd_dot_fn = struct {
                fn call(ctx: @TypeOf(simd_context)) !f32 {
                    return ctx.simdDot();
                }
            }.call;

            const scalar_dot_fn = struct {
                fn call(ctx: @TypeOf(simd_context)) !f32 {
                    return ctx.scalarDot();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Dot Product ({} elements)", .{size}), "SIMD", simd_dot_fn, simd_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Dot Product ({} elements)", .{size}), "SIMD", scalar_dot_fn, simd_context);

            // SIMD vector addition
            const add_context = struct {
                fn simdAdd(context: @This()) !void {
                    addVectorsSIMD(context.a, context.b, context.result);
                }
                fn scalarAdd(context: @This()) !void {
                    for (context.a, context.b, 0..) |val_a, val_b, i| {
                        context.result[i] = val_a + val_b;
                    }
                }
                a: []f32,
                b: []f32,
                result: []f32,
            }{
                .a = test_vectors.a,
                .b = test_vectors.b,
                .result = test_vectors.result,
            };

            // Create wrapper functions for benchmark framework
            const simd_add_fn = struct {
                fn call(ctx: @TypeOf(add_context)) !void {
                    return ctx.simdAdd();
                }
            }.call;

            const scalar_add_fn = struct {
                fn call(ctx: @TypeOf(add_context)) !void {
                    return ctx.scalarAdd();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Vector Add ({} elements)", .{size}), "SIMD", simd_add_fn, add_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Vector Add ({} elements)", .{size}), "SIMD", scalar_add_fn, add_context);
        }
    }

    fn benchmarkVectorDatabase(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("🗄️ Benchmarking Vector Database Operations", .{});

        for (self.config.database_sizes) |db_size| {
            const dimensions = 128;

            // Create test vectors
            const vectors = try self.allocator.alloc([dimensions]f32, db_size);
            defer self.allocator.free(vectors);

            // Initialize with test data
            for (vectors, 0..) |*vector, i| {
                for (vector, 0..) |*val, j| {
                    val.* = @as(f32, @floatFromInt((i * dimensions + j) % 1000)) / 1000.0;
                }
            }

            // Create query vector
            var query: [dimensions]f32 = undefined;
            for (&query, 0..) |*val, i| {
                val.* = @sin(@as(f32, @floatFromInt(i)) / 10.0);
            }

            // Benchmark vector search
            const search_context = struct {
                fn vectorSearch(context: @This()) !usize {
                    var best_idx: usize = 0;
                    var best_similarity: f32 = -1.0;

                    for (context.vectors, 0..) |vector, i| {
                        const similarity = cosineSimilarity(context.query, &vector);
                        if (similarity > best_similarity) {
                            best_similarity = similarity;
                            best_idx = i;
                        }
                    }
                    return best_idx;
                }
                vectors: []const [dimensions]f32,
                query: *const [dimensions]f32,
            }{
                .vectors = vectors,
                .query = &query,
            };

            // Create wrapper function for benchmark framework
            const vector_search_fn = struct {
                fn call(ctx: @TypeOf(search_context)) !usize {
                    return ctx.vectorSearch();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Vector Similarity Search ({} vectors)", .{db_size}), "Database", vector_search_fn, search_context);

            // Benchmark vector insertion simulation
            const insert_context = struct {
                fn vectorInsertion(context: @This()) !void {
                    // Simulate vector insertion overhead
                    var dummy_vector = [_]f32{0.0} ** 128;
                    for (0..context.vector_size) |i| {
                        dummy_vector[i % 128] = @as(f32, @floatFromInt(i)) * 0.01;
                    }
                }
                vector_size: usize,
            }{
                .vector_size = dimensions,
            };

            // Create wrapper function for benchmark framework
            const vector_insert_fn = struct {
                fn call(ctx: @TypeOf(insert_context)) !void {
                    return ctx.vectorInsertion();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Vector Insertion ({}D)", .{dimensions}), "Database", vector_insert_fn, insert_context);
        }
    }

    fn benchmarkLockFreeOperations(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("🔒 Benchmarking Lock-free Data Structures", .{});

        // Atomic operations benchmark
        const atomic_context = struct {
            fn atomicIncrement(_: @This()) !u64 {
                var counter: u64 = 0;
                _ = @atomicRmw(u64, &counter, .Add, 1, .monotonic);
                return counter;
            }
            fn compareAndSwap(_: @This()) !bool {
                var value: u64 = 42;
                return @cmpxchgWeak(u64, &value, 42, 43, .acquire, .monotonic) == null;
            }
        }{};

        // Create wrapper functions for benchmark framework
        const atomic_inc_fn = struct {
            fn call(ctx: @TypeOf(atomic_context)) !u64 {
                return ctx.atomicIncrement();
            }
        }.call;

        const cas_fn = struct {
            fn call(ctx: @TypeOf(atomic_context)) !bool {
                return ctx.compareAndSwap();
            }
        }.call;

        try self.framework_suite.runBenchmark("Atomic Increment", "Concurrency", atomic_inc_fn, atomic_context);
        try self.framework_suite.runBenchmark("Compare-and-Swap", "Concurrency", cas_fn, atomic_context);

        // Lock-free queue simulation
        const queue_context = struct {
            fn lockFreeQueue(_: @This()) !void {
                // Simulate lock-free queue operations
                var head: ?*u64 = null;
                var node: u64 = 123;
                _ = @cmpxchgWeak(?*u64, &head, null, &node, .release, .acquire);
            }
        }{};

        // Create wrapper function for benchmark framework
        const queue_fn = struct {
            fn call(ctx: @TypeOf(queue_context)) !void {
                return ctx.lockFreeQueue();
            }
        }.call;

        try self.framework_suite.runBenchmark("Lock-free Queue Operations", "Concurrency", queue_fn, queue_context);
    }

    fn benchmarkTextProcessing(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("📝 Benchmarking Text Processing", .{});

        for (self.config.text_sizes) |size| {
            const text = try framework.BenchmarkUtils.createTestText(self.allocator, size);
            defer self.allocator.free(text);

            // Benchmark different text operations
            const text_context = struct {
                fn tokenize(context: @This()) !usize {
                    var tokens: usize = 0;
                    var in_token = false;

                    for (context.text) |char| {
                        if (char == ' ' or char == '\n' or char == '\t') {
                            in_token = false;
                        } else if (!in_token) {
                            tokens += 1;
                            in_token = true;
                        }
                    }
                    return tokens;
                }
                fn search(context: @This()) !?usize {
                    const needle = "test";
                    return std.mem.indexOf(u8, context.text, needle);
                }
                fn hash(context: @This()) !u64 {
                    return std.hash_map.hashString(context.text);
                }
                text: []u8,
            }{
                .text = text,
            };

            // Create wrapper functions for benchmark framework
            const tokenize_fn = struct {
                fn call(ctx: @TypeOf(text_context)) !usize {
                    return ctx.tokenize();
                }
            }.call;

            const search_fn = struct {
                fn call(ctx: @TypeOf(text_context)) !?usize {
                    return ctx.search();
                }
            }.call;

            const hash_fn = struct {
                fn call(ctx: @TypeOf(text_context)) !u64 {
                    return ctx.hash();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Text Tokenization ({} bytes)", .{size}), "Text", tokenize_fn, text_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Text Search ({} bytes)", .{size}), "Text", search_fn, text_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Text Hashing ({} bytes)", .{size}), "Text", hash_fn, text_context);
        }
    }

    fn benchmarkMemoryOperations(self: *EnhancedPerformanceBenchmarkSuite) !void {
        std.log.info("💾 Benchmarking Memory Operations", .{});

        for (self.config.alloc_sizes) |size| {
            const alloc_context = struct {
                fn memoryAllocation(context: @This()) !void {
                    const memory = try context.allocator.alloc(u8, context.size);
                    defer context.allocator.free(memory);
                    // Touch the memory to ensure it's allocated
                    @memset(memory, 0);
                }
                allocator: std.mem.Allocator,
                size: usize,
            }{
                .allocator = self.allocator,
                .size = size,
            };

            // Create wrapper function for benchmark framework
            const alloc_fn = struct {
                fn call(ctx: @TypeOf(alloc_context)) !void {
                    return ctx.memoryAllocation();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Memory Allocation ({} bytes)", .{size}), "Memory", alloc_fn, alloc_context);
        }
    }
};

// SIMD implementations for benchmarks
fn dotProductSIMD(a: []const f32, b: []const f32) f32 {
    const SIMD_WIDTH = 4;
    const F32Vector = @Vector(SIMD_WIDTH, f32);

    var sum: f32 = 0.0;
    var i: usize = 0;

    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: F32Vector = a[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const vb: F32Vector = b[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const product = va * vb;
        sum += @reduce(.Add, product);
    }

    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

fn dotProductScalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |val_a, val_b| {
        sum += val_a * val_b;
    }
    return sum;
}

fn addVectorsSIMD(a: []const f32, b: []const f32, result: []f32) void {
    const SIMD_WIDTH = 4;
    const F32Vector = @Vector(SIMD_WIDTH, f32);

    var i: usize = 0;

    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: F32Vector = a[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const vb: F32Vector = b[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].*;
        const sum = va + vb;
        result[i .. i + SIMD_WIDTH][0..SIMD_WIDTH].* = sum;
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0.0;

    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (a, b) |val_a, val_b| {
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }

    const magnitude = @sqrt(norm_a * norm_b);
    return if (magnitude > 0) dot_product / magnitude else 0.0;
}

// Main benchmark runner
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = PerformanceBenchmarkConfig{
        .framework_config = .{
            .warmup_iterations = 50,
            .measurement_iterations = 500,
            .samples = 5,
        },
    };

    var suite = try EnhancedPerformanceBenchmarkSuite.init(allocator, config);
    defer suite.deinit();

    try suite.runAllBenchmarks();
}
