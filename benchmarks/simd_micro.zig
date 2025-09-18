//! Enhanced SIMD Micro-benchmarks
//!
//! This benchmark provides detailed SIMD performance analysis:
//! - Vector operations with statistical analysis
//! - Cross-platform SIMD performance comparison
//! - Memory access pattern optimization
//! - Integration with the standardized benchmark framework

const std = @import("std");
const framework = @import("benchmark_framework.zig");
const utils = @import("abi").utils;
const abi = @import("abi");

/// Enhanced SIMD micro-benchmark configuration
pub const SIMDMicroBenchmarkConfig = struct {
    framework_config: framework.BenchmarkConfig = .{
        .warmup_iterations = 50,
        .measurement_iterations = 1000,
        .samples = 10,
        .enable_memory_tracking = true,
        .enable_detailed_stats = true,
        .output_format = .console,
    },
    vector_sizes: []const usize = &[_]usize{ 100_000, 1_000_000, 10_000_000 },
    matrix_sizes: []const usize = &[_]usize{ 32, 64, 128, 256 },
};

/// Enhanced SIMD micro-benchmark suite
pub const EnhancedSIMDMicroBenchmarkSuite = struct {
    framework_suite: *framework.BenchmarkSuite,
    config: SIMDMicroBenchmarkConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SIMDMicroBenchmarkConfig) !*EnhancedSIMDMicroBenchmarkSuite {
        const framework_suite = try framework.BenchmarkSuite.init(allocator, config.framework_config);
        const self = try allocator.create(EnhancedSIMDMicroBenchmarkSuite);
        self.* = .{
            .framework_suite = framework_suite,
            .config = config,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *EnhancedSIMDMicroBenchmarkSuite) void {
        self.framework_suite.deinit();
        self.allocator.destroy(self);
    }

    pub fn runAllBenchmarks(self: *EnhancedSIMDMicroBenchmarkSuite) !void {
        std.log.info("⚡ Running Enhanced SIMD Micro-benchmarks", .{});
        std.log.info("=========================================", .{});

        // Vector operations
        try self.benchmarkVectorOperations();

        // Matrix operations
        try self.benchmarkMatrixOperations();

        // Mathematical functions
        try self.benchmarkMathFunctions();

        // Print comprehensive report
        try self.framework_suite.printReport();
    }

    fn benchmarkVectorOperations(self: *EnhancedSIMDMicroBenchmarkSuite) !void {
        std.log.info("📐 Benchmarking Vector Operations", .{});

        for (self.config.vector_sizes) |size| {
            const vectors = try self.createTestVectors(size);
            defer {
                self.allocator.free(vectors.a);
                self.allocator.free(vectors.b);
                self.allocator.free(vectors.result);
            }

            // Dot product
            const dot_product_context = struct {
                fn dotProduct(context: @This()) !f32 {
                    return abi.simd.VectorOps.dotProduct(context.a, context.b);
                }
                a: []f32,
                b: []f32,
            }{
                .a = vectors.a,
                .b = vectors.b,
            };

            // Create wrapper function for benchmark framework
            const dot_product_fn = struct {
                fn call(ctx: @TypeOf(dot_product_context)) !f32 {
                    return ctx.dotProduct();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Dot Product ({} elements)", .{size}), "Vector", dot_product_fn, dot_product_context);

            // Cosine similarity
            const cosine_context = struct {
                fn cosineSimilarity(context: @This()) !f32 {
                    // Simple cosine similarity implementation
                    const dot = abi.simd.VectorOps.dotProduct(context.a, context.b);
                    const norm_a = abi.simd.VectorOps.dotProduct(context.a, context.a);
                    const norm_b = abi.simd.VectorOps.dotProduct(context.b, context.b);
                    return dot / (@sqrt(norm_a) * @sqrt(norm_b));
                }
                a: []f32,
                b: []f32,
            }{
                .a = vectors.a,
                .b = vectors.b,
            };

            // Create wrapper function for benchmark framework
            const cosine_fn = struct {
                fn call(ctx: @TypeOf(cosine_context)) !f32 {
                    return ctx.cosineSimilarity();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Cosine Similarity ({} elements)", .{size}), "Vector", cosine_fn, cosine_context);

            // Vector addition
            const add_context = struct {
                fn vectorAdd(context: @This()) !void {
                    for (context.result, context.a, 0..) |*rv, av, i| {
                        rv.* = av + context.b[i % context.b.len];
                    }
                }
                a: []f32,
                b: []f32,
                result: []f32,
            }{
                .a = vectors.a,
                .b = vectors.b,
                .result = vectors.result,
            };

            // Create wrapper function for benchmark framework
            const add_fn = struct {
                fn call(ctx: @TypeOf(add_context)) !void {
                    return ctx.vectorAdd();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Vector Addition ({} elements)", .{size}), "Vector", add_fn, add_context);

            // Vector sum
            const sum_context = struct {
                fn vectorSum(context: @This()) !f32 {
                    var sum: f32 = 0;
                    for (context.a) |v| sum += v;
                    return sum;
                }
                a: []f32,
            }{
                .a = vectors.a,
            };

            // Create wrapper function for benchmark framework
            const sum_fn = struct {
                fn call(ctx: @TypeOf(sum_context)) !f32 {
                    return ctx.vectorSum();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Vector Sum ({} elements)", .{size}), "Vector", sum_fn, sum_context);
        }
    }

    fn benchmarkMatrixOperations(self: *EnhancedSIMDMicroBenchmarkSuite) !void {
        std.log.info("🔢 Benchmarking Matrix Operations", .{});

        for (self.config.matrix_sizes) |size| {
            const matrices = try self.createTestMatrices(size, size, size);
            defer {
                self.allocator.free(matrices.a);
                self.allocator.free(matrices.b);
                self.allocator.free(matrices.result);
            }

            // Matrix multiplication
            const mm_context = struct {
                fn matrixMultiply(context: @This()) !void {
                    const M = context.rows_a;
                    const K = context.cols_a;
                    const N = context.cols_b;

                    for (0..M) |i| {
                        for (0..N) |j| {
                            var sum: f32 = 0;
                            for (0..K) |k| {
                                sum += context.a[i * K + k] * context.b[k * N + j];
                            }
                            context.result[i * N + j] = sum;
                        }
                    }
                }
                a: []f32,
                b: []f32,
                result: []f32,
                rows_a: usize,
                cols_a: usize,
                cols_b: usize,
            }{
                .a = matrices.a,
                .b = matrices.b,
                .result = matrices.result,
                .rows_a = size,
                .cols_a = size,
                .cols_b = size,
            };

            // Create wrapper function for benchmark framework
            const mm_fn = struct {
                fn call(ctx: @TypeOf(mm_context)) !void {
                    return ctx.matrixMultiply();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Matrix Multiply ({}x{})", .{ size, size }), "Matrix", mm_fn, mm_context);
        }
    }

    fn benchmarkMathFunctions(self: *EnhancedSIMDMicroBenchmarkSuite) !void {
        std.log.info("🧮 Benchmarking Mathematical Functions", .{});

        const math_context = struct {
            fn mathOperations(context: @This()) !f32 {
                var result: f32 = 0.0;
                for (context.data) |val| {
                    result += @sin(val) + @cos(val) + @sqrt(if (val < 0) -val else val);
                }
                return result;
            }
            data: []f32,
        }{
            .data = try self.createTestData(10000),
        };
        defer self.allocator.free(math_context.data);

        // Create wrapper function for benchmark framework
        const math_fn = struct {
            fn call(ctx: @TypeOf(math_context)) !f32 {
                return ctx.mathOperations();
            }
        }.call;

        try self.framework_suite.runBenchmark("Mathematical Functions (sin, cos, sqrt)", "Math", math_fn, math_context);
    }

    // Helper functions
    fn createTestVectors(self: *EnhancedSIMDMicroBenchmarkSuite, size: usize) !struct { a: []f32, b: []f32, result: []f32 } {
        const a = try self.allocator.alloc(f32, size);
        const b = try self.allocator.alloc(f32, size);
        const result = try self.allocator.alloc(f32, size);

        fillLinear(a, 1.0);
        fillLinear(b, 0.5);

        return .{ .a = a, .b = b, .result = result };
    }

    fn createTestMatrices(self: *EnhancedSIMDMicroBenchmarkSuite, rows: usize, cols_a: usize, cols_b: usize) !struct { a: []f32, b: []f32, result: []f32 } {
        const a = try self.allocator.alloc(f32, rows * cols_a);
        const b = try self.allocator.alloc(f32, cols_a * cols_b);
        const result = try self.allocator.alloc(f32, rows * cols_b);

        for (a, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i * 7) % 31)) * 0.03125;
        }

        for (b, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i * 11) % 29)) * 0.03448;
        }

        return .{ .a = a, .b = b, .result = result };
    }

    fn createTestData(self: *EnhancedSIMDMicroBenchmarkSuite, size: usize) ![]f32 {
        const data = try self.allocator.alloc(f32, size);
        for (data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 1000)) * 0.001;
        }
        return data;
    }
};

fn fillLinear(buf: []f32, mul: f32) void {
    for (buf, 0..) |*v, i| v.* = mul * @as(f32, @floatFromInt(i % 100));
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = SIMDMicroBenchmarkConfig{};
    var suite = try EnhancedSIMDMicroBenchmarkSuite.init(allocator, config);
    defer suite.deinit();

    try suite.runAllBenchmarks();
}
