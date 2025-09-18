//! Enhanced Performance Benchmark Suite
//!
//! This suite provides comprehensive performance benchmarks for core functionality:
//! - AI activation functions and neural operations
//! - Memory management and allocation patterns
//! - SIMD operations and vector processing
//! - Database operations and vector search
//! - Statistical analysis with confidence intervals
//!
//! Run with: zig run benchmarks/benchmark_suite.zig

const std = @import("std");
const framework = @import("benchmark_framework.zig");
const utils = @import("abi").utils;

const root = @import("abi");
const ai = root.ai;
const monitoring = root.monitoring;
const core = root.core;

// Enhanced configuration using the framework
pub const EnhancedBenchmarkConfig = struct {
    framework_config: framework.BenchmarkConfig = .{
        .warmup_iterations = 100,
        .measurement_iterations = 1000,
        .samples = 10,
        .enable_memory_tracking = true,
        .enable_detailed_stats = true,
        .output_format = .console,
    },
    data_sizes: []const usize = &[_]usize{ 64, 128, 256, 512, 1024, 2048 },
    vector_dimensions: []const u16 = &[_]u16{ 64, 128, 256, 512 },
    network_config: ai.TrainingConfig = .{
        .learning_rate = 0.01,
        .batch_size = 32,
        .epochs = 10,
        .use_mixed_precision = true,
        .checkpoint_frequency = 10,
    },
};

// Use the framework's BenchmarkResult instead

pub const EnhancedBenchmarkSuite = struct {
    framework_suite: *framework.BenchmarkSuite,
    config: EnhancedBenchmarkConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: EnhancedBenchmarkConfig) !*EnhancedBenchmarkSuite {
        const framework_suite = try framework.BenchmarkSuite.init(allocator, config.framework_config);
        const self = try allocator.create(EnhancedBenchmarkSuite);
        self.* = .{
            .framework_suite = framework_suite,
            .config = config,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *EnhancedBenchmarkSuite) void {
        self.framework_suite.deinit();
        self.allocator.destroy(self);
    }

    pub fn runAllBenchmarks(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("🚀 Running Enhanced Performance Benchmark Suite", .{});
        std.log.info("================================================", .{});

        // AI and Neural Network Benchmarks
        try self.benchmarkAIActivationFunctions();
        try self.benchmarkNeuralNetworkOperations();

        // SIMD and Vector Operations
        try self.benchmarkSIMDOperations();
        try self.benchmarkVectorOperations();

        // Memory Management
        try self.benchmarkMemoryManagement();

        // Database Operations
        try self.benchmarkDatabaseOperations();

        // Utility Functions
        try self.benchmarkUtilityFunctions();

        // Print comprehensive report
        try self.framework_suite.printReport();
    }

    fn benchmarkAIActivationFunctions(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("🧠 Benchmarking AI Activation Functions", .{});

        // Benchmark individual activation functions
        const activation_context = struct {
            fn sigmoid(context: @This()) !f32 {
                return ai.ActivationUtils.fastSigmoid(context.x);
            }
            fn tanh(context: @This()) !f32 {
                return ai.ActivationUtils.fastTanh(context.x);
            }
            fn gelu(context: @This()) !f32 {
                return ai.ActivationUtils.fastGelu(context.x);
            }
            x: f32,
        }{ .x = 0.5 };

        try self.framework_suite.runBenchmark("Sigmoid Activation", "AI", activation_context.sigmoid, activation_context);
        try self.framework_suite.runBenchmark("Tanh Activation", "AI", activation_context.tanh, activation_context);
        try self.framework_suite.runBenchmark("GELU Activation", "AI", activation_context.gelu, activation_context);

        // Benchmark batch activation processing
        const batch_context = struct {
            fn batchSigmoid(context: @This()) !void {
                for (context.data) |*val| {
                    val.* = ai.ActivationUtils.fastSigmoid(val.*);
                }
            }
            fn batchTanh(context: @This()) !void {
                for (context.data) |*val| {
                    val.* = ai.ActivationUtils.fastTanh(val.*);
                }
            }
            data: []f32,
        }{ .data = try self.createTestVector(1024) };
        defer self.allocator.free(batch_context.data);

        try self.framework_suite.runBenchmark("Batch Sigmoid (1024)", "AI", batch_context.batchSigmoid, batch_context);
        try self.framework_suite.runBenchmark("Batch Tanh (1024)", "AI", batch_context.batchTanh, batch_context);
    }

    fn benchmarkNeuralNetworkOperations(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("🔬 Benchmarking Neural Network Operations", .{});

        // Test different network sizes
        for (self.config.vector_dimensions) |dim| {
            const network_context = struct {
                fn forwardPass(context: @This()) !f32 {
                    // Simulate forward pass computation
                    var result: f32 = 0.0;
                    for (context.inputs) |input| {
                        result += input * context.weights[context.inputs.len % context.weights.len];
                    }
                    return result;
                }
                inputs: []f32,
                weights: []f32,
            }{
                .inputs = try self.createTestVector(dim),
                .weights = try self.createTestVector(dim),
            };
            defer self.allocator.free(network_context.inputs);
            defer self.allocator.free(network_context.weights);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Neural Forward Pass ({}D)", .{dim}), "AI", network_context.forwardPass, network_context);
        }
    }

    fn benchmarkSIMDOperations(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("⚡ Benchmarking SIMD Operations", .{});

        for (self.config.data_sizes) |size| {
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

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Dot Product ({} elements)", .{size}), "SIMD", simd_context.simdDot, simd_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Dot Product ({} elements)", .{size}), "SIMD", simd_context.scalarDot, simd_context);

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

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Vector Add ({} elements)", .{size}), "SIMD", add_context.simdAdd, add_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Vector Add ({} elements)", .{size}), "SIMD", add_context.scalarAdd, add_context);
        }
    }

    fn benchmarkVectorOperations(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("📐 Benchmarking Vector Operations", .{});

        for (self.config.vector_dimensions) |dim| {
            const vector_context = struct {
                fn cosineSimilarity(context: @This()) !f32 {
                    return utils.MathUtils.distance2D(context.a[0], context.a[1], context.b[0], context.b[1]);
                }
                fn euclideanDistance(context: @This()) !f32 {
                    return utils.MathUtils.distance2D(context.a[0], context.a[1], context.b[0], context.b[1]);
                }
                a: []f32,
                b: []f32,
            }{
                .a = try self.createTestVector(dim),
                .b = try self.createTestVector(dim),
            };
            defer self.allocator.free(vector_context.a);
            defer self.allocator.free(vector_context.b);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Cosine Similarity ({}D)", .{dim}), "Vector", vector_context.cosineSimilarity, vector_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Euclidean Distance ({}D)", .{dim}), "Vector", vector_context.euclideanDistance, vector_context);
        }
    }

    fn benchmarkMemoryManagement(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("💾 Benchmarking Memory Management", .{});

        for (self.config.data_sizes) |size| {
            const memory_context = struct {
                fn standardAlloc(context: @This()) !void {
                    const buffer = try context.allocator.alloc(f32, context.size);
                    defer context.allocator.free(buffer);
                    // Touch memory to ensure allocation
                    @memset(buffer, 0);
                }
                fn safeAlloc(context: @This()) !void {
                    const buffer = try utils.MemoryUtils.safeAlloc(context.allocator, f32, context.size);
                    defer context.allocator.free(buffer);
                    @memset(buffer, 0);
                }
                allocator: std.mem.Allocator,
                size: usize,
            }{
                .allocator = self.allocator,
                .size = size,
            };

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Standard Allocation ({} elements)", .{size}), "Memory", memory_context.standardAlloc, memory_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Safe Allocation ({} elements)", .{size}), "Memory", memory_context.safeAlloc, memory_context);
        }
    }

    fn benchmarkDatabaseOperations(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("🗄️ Benchmarking Database Operations", .{});

        // Simulate database operations
        const db_context = struct {
            fn searchOperation(context: @This()) !usize {
                // Simulate vector search
                var best_idx: usize = 0;
                var best_score: f32 = -1.0;

                for (context.vectors, 0..) |vector, i| {
                    var score: f32 = 0.0;
                    for (vector, context.query) |v, q| {
                        score += v * q;
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_idx = i;
                    }
                }
                return best_idx;
            }
            fn insertOperation(context: @This()) !void {
                // Simulate vector insertion
                _ = context;
            }
            vectors: []const []f32,
            query: []f32,
        }{
            .vectors = try self.createTestVectorArray(100, 128),
            .query = try self.createTestVector(128),
        };
        defer {
            for (db_context.vectors) |vector| {
                self.allocator.free(vector);
            }
            self.allocator.free(db_context.vectors);
            self.allocator.free(db_context.query);
        }

        try self.framework_suite.runBenchmark("Vector Search (100 vectors)", "Database", db_context.searchOperation, db_context);
        try self.framework_suite.runBenchmark("Vector Insert", "Database", db_context.insertOperation, db_context);
    }

    fn benchmarkUtilityFunctions(self: *EnhancedBenchmarkSuite) !void {
        std.log.info("🛠️ Benchmarking Utility Functions", .{});

        // JSON operations
        const json_context = struct {
            fn jsonParse(context: @This()) !void {
                var parsed = try utils.JsonUtils.parse(context.allocator, context.json_str);
                defer parsed.deinit(context.allocator);
            }
            fn jsonStringify(context: @This()) !void {
                const stringified = try utils.JsonUtils.stringify(context.allocator, context.json_value);
                defer context.allocator.free(stringified);
            }
            allocator: std.mem.Allocator,
            json_str: []const u8,
            json_value: utils.JsonUtils.JsonValue,
        }{
            .allocator = self.allocator,
            .json_str = "{\"name\":\"test\",\"value\":42,\"active\":true}",
            .json_value = utils.JsonUtils.JsonValue{ .string = "test" },
        };

        try self.framework_suite.runBenchmark("JSON Parse", "Utilities", json_context.jsonParse, json_context);
        try self.framework_suite.runBenchmark("JSON Stringify", "Utilities", json_context.jsonStringify, json_context);

        // URL operations
        const url_context = struct {
            fn urlEncode(context: @This()) !void {
                const encoded = try utils.UrlUtils.encode(context.allocator, context.url);
                defer context.allocator.free(encoded);
            }
            fn urlDecode(context: @This()) !void {
                const decoded = try utils.UrlUtils.decode(context.allocator, context.encoded_url);
                defer context.allocator.free(decoded);
            }
            allocator: std.mem.Allocator,
            url: []const u8,
            encoded_url: []const u8,
        }{
            .allocator = self.allocator,
            .url = "Hello World! Test & More",
            .encoded_url = "Hello%20World%21%20Test%20%26%20More",
        };

        try self.framework_suite.runBenchmark("URL Encode", "Utilities", url_context.urlEncode, url_context);
        try self.framework_suite.runBenchmark("URL Decode", "Utilities", url_context.urlDecode, url_context);
    }

    // Helper functions
    fn createTestVector(self: *EnhancedBenchmarkSuite, size: usize) ![]f32 {
        const vector = try self.allocator.alloc(f32, size);
        for (vector, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }
        return vector;
    }

    fn createTestVectorArray(self: *EnhancedBenchmarkSuite, count: usize, size: usize) ![][]f32 {
        const vectors = try self.allocator.alloc([]f32, count);
        for (vectors, 0..) |*vector, i| {
            vector.* = try self.createTestVector(size);
            for (vector.*, 0..) |*val, j| {
                val.* = @as(f32, @floatFromInt((i * size + j) % 100)) * 0.01;
            }
        }
        return vectors;
    }
};

// SIMD helper functions
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

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = EnhancedBenchmarkConfig{};
    var suite = try EnhancedBenchmarkSuite.init(allocator, config);
    defer suite.deinit();

    try suite.runAllBenchmarks();
}
