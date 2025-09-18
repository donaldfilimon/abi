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
    network_config: ai.model_registry.ModelEntry.TrainingConfig = .{
        .learning_rate = 0.01,
        .batch_size = 32,
        .epochs = 10,
        .optimizer = "adam",
        .loss_function = "mse",
        .dataset = "benchmark_data",
        .total_samples = 1000,
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
        // std.log.info("🚀 Running Enhanced Performance Benchmark Suite", .{});
        // std.log.info("================================================", .{});

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
        // std.log.info("🧠 Benchmarking AI Activation Functions", .{});

        // Benchmark individual activation functions
        const ActivationContext = struct {
            fn sigmoid(context: @This()) !f32 {
                return 1.0 / (1.0 + std.math.exp(-context.x));
            }
            fn tanh(context: @This()) !f32 {
                return std.math.tanh(context.x);
            }
            fn gelu(context: @This()) !f32 {
                const x = context.x;
                return 0.5 * x * (1.0 + std.math.tanh(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x * x * x)));
            }
            x: f32,
        };
        const activation_context = ActivationContext{ .x = 0.5 };

        // Create wrapper functions for benchmark framework
        const sigmoid_fn = struct {
            fn call(ctx: ActivationContext) !f32 {
                return ctx.sigmoid();
            }
        }.call;

        const tanh_fn = struct {
            fn call(ctx: ActivationContext) !f32 {
                return ctx.tanh();
            }
        }.call;

        const gelu_fn = struct {
            fn call(ctx: ActivationContext) !f32 {
                return ctx.gelu();
            }
        }.call;

        try self.framework_suite.runBenchmark("Sigmoid Activation", "AI", sigmoid_fn, activation_context);
        try self.framework_suite.runBenchmark("Tanh Activation", "AI", tanh_fn, activation_context);
        try self.framework_suite.runBenchmark("GELU Activation", "AI", gelu_fn, activation_context);

        // Benchmark batch activation processing
        const BatchContext = struct {
            fn batchSigmoid(context: @This()) !void {
                for (context.data) |*val| {
                    val.* = 1.0 / (1.0 + std.math.exp(-val.*));
                }
            }
            fn batchTanh(context: @This()) !void {
                for (context.data) |*val| {
                    val.* = std.math.tanh(val.*);
                }
            }
            data: []f32,
        };
        const batch_context = BatchContext{ .data = try self.createTestVector(1024) };
        defer self.allocator.free(batch_context.data);

        // Create wrapper functions for benchmark framework
        const batch_sigmoid_fn = struct {
            fn call(ctx: BatchContext) !void {
                return ctx.batchSigmoid();
            }
        }.call;

        const batch_tanh_fn = struct {
            fn call(ctx: BatchContext) !void {
                return ctx.batchTanh();
            }
        }.call;

        try self.framework_suite.runBenchmark("Batch Sigmoid (1024)", "AI", batch_sigmoid_fn, batch_context);
        try self.framework_suite.runBenchmark("Batch Tanh (1024)", "AI", batch_tanh_fn, batch_context);
    }

    fn benchmarkNeuralNetworkOperations(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("🔬 Benchmarking Neural Network Operations", .{});

        // Test different network sizes
        for (self.config.vector_dimensions) |dim| {
            const NetworkContext = struct {
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
            };
            const network_context = NetworkContext{
                .inputs = try self.createTestVector(dim),
                .weights = try self.createTestVector(dim),
            };
            defer self.allocator.free(network_context.inputs);
            defer self.allocator.free(network_context.weights);

            // Create wrapper function for benchmark framework
            const forward_pass_fn = struct {
                fn call(ctx: NetworkContext) !f32 {
                    return ctx.forwardPass();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Neural Forward Pass ({}D)", .{dim}), "AI", forward_pass_fn, network_context);
        }
    }

    fn benchmarkSIMDOperations(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("⚡ Benchmarking SIMD Operations", .{});

        for (self.config.data_sizes) |size| {
            const test_vectors = try framework.BenchmarkUtils.createTestVectors(self.allocator, size);
            defer {
                self.allocator.free(test_vectors.a);
                self.allocator.free(test_vectors.b);
                self.allocator.free(test_vectors.result);
            }

            // SIMD dot product
            const SimdContext = struct {
                fn simdDot(context: @This()) !f32 {
                    return dotProductSIMD(context.a, context.b);
                }
                fn scalarDot(context: @This()) !f32 {
                    return dotProductScalar(context.a, context.b);
                }
                a: []f32,
                b: []f32,
            };
            const simd_context = SimdContext{
                .a = test_vectors.a,
                .b = test_vectors.b,
            };

            // Create wrapper functions for benchmark framework
            const simd_dot_fn = struct {
                fn call(ctx: SimdContext) !f32 {
                    return ctx.simdDot();
                }
            }.call;

            const scalar_dot_fn = struct {
                fn call(ctx: SimdContext) !f32 {
                    return ctx.scalarDot();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Dot Product ({} elements)", .{size}), "SIMD", simd_dot_fn, simd_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Dot Product ({} elements)", .{size}), "SIMD", scalar_dot_fn, simd_context);

            // SIMD vector addition
            const AddContext = struct {
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
            };
            const add_context = AddContext{
                .a = test_vectors.a,
                .b = test_vectors.b,
                .result = test_vectors.result,
            };

            // Create wrapper functions for benchmark framework
            const simd_add_fn = struct {
                fn call(ctx: AddContext) !void {
                    return ctx.simdAdd();
                }
            }.call;

            const scalar_add_fn = struct {
                fn call(ctx: AddContext) !void {
                    return ctx.scalarAdd();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "SIMD Vector Add ({} elements)", .{size}), "SIMD", simd_add_fn, add_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Scalar Vector Add ({} elements)", .{size}), "SIMD", scalar_add_fn, add_context);
        }
    }

    fn benchmarkVectorOperations(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("📐 Benchmarking Vector Operations", .{});

        for (self.config.vector_dimensions) |dim| {
            const VectorContext = struct {
                fn cosineSimilarity(context: @This()) !f32 {
                    const dx = context.a[0] - context.b[0];
                    const dy = context.a[1] - context.b[1];
                    return @sqrt(dx * dx + dy * dy);
                }
                fn euclideanDistance(context: @This()) !f32 {
                    const dx = context.a[0] - context.b[0];
                    const dy = context.a[1] - context.b[1];
                    return @sqrt(dx * dx + dy * dy);
                }
                a: []f32,
                b: []f32,
            };
            const vector_context = VectorContext{
                .a = try self.createTestVector(dim),
                .b = try self.createTestVector(dim),
            };
            defer self.allocator.free(vector_context.a);
            defer self.allocator.free(vector_context.b);

            // Create wrapper functions for benchmark framework
            const cosine_similarity_fn = struct {
                fn call(ctx: VectorContext) !f32 {
                    return ctx.cosineSimilarity();
                }
            }.call;

            const euclidean_distance_fn = struct {
                fn call(ctx: VectorContext) !f32 {
                    return ctx.euclideanDistance();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Cosine Similarity ({}D)", .{dim}), "Vector", cosine_similarity_fn, vector_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Euclidean Distance ({}D)", .{dim}), "Vector", euclidean_distance_fn, vector_context);
        }
    }

    fn benchmarkMemoryManagement(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("💾 Benchmarking Memory Management", .{});

        for (self.config.data_sizes) |size| {
            const MemoryContext = struct {
                fn standardAlloc(context: @This()) !void {
                    const buffer = try context.allocator.alloc(f32, context.size);
                    defer context.allocator.free(buffer);
                    // Touch memory to ensure allocation
                    @memset(buffer, 0);
                }
                fn safeAlloc(context: @This()) !void {
                    const buffer = try context.allocator.alloc(f32, context.size);
                    defer context.allocator.free(buffer);
                    @memset(buffer, 0);
                }
                allocator: std.mem.Allocator,
                size: usize,
            };
            const memory_context = MemoryContext{
                .allocator = self.allocator,
                .size = size,
            };

            // Create wrapper functions for benchmark framework
            const standard_alloc_fn = struct {
                fn call(ctx: MemoryContext) !void {
                    return ctx.standardAlloc();
                }
            }.call;

            const safe_alloc_fn = struct {
                fn call(ctx: MemoryContext) !void {
                    return ctx.safeAlloc();
                }
            }.call;

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Standard Allocation ({} elements)", .{size}), "Memory", standard_alloc_fn, memory_context);

            try self.framework_suite.runBenchmark(try std.fmt.allocPrint(self.allocator, "Safe Allocation ({} elements)", .{size}), "Memory", safe_alloc_fn, memory_context);
        }
    }

    fn benchmarkDatabaseOperations(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("🗄️ Benchmarking Database Operations", .{});

        // Simulate database operations
        const DbContext = struct {
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
        };
        const db_context = DbContext{
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

        // Create wrapper functions for benchmark framework
        const search_operation_fn = struct {
            fn call(ctx: DbContext) !usize {
                return ctx.searchOperation();
            }
        }.call;

        const insert_operation_fn = struct {
            fn call(ctx: DbContext) !void {
                return ctx.insertOperation();
            }
        }.call;

        try self.framework_suite.runBenchmark("Vector Search (100 vectors)", "Database", search_operation_fn, db_context);
        try self.framework_suite.runBenchmark("Vector Insert", "Database", insert_operation_fn, db_context);
    }

    fn benchmarkUtilityFunctions(self: *EnhancedBenchmarkSuite) !void {
        // std.log.info("🛠️ Benchmarking Utility Functions", .{});

        // JSON operations
        const JsonContext = struct {
            fn jsonParse(context: @This()) !void {
                var parsed = try utils.json.JsonUtils.parse(context.allocator, context.json_str);
                defer parsed.deinit(context.allocator);
            }
            fn jsonStringify(context: @This()) !void {
                const stringified = try utils.json.JsonUtils.stringify(context.allocator, context.json_value);
                defer context.allocator.free(stringified);
            }
            allocator: std.mem.Allocator,
            json_str: []const u8,
            json_value: utils.json.JsonValue,
        };
        const json_context = JsonContext{
            .allocator = self.allocator,
            .json_str = "{\"name\":\"test\",\"value\":42,\"active\":true}",
            .json_value = utils.json.JsonValue{ .string = "test" },
        };

        // Create wrapper functions for benchmark framework
        const json_parse_fn = struct {
            fn call(ctx: JsonContext) !void {
                return ctx.jsonParse();
            }
        }.call;

        const json_stringify_fn = struct {
            fn call(ctx: JsonContext) !void {
                return ctx.jsonStringify();
            }
        }.call;

        try self.framework_suite.runBenchmark("JSON Parse", "Utilities", json_parse_fn, json_context);
        try self.framework_suite.runBenchmark("JSON Stringify", "Utilities", json_stringify_fn, json_context);

        // URL operations - commented out as URL utilities not implemented in this refactor
        // TODO: Implement URL utilities module
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
