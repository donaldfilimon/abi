//! Test entrypoint for ABI framework tests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Force-reference submodules to include their tests
test {
    // LLM module tests (when enabled)
    if (build_options.enable_llm) {
        // I/O submodules
        std.testing.refAllDecls(abi.ai.llm.io.mmap);
        std.testing.refAllDecls(abi.ai.llm.io.gguf);
        std.testing.refAllDecls(abi.ai.llm.io.tensor_loader);

        // Tensor submodules
        std.testing.refAllDecls(abi.ai.llm.tensor.tensor);
        std.testing.refAllDecls(abi.ai.llm.tensor.view);
        std.testing.refAllDecls(abi.ai.llm.tensor.quantized);

        // Tokenizer submodules
        std.testing.refAllDecls(abi.ai.llm.tokenizer.bpe);
        std.testing.refAllDecls(abi.ai.llm.tokenizer.vocab);
        std.testing.refAllDecls(abi.ai.llm.tokenizer.special_tokens);

        // Ops submodules
        std.testing.refAllDecls(abi.ai.llm.ops.matmul);
        std.testing.refAllDecls(abi.ai.llm.ops.matmul_quant);
        std.testing.refAllDecls(abi.ai.llm.ops.attention);
        std.testing.refAllDecls(abi.ai.llm.ops.rope);
        std.testing.refAllDecls(abi.ai.llm.ops.rmsnorm);
        std.testing.refAllDecls(abi.ai.llm.ops.activations);
        std.testing.refAllDecls(abi.ai.llm.ops.ffn);

        // Cache submodules
        std.testing.refAllDecls(abi.ai.llm.cache.kv_cache);
        std.testing.refAllDecls(abi.ai.llm.cache.ring_buffer);

        // Model submodules
        std.testing.refAllDecls(abi.ai.llm.model.llama);
        std.testing.refAllDecls(abi.ai.llm.model.layer);
        std.testing.refAllDecls(abi.ai.llm.model.weights);
        std.testing.refAllDecls(abi.ai.llm.model.config);

        // Generation submodules
        std.testing.refAllDecls(abi.ai.llm.generation.generator);
        std.testing.refAllDecls(abi.ai.llm.generation.sampler);
        std.testing.refAllDecls(abi.ai.llm.generation.batch);
    }
    // Note: Explore module tests skipped - requires Zig 0.16 I/O API migration
}

// Property-based testing framework
pub const proptest = @import("proptest.zig");

pub const Generator = proptest.Generator;
pub const Generators = proptest.Generators;
pub const PropTest = proptest.PropTest;
pub const PropTestConfig = proptest.PropTestConfig;
pub const PropTestResult = proptest.PropTestResult;
pub const Assertions = proptest.Assertions;
pub const Fuzzer = proptest.Fuzzer;
pub const forAll = proptest.forAll;

test "abi version returns build package version" {
    try std.testing.expectEqualStrings("0.1.0", abi.version());
}

test "abi exports required symbols" {
    _ = abi;

    std.testing.refAllDecls(abi);

    if (@hasDecl(abi, "GpuDevice")) {
        std.log.info("GPU module symbols found", .{});
    } else {
        std.log.info("GPU module not compiled or disabled", .{});
    }

    if (@hasDecl(abi, "HnswIndex")) {
        std.log.info("HNSW module symbols found", .{});
    }
}

test "framework initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_instance = try abi.createDefaultFramework(gpa.allocator());
    defer framework_instance.deinit();

    try std.testing.expect(framework_instance.isRunning());
}

test "framework minimal initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_minimal = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = false,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = false,
    });
    defer framework_minimal.deinit();

    try std.testing.expect(framework_minimal.isRunning());
}

test "framework with gpu enabled" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = true,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = false,
    });
    defer framework.deinit();

    try std.testing.expect(framework.isFeatureEnabled(.gpu));
}

test "framework feature flags" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = build_options.enable_gpu,
        .enable_ai = build_options.enable_ai,
        .enable_web = build_options.enable_web,
        .enable_database = build_options.enable_database,
        .enable_network = build_options.enable_network,
        .enable_profiling = build_options.enable_profiling,
    });
    defer framework.deinit();

    try std.testing.expectEqual(build_options.enable_gpu, framework.isFeatureEnabled(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, framework.isFeatureEnabled(.ai));
    try std.testing.expectEqual(build_options.enable_web, framework.isFeatureEnabled(.web));
    try std.testing.expectEqual(build_options.enable_database, framework.isFeatureEnabled(.database));
    try std.testing.expectEqual(build_options.enable_network, framework.isFeatureEnabled(.network));
    try std.testing.expectEqual(build_options.enable_profiling, framework.isFeatureEnabled(.monitoring));
}
