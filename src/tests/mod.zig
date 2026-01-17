//! Test entrypoint for ABI framework tests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Force-reference submodules to include their tests
comptime {
    // LLM module tests (when enabled)
    if (build_options.enable_llm) {
        _ = abi.ai.llm.io;
        _ = abi.ai.llm.tensor;
        _ = abi.ai.llm.tokenizer;
        _ = abi.ai.llm.ops;
        _ = abi.ai.llm.cache;
        _ = abi.ai.llm.model;
        _ = abi.ai.llm.generation;
    }
    // Explore module tests (when enabled)
    if (build_options.enable_explore) {
        _ = abi.ai.explore;
    }
    // Connector tests
    _ = @import("connectors_test.zig");
    // Integration test matrix
    _ = @import("test_matrix.zig");
    // Include training demo test
    _ = @import("training_demo.zig");
    // LLM reference vectors for llama-cpp compatibility
    if (build_options.enable_llm) {
        _ = @import("llm_reference_vectors.zig");
    }
    // Note: property_tests.zig uses internal APIs (hnsw, index) and should be run directly:
    // zig test src/tests/property_tests.zig
}

// Connector tests
pub const connectors_test = @import("connectors_test.zig");

// Integration test matrix
pub const test_matrix = @import("test_matrix.zig");
pub const TestMatrix = test_matrix.TestMatrix;

// Property-based testing framework
pub const proptest = @import("proptest.zig");

// LLM reference vectors for llama-cpp compatibility testing
pub const llm_reference_vectors = if (build_options.enable_llm) @import("llm_reference_vectors.zig") else struct {};

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
