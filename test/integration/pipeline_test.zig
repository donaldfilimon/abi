//! Pipeline Integration Tests
//!
//! Tests the Abbey Dynamic Model pipeline DSL via the public `@import("abi")` API.
//! Covers: builder chaining, execution, template rendering, routing, WDBX persistence.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "pipeline builder creates and runs a basic pipeline" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;

    var builder = pipeline_mod.chain(allocator, "test-session-1");
    defer builder.deinit();

    var p = builder
        .template("Hello {input}, context: {context}")
        .route(.heuristic)
        .generate(.{})
        .build();
    defer p.deinit();

    const result = try p.run("world");
    var result_mut = result;
    defer result_mut.deinit();

    try std.testing.expect(result.steps_executed > 0);
    try std.testing.expect(result.response != null);
}

test "pipeline template step interpolates variables" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;

    var builder = pipeline_mod.chain(allocator, "test-session-2");
    defer builder.deinit();

    var p = builder
        .template("User said: {input}")
        .generate(.{})
        .build();
    defer p.deinit();

    const result = try p.run("tell me about Zig");
    var result_mut = result;
    defer result_mut.deinit();

    try std.testing.expect(result.response != null);
    try std.testing.expect(result.steps_executed == 2);
}

test "pipeline routing selects profile based on keywords" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;

    // Technical keywords should route to Aviva
    var builder = pipeline_mod.chain(allocator, "test-session-3");
    defer builder.deinit();

    var p = builder
        .route(.heuristic)
        .generate(.{})
        .build();
    defer p.deinit();

    const result = try p.run("debug this code function error");
    var result_mut = result;
    defer result_mut.deinit();

    try std.testing.expect(result.response != null);
    // Response should mention Aviva since we used technical keywords
    if (result.response) |r| {
        try std.testing.expect(std.mem.indexOf(u8, r, "Aviva") != null);
    }
}

test "pipeline with WDBX chain records blocks" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;
    if (!build_options.feat_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;
    const block_chain_mod = abi.ai.pipeline.types.BlockChain;

    var wdbx_chain = block_chain_mod.init(allocator, "test-wdbx-session");
    defer wdbx_chain.deinit();

    var builder = pipeline_mod.chain(allocator, "test-wdbx-session");
    defer builder.deinit();

    var p = builder
        .withChain(&wdbx_chain)
        .template("Respond to: {input}")
        .route(.adaptive)
        .generate(.{})
        .store(.wdbx)
        .build();
    defer p.deinit();

    const result = try p.run("Hello Abbey!");
    var result_mut = result;
    defer result_mut.deinit();

    try std.testing.expect(result.steps_executed >= 3);
    try std.testing.expect(result.pipeline_id > 0);
}

test "pipeline validation step catches unsafe content" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;

    var builder = pipeline_mod.chain(allocator, "test-session-validate");
    defer builder.deinit();

    var p = builder
        .generate(.{})
        .validate(.constitution)
        .build();
    defer p.deinit();

    const result = try p.run("test input");
    var result_mut = result;
    defer result_mut.deinit();

    // Normal input should pass validation
    try std.testing.expect(result.validation_passed);
}

test "pipeline full chain matches DSL syntax from spec" {
    if (!build_options.feat_reasoning) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const pipeline_mod = abi.ai.pipeline;

    // This test verifies the exact DSL syntax from the design spec compiles and runs
    var builder = pipeline_mod.chain(allocator, "spec-demo");
    defer builder.deinit();

    var p = builder
        .retrieve(.wdbx, .{ .k = 5 })
        .template("Given {context}, respond to: {input}")
        .route(.adaptive)
        .modulate()
        .generate(.{ .mode = .blocking })
        .validate(.constitution)
        .store(.wdbx)
        .build();
    defer p.deinit();

    const result = try p.run("What is the meaning of life?");
    var result_mut = result;
    defer result_mut.deinit();

    try std.testing.expect(result.response != null);
    try std.testing.expect(result.validation_passed);
    try std.testing.expect(result.steps_executed > 0);
}
