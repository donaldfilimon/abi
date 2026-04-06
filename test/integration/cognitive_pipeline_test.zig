const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "cognitive pipeline: vision -> reasoning -> agent -> tool" {
    if (!build_options.feat_ai or !build_options.feat_vision) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // 1. Initialize the abi framework with AI features
    // Note: In integration tests, we use abi.ai.init or similar if available,
    // but usually the framework is initialized via a central point or features are used directly.
    // Based on ai_test.zig, we can use abi.ai.init.
    try abi.ai.init(allocator, .{});
    defer abi.ai.deinit();

    // 2. Setup Vision
    // We'll simulate a vision result. In a real scenario, we'd load an image.
    // Here we just verify we can create the context and an image.
    var vision_ctx = try abi.ai.vision.Context.init(allocator);
    defer vision_ctx.deinit();

    // Create a mock "screenshot" (blank image)
    var screenshot = try vision_ctx.createImage(224, 224, 3);
    defer screenshot.deinit();

    // Simulate detection: "Detected a bug in the UI rendering logic."
    const vision_analysis = "Detected a bug in the UI rendering logic: overlapping elements in the sidebar.";

    // 3. Initialize Reasoning (Abbey)
    var reasoning_ctx = try abi.ai.reasoning.Context.init(allocator, .{});
    defer reasoning_ctx.deinit();

    var chain = reasoning_ctx.createChain(vision_analysis);
    defer chain.deinit();

    // Add reasoning steps based on the vision input
    try chain.addStep(.assessment, "Analyzing vision report of overlapping sidebar elements", .{
        .level = .high,
        .score = 0.9,
        .reasoning = "Visual evidence is clear",
    });

    try chain.addStep(.analysis, "Determining root cause: CSS z-index conflict in Sidebar.zig", .{
        .level = .medium,
        .score = 0.7,
        .reasoning = "Sidebar.zig is the most likely culprit for sidebar rendering",
    });

    try chain.addStep(.synthesis, "Proposal: Dispatch agent to fix z-index in Sidebar.zig", .{
        .level = .high,
        .score = 0.85,
        .reasoning = "Targeted fix identified",
    });

    try chain.finalize();
    const reasoning_summary = try chain.getSummary(allocator);
    defer allocator.free(reasoning_summary);

    // 4. Initialize Agent
    var agents_ctx = try abi.ai.agents.Context.init(allocator, .{});
    defer agents_ctx.deinit();

    const agent = try agents_ctx.createAgent("FixerAgent");
    agent.setBackend(.echo); // Use echo backend for predictable testing

    // 5. Setup a mock "fix" tool
    const FixTool = struct {
        pub fn execute(_: *abi.ai.tools.Context, args: std.json.Value) abi.ai.tools.ToolExecutionError!abi.ai.tools.ToolResult {
            _ = args;
            // In a real test, we'd check if args contains the file and fix
            return abi.ai.tools.ToolResult.init(std.testing.allocator, true, "Fixed z-index in Sidebar.zig");
        }
    };

    const fix_tool = abi.ai.tools.Tool{
        .name = "fix_ui_bug",
        .description = "Fixes UI bugs by applying CSS/code changes",
        .parameters = &.{}, // Simplified for test
        .execute = FixTool.execute,
    };

    try agents_ctx.registerTool(fix_tool);

    // 6. Execute the task
    const task = "Fix the UI bug identified by the reasoning engine: " ++ vision_analysis;
    const agent_response = try agent.process(task, allocator);
    defer allocator.free(agent_response);

    // Verify agent used the echo backend correctly (as a proxy for execution)
    try std.testing.expect(std.mem.indexOf(u8, agent_response, "Echo:") != null);

    // Manually trigger the tool to verify integration
    const registry = try agents_ctx.getToolRegistry();
    const tool = registry.get("fix_ui_bug").?;

    var io_backend = try abi.foundation.io.IoBackend.init(allocator);
    defer io_backend.deinit();

    var tool_ctx = abi.ai.tools.createContext(allocator, ".", &io_backend.io);

    var result = try tool.execute(&tool_ctx, .null);
    defer result.deinit();

    // 7. Verify final outcome
    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("Fixed z-index in Sidebar.zig", result.output);

    // Final check on reasoning confidence
    try std.testing.expect(chain.getConfidence().score > 0.5);
}
