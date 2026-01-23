//! Multi-Model Orchestration Example
//!
//! Demonstrates the multi-model orchestration capabilities including:
//! - Registering multiple LLM backends
//! - Task-based routing
//! - Round-robin load balancing
//! - Fallback handling
//! - Ensemble mode for combining outputs

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check if AI is enabled
    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    std.debug.print("=== Multi-Model Orchestration Demo ===\n\n", .{});

    // Example 1: Basic orchestrator setup
    try basicOrchestration(allocator);

    // Example 2: Task-based routing
    try taskBasedRouting(allocator);

    // Example 3: Fallback demonstration
    try fallbackDemo(allocator);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

/// Example 1: Basic orchestrator setup with multiple models
fn basicOrchestration(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Example 1: Basic Orchestration ---\n", .{});

    // Create orchestrator with round-robin strategy
    var orchestrator = abi.ai.orchestration.Orchestrator.init(allocator, .{
        .strategy = .round_robin,
        .enable_fallback = true,
        .max_concurrent_requests = 10,
    }) catch |err| {
        std.debug.print("Failed to create orchestrator: {}\n", .{err});
        return;
    };
    defer orchestrator.deinit();

    // Register multiple models
    try orchestrator.registerModel(.{
        .id = "gpt-4",
        .name = "GPT-4",
        .backend = .openai,
        .model_name = "gpt-4-turbo-preview",
        .capabilities = &.{ .reasoning, .coding, .creative },
        .priority = 1,
        .cost_per_1k_tokens = 0.03,
    });

    try orchestrator.registerModel(.{
        .id = "claude-3",
        .name = "Claude 3 Opus",
        .backend = .anthropic,
        .model_name = "claude-3-opus-20240229",
        .capabilities = &.{ .reasoning, .creative, .analysis },
        .priority = 2,
        .cost_per_1k_tokens = 0.015,
    });

    try orchestrator.registerModel(.{
        .id = "llama-3",
        .name = "Llama 3 70B",
        .backend = .ollama,
        .model_name = "llama3:70b",
        .capabilities = &.{ .reasoning, .coding },
        .priority = 3,
        .cost_per_1k_tokens = 0.0,
    });

    // Get orchestrator stats
    const stats = orchestrator.getStats();
    std.debug.print("Registered models: {d}\n", .{stats.total_models});
    std.debug.print("Available models: {d}\n", .{stats.available_models});

    // Route a request (round-robin will select first available)
    const result1 = try orchestrator.route("Hello, world!", null);
    std.debug.print("First route: {s}\n", .{result1.model_id});

    const result2 = try orchestrator.route("How are you?", null);
    std.debug.print("Second route: {s}\n", .{result2.model_id});

    const result3 = try orchestrator.route("What is Zig?", null);
    std.debug.print("Third route: {s}\n", .{result3.model_id});

    std.debug.print("\n", .{});
}

/// Example 2: Task-based routing
fn taskBasedRouting(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Example 2: Task-Based Routing ---\n", .{});

    // Create orchestrator with task-based strategy
    var orchestrator = abi.ai.orchestration.Orchestrator.init(allocator, .{
        .strategy = .task_based,
        .enable_fallback = true,
    }) catch |err| {
        std.debug.print("Failed to create orchestrator: {}\n", .{err});
        return;
    };
    defer orchestrator.deinit();

    // Register specialized models
    try orchestrator.registerModel(.{
        .id = "codellama",
        .name = "Code Llama",
        .backend = .ollama,
        .model_name = "codellama:34b",
        .capabilities = &.{.coding},
        .priority = 1,
    });

    try orchestrator.registerModel(.{
        .id = "creative-writer",
        .name = "Creative Writer",
        .backend = .openai,
        .model_name = "gpt-4",
        .capabilities = &.{ .creative, .summarization },
        .priority = 1,
    });

    try orchestrator.registerModel(.{
        .id = "math-solver",
        .name = "Math Solver",
        .backend = .openai,
        .model_name = "gpt-4",
        .capabilities = &.{.math},
        .priority = 1,
    });

    try orchestrator.registerModel(.{
        .id = "general",
        .name = "General Assistant",
        .backend = .ollama,
        .model_name = "llama3:8b",
        .capabilities = &.{}, // No specific capabilities = handles general tasks
        .priority = 2,
    });

    // Test task type detection
    const prompts = [_]struct { text: []const u8, task: ?abi.ai.TaskType }{
        .{ .text = "Write a function to sort an array", .task = .coding },
        .{ .text = "Write a poem about the ocean", .task = .creative },
        .{ .text = "Calculate the derivative of x^2 + 3x", .task = .math },
        .{ .text = "Summarize this article about AI", .task = .summarization },
        .{ .text = "Hello, how are you?", .task = null },
    };

    for (prompts) |p| {
        // Detect task type if not provided
        const task_type = p.task orelse abi.ai.TaskType.detect(p.text);
        const result = try orchestrator.route(p.text, task_type);

        std.debug.print("Prompt: \"{s}\"\n", .{p.text});
        std.debug.print("  Task: {s}\n", .{@tagName(task_type)});
        std.debug.print("  Routed to: {s}\n", .{result.model_id});
    }

    std.debug.print("\n", .{});
}

/// Example 3: Fallback handling demonstration
fn fallbackDemo(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Example 3: Fallback Handling ---\n", .{});

    // Create orchestrator with fallback enabled
    var orchestrator = abi.ai.orchestration.Orchestrator.init(allocator, .{
        .strategy = .priority,
        .enable_fallback = true,
        .max_retries = 3,
    }) catch |err| {
        std.debug.print("Failed to create orchestrator: {}\n", .{err});
        return;
    };
    defer orchestrator.deinit();

    // Register models with different priorities
    try orchestrator.registerModel(.{
        .id = "primary",
        .name = "Primary Model",
        .backend = .openai,
        .priority = 1, // Highest priority
    });

    try orchestrator.registerModel(.{
        .id = "secondary",
        .name = "Secondary Model",
        .backend = .anthropic,
        .priority = 2,
    });

    try orchestrator.registerModel(.{
        .id = "fallback",
        .name = "Fallback Model",
        .backend = .ollama,
        .priority = 3,
    });

    std.debug.print("Initial state:\n", .{});
    printModelStatuses(&orchestrator);

    // Simulate primary model becoming unhealthy
    std.debug.print("\nSimulating primary model failure...\n", .{});
    try orchestrator.setModelHealth("primary", .unhealthy);

    std.debug.print("After primary failure:\n", .{});
    printModelStatuses(&orchestrator);

    // Route request - should go to secondary
    const result = try orchestrator.route("Test request", null);
    std.debug.print("Request routed to: {s}\n", .{result.model_id});

    // Simulate secondary also failing
    std.debug.print("\nSimulating secondary model failure...\n", .{});
    try orchestrator.setModelHealth("secondary", .degraded);

    std.debug.print("After secondary degradation:\n", .{});
    printModelStatuses(&orchestrator);

    // Route request - should go to fallback
    const result2 = try orchestrator.route("Another request", null);
    std.debug.print("Request routed to: {s}\n", .{result2.model_id});

    // Restore primary
    std.debug.print("\nRestoring primary model...\n", .{});
    try orchestrator.setModelHealth("primary", .healthy);

    const result3 = try orchestrator.route("Final request", null);
    std.debug.print("Request routed to: {s}\n", .{result3.model_id});

    std.debug.print("\n", .{});
}

fn printModelStatuses(orchestrator: *abi.ai.orchestration.Orchestrator) void {
    const models = orchestrator.listModels(std.heap.page_allocator) catch return;
    defer std.heap.page_allocator.free(models);

    for (models) |model_id| {
        if (orchestrator.getModel(model_id)) |model| {
            std.debug.print("  {s}: {s} (priority: {d})\n", .{
                model_id,
                model.status.toString(),
                model.config.priority,
            });
        }
    }
}
