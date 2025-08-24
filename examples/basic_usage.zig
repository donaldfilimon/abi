//! Basic usage example for the Abi AI framework

const std = @import("std");
const agent = @import("../src/agent.zig");
const platform = @import("../src/platform.zig");
const localml = @import("../src/localml.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Abi AI Framework Demo ===\n", .{});

    // Initialize platform
    try platform.PlatformLayer.initializePlatform();
    const sys_info = platform.PlatformLayer.getSystemInfo();
    std.debug.print("Platform: {s}\n", .{sys_info.platform});
    std.debug.print("Memory: {d} MB\n", .{sys_info.memory / 1024 / 1024});
    std.debug.print("CPU Count: {d}\n", .{sys_info.cpu_count});

    // Demo 1: AI Agent with different personas
    try demoAIAgent(allocator);

    // Demo 2: Machine Learning
    try demoMachineLearning();

    // Demo 3: Platform optimizations
    try demoPlatformOptimizations(allocator);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

fn demoAIAgent(allocator: std.mem.Allocator) !void {
    std.debug.print("\n--- AI Agent Demo ---\n", .{});

    var ai_agent = try agent.Agent.init(allocator, .{});
    defer ai_agent.deinit();

    const personas = [_]agent.PersonaType{
        .EmpatheticAnalyst,
        .DirectExpert,
        .CreativeWriter,
        .TechnicalAdvisor,
    };

    const messages = [_][]const u8{
        "Hello, how are you today?",
        "Can you explain quantum computing?",
        "Write a short story about a robot",
        "What's the best way to learn Zig?",
    };

    for (personas, 0..) |persona, i| {
        ai_agent.setPersona(persona);
        const response = try ai_agent.generateResponse(messages[i]);
        defer allocator.free(response);

        std.debug.print("Persona: {s}\n", .{persona.toString()});
        std.debug.print("Message: {s}\n", .{messages[i]});
        std.debug.print("Response: {s}\n\n", .{response});
    }
}

fn demoMachineLearning() !void {
    std.debug.print("--- Machine Learning Demo ---\n", .{});

    // Create sample training data
    const training_data = [_]localml.DataRow{
        .{ .x1 = 1.0, .x2 = 2.0, .label = 1.0 },
        .{ .x1 = 2.0, .x2 = 3.0, .label = 1.0 },
        .{ .x1 = 3.0, .x2 = 4.0, .label = 0.0 },
        .{ .x1 = 4.0, .x2 = 5.0, .label = 0.0 },
        .{ .x1 = 1.5, .x2 = 2.5, .label = 1.0 },
        .{ .x1 = 3.5, .x2 = 4.5, .label = 0.0 },
    };

    // Initialize and train model
    var model = localml.Model.init();
    try model.train(&training_data, 0.01, 1000);

    std.debug.print("Model trained successfully!\n", .{});

    // Make predictions
    const test_cases = [_]localml.DataRow{
        .{ .x1 = 1.2, .x2 = 2.3, .label = 0.0 },
        .{ .x1 = 3.8, .x2 = 4.9, .label = 0.0 },
    };

    for (test_cases, 0..) |test_case, i| {
        const prediction = try model.predict(test_case);
        std.debug.print("Test {d}: input=({d:.1}, {d:.1}), prediction={d:.3}\n", .{
            i + 1, test_case.x1, test_case.x2, prediction,
        });
    }
}

fn demoPlatformOptimizations(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Platform Optimizations Demo ---\n", .{});

    // Demonstrate platform-specific memory allocation
    const test_size = 1024 * 1024; // 1MB
    const optimized_memory = try platform.PlatformLayer.allocateOptimizedMemory(test_size);
    defer allocator.free(optimized_memory);

    std.debug.print("Allocated {d} bytes using platform-optimized allocator\n", .{optimized_memory.len});

    // Demonstrate SIMD operations (if available)
    if (std.Target.x86.featureSetHas(std.Target.x86.features, .avx2)) {
        std.debug.print("AVX2 support detected - SIMD optimizations available\n", .{});
    } else {
        std.debug.print("SIMD optimizations not available on this platform\n", .{});
    }

    // Performance benchmark
    const iterations = 1000000;
    const timer = std.time.Timer.start() catch return;

    var sum: f64 = 0.0;
    for (0..iterations) |i| {
        sum += @as(f64, @floatFromInt(i));
    }

    const elapsed = timer.read();
    const ops_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0);

    std.debug.print("Performance test: {d:.0} operations/second\n", .{ops_per_second});
    std.debug.print("Sum: {d}\n", .{sum});
}

test "basic usage example" {
    // This test ensures the example compiles and runs
    try main();
}
