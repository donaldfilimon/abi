//! Example of using Abi AI Framework as a module in your Zig project
//!
//! To use this in your project:
//! 1. Add Abi as a dependency in your build.zig.zon
//! 2. Import the module in your build.zig
//! 3. Use the framework in your code as shown below

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Abi Framework Module Usage Example ===\n\n", .{});

    // Example 1: Using individual modules
    try useIndividualModules(allocator);

    // Example 2: Building an AI-powered application
    try buildAIApplication(allocator);

    // Example 3: High-performance data processing
    try performanceProcessing(allocator);
}

/// Example 1: Using individual Abi modules
fn useIndividualModules(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Using Individual Modules ---\n", .{});

    // Use SIMD text processing directly
    const text = "Hello, world!\nThis is a test.\nWith multiple lines.";
    const line_count = abi.simd_text.SIMDTextProcessor.countLines(text);
    std.debug.print("Line count: {}\n", .{line_count});

    // Use lock-free queue
    var queue = try abi.lockfree.LockFreeQueue(i32).init(allocator);
    defer queue.deinit();

    try queue.enqueue(42);
    try queue.enqueue(100);

    while (queue.dequeue()) |value| {
        std.debug.print("Dequeued: {}\n", .{value});
    }

    // Use neural network
    var nn = try abi.neural.NeuralNetwork.init(allocator);
    defer nn.deinit();

    try nn.addLayer(.{
        .type = .Dense,
        .input_size = 2,
        .output_size = 3,
        .activation = .ReLU,
    });

    std.debug.print("\n", .{});
}

/// Example 2: Building an AI-powered application
fn buildAIApplication(allocator: std.mem.Allocator) !void {
    std.debug.print("--- AI-Powered Application ---\n", .{});

    // Initialize the AI agent
    var agent = try abi.agent.Agent.init(allocator, .{
        .max_response_length = 512,
        .max_history_length = 10,
    });
    defer agent.deinit();

    // Create a vector database for context storage
    var db = try abi.database.Db.open("app_context.wdbx", true);
    defer db.close();
    try db.init(384); // Standard embedding dimension

    // Simulate a conversation
    const queries = [_]struct { persona: abi.agent.PersonaType, query: []const u8 }{
        .{ .persona = .TechnicalAdvisor, .query = "How do I optimize database queries?" },
        .{ .persona = .CreativeWriter, .query = "Write a haiku about programming" },
        .{ .persona = .ProblemSolver, .query = "Debug this code: var x = null.field" },
    };

    for (queries) |q| {
        agent.setPersona(q.persona);

        const response = agent.generateResponse(q.query) catch |err| {
            std.debug.print("Error: {}\n", .{err});
            continue;
        };
        defer allocator.free(response);

        std.debug.print("{s}: {s}\n", .{ @tagName(q.persona), q.query });
        std.debug.print("Response: {s}\n\n", .{response});

        // Store the interaction in the database (simplified)
        // In a real app, you'd generate embeddings first
        const dummy_embedding = [_]f32{0.1} ** 384;
        _ = try db.addEmbedding(&dummy_embedding);
    }
}

/// Example 3: High-performance data processing
fn performanceProcessing(allocator: std.mem.Allocator) !void {
    std.debug.print("--- High-Performance Processing ---\n", .{});

    // Process large text data
    const large_text = try allocator.alloc(u8, 1024 * 1024); // 1MB
    defer allocator.free(large_text);
    @memset(large_text, 'A');

    var timer = try std.time.Timer.start();

    // SIMD text processing
    const lines = abi.simd_text.SIMDTextProcessor.countLines(large_text);
    const text_time = timer.read();

    std.debug.print("Processed 1MB text in {}ns ({} lines)\n", .{ text_time, lines });

    // Vector operations
    const vec_size = 10000;
    const vec_a = try allocator.alloc(f32, vec_size);
    defer allocator.free(vec_a);
    const vec_b = try allocator.alloc(f32, vec_size);
    defer allocator.free(vec_b);

    for (vec_a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (vec_b, 0..) |*v, i| v.* = @floatFromInt(i * 2);

    timer.reset();
    const distance = abi.simd_vector.distanceSquaredSIMD(vec_a, vec_b);
    const vec_time = timer.read();

    std.debug.print("Vector distance ({} elements) in {}ns: {d:.2}\n", .{ vec_size, vec_time, distance });

    // Platform info
    const sys_info = abi.platform.PlatformLayer.getSystemInfo();
    std.debug.print("\nPlatform: {s}, CPUs: {}, Memory: {}MB\n", .{
        sys_info.platform,
        sys_info.cpu_count,
        sys_info.memory / 1024 / 1024,
    });
}

/// Example build.zig configuration for using Abi
pub const example_build_zig =
    \\const std = @import("std");
    \\
    \\pub fn build(b: *std.Build) void {
    \\    const target = b.standardTargetOptions(.{});
    \\    const optimize = b.standardOptimizeOption(.{});
    \\
    \\    // Add Abi dependency
    \\    const abi_dep = b.dependency("abi", .{
    \\        .target = target,
    \\        .optimize = optimize,
    \\    });
    \\
    \\    const exe = b.addExecutable(.{
    \\        .name = "my-app",
    \\        .root_source_file = b.path("src/main.zig"),
    \\        .target = target,
    \\        .optimize = optimize,
    \\    });
    \\
    \\    // Add Abi module
    \\    exe.root_module.addImport("abi", abi_dep.module("abi"));
    \\
    \\    // Or use specific modules
    \\    exe.root_module.addImport("abi-agent", abi_dep.module("agent"));
    \\    exe.root_module.addImport("abi-database", abi_dep.module("database"));
    \\
    \\    b.installArtifact(exe);
    \\}
;
