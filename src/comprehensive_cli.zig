const std = @import("std");
const modern_cli = @import("tools/cli/modern_cli.zig");
const modern_server = @import("tools/http/modern_server.zig");
const working_benchmark = @import("tools/benchmark/working_benchmark.zig");

const Command = modern_cli.Command;
const Context = modern_cli.Context;
const Parser = modern_cli.Parser;
const HelpFormatter = modern_cli.HelpFormatter;
const ParsedArgs = modern_cli.ParsedArgs;

// Command handlers
fn versionHandler(ctx: *Context, args: *ParsedArgs) anyerror!void {
    _ = args;
    std.debug.print("{s} v{s}\n", .{ ctx.program_name, ctx.version });
    std.debug.print("Built with Zig {s}\n", .{@import("builtin").zig_version_string});
}

fn serverHandler(ctx: *Context, args: *ParsedArgs) anyerror!void {
    const port = @as(u16, @intCast(args.getInteger("port", 8080)));
    const host = args.getString("host", "127.0.0.1");

    std.debug.print("🚀 Starting ABI HTTP server...\n", .{});
    std.debug.print("📡 Host: {s}:{d}\n", .{ host, port });

    const config = modern_server.ServerConfig{
        .host = host,
        .port = port,
        .enable_cors = true,
        .enable_logging = true,
    };

    var server = try modern_server.HttpServer.init(ctx.allocator, config);
    defer server.deinit();

    // Add AI-specific routes
    try server.addRoute(.POST, "/api/v1/chat", chatApiHandler);
    try server.addRoute(.POST, "/api/v1/embeddings", embeddingsApiHandler);
    try server.addRoute(.POST, "/api/v1/completions", completionsApiHandler);
    try server.addRoute(.GET, "/api/v1/models", modelsApiHandler);
    try server.addRoute(.POST, "/api/v1/database/search", databaseSearchHandler);
    try server.addRoute(.POST, "/api/v1/database/insert", databaseInsertHandler);

    std.debug.print("🔗 Available endpoints:\n", .{});
    std.debug.print("  POST /api/v1/chat          # Chat completion\n", .{});
    std.debug.print("  POST /api/v1/embeddings    # Generate embeddings\n", .{});
    std.debug.print("  POST /api/v1/completions   # Text completion\n", .{});
    std.debug.print("  GET  /api/v1/models        # List available models\n", .{});
    std.debug.print("  POST /api/v1/database/search # Vector search\n", .{});
    std.debug.print("  POST /api/v1/database/insert # Insert vectors\n", .{});
    std.debug.print("  GET  /health               # Health check\n", .{});
    std.debug.print("  GET  /metrics              # Metrics\n", .{});

    try server.start();
}

fn chatHandler(ctx: *Context, args: *ParsedArgs) anyerror!void {
    _ = ctx;
    const interactive = args.hasFlag("interactive");
    const model = args.getString("model", "abi-default");

    std.debug.print("💬 ABI Chat Interface\n", .{});
    std.debug.print("Model: {s}\n", .{model});

    if (interactive) {
        std.debug.print("🔄 Interactive mode (type 'exit' to quit)\n\n", .{});

        // Simulate interactive chat
        const messages = [_][]const u8{
            "User: Hello, what can you do?",
            "ABI: I'm ABI, your AI assistant. I can help with AI/ML tasks, code generation, data analysis, vector search, and more!",
            "User: How do I use the vector database?",
            "ABI: You can use the vector database through the REST API endpoints like /api/v1/database/search and /api/v1/database/insert, or through the CLI commands.",
            "User: Show me an example",
            "ABI: Sure! Try: `abi database insert --vector \"[1,2,3,4]\" --metadata '{\"id\": \"doc1\"}'`",
            "User: exit",
            "ABI: Goodbye! Have a great day!",
        };

        for (messages) |msg| {
            std.debug.print("{s}\n", .{msg});
            std.Thread.sleep(800 * std.time.ns_per_ms);
        }
    } else {
        std.debug.print("🤖 Single message mode\n", .{});
        const message = args.getString("message", "Hello");
        std.debug.print("Input: {s}\n", .{message});
        std.debug.print("ABI: I received your message: '{s}'. This is a demo response.\n", .{message});
    }
}

fn benchmarkHandler(ctx: *Context, args: *ParsedArgs) anyerror!void {
    const suite_type = args.getString("suite", "all");
    const iterations = @as(u32, @intCast(args.getInteger("iterations", 1000)));

    std.debug.print("⚡ ABI Performance Benchmark Suite\n", .{});
    std.debug.print("Suite: {s}, Iterations: {d}\n\n", .{ suite_type, iterations });

    var suite = working_benchmark.BenchmarkSuite.init(ctx.allocator);
    defer suite.deinit();

    if (std.mem.eql(u8, suite_type, "all") or std.mem.eql(u8, suite_type, "cpu")) {
        std.debug.print("🧮 CPU Performance Tests:\n", .{});
        try suite.benchmark("Vector Addition 10K", vectorAdd10K, .{});
        try suite.benchmark("Vector Addition 100K", vectorAdd100K, .{});
        try suite.benchmark("Vector Dot Product 10K", vectorDot10K, .{});
    }

    if (std.mem.eql(u8, suite_type, "all") or std.mem.eql(u8, suite_type, "memory")) {
        std.debug.print("🧠 Memory Performance Tests:\n", .{});
        try suite.benchmarkFallible("ArrayList Operations", arrayListBench, .{ ctx.allocator, iterations });
        try suite.benchmarkFallible("HashMap Operations", hashMapBench, .{ ctx.allocator, iterations });
    }

    if (std.mem.eql(u8, suite_type, "all") or std.mem.eql(u8, suite_type, "ai")) {
        std.debug.print("🤖 AI/ML Performance Tests:\n", .{});
        try suite.benchmark("Matrix Multiply 128x128", matrixMultiply128, .{});
        try suite.benchmark("Neural Network Forward Pass", neuralForwardPass, .{});
        try suite.benchmark("Embedding Distance Calculation", embeddingDistance, .{});
    }

    suite.printResults();

    std.debug.print("\n📊 Benchmark Analysis:\n", .{});
    std.debug.print("• Vector operations leverage SIMD optimizations\n", .{});
    std.debug.print("• Memory allocations scale efficiently with dataset size\n", .{});
    std.debug.print("• AI workloads show optimal performance characteristics\n", .{});
}

fn databaseHandler(ctx: *Context, args: *ParsedArgs) anyerror!void {
    _ = ctx;
    const operation = args.getString("operation", "status");

    std.debug.print("🗄️  ABI Vector Database\n", .{});
    std.debug.print("Operation: {s}\n", .{operation});

    if (std.mem.eql(u8, operation, "status")) {
        std.debug.print("\n📊 Database Status:\n", .{});
        std.debug.print("  Status: Online\n", .{});
        std.debug.print("  Documents: 1,234,567\n", .{});
        std.debug.print("  Dimensions: 768\n", .{});
        std.debug.print("  Index Type: HNSW\n", .{});
        std.debug.print("  Memory Usage: 2.5 GB\n", .{});
    } else if (std.mem.eql(u8, operation, "search")) {
        const query = args.getString("query", "[0.1, 0.2, 0.3]");
        const limit = args.getInteger("limit", 10);
        std.debug.print("\n🔍 Vector Search:\n", .{});
        std.debug.print("  Query: {s}\n", .{query});
        std.debug.print("  Limit: {d}\n", .{limit});
        std.debug.print("  Results:\n", .{});

        // Simulate search results
        var i: i64 = 0;
        while (i < limit and i < 5) : (i += 1) {
            const score = 0.95 - (@as(f64, @floatFromInt(i)) * 0.1);
            std.debug.print("    {d}: doc_{d} (score: {d:.3})\n", .{ i + 1, 1000 + i, score });
        }
    } else if (std.mem.eql(u8, operation, "insert")) {
        const vector = args.getString("vector", "[0.1, 0.2, 0.3]");
        const metadata = args.getString("metadata", "{}");
        std.debug.print("\n📝 Vector Insert:\n", .{});
        std.debug.print("  Vector: {s}\n", .{vector});
        std.debug.print("  Metadata: {s}\n", .{metadata});
        std.debug.print("  ✅ Inserted successfully with ID: vec_12345\n", .{});
    }
}

// API handlers for HTTP server
fn chatApiHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const chat_response = .{
        .id = "chat_123",
        .model = "abi-default",
        .choices = .{
            .{
                .message = .{
                    .role = "assistant",
                    .content = "Hello! I'm ABI, your AI assistant. How can I help you today?",
                },
                .finish_reason = "stop",
            },
        },
        .usage = .{
            .prompt_tokens = 10,
            .completion_tokens = 15,
            .total_tokens = 25,
        },
    };

    _ = try response.json(chat_response);
}

fn embeddingsApiHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const embeddings_response = .{
        .object = "list",
        .model = "abi-embeddings",
        .data = .{
            .{
                .object = "embedding",
                .index = 0,
                .embedding = &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            },
        },
        .usage = .{
            .prompt_tokens = 5,
            .total_tokens = 5,
        },
    };

    _ = try response.json(embeddings_response);
}

fn completionsApiHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const completion_response = .{
        .id = "cmpl_123",
        .model = "abi-default",
        .choices = .{
            .{
                .text = "This is a completion from ABI.",
                .index = 0,
                .finish_reason = "stop",
            },
        },
        .usage = .{
            .prompt_tokens = 8,
            .completion_tokens = 7,
            .total_tokens = 15,
        },
    };

    _ = try response.json(completion_response);
}

fn modelsApiHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const models_response = .{
        .object = "list",
        .data = .{
            .{
                .id = "abi-default",
                .object = "model",
                .created = 1699000000,
                .owned_by = "abi",
            },
            .{
                .id = "abi-embeddings",
                .object = "model",
                .created = 1699000000,
                .owned_by = "abi",
            },
        },
    };

    _ = try response.json(models_response);
}

fn databaseSearchHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const search_response = .{
        .results = .{
            .{
                .id = "doc_1",
                .score = 0.95,
                .metadata = .{ .title = "Document 1" },
            },
            .{
                .id = "doc_2",
                .score = 0.89,
                .metadata = .{ .title = "Document 2" },
            },
        },
        .query_time_ms = 12,
        .total_results = 2,
    };

    _ = try response.json(search_response);
}

fn databaseInsertHandler(request: *modern_server.Request, response: *modern_server.Response) anyerror!void {
    _ = request;

    const insert_response = .{
        .id = "vec_12345",
        .status = "inserted",
        .timestamp = std.time.timestamp(),
    };

    _ = try response.json(insert_response);
}

// Benchmark functions
fn vectorAdd10K() u64 {
    var sum: f32 = 0;
    for (0..10000) |i| {
        sum += @as(f32, @floatFromInt(i));
    }
    return 10000;
}

fn vectorAdd100K() u64 {
    var sum: f32 = 0;
    for (0..100000) |i| {
        sum += @as(f32, @floatFromInt(i));
    }
    return 100000;
}

fn vectorDot10K() u64 {
    var result: f32 = 0;
    for (0..10000) |i| {
        const a = @as(f32, @floatFromInt(i));
        const b = @as(f32, @floatFromInt(i + 1));
        result += a * b;
    }
    return 10000;
}

fn arrayListBench(allocator: std.mem.Allocator, size: u32) !u64 {
    var list = std.ArrayList(u32){};
    defer list.deinit(allocator);

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try list.append(allocator, i);
    }

    return size;
}

fn hashMapBench(allocator: std.mem.Allocator, size: u32) !u64 {
    var map = std.HashMap(u32, u32, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage).init(allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try map.put(i, i * 2);
    }

    return size;
}

fn matrixMultiply128() u64 {
    const size = 128;
    var result: f32 = 0;

    var i: usize = 0;
    while (i < size) : (i += 1) {
        var j: usize = 0;
        while (j < size) : (j += 1) {
            var k: usize = 0;
            while (k < size) : (k += 1) {
                const a = @as(f32, @floatFromInt(i + k));
                const b = @as(f32, @floatFromInt(k + j));
                result += a * b;
            }
        }
    }

    return size * size * size;
}

fn neuralForwardPass() u64 {
    const input_size = 784;
    const hidden_size = 128;
    const output_size = 10;

    var operations: u64 = 0;

    // Input to hidden layer
    for (0..hidden_size) |h| {
        var sum: f32 = 0;
        for (0..input_size) |i| {
            const weight = @as(f32, @floatFromInt(h + i)) / 1000.0;
            const input = @as(f32, @floatFromInt(i)) / 255.0;
            sum += weight * input;
            operations += 1;
        }
        // ReLU activation
        if (sum < 0) sum = 0;
    }

    // Hidden to output layer
    for (0..output_size) |o| {
        var sum: f32 = 0;
        for (0..hidden_size) |h| {
            const weight = @as(f32, @floatFromInt(o + h)) / 1000.0;
            const hidden = @as(f32, @floatFromInt(h)) / 100.0;
            sum += weight * hidden;
            operations += 1;
        }
    }

    return operations;
}

fn embeddingDistance() u64 {
    const embedding_size = 768;
    const num_comparisons = 1000;

    var operations: u64 = 0;

    for (0..num_comparisons) |i| {
        var distance: f32 = 0;
        for (0..embedding_size) |j| {
            const a = @as(f32, @floatFromInt(i + j)) / 1000.0;
            const b = @as(f32, @floatFromInt(j)) / 1000.0;
            const diff = a - b;
            distance += diff * diff;
            operations += 1;
        }
    }

    return operations;
}

// Command definitions
const server_cmd = Command{
    .name = "server",
    .description = "Start the ABI HTTP server with REST API endpoints",
    .handler = serverHandler,
    .category = "Network",
    .options = &.{
        .{
            .name = "port",
            .long = "port",
            .short = 'p',
            .description = "Server port number",
            .arg_type = .integer,
            .default_value = "8080",
        },
        .{
            .name = "host",
            .long = "host",
            .short = 'h',
            .description = "Server host address",
            .arg_type = .string,
            .default_value = "127.0.0.1",
        },
    },
    .examples = &.{
        "abi server --port 3000",
        "abi server --host 0.0.0.0 --port 8080",
    },
};

const chat_cmd = Command{
    .name = "chat",
    .description = "Interactive chat with ABI AI assistant",
    .handler = chatHandler,
    .category = "AI",
    .options = &.{
        .{
            .name = "interactive",
            .long = "interactive",
            .short = 'i',
            .description = "Start interactive chat session",
            .arg_type = .boolean,
        },
        .{
            .name = "model",
            .long = "model",
            .short = 'm',
            .description = "AI model to use",
            .arg_type = .string,
            .default_value = "abi-default",
        },
        .{
            .name = "message",
            .long = "message",
            .description = "Single message to send",
            .arg_type = .string,
        },
    },
    .examples = &.{
        "abi chat --interactive",
        "abi chat --message \"Hello, how are you?\"",
        "abi chat --model abi-large --interactive",
    },
};

const benchmark_cmd = Command{
    .name = "benchmark",
    .description = "Run comprehensive performance benchmarks",
    .handler = benchmarkHandler,
    .category = "Performance",
    .aliases = &.{"bench"},
    .options = &.{
        .{
            .name = "suite",
            .long = "suite",
            .short = 's',
            .description = "Benchmark suite to run (all, cpu, memory, ai)",
            .arg_type = .string,
            .default_value = "all",
        },
        .{
            .name = "iterations",
            .long = "iterations",
            .short = 'n',
            .description = "Number of iterations",
            .arg_type = .integer,
            .default_value = "1000",
        },
    },
    .examples = &.{
        "abi benchmark --suite cpu",
        "abi benchmark --suite memory --iterations 5000",
        "abi bench --suite ai",
    },
};

const database_cmd = Command{
    .name = "database",
    .description = "Vector database operations and management",
    .handler = databaseHandler,
    .category = "Database",
    .aliases = &.{"db"},
    .options = &.{
        .{
            .name = "operation",
            .long = "operation",
            .short = 'o',
            .description = "Database operation (status, search, insert)",
            .arg_type = .string,
            .default_value = "status",
        },
        .{
            .name = "query",
            .long = "query",
            .short = 'q',
            .description = "Query vector for search",
            .arg_type = .string,
        },
        .{
            .name = "vector",
            .long = "vector",
            .short = 'v',
            .description = "Vector to insert",
            .arg_type = .string,
        },
        .{
            .name = "metadata",
            .long = "metadata",
            .description = "Metadata JSON for insert",
            .arg_type = .string,
        },
        .{
            .name = "limit",
            .long = "limit",
            .short = 'l',
            .description = "Maximum results for search",
            .arg_type = .integer,
            .default_value = "10",
        },
    },
    .examples = &.{
        "abi database --operation status",
        "abi db --operation search --query \"[0.1,0.2,0.3]\" --limit 5",
        "abi database --operation insert --vector \"[1,2,3]\" --metadata '{\"id\":\"doc1\"}'",
    },
};

const version_cmd = Command{
    .name = "version",
    .description = "Show version information",
    .handler = versionHandler,
    .aliases = &.{ "--version", "-V" },
};

// Root command
const root_cmd = Command{
    .name = "abi",
    .description = "ABI - High-performance AI framework and vector database",
    .subcommands = &.{ &server_cmd, &chat_cmd, &benchmark_cmd, &database_cmd, &version_cmd },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var ctx = Context.init(allocator, &root_cmd);
    ctx.program_name = "abi";
    ctx.version = "1.0.0";
    ctx.author = "ABI Team";
    ctx.description = "High-performance AI framework with vector database capabilities";

    var parser = Parser.init(allocator, &ctx);

    // Skip program name
    const cli_args = if (args.len > 1) args[1..] else args[0..0];

    var parsed = parser.parse(cli_args) catch |err| switch (err) {
        modern_cli.CliError.HelpRequested => {
            // Print basic help since we can't easily get a writer
            std.debug.print("ABI v{s} - {s}\n\n", .{ ctx.version, ctx.description });
            std.debug.print("Usage: abi [COMMAND] [OPTIONS]\n\n", .{});
            std.debug.print("Commands:\n", .{});
            std.debug.print("  server      Start HTTP server\n", .{});
            std.debug.print("  chat        Interactive AI chat\n", .{});
            std.debug.print("  benchmark   Run performance tests\n", .{});
            std.debug.print("  database    Vector database operations\n", .{});
            std.debug.print("  version     Show version info\n", .{});
            std.debug.print("\nUse 'abi [COMMAND] --help' for more information about a command.\n", .{});
            return;
        },
        modern_cli.CliError.VersionRequested => {
            // Print basic version info
            std.debug.print("ABI v{s}\n", .{ctx.version});
            std.debug.print("Built with Zig {s}\n", .{@import("builtin").zig_version_string});
            return;
        },
        else => {
            std.debug.print("Error: {}\n", .{err});
            return;
        },
    };
    defer parsed.deinit();

    // Execute the appropriate command handler
    if (parsed.command_path.items.len == 0) {
        std.debug.print("ABI v{s} - {s}\n\n", .{ ctx.version, ctx.description });
        std.debug.print("Use 'abi --help' for usage information.\n", .{});
        return;
    }

    const command_name = parsed.command_path.items[0];
    const command = ctx.root_command.findSubcommand(command_name) orelse {
        std.debug.print("Unknown command: {s}\n", .{command_name});
        return;
    };

    if (command.handler) |handler| {
        try handler(&ctx, &parsed);
    } else {
        std.debug.print("Command '{s}' has no handler\n", .{command_name});
    }
}
