//! WDBX Vector Database - Main Entry Point
//!
//! This is the main entry point for the WDBX vector database application.
//! It provides a unified interface for all database operations through
//! various APIs (CLI, HTTP, TCP, WebSocket).

const std = @import("std");
const core = @import("core/mod.zig");
const api = @import("api/mod.zig");

/// Application configuration
const AppConfig = struct {
    /// Command to execute
    command: Command,
    /// Database configuration
    db_config: ?core.DatabaseConfig = null,
    /// API configuration
    api_config: api.ApiConfig = .{},
    /// Additional command-specific options
    options: CommandOptions = .{},
};

/// Available commands
const Command = enum {
    help,
    version,
    init,
    serve,
    cli,
    benchmark,
    test,

    pub fn fromString(s: []const u8) ?Command {
        return std.meta.stringToEnum(Command, s);
    }
};

/// Command-specific options
const CommandOptions = struct {
    /// Server options
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    protocol: Protocol = .http,
    
    /// CLI options
    db_path: ?[]const u8 = null,
    verbose: bool = false,
    quiet: bool = false,
    
    /// Benchmark options
    vector_count: usize = 10000,
    dimensions: u32 = 384,
    query_count: usize = 1000,
};

/// Server protocol
const Protocol = enum {
    http,
    tcp,
    websocket,
    grpc,
};

/// Application errors
const AppError = error{
    InvalidCommand,
    MissingArgument,
    InvalidConfiguration,
    DatabaseError,
    ServerError,
};

/// Main entry point
pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const config = try parseArgs(allocator);
    defer if (config.options.db_path) |path| allocator.free(path);

    // Execute command
    try executeCommand(allocator, config);
}

/// Parse command line arguments
fn parseArgs(allocator: std.mem.Allocator) !AppConfig {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // Skip executable name

    // Get command
    const cmd_str = args.next() orelse "help";
    const command = Command.fromString(cmd_str) orelse {
        std.debug.print("Unknown command: {s}\n", .{cmd_str});
        return AppError.InvalidCommand;
    };

    var config = AppConfig{ .command = command };

    // Parse remaining arguments
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--host")) {
            config.options.host = args.next() orelse return AppError.MissingArgument;
        } else if (std.mem.eql(u8, arg, "--port")) {
            const port_str = args.next() orelse return AppError.MissingArgument;
            config.options.port = try std.fmt.parseInt(u16, port_str, 10);
        } else if (std.mem.eql(u8, arg, "--db")) {
            const path = args.next() orelse return AppError.MissingArgument;
            config.options.db_path = try allocator.dupe(u8, path);
        } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            config.options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--quiet") or std.mem.eql(u8, arg, "-q")) {
            config.options.quiet = true;
        } else if (std.mem.eql(u8, arg, "--dimensions") or std.mem.eql(u8, arg, "-d")) {
            const dim_str = args.next() orelse return AppError.MissingArgument;
            config.options.dimensions = try std.fmt.parseInt(u32, dim_str, 10);
        } else if (std.mem.eql(u8, arg, "--protocol")) {
            const proto_str = args.next() orelse return AppError.MissingArgument;
            config.options.protocol = std.meta.stringToEnum(Protocol, proto_str) orelse .http;
        }
    }

    return config;
}

/// Execute the specified command
fn executeCommand(allocator: std.mem.Allocator, config: AppConfig) !void {
    switch (config.command) {
        .help => try showHelp(),
        .version => try showVersion(),
        .init => try initDatabase(allocator, config),
        .serve => try startServer(allocator, config),
        .cli => try runCli(allocator, config),
        .benchmark => try runBenchmark(allocator, config),
        .test => try runTests(allocator),
    }
}

/// Show help message
fn showHelp() !void {
    const help_text =
        \\WDBX Vector Database v{s}
        \\
        \\Usage: wdbx <command> [options]
        \\
        \\Commands:
        \\  help       Show this help message
        \\  version    Show version information
        \\  init       Initialize a new database
        \\  serve      Start a server (HTTP/TCP/WebSocket)
        \\  cli        Run interactive CLI mode
        \\  benchmark  Run performance benchmarks
        \\  test       Run test suite
        \\
        \\Common Options:
        \\  --db <path>         Database file path
        \\  --host <address>    Server host address (default: 127.0.0.1)
        \\  --port <number>     Server port (default: 8080)
        \\  --protocol <type>   Server protocol: http, tcp, websocket (default: http)
        \\  --verbose, -v       Enable verbose output
        \\  --quiet, -q         Suppress output
        \\
        \\Examples:
        \\  wdbx init --db vectors.wdbx --dimensions 384
        \\  wdbx serve --db vectors.wdbx --protocol http --port 8080
        \\  wdbx cli --db vectors.wdbx
        \\
        \\For more information, visit: https://github.com/yourusername/wdbx
        \\
    ;

    const stdout = std.io.getStdOut().writer();
    try stdout.print(help_text, .{core.VERSION});
}

/// Show version information
fn showVersion() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("WDBX Vector Database v{s}\n", .{core.VERSION});
    try stdout.print("Build: {s}\n", .{@tagName(std.builtin.mode)});
    try stdout.print("Zig version: {}\n", .{std.builtin.zig_version});
}

/// Initialize a new database
fn initDatabase(allocator: std.mem.Allocator, config: AppConfig) !void {
    const db_path = config.options.db_path orelse {
        std.debug.print("Error: Database path required. Use --db <path>\n", .{});
        return AppError.MissingArgument;
    };

    // Create database configuration
    const db_config = core.DatabaseConfig{
        .dimensions = config.options.dimensions,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
        .enable_simd = true,
    };

    // Create and initialize database
    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(db_config);

    std.debug.print("Database initialized successfully at: {s}\n", .{db_path});
    std.debug.print("  Dimensions: {d}\n", .{db_config.dimensions});
    std.debug.print("  Index type: {s}\n", .{@tagName(db_config.index_type)});
    std.debug.print("  Distance metric: {s}\n", .{@tagName(db_config.distance_metric)});
}

/// Start a server
fn startServer(allocator: std.mem.Allocator, config: AppConfig) !void {
    const db_path = config.options.db_path orelse {
        std.debug.print("Error: Database path required. Use --db <path>\n", .{});
        return AppError.MissingArgument;
    };

    std.debug.print("Starting {s} server on {s}:{d}\n", .{
        @tagName(config.options.protocol),
        config.options.host,
        config.options.port,
    });

    switch (config.options.protocol) {
        .http => {
            var server = try api.HttpServer.init(allocator, .{
                .db_path = db_path,
                .host = config.options.host,
                .port = config.options.port,
                .api_config = config.api_config,
            });
            defer server.deinit();
            try server.run();
        },
        .tcp => {
            var server = try api.TcpServer.init(allocator, .{
                .db_path = db_path,
                .host = config.options.host,
                .port = config.options.port,
            });
            defer server.deinit();
            try server.run();
        },
        else => {
            std.debug.print("Protocol {s} not yet implemented\n", .{@tagName(config.options.protocol)});
            return AppError.InvalidConfiguration;
        },
    }
}

/// Run interactive CLI
fn runCli(allocator: std.mem.Allocator, config: AppConfig) !void {
    const cli_config = api.cli.CLI.Config{
        .db_path = config.options.db_path,
        .verbose = config.options.verbose,
        .quiet = config.options.quiet,
    };

    const cli = try api.CLI.init(allocator, cli_config);
    defer cli.deinit();

    // If we have remaining args, run them as a command
    // Otherwise, start interactive mode
    std.debug.print("Starting WDBX CLI...\n", .{});
    
    // For now, just show help
    try cli.run(&[_][]const u8{"help"});
}

/// Run benchmarks
fn runBenchmark(allocator: std.mem.Allocator, config: AppConfig) !void {
    std.debug.print("Running WDBX benchmarks...\n", .{});
    std.debug.print("  Vector count: {d}\n", .{config.options.vector_count});
    std.debug.print("  Dimensions: {d}\n", .{config.options.dimensions});
    std.debug.print("  Query count: {d}\n", .{config.options.query_count});

    // Create temporary database
    const tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/benchmark.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    // Initialize database
    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(.{
        .dimensions = config.options.dimensions,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
        .enable_simd = true,
    });

    // Generate random vectors
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();

    std.debug.print("\nInsertion benchmark:\n", .{});
    const insert_start = std.time.nanoTimestamp();

    var i: usize = 0;
    while (i < config.options.vector_count) : (i += 1) {
        const vec = try allocator.alloc(f32, config.options.dimensions);
        defer allocator.free(vec);

        for (vec) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0; // Range: [-1, 1]
        }

        _ = try db.addVector(vec, null);

        if ((i + 1) % 1000 == 0) {
            std.debug.print("  Inserted {d} vectors...\r", .{i + 1});
        }
    }

    const insert_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - insert_start)) / 1e9;
    const insert_rate = @as(f64, @floatFromInt(config.options.vector_count)) / insert_time;
    
    std.debug.print("\n  Total time: {d:.2} seconds\n", .{insert_time});
    std.debug.print("  Rate: {d:.0} vectors/second\n", .{insert_rate});

    // Query benchmark
    std.debug.print("\nQuery benchmark:\n", .{});
    const query_start = std.time.nanoTimestamp();

    i = 0;
    while (i < config.options.query_count) : (i += 1) {
        const query = try allocator.alloc(f32, config.options.dimensions);
        defer allocator.free(query);

        for (query) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }

        const results = try db.search(query, 10, allocator);
        allocator.free(results);
    }

    const query_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - query_start)) / 1e9;
    const query_rate = @as(f64, @floatFromInt(config.options.query_count)) / query_time;
    const avg_query_time = query_time / @as(f64, @floatFromInt(config.options.query_count)) * 1000.0;

    std.debug.print("  Total time: {d:.2} seconds\n", .{query_time});
    std.debug.print("  Rate: {d:.0} queries/second\n", .{query_rate});
    std.debug.print("  Average query time: {d:.2} ms\n", .{avg_query_time});

    // Database statistics
    const stats = db.getStats();
    std.debug.print("\nDatabase statistics:\n", .{});
    std.debug.print("  Total vectors: {d}\n", .{stats.total_vectors});
    std.debug.print("  Total searches: {d}\n", .{stats.total_searches});
    std.debug.print("  Average search time: {d} ns\n", .{stats.avg_search_time_ns});
}

/// Run test suite
fn runTests(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running WDBX test suite...\n", .{});
    
    // In a real implementation, this would run all tests
    // For now, we'll just indicate that tests should be run with `zig test`
    std.debug.print("\nTo run tests, use: zig test src/main_refactored.zig\n", .{});
}

// Tests
test "Command parsing" {
    const testing = std.testing;
    
    try testing.expectEqual(Command.help, Command.fromString("help").?);
    try testing.expectEqual(Command.serve, Command.fromString("serve").?);
    try testing.expectEqual(null, Command.fromString("invalid"));
}

test "Basic database operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/test.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    // Create and initialize database
    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(.{
        .dimensions = 4,
        .index_type = .flat,
        .distance_metric = .euclidean,
    });

    // Add vectors
    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    
    _ = try db.addVector(&v1, null);
    _ = try db.addVector(&v2, null);

    // Search
    const query = [_]f32{ 0.9, 0.1, 0.0, 0.0 };
    const results = try db.search(&query, 1, allocator);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
}