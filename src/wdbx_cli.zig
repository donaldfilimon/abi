//! WDBX Vector Database - Command Line Interface
//!
//! This module provides a comprehensive CLI for the WDBX vector database,
//! including vector operations, server management, and database administration.
//
//! ---
//! @Web @Definitions
//!
//! In Zig, **structs** and **enums** are fundamental constructs for defining complex data types and managing related data efficiently.
//!
//! - **Structs** group related data fields together, similar to C structures. They can contain fields of various types and include methods (functions) that operate on the struct's data. For example, a `Person` struct might include fields for `name` and `age`, along with a method to display this information. Structs promote modularity and clarity in code by encapsulating data and behavior together. [programming.muthu.co](https://programming.muthu.co/posts/beginners-guide-to-zig/?utm_source=openai)
//!
//! - **Enums** (enumerations) define a type with a fixed set of named values, representing discrete options. In Zig, enums can also store associated data, making them more flexible than in some other languages. For instance, a `Direction` enum could have values like `North`, `East`, `South`, and `West`. Enums can also have methods, allowing for encapsulation of behavior alongside data. [piembsystech.com](https://piembsystech.com/using-structs-unions-and-enums-in-zig-programming-language/?utm_source=openai)
//!
//! For more detailed information on Zig's structs and enums, including advanced usage and examples, refer to the official Zig documentation: [ziglang.org](https://ziglang.org/documentation/0.1.1/?utm_source=openai)
//! ---

const std = @import("std");
const database = @import("database.zig");

/// The current version string for the WDBX CLI.
pub const version_string = "WDBX Vector Database v1.0.0";

/// CLI commands for WDBX, representing all supported user actions.
///
/// In Zig, enums are used to define a type with a fixed set of named values.
/// Each value represents a discrete command the CLI can execute.
/// Enums can also have methods, such as parsing from a string or providing descriptions.
pub const Command = enum {
    /// Show help information for the CLI.
    help,
    /// Show version information.
    version,
    /// Query k-nearest neighbors for a given vector.
    knn,
    /// Query the single nearest neighbor for a given vector.
    query,
    /// Add a new vector to the database.
    add,
    /// Show database statistics.
    stats,
    /// Show performance metrics (not yet implemented).
    monitor,
    /// Run machine learning optimization (not yet implemented).
    optimize,
    /// Save the database to a file.
    save,
    /// Load the database from a file.
    load,
    /// Start the HTTP REST API server.
    http,
    /// Start the TCP binary protocol server.
    tcp,
    /// Start the WebSocket server.
    ws,
    /// Generate a JWT authentication token.
    gen_token,

    /// Parse a string into a Command, or return null if invalid.
    ///
    /// This method demonstrates how enums in Zig can have associated methods,
    /// allowing for convenient parsing and conversion operations.
    pub fn fromString(str: []const u8) ?Command {
        inline for (std.meta.fields(Command)) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }

    /// Get a human-readable description of the command.
    ///
    /// This method provides a string description for each command,
    /// showcasing how enums can encapsulate both data and behavior.
    pub fn getDescription(self: Command) []const u8 {
        return switch (self) {
            .help => "Show help information",
            .version => "Show version information",
            .knn => "Query k-nearest neighbors",
            .query => "Query nearest neighbor",
            .add => "Add vector to database",
            .stats => "Show database statistics",
            .monitor => "Show performance metrics",
            .optimize => "Run ML optimization",
            .save => "Save database to file",
            .load => "Load database from file",
            .http => "Start HTTP REST API server",
            .tcp => "Start TCP binary protocol server",
            .ws => "Start WebSocket server",
            .gen_token => "Generate JWT authentication token",
        };
    }
};

/// Output format for CLI and API responses.
///
/// This enum demonstrates how Zig enums can represent discrete options for output formatting.
/// Each variant can be used to control how results are displayed or serialized.
pub const OutputFormat = enum {
    /// Human-readable text output.
    text,
    /// JSON output.
    json,
    /// CSV output.
    csv,

    /// Get the string representation of the output format.
    pub fn toString(self: OutputFormat) []const u8 {
        return @tagName(self);
    }
};

/// CLI options, representing all user-configurable parameters.
///
/// In Zig, structs are used to group related data fields together.
/// This struct encapsulates all the options that can be set by the user via the CLI.
pub const Options = struct {
    /// The command to execute.
    command: Command = .help,
    /// Enable verbose output.
    verbose: bool = false,
    /// Suppress output.
    quiet: bool = false,
    /// Path to the database file.
    db_path: ?[]const u8 = null,
    /// Port for server commands.
    port: u16 = 8080,
    /// Host for server commands.
    host: []const u8 = "127.0.0.1",
    /// Number of neighbors for knn queries.
    k: usize = 5,
    /// Vector string for add/query/knn commands.
    vector: ?[]const u8 = null,
    /// User role for token generation.
    role: []const u8 = "admin",
    /// Output format for results.
    output_format: OutputFormat = .text,

    /// Free any heap-allocated option fields.
    ///
    /// This method demonstrates how structs in Zig can have methods for resource management,
    /// such as freeing heap-allocated memory associated with struct fields.
    pub fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        if (self.db_path) |path| allocator.free(path);
        if (self.vector) |vec| allocator.free(vec);
        if (!std.mem.eql(u8, self.host, "127.0.0.1")) allocator.free(self.host);
        if (!std.mem.eql(u8, self.role, "admin")) allocator.free(self.role);
    }
};

/// Application context, holding allocator, options, and database handle.
///
/// This struct demonstrates how Zig structs can encapsulate both data and methods,
/// providing a convenient way to manage application state and resources.
pub const AppContext = struct {
    /// Allocator for all heap allocations.
    allocator: std.mem.Allocator,
    /// CLI options for this run.
    options: Options,
    /// Database handle, if opened.
    db: ?*database.Db,

    /// Initialize a new application context.
    pub fn init(allocator: std.mem.Allocator, options: Options) !AppContext {
        return AppContext{
            .allocator = allocator,
            .options = options,
            .db = null,
        };
    }

    /// Clean up resources held by the context.
    pub fn deinit(self: *AppContext) void {
        if (self.db) |db| {
            db.close();
        }
        self.options.deinit(self.allocator);
    }

    /// Open the database at the given path, creating if necessary.
    pub fn openDatabase(self: *AppContext, path: []const u8) !void {
        self.db = try database.Db.open(path, true);
        if (self.db.?.getDimension() == 0) {
            try self.db.?.init(8); // Default to 8 dimensions
        }
    }

    /// Run the command specified in options.
    pub fn run(self: *AppContext) !void {
        switch (self.options.command) {
            .help => try self.showHelp(),
            .version => try self.showVersion(),
            .knn => try self.runKnn(),
            .query => try self.runQuery(),
            .add => try self.runAdd(),
            .stats => try self.runStats(),
            .monitor => try self.runMonitor(),
            .optimize => try self.runOptimize(),
            .save => try self.runSave(),
            .load => try self.runLoad(),
            .http => try self.runHttpServer(),
            .tcp => try self.runTcpServer(),
            .ws => try self.runWebSocketServer(),
            .gen_token => try self.runGenToken(),
        }
    }

    /// Print help information for all commands and options.
    ///
    /// This function demonstrates how documentation and usage information can be
    /// provided to users, leveraging the enum and struct definitions above.
    fn showHelp(_: *AppContext) !void {
        std.debug.print(
            \\WDBX Vector Database - Command Line Interface
            \\
            \\Usage: wdbx <command> [options]
            \\
            \\Commands:
            \\  knn <vector> [k]     Query k-nearest neighbors (default k=5)
            \\  query <vector>       Query nearest neighbor
            \\  add <vector>         Add vector to database
            \\  stats                Show database statistics
            \\  monitor              Show performance metrics
            \\  optimize             Run ML optimization
            \\  save <file>          Save database to file
            \\  load <file>          Load database from file
            \\  http [port]          Start HTTP REST API server
            \\  tcp [port]           Start TCP binary protocol server
            \\  ws [port]            Start WebSocket server
            \\  gen_token [role]     Generate JWT authentication token
            \\
            \\Options:
            \\  --db <path>          Database file path
            \\  --host <host>        Server host (default: 127.0.0.1)
            \\  --port <port>        Server port (default: 8080)
            \\  --role <role>        User role for token generation
            \\  --format <format>    Output format: text, json, csv
            \\  --verbose            Enable verbose output
            \\  --quiet              Suppress output
            \\
            \\Examples:
            \\  wdbx knn "1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1" 5
            \\  wdbx http 8080
            \\  wdbx gen_token admin
            \\
            \\
            \\--- Zig Structs and Enums ---
            \\Structs group related data fields and can include methods.
            \\Enums define a type with a fixed set of named values and can also have methods.
            \\See: https://ziglang.org/documentation/0.1.1/?utm_source=openai
            \\
        , .{});
    }

    /// Print the current version string.
    fn showVersion(_: *AppContext) !void {
        std.debug.print("{s}\n", .{version_string});
    }

    /// Run the k-nearest neighbors query.
    fn runKnn(self: *AppContext) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for knn command\n", .{});
            return;
        }

        const vector_str = self.options.vector.?;
        const k = self.options.k;

        // Parse vector string
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);

        // Open database
        if (self.options.db_path) |path| {
            try self.openDatabase(path);
        } else {
            try self.openDatabase("vectors.wdbx");
        }

        const db = self.db.?;

        // Perform k-nearest neighbor search
        const results = try db.search(vector, k, self.allocator);
        defer self.allocator.free(results);

        // Display results
        std.debug.print("=== K-Nearest Neighbors (k={d}) ===\n", .{k});
        for (results, 0..) |result, i| {
            std.debug.print("{d:2}. Index: {d:6}, Distance: {d:.6f}\n", .{ i + 1, result.index, result.score });
        }
    }

    /// Run the nearest neighbor query.
    fn runQuery(self: *AppContext) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for query command\n", .{});
            return;
        }

        const vector_str = self.options.vector.?;

        // Parse vector string
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);

        // Open database
        if (self.options.db_path) |path| {
            try self.openDatabase(path);
        } else {
            try self.openDatabase("vectors.wdbx");
        }

        const db = self.db.?;

        // Perform nearest neighbor search
        const results = try db.search(vector, 1, self.allocator);
        defer self.allocator.free(results);

        if (results.len > 0) {
            std.debug.print("=== Nearest Neighbor ===\n", .{});
            std.debug.print("Index: {d}, Distance: {d:.6f}\n", .{ results[0].index, results[0].score });
        } else {
            std.debug.print("No vectors found in database\n", .{});
        }
    }

    /// Add a new vector to the database.
    fn runAdd(self: *AppContext) !void {
        if (self.options.vector == null) {
            std.debug.print("Error: Vector required for add command\n", .{});
            return;
        }

        const vector_str = self.options.vector.?;

        // Parse vector string
        const vector = try self.parseVector(vector_str);
        defer self.allocator.free(vector);

        // Open database
        if (self.options.db_path) |path| {
            try self.openDatabase(path);
        } else {
            try self.openDatabase("vectors.wdbx");
        }

        const db = self.db.?;

        // Add vector to database
        const row_id = try db.addEmbedding(vector);
        std.debug.print("Vector added successfully at row {d}\n", .{row_id});
    }

    /// Show statistics about the database.
    fn runStats(self: *AppContext) !void {
        // Open database
        if (self.options.db_path) |path| {
            try self.openDatabase(path);
        } else {
            try self.openDatabase("vectors.wdbx");
        }

        const db = self.db.?;
        const stats = db.getStats();

        std.debug.print(
            \\=== WDBX Database Statistics ===
            \\Vectors stored: {d}
            \\Vector dimension: {d}
            \\Searches performed: {d}
            \\Average search time: {d}μs
            \\Writes performed: {d}
            \\Initializations: {d}
            \\
        , .{
            db.getRowCount(),
            db.getDimension(),
            stats.search_count,
            stats.getAverageSearchTime(),
            stats.write_count,
            stats.initialization_count,
        });
    }

    /// Show performance metrics (not yet implemented).
    fn runMonitor(_: *AppContext) !void {
        std.debug.print("Performance monitoring not yet implemented\n", .{});
    }

    /// Run ML optimization (not yet implemented).
    fn runOptimize(_: *AppContext) !void {
        std.debug.print("ML optimization not yet implemented\n", .{});
    }

    /// Save the database to a file (not yet implemented).
    fn runSave(_: *AppContext) !void {
        std.debug.print("Database save not yet implemented\n", .{});
    }

    /// Load the database from a file (not yet implemented).
    fn runLoad(_: *AppContext) !void {
        std.debug.print("Database load not yet implemented\n", .{});
    }

    /// Start the HTTP REST API server.
    fn runHttpServer(self: *AppContext) !void {
        std.debug.print("Starting HTTP server on {}:{}\n", .{ self.options.host, self.options.port });

        const http_server = @import("wdbx_http_server.zig");
        const config = http_server.ServerConfig{
            .host = self.options.host,
            .port = self.options.port,
        };

        var server = try http_server.WdbxHttpServer.init(self.allocator, config);
        defer server.deinit();

        // Open database if specified
        if (self.options.db_path) |path| {
            try server.openDatabase(path);
        } else {
            try server.openDatabase("vectors.wdbx");
        }

        try server.start();
    }

    /// Start the TCP binary protocol server (not yet implemented).
    fn runTcpServer(self: *AppContext) !void {
        std.debug.print("Starting TCP server on {}:{}\n", .{ self.options.host, self.options.port });
        std.debug.print("TCP server not yet implemented\n", .{});
    }

    /// Start the WebSocket server (not yet implemented).
    fn runWebSocketServer(self: *AppContext) !void {
        std.debug.print("Starting WebSocket server on {}:{}\n", .{ self.options.host, self.options.port });
        std.debug.print("WebSocket server not yet implemented\n", .{});
    }

    /// Generate a JWT authentication token (stub).
    fn runGenToken(self: *AppContext) !void {
        const role = self.options.role;
        std.debug.print("Generating JWT token for role: {s}\n", .{role});
        std.debug.print("JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example.token\n", .{});
    }

    /// Parse a comma-separated string into a vector of f32.
    ///
    /// This function demonstrates how to use Zig's standard library to split and parse strings,
    /// and how to use ArrayList (a struct) for dynamic arrays.
    fn parseVector(self: *AppContext, vector_str: []const u8) ![]f32 {
        var list = try std.ArrayList(f32).initCapacity(self.allocator, 8);
        defer list.deinit(self.allocator);

        var iter = std.mem.splitSequence(u8, vector_str, ",");
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\n\r");
            if (trimmed.len > 0) {
                const value = try std.fmt.parseFloat(f32, trimmed);
                try list.append(self.allocator, value);
            }
        }

        return try list.toOwnedSlice(self.allocator);
    }
};

/// Parse command-line arguments into Options.
/// Returns an Options struct, or an error if parsing fails.
///
/// This function demonstrates how to use structs and enums to represent and parse CLI arguments,
/// leveraging Zig's type system for safety and clarity.
pub fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var options = Options{};

    if (args.len < 2) {
        return options;
    }

    // Parse command
    if (Command.fromString(args[1])) |command| {
        options.command = command;
    } else {
        return error.InvalidCommand;
    }

    // Parse arguments based on command
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        switch (options.command) {
            .knn => {
                if (options.vector == null) {
                    options.vector = try allocator.dupe(u8, arg);
                } else if (options.k == 5) {
                    options.k = try std.fmt.parseInt(usize, arg, 10);
                }
            },
            .query, .add => {
                if (options.vector == null) {
                    options.vector = try allocator.dupe(u8, arg);
                }
            },
            .http, .tcp, .ws => {
                if (i == 2) {
                    options.port = try std.fmt.parseInt(u16, arg, 10);
                }
            },
            .gen_token => {
                if (i == 2) {
                    options.role = try allocator.dupe(u8, arg);
                }
            },
            .save, .load => {
                if (options.db_path == null) {
                    options.db_path = try allocator.dupe(u8, arg);
                }
            },
            else => {},
        }
    }

    // Parse global options
    i = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--db")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.db_path = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--host")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.host = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--role")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.role = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            if (std.mem.eql(u8, args[i], "json")) {
                options.output_format = .json;
            } else if (std.mem.eql(u8, args[i], "csv")) {
                options.output_format = .csv;
            }
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            options.quiet = true;
        }
    }

    return options;
}

/// Entry point for the WDBX CLI application.
///
/// This function demonstrates how to use Zig's structs and enums to manage application state,
/// parse command-line arguments, and execute the appropriate command.
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const options = parseArgs(allocator) catch |err| {
        std.debug.print("Error parsing arguments: {}\n", .{err});
        std.process.exit(1);
    };

    var context = try AppContext.init(allocator, options);
    defer context.deinit();

    context.run() catch |err| {
        std.debug.print("Error: {}\n", .{err});
        std.process.exit(1);
    };
}
