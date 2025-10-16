//! ABI Framework CLI - Zig 0.16 Modernized Version
//!
//! This is the main CLI interface for the ABI framework, fully compliant with
//! Zig 0.16 formatting standards and best practices.

const std = @import("std");
const builtin = @import("builtin");

// Use centralized imports for consistency
const imports = @import("shared/imports.zig");
const patterns = @import("shared/patterns/common.zig");
const errors = @import("shared/errors/framework_errors.zig");

// Framework imports
const abi = @import("abi");
const Framework = abi.framework.runtime.Framework;
const FrameworkOptions = abi.framework.config.FrameworkOptions;
const Feature = abi.framework.config.Feature;
const Agent = abi.ai.agent.Agent;
const AgentConfig = abi.ai.agent.AgentConfig;

// Type aliases for better readability
const Allocator = imports.Allocator;
const ArrayList = imports.ArrayList;
const Writer = imports.Writer;
const Logger = patterns.Logger;
const ErrorContext = patterns.ErrorContext;

// Build manifest handling
const manifest = @import("../build.zig.zon");
const ManifestDependencies = @TypeOf(manifest.dependencies);
const manifest_dependency_fields = std.meta.fields(ManifestDependencies);

/// Dependency information structure
const DependencyInfo = struct {
    name: []const u8,
    url: ?[]const u8 = null,
    hash: ?[]const u8 = null,
    
    pub fn deinit(self: DependencyInfo, allocator: Allocator) void {
        allocator.free(self.name);
        if (self.url) |url| allocator.free(url);
        if (self.hash) |hash| allocator.free(hash);
    }
};

/// CLI exit codes following POSIX conventions
pub const ExitCode = enum(u8) {
    success = 0,
    usage = 1,
    config = 2,
    runtime = 3,
    io = 4,
    backend_missing = 5,
    
    pub fn toInt(self: ExitCode) u8 {
        return @intFromEnum(self);
    }
};

/// I/O channels for testable CLI operations
pub const Channels = struct {
    out: Writer,
    err: Writer,
    
    pub fn init(out: Writer, err: Writer) Channels {
        return .{ .out = out, .err = err };
    }
    
    pub fn printJson(self: Channels, comptime fmt: []const u8, args: anytype) !void {
        try self.out.print(fmt, args);
        try self.out.print("\n", .{});
    }
};

/// Session-based vector database for CLI operations
const SessionDatabase = struct {
    allocator: Allocator,
    dim: ?usize = null,
    next_id: u64 = 1,
    entries: ArrayList(VectorEntry),
    
    const VectorEntry = struct {
        id: u64,
        values: []f32,
        metadata: ?[]u8,
        
        pub fn deinit(self: VectorEntry, allocator: Allocator) void {
            allocator.free(self.values);
            if (self.metadata) |meta| {
                allocator.free(meta);
            }
        }
    };
    
    pub const SearchResult = struct {
        id: u64,
        distance: f32,
    };
    
    pub const Error = error{
        Empty,
        InvalidVector,
        DimensionMismatch,
        InvalidK,
        OutOfMemory,
    };
    
    pub fn init(allocator: Allocator) SessionDatabase {
        return .{
            .allocator = allocator,
            .entries = ArrayList(VectorEntry).init(allocator),
        };
    }
    
    pub fn deinit(self: *SessionDatabase) void {
        for (self.entries.items) |entry| {
            entry.deinit(self.allocator);
        }
        self.entries.deinit();
    }
    
    pub fn insert(self: *SessionDatabase, vector: []const f32, metadata: ?[]const u8) Error!u64 {
        if (vector.len == 0) return Error.InvalidVector;
        
        // Check dimension consistency
        if (self.dim) |dim| {
            if (vector.len != dim) return Error.DimensionMismatch;
        } else {
            self.dim = vector.len;
        }
        
        // Allocate and copy vector data
        const stored_vector = self.allocator.dupe(f32, vector) catch return Error.OutOfMemory;
        errdefer self.allocator.free(stored_vector);
        
        // Handle metadata if provided
        var stored_metadata: ?[]u8 = null;
        if (metadata) |meta| {
            stored_metadata = self.allocator.dupe(u8, meta) catch return Error.OutOfMemory;
        }
        errdefer if (stored_metadata) |meta| self.allocator.free(meta);
        
        // Create entry
        const id = self.next_id;
        self.next_id += 1;
        
        const entry = VectorEntry{
            .id = id,
            .values = stored_vector,
            .metadata = stored_metadata,
        };
        
        self.entries.append(entry) catch return Error.OutOfMemory;
        return id;
    }
    
    pub fn count(self: *const SessionDatabase) usize {
        return self.entries.items.len;
    }
    
    pub fn search(self: *SessionDatabase, query: []const f32, k: usize) Error![]SearchResult {
        if (self.dim == null or self.entries.items.len == 0) return Error.Empty;
        if (query.len == 0) return Error.InvalidVector;
        if (query.len != self.dim.?) return Error.DimensionMismatch;
        if (k == 0) return Error.InvalidK;
        
        var results = ArrayList(SearchResult).init(self.allocator);
        defer results.deinit();
        
        // Calculate distances
        for (self.entries.items) |entry| {
            var sum: f32 = 0.0;
            for (query, entry.values) |q, v| {
                const diff = q - v;
                sum += diff * diff;
            }
            const distance = @sqrt(sum);
            
            try results.append(.{ .id = entry.id, .distance = distance });
        }
        
        // Sort by distance (ascending)
        const SortContext = struct {
            pub fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                if (a.distance == b.distance) return a.id < b.id;
                return a.distance < b.distance;
            }
        };
        std.mem.sort(SearchResult, results.items, {}, SortContext.lessThan);
        
        // Return top k results
        const result_count = @min(results.items.len, k);
        const owned_results = try self.allocator.alloc(SearchResult, result_count);
        @memcpy(owned_results, results.items[0..result_count]);
        
        return owned_results;
    }
};

/// Main CLI context with dependency injection
pub const Cli = struct {
    allocator: Allocator,
    channels: Channels,
    logger: Logger,
    framework: Framework,
    database: SessionDatabase,
    json_mode: bool,
    
    pub fn init(
        allocator: Allocator,
        channels: Channels,
        json_mode: bool,
        log_level: Logger.Level,
    ) !Cli {
        const framework = Framework.init(allocator, FrameworkOptions{}) catch |err| {
            const error_ctx = ErrorContext.init("Failed to initialize framework")
                .withLocation(@src())
                .withCause(err);
            std.log.err("{}", .{error_ctx});
            return err;
        };
        
        return .{
            .allocator = allocator,
            .channels = channels,
            .logger = Logger.init(channels.err, log_level),
            .framework = framework,
            .database = SessionDatabase.init(allocator),
            .json_mode = json_mode,
        };
    }
    
    pub fn deinit(self: *Cli) void {
        self.database.deinit();
        self.framework.deinit();
    }
    
    pub fn dispatch(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printHelp();
            return .usage;
        }
        
        const command = args[0];
        const tail = args[1..];
        
        // Command routing with error handling
        if (imports.string.eql(command, "help")) {
            try self.printHelp();
            return .success;
        } else if (imports.string.eql(command, "features")) {
            return self.handleFeatures(tail) catch |err| {
                const error_ctx = ErrorContext.init("Features command failed")
                    .withLocation(@src())
                    .withCause(err);
                try self.logger.err("{}", .{error_ctx});
                return .runtime;
            };
        } else if (imports.string.eql(command, "agent")) {
            return self.handleAgent(tail) catch |err| {
                const error_ctx = ErrorContext.init("Agent command failed")
                    .withLocation(@src())
                    .withCause(err);
                try self.logger.err("{}", .{error_ctx});
                return .runtime;
            };
        } else if (imports.string.eql(command, "db")) {
            return self.handleDatabase(tail) catch |err| {
                const error_ctx = ErrorContext.init("Database command failed")
                    .withLocation(@src())
                    .withCause(err);
                try self.logger.err("{}", .{error_ctx});
                return .runtime;
            };
        } else if (imports.string.eql(command, "gpu")) {
            return self.handleGpu(tail) catch |err| {
                const error_ctx = ErrorContext.init("GPU command failed")
                    .withLocation(@src())
                    .withCause(err);
                try self.logger.err("{}", .{error_ctx});
                return .runtime;
            };
        } else if (imports.string.eql(command, "deps")) {
            return self.handleDeps(tail) catch |err| {
                const error_ctx = ErrorContext.init("Dependencies command failed")
                    .withLocation(@src())
                    .withCause(err);
                try self.logger.err("{}", .{error_ctx});
                return .runtime;
            };
        }
        
        try self.logger.err("Unknown command: {s}", .{command});
        return .usage;
    }
    
    // Command handlers with modern error handling
    fn handleFeatures(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0 or isHelp(args)) {
            try self.printFeaturesHelp();
            return .success;
        }
        
        const subcommand = args[0];
        const tail = args[1..];
        
        if (imports.string.eql(subcommand, "list")) {
            if (tail.len != 0) {
                try self.logger.err("features list does not accept extra arguments", .{});
                return .usage;
            }
            try self.emitFeatureList();
            return .success;
        }
        
        if (imports.string.eql(subcommand, "enable")) {
            return try self.toggleFeatures(tail, true);
        }
        
        if (imports.string.eql(subcommand, "disable")) {
            return try self.toggleFeatures(tail, false);
        }
        
        try self.logger.err("Unknown features subcommand: {s}", .{subcommand});
        return .usage;
    }
    
    fn emitFeatureList(self: *Cli) !void {
        if (self.json_mode) {
            var buffer = ArrayList(u8).init(self.allocator);
            defer buffer.deinit();
            
            try buffer.appendSlice("{\"features\":{");
            var first = true;
            
            inline for (std.meta.fields(Feature)) |field| {
                const feature = @as(Feature, @enumFromInt(field.value));
                const enabled = self.framework.isFeatureEnabled(feature);
                
                if (!first) try buffer.appendSlice(",");
                first = false;
                
                try buffer.writer().print("\"{s}\":{s}", .{ 
                    field.name, 
                    if (enabled) "true" else "false" 
                });
            }
            
            try buffer.appendSlice("}}");
            try self.channels.printJson("{s}", .{buffer.items});
        } else {
            try self.logger.info("Enabled features ({d}):", .{self.framework.featureCount()});
            
            inline for (std.meta.fields(Feature)) |field| {
                const feature = @as(Feature, @enumFromInt(field.value));
                const enabled = self.framework.isFeatureEnabled(feature);
                const status = if (enabled) "enabled" else "disabled";
                
                try self.logger.info("  - {s}: {s}", .{ field.name, status });
            }
        }
    }
    
    fn toggleFeatures(self: *Cli, args: [][]const u8, enabled: bool) !ExitCode {
        if (args.len == 0) {
            try self.logger.err("Specify at least one feature to toggle", .{});
            return .usage;
        }
        
        var changed = ArrayList([]const u8).init(self.allocator);
        defer changed.deinit();
        
        for (args) |token| {
            const feature = parseFeature(token) orelse {
                try self.logger.err("Unknown feature: {s}", .{token});
                return .usage;
            };
            
            const modified = if (enabled)
                self.framework.enableFeature(feature)
            else
                self.framework.disableFeature(feature);
                
            if (modified) try changed.append(token);
        }
        
        if (self.json_mode) {
            var list_buffer = ArrayList(u8).init(self.allocator);
            defer list_buffer.deinit();
            
            try list_buffer.appendSlice("[");
            for (changed.items, 0..) |name, idx| {
                if (idx != 0) try list_buffer.appendSlice(",");
                try list_buffer.writer().print("\"{s}\"", .{name});
            }
            try list_buffer.appendSlice("]");
            
            const action = if (enabled) "enable" else "disable";
            try self.channels.printJson(
                "{{\"status\":\"ok\",\"action\":\"{s}\",\"updated\":{s}}}",
                .{ action, list_buffer.items }
            );
        } else {
            const label = if (enabled) "enabled" else "disabled";
            if (changed.items.len == 0) {
                try self.logger.warn("No features were {s}", .{label});
            } else {
                const action_word = if (enabled) "Enabled" else "Disabled";
                try self.logger.info("{s} {d} feature(s)", .{ action_word, changed.items.len });
            }
        }
        
        return .success;
    }
    
    // Additional command handlers would follow the same pattern...
    // (Truncated for brevity, but would include handleAgent, handleDatabase, etc.)
    
    fn printHelp(self: *Cli) !void {
        const message =
            \\Usage: abi <command> [options]
            \\
            \\Commands:
            \\  features   list|enable|disable
            \\  agent      run
            \\  db         insert|search
            \\  gpu        bench
            \\  deps       list|update
            \\
        ;
        try self.logger.info("{s}", .{message});
    }
    
    fn printFeaturesHelp(self: *Cli) !void {
        const text = 
            \\features <list|enable|disable> [feature...]
            \\Features: ai, database, gpu, web, monitoring, connectors, simd
            \\
        ;
        try self.logger.info("{s}", .{text});
    }
    
    // Remaining methods would be implemented following the same patterns...
};

// Utility functions
fn parseFeature(name: []const u8) ?Feature {
    inline for (std.meta.fields(Feature)) |field| {
        if (std.ascii.eqlIgnoreCase(name, field.name)) {
            return @as(Feature, @enumFromInt(field.value));
        }
    }
    return null;
}

fn isHelp(args: [][]const u8) bool {
    if (args.len == 0) return false;
    return imports.string.eql(args[0], "--help") or imports.string.eql(args[0], "-h");
}

/// Main entry point with proper error handling
pub fn main() !void {
    var gpa = imports.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse command line arguments
    const raw_args = std.process.argsAlloc(allocator) catch |err| {
        std.log.err("Failed to parse arguments: {s}", .{@errorName(err)});
        imports.process.exit(ExitCode.runtime.toInt());
    };
    defer std.process.argsFree(allocator, raw_args);
    
    if (raw_args.len == 0) return;
    
    // Parse global flags
    var idx: usize = 1;
    var json_mode = false;
    var help_flag = false;
    var log_level: Logger.Level = .info;
    
    while (idx < raw_args.len) {
        const arg = raw_args[idx];
        
        if (imports.string.eql(arg, "--json")) {
            json_mode = true;
            idx += 1;
            continue;
        }
        
        if (imports.string.eql(arg, "--help") or imports.string.eql(arg, "-h")) {
            help_flag = true;
            idx += 1;
            continue;
        }
        
        if (imports.string.startsWith(arg, "--log-level=")) {
            const value = arg["--log-level=".len..];
            log_level = std.meta.stringToEnum(Logger.Level, value) orelse {
                std.log.err("Invalid log level: {s}", .{value});
                imports.process.exit(ExitCode.usage.toInt());
            };
            idx += 1;
            continue;
        }
        
        break;
    }
    
    // Adjust log level for JSON mode
    if (json_mode and log_level == .info) {
        log_level = .@"error";
    }
    
    // Initialize CLI with dependency injection
    const channels = Channels.init(
        std.io.getStdOut().writer().any(),
        std.io.getStdErr().writer().any(),
    );
    
    var cli = Cli.init(allocator, channels, json_mode, log_level) catch |err| {
        std.log.err("Failed to initialize CLI: {s}", .{@errorName(err)});
        imports.process.exit(ExitCode.runtime.toInt());
    };
    defer cli.deinit();
    
    // Handle help or execute command
    if (help_flag or idx >= raw_args.len) {
        try cli.printHelp();
        return;
    }
    
    // Dispatch command
    const exit_code = cli.dispatch(raw_args[idx..]) catch |err| {
        const error_ctx = ErrorContext.init("Command execution failed")
            .withLocation(@src())
            .withCause(err);
        try cli.logger.err("{}", .{error_ctx});
        imports.process.exit(ExitCode.runtime.toInt());
    };
    
    if (exit_code != .success) {
        imports.process.exit(exit_code.toInt());
    }
}

// Tests following Zig 0.16 patterns
test "SessionDatabase handles vector operations correctly" {
    const testing = imports.testing;
    
    var db = SessionDatabase.init(testing.allocator);
    defer db.deinit();
    
    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const id = try db.insert(&vector, null);
    
    try testing.expectEqual(@as(u64, 1), id);
    try testing.expectEqual(@as(usize, 1), db.count());
    
    const results = try db.search(&vector, 1);
    defer testing.allocator.free(results);
    
    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(u64, 1), results[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].distance, 0.001);
}

test "CLI initialization with dependency injection" {
    const testing = imports.testing;
    
    var buffer_out = ArrayList(u8).init(testing.allocator);
    defer buffer_out.deinit();
    var buffer_err = ArrayList(u8).init(testing.allocator);
    defer buffer_err.deinit();
    
    const channels = Channels.init(
        buffer_out.writer().any(),
        buffer_err.writer().any(),
    );
    
    var cli = try Cli.init(testing.allocator, channels, false, .info);
    defer cli.deinit();
    
    try testing.expect(!cli.json_mode);
    try testing.expectEqual(@as(usize, 0), cli.database.count());
}

test "Feature parsing works correctly" {
    const testing = imports.testing;
    
    const ai_feature = parseFeature("ai");
    try testing.expect(ai_feature != null);
    try testing.expectEqual(Feature.ai, ai_feature.?);
    
    const invalid_feature = parseFeature("invalid");
    try testing.expect(invalid_feature == null);
}