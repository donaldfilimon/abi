//! Call graph analysis for codebases
//!
//! Provides functionality to build and analyze function call graphs across
//! Zig, Rust, and TypeScript/JavaScript codebases.

const std = @import("std");
const AstParser = @import("ast.zig").AstParser;
const AstNode = @import("ast.zig").AstNode;
const ParsedFile = @import("ast.zig").ParsedFile;

/// Represents a function in the call graph
pub const Function = struct {
    name: []const u8,
    file_path: []const u8,
    line: u32,
};

/// Represents an edge in the call graph (caller -> callee)
pub const CallEdge = struct {
    caller: Function,
    callee: Function,
};

/// Call graph data structure
pub const CallGraph = struct {
    allocator: std.mem.Allocator,
    /// Map from function name to list of functions it calls
    calls: std.StringHashMap(std.ArrayList(Function)),
    /// Map from function name to list of functions that call it
    called_by: std.StringHashMap(std.ArrayList(Function)),
    /// All functions in the graph
    all_functions: std.ArrayList(Function),
    /// All edges in the graph
    edges: std.ArrayList(CallEdge),

    pub fn init(allocator: std.mem.Allocator) CallGraph {
        return .{
            .allocator = allocator,
            .calls = std.StringHashMap(std.ArrayList(Function)).init(allocator),
            .called_by = std.StringHashMap(std.ArrayList(Function)).init(allocator),
            .all_functions = std.ArrayList(Function){},
            .edges = std.ArrayList(CallEdge){},
        };
    }

    pub fn deinit(self: *CallGraph) void {
        var calls_iter = self.calls.valueIterator();
        while (calls_iter.next()) |list| {
            list.deinit();
        }
        self.calls.deinit();

        var called_by_iter = self.called_by.valueIterator();
        while (called_by_iter.next()) |list| {
            list.deinit();
        }
        self.called_by.deinit();

        self.all_functions.deinit();
        self.edges.deinit();
    }

    /// Add a function to the graph
    pub fn addFunction(self: *CallGraph, func: Function) !void {
        try self.all_functions.append(self.allocator, func);

        const key = try self.allocator.dupe(u8, func.name);
        errdefer self.allocator.free(key);

        try self.calls.put(key, std.ArrayList(Function){});
        try self.called_by.put(key, std.ArrayList(Function){});
    }

    /// Add a call relationship (caller -> callee)
    pub fn addCall(self: *CallGraph, caller: Function, callee: Function) !void {
        try self.edges.append(self.allocator, .{ .caller = caller, .callee = callee });

        if (self.calls.get(caller.name)) |callers_list| {
            try callers_list.append(self.allocator, callee);
        }

        if (self.called_by.get(callee.name)) |called_by_list| {
            try called_by_list.append(self.allocator, caller);
        }
    }

    /// Get all functions called by a given function
    pub fn getCallees(self: *const CallGraph, function_name: []const u8) ?[]const Function {
        if (self.calls.get(function_name)) |list| {
            return list.items;
        }
        return null;
    }

    /// Get all functions that call a given function
    pub fn getCallers(self: *const CallGraph, function_name: []const u8) ?[]const Function {
        if (self.called_by.get(function_name)) |list| {
            return list.items;
        }
        return null;
    }

    /// Check if there's a call path from function A to function B
    pub fn hasPathTo(self: *const CallGraph, from: []const u8, to: []const u8) bool {
        var visited = std.StringHashMap(void).init(self.allocator);
        defer visited.deinit();

        return self.dfsPath(from, to, &visited);
    }

    fn dfsPath(self: *const CallGraph, current: []const u8, target: []const u8, visited: *std.StringHashMap(void)) bool {
        if (std.mem.eql(u8, current, target)) return true;
        if (visited.get(current) != null) return false;

        try visited.put(current, {});

        if (self.calls.get(current)) |callees| {
            for (callees.items) |callee| {
                if (self.dfsPath(callee.name, target, visited)) return true;
            }
        }

        return false;
    }

    /// Export graph to DOT format for visualization
    pub fn toDot(self: *const CallGraph, writer: anytype) !void {
        try writer.writeAll("digraph CallGraph {\n");
        try writer.writeAll("  rankdir=LR;\n");
        try writer.writeAll("  node [shape=box];\n\n");

        for (self.edges.items) |edge| {
            try writer.print("  \"{s}\" -> \"{s}\";\n", .{ edge.caller.name, edge.callee.name });
        }

        try writer.writeAll("}\n");
    }
};

/// Builder for creating call graphs from parsed AST
pub const CallGraphBuilder = struct {
    allocator: std.mem.Allocator,
    parser: AstParser,
    graph: CallGraph,

    pub fn init(allocator: std.mem.Allocator) CallGraphBuilder {
        return .{
            .allocator = allocator,
            .parser = AstParser.init(allocator),
            .graph = CallGraph.init(allocator),
        };
    }

    pub fn deinit(self: *CallGraphBuilder) void {
        self.parser.deinit();
        self.graph.deinit();
    }

    /// Build call graph from a single parsed file
    pub fn buildFromFile(self: *CallGraphBuilder, parsed_file: *const ParsedFile) !void {
        const file_path = parsed_file.file_path;

        try self.extractFunctions(parsed_file, file_path);
        try self.extractCalls(parsed_file, file_path);
    }

    /// Build call graph from multiple parsed files
    pub fn buildFromFiles(self: *CallGraphBuilder, files: []const *ParsedFile) !void {
        for (files) |file| {
            try self.buildFromFile(file);
        }
    }

    fn extractFunctions(self: *CallGraphBuilder, parsed_file: *const ParsedFile, file_path: []const u8) !void {
        for (parsed_file.nodes.items) |node| {
            if (node.type == .function_def) {
                const func: Function = .{
                    .name = node.name orelse continue,
                    .file_path = file_path,
                    .line = node.line,
                };
                try self.graph.addFunction(func);
            }
        }
    }

    fn extractCalls(self: *CallGraphBuilder, parsed_file: *const ParsedFile, file_path: []const u8) !void {
        for (parsed_file.nodes.items) |node| {
            if (node.type == .function_def) {
                const caller: Function = .{
                    .name = node.name orelse continue,
                    .file_path = file_path,
                    .line = node.line,
                };

                if (node.children) |children| {
                    for (children) |child| {
                        if (child.type == .function_call) {
                            const callee_name = child.name orelse continue;

                            // Find or create callee function
                            const callee: Function = .{
                                .name = callee_name,
                                .file_path = file_path,
                                .line = child.line,
                            };

                            try self.graph.addCall(caller, callee);
                        }
                    }
                }
            }
        }
    }

    /// Get the built graph
    pub fn getGraph(self: *const CallGraphBuilder) *const CallGraph {
        return &self.graph;
    }
};

/// Convenience function to build call graph from file paths
pub fn buildCallGraph(allocator: std.mem.Allocator, file_paths: []const []const u8) !CallGraph {
    var builder = CallGraphBuilder.init(allocator);
    defer builder.deinit();

    var parsed_files = std.ArrayList(*ParsedFile).init(allocator);
    defer {
        for (parsed_files.items) |file| {
            file.deinit();
            allocator.destroy(file);
        }
        parsed_files.deinit();
    }

    for (file_paths) |file_path| {
        const parsed_file = try allocator.create(ParsedFile);
        parsed_file.* = try builder.parser.parseFile(file_path);
        try parsed_files.append(parsed_file);
    }

    try builder.buildFromFiles(parsed_files.items);

    // Copy the graph to return it
    var result = CallGraph.init(allocator);
    errdefer result.deinit();

    // Copy all functions
    for (builder.graph.all_functions.items) |func| {
        try result.addFunction(func);
    }

    // Copy all edges
    for (builder.graph.edges.items) |edge| {
        try result.addCall(edge.caller, edge.callee);
    }

    return result;
}
