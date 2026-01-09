//! Dependency analysis for codebases
//!
//! Provides functionality to analyze module dependencies, imports, and
//! build dependency graphs across Zig, Rust, and TypeScript/JavaScript codebases.

const std = @import("std");
const AstParser = @import("ast.zig").AstParser;
const AstNode = @import("ast.zig").AstNode;
const ParsedFile = @import("ast.zig").ParsedFile;

/// Represents a module in the dependency graph
pub const Module = struct {
    name: []const u8,
    file_path: []const u8,
    file_type: []const u8,
};

/// Represents a dependency edge (module -> imports module)
pub const DependencyEdge = struct {
    from: Module,
    to: Module,
    import_type: ImportType,
};

/// Types of imports/dependencies
pub const ImportType = enum {
    /// Standard library import
    std,
    /// External package/library import
    external,
    /// Local/project import
    local,
    /// Relative path import
    relative,
};

/// Dependency graph data structure
pub const DependencyGraph = struct {
    allocator: std.mem.Allocator,
    /// Map from module name to list of modules it depends on
    dependencies: std.StringHashMap(std.ArrayListUnmanaged(ModuleDependency)),
    /// Map from module name to list of modules that depend on it
    dependents: std.StringHashMap(std.ArrayListUnmanaged(ModuleDependency)),
    /// All modules in the graph
    all_modules: std.ArrayListUnmanaged(Module),
    /// All edges in the graph
    edges: std.ArrayListUnmanaged(DependencyEdge),

    /// Represents a dependency relationship
    pub const ModuleDependency = struct {
        module: Module,
        import_type: ImportType,
    };

    pub fn init(allocator: std.mem.Allocator) DependencyGraph {
        return .{
            .allocator = allocator,
            .dependencies = std.StringHashMap(std.ArrayListUnmanaged(ModuleDependency)).init(allocator),
            .dependents = std.StringHashMap(std.ArrayListUnmanaged(ModuleDependency)).init(allocator),
            .all_modules = .{},
            .edges = .{},
        };
    }

    pub fn deinit(self: *DependencyGraph) void {
        var deps_iter = self.dependencies.valueIterator();
        while (deps_iter.next()) |list| {
            list.deinit(self.allocator);
        }
        self.dependencies.deinit();

        var dependents_iter = self.dependents.valueIterator();
        while (dependents_iter.next()) |list| {
            list.deinit(self.allocator);
        }
        self.dependents.deinit();

        self.all_modules.deinit(self.allocator);
        self.edges.deinit(self.allocator);
    }

    /// Add a module to the graph
    pub fn addModule(self: *DependencyGraph, module: Module) !void {
        try self.all_modules.append(self.allocator, module);

        const key = try self.allocator.dupe(u8, module.name);
        errdefer self.allocator.free(key);

        try self.dependencies.put(key, .{});
        try self.dependents.put(key, .{});
    }

    /// Add a dependency relationship (from -> to)
    pub fn addDependency(self: *DependencyGraph, from: Module, to: Module, import_type: ImportType) !void {
        const edge: DependencyEdge = .{
            .from = from,
            .to = to,
            .import_type = import_type,
        };
        try self.edges.append(self.allocator, edge);

        const dep = ModuleDependency{ .module = to, .import_type = import_type };

        if (self.dependencies.get(from.name)) |deps_list| {
            try deps_list.append(self.allocator, dep);
        }

        const dependent = ModuleDependency{ .module = from, .import_type = import_type };
        if (self.dependents.get(to.name)) |dependents_list| {
            try dependents_list.append(self.allocator, dependent);
        }
    }

    /// Get all modules that a given module depends on
    pub fn getDependencies(self: *const DependencyGraph, module_name: []const u8) ?[]const ModuleDependency {
        if (self.dependencies.get(module_name)) |list| {
            return list.items;
        }
        return null;
    }

    /// Get all modules that depend on a given module
    pub fn getDependents(self: *const DependencyGraph, module_name: []const u8) ?[]const ModuleDependency {
        if (self.dependents.get(module_name)) |list| {
            return list.items;
        }
        return null;
    }

    /// Detect circular dependencies
    pub fn findCircularDependencies(self: *const DependencyGraph) !std.ArrayList([]const []const u8) {
        var cycles = std.ArrayList([]const []const u8).init(self.allocator);
        var visited = std.StringHashMap(void).init(self.allocator);
        defer visited.deinit();

        for (self.all_modules.items) |module| {
            var path = std.ArrayList([]const u8).init(self.allocator);
            defer path.deinit();

            try self.findCyclesDFS(module.name, &visited, &path, &cycles);
        }

        return cycles;
    }

    fn findCyclesDFS(
        self: *const DependencyGraph,
        current: []const u8,
        visited: *std.StringHashMap(void),
        path: *std.ArrayList([]const u8),
        cycles: *std.ArrayList([]const []const u8),
    ) !void {
        if (path.items.len > 0 and std.mem.eql(u8, current, path.items[0])) {
            var cycle = std.ArrayList([]const u8).init(self.allocator);
            try cycle.appendSlice(path.items);
            try cycles.append(cycle.toOwnedSlice());
            return;
        }

        if (visited.get(current) != null) return;

        for (path.items) |item| {
            if (std.mem.eql(u8, item, current)) {
                var cycle = std.ArrayList([]const u8).init(self.allocator);
                const start_idx = for (path.items, 0..) |item2, i| {
                    if (std.mem.eql(u8, item2, current)) break i;
                } else unreachable;
                try cycle.appendSlice(path.items[start_idx..]);
                try cycles.append(cycle.toOwnedSlice());
                return;
            }
        }

        try path.append(current);

        if (self.dependencies.get(current)) |deps| {
            for (deps.items) |dep| {
                try self.findCyclesDFS(dep.module.name, visited, path, cycles);
            }
        }

        _ = path.pop();
        try visited.put(current, {});
    }

    /// Get topological order of modules
    pub fn topologicalSort(self: *const DependencyGraph) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).init(self.allocator);
        var visited = std.StringHashMap(void).init(self.allocator);
        defer visited.deinit();

        for (self.all_modules.items) |module| {
            if (visited.get(module.name) == null) {
                try self.topologicalSortDFS(module.name, &visited, &result);
            }
        }

        std.mem.reverse([]const u8, result.items);
        return result;
    }

    fn topologicalSortDFS(
        self: *const DependencyGraph,
        current: []const u8,
        visited: *std.StringHashMap(void),
        result: *std.ArrayList([]const u8),
    ) !void {
        if (visited.get(current) != null) return;

        try visited.put(current, {});

        if (self.dependencies.get(current)) |deps| {
            for (deps.items) |dep| {
                try self.topologicalSortDFS(dep.module.name, visited, result);
            }
        }

        try result.append(current);
    }

    /// Export graph to DOT format for visualization
    pub fn toDot(self: *const DependencyGraph, writer: anytype) !void {
        try writer.writeAll("digraph DependencyGraph {\n");
        try writer.writeAll("  rankdir=LR;\n");
        try writer.writeAll("  node [shape=box];\n\n");

        for (self.edges.items) |edge| {
            const style = switch (edge.import_type) {
                .std => " [color=blue]",
                .external => " [color=green]",
                .local => " [color=black]",
                .relative => " [color=orange, style=dashed]",
            };
            try writer.print("  \"{s}\" -> \"{s}\"{s};\n", .{ edge.from.name, edge.to.name, style });
        }

        try writer.writeAll("}\n");
    }
};

/// Builder for creating dependency graphs from parsed AST
pub const DependencyAnalyzer = struct {
    allocator: std.mem.Allocator,
    parser: AstParser,
    graph: DependencyGraph,

    pub fn init(allocator: std.mem.Allocator) DependencyAnalyzer {
        return .{
            .allocator = allocator,
            .parser = AstParser.init(allocator),
            .graph = DependencyGraph.init(allocator),
        };
    }

    pub fn deinit(self: *DependencyAnalyzer) void {
        self.parser.deinit();
        self.graph.deinit();
    }

    /// Build dependency graph from a single parsed file
    pub fn buildFromFile(self: *DependencyAnalyzer, parsed_file: *const ParsedFile) !void {
        const module_name = try self.getModuleName(parsed_file);
        defer self.allocator.free(module_name);

        const module: Module = .{
            .name = module_name,
            .file_path = parsed_file.file_path,
            .file_type = parsed_file.file_type,
        };
        try self.graph.addModule(module);

        try self.extractImports(parsed_file, module);
    }

    /// Build dependency graph from multiple parsed files
    pub fn buildFromFiles(self: *DependencyAnalyzer, files: []const *ParsedFile) !void {
        for (files) |file| {
            try self.buildFromFile(file);
        }
    }

    fn getModuleName(self: *DependencyAnalyzer, parsed_file: *const ParsedFile) ![]const u8 {
        const file_path = parsed_file.file_path;

        if (std.mem.eql(u8, parsed_file.file_type, ".zig")) {
            if (std.mem.lastIndexOf(u8, file_path, ".zig")) |idx| {
                return self.allocator.dupe(u8, file_path[0..idx]);
            }
        } else if (std.mem.eql(u8, parsed_file.file_type, ".rs")) {
            if (std.mem.lastIndexOf(u8, file_path, ".rs")) |idx| {
                return self.allocator.dupe(u8, file_path[0..idx]);
            }
        } else if (std.mem.eql(u8, parsed_file.file_type, ".ts") or std.mem.eql(u8, parsed_file.file_type, ".js")) {
            if (std.mem.lastIndexOf(u8, file_path, ".ts")) |idx| {
                return self.allocator.dupe(u8, file_path[0..idx]);
            }
            if (std.mem.lastIndexOf(u8, file_path, ".js")) |idx| {
                return self.allocator.dupe(u8, file_path[0..idx]);
            }
        }

        return self.allocator.dupe(u8, file_path);
    }

    fn extractImports(self: *DependencyAnalyzer, parsed_file: *const ParsedFile, from_module: Module) !void {
        for (parsed_file.imports.items) |import_path| {
            const import_type = self.classifyImport(import_path, parsed_file.file_type);

            const to_module: Module = .{
                .name = import_path,
                .file_path = import_path,
                .file_type = parsed_file.file_type,
            };

            try self.graph.addDependency(from_module, to_module, import_type);
        }
    }

    fn classifyImport(self: *DependencyAnalyzer, import_path: []const u8, language: ParsedFile.Language) ImportType {
        _ = self;

        if (language == .zig) {
            if (std.mem.startsWith(u8, import_path, "std.")) {
                return .std;
            }
            if (std.mem.startsWith(u8, import_path, "@import(\"std\")")) {
                return .std;
            }
            return .local;
        } else if (language == .rust) {
            if (std.mem.startsWith(u8, import_path, "std::") or
                std.mem.startsWith(u8, import_path, "core::") or
                std.mem.startsWith(u8, import_path, "alloc::"))
            {
                return .std;
            }
            if (std.mem.indexOf(u8, import_path, "::") != null) {
                return .external;
            }
            if (std.mem.startsWith(u8, import_path, "crate::") or
                std.mem.startsWith(u8, import_path, "super::") or
                std.mem.startsWith(u8, import_path, "self::"))
            {
                return .local;
            }
        } else if (language == .typescript or language == .javascript) {
            if (std.mem.startsWith(u8, import_path, "./") or
                std.mem.startsWith(u8, import_path, "../"))
            {
                return .relative;
            }
            if (std.mem.startsWith(u8, import_path, "@")) {
                return .external;
            }
            return .external;
        }

        return .local;
    }

    /// Get the built graph
    pub fn getGraph(self: *const DependencyAnalyzer) *const DependencyGraph {
        return &self.graph;
    }
};

/// Convenience function to build dependency graph from file paths
pub fn buildDependencyGraph(allocator: std.mem.Allocator, file_paths: []const []const u8) !DependencyGraph {
    var analyzer = DependencyAnalyzer.init(allocator);
    defer analyzer.deinit();

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
        parsed_file.* = try analyzer.parser.parseFile(file_path);
        try parsed_files.append(parsed_file);
    }

    try analyzer.buildFromFiles(parsed_files.items);

    var result = DependencyGraph.init(allocator);
    errdefer result.deinit();

    for (analyzer.graph.all_modules.items) |module| {
        try result.addModule(module);
    }

    for (analyzer.graph.edges.items) |edge| {
        try result.addDependency(edge.from, edge.to, edge.import_type);
    }

    return result;
}
