//! AI explore stub — disabled at compile time.

const std = @import("std");
const stub_root = @This();

// --- Submodule Namespace Re-exports ---

pub const config = struct {
    pub const ExploreConfig = stub_root.ExploreConfig;
    pub const ExploreLevel = stub_root.ExploreLevel;
    pub const OutputFormat = stub_root.OutputFormat;
    pub const FileType = stub_root.FileType;
    pub const FileFilter = stub_root.FileFilter;
    pub const SearchScope = stub_root.SearchScope;
    pub const SearchOptions = stub_root.SearchOptions;
};
pub const results = struct {
    pub const ExploreResult = stub_root.ExploreResult;
    pub const Match = stub_root.Match;
    pub const MatchType = stub_root.MatchType;
    pub const ExploreError = stub_root.ExploreError;
    pub const ExplorationStats = stub_root.ExplorationStats;
};
pub const fs = struct {
    pub const FileVisitor = stub_root.FileVisitor;
    pub const FileStats = stub_root.FileStats;
};
pub const search = struct {
    pub const SearchPattern = stub_root.SearchPattern;
    pub const PatternType = stub_root.PatternType;
    pub const PatternCompiler = stub_root.PatternCompiler;
};
pub const agent = struct {
    pub const ExploreAgent = stub_root.ExploreAgent;
    pub const createDefaultAgent = stub_root.createDefaultAgent;
    pub const createQuickAgent = stub_root.createQuickAgent;
    pub const createThoroughAgent = stub_root.createThoroughAgent;
};
pub const query = struct {
    pub const QueryIntent = stub_root.QueryIntent;
    pub const ParsedQuery = stub_root.ParsedQuery;
    pub const QueryUnderstanding = stub_root.QueryUnderstanding;
};
pub const ast = struct {
    pub const AstNode = stub_root.AstNode;
    pub const AstNodeType = stub_root.AstNodeType;
    pub const ParsedFile = stub_root.ParsedFile;
    pub const AstParser = stub_root.AstParser;
};
pub const parallel = struct {
    pub const ParallelExplorer = stub_root.ParallelExplorer;
    pub const WorkItem = stub_root.WorkItem;
    pub const parallelExplore = stub_root.parallelExplore;
};
pub const callgraph = struct {
    pub const Function = stub_root.Function;
    pub const CallEdge = stub_root.CallEdge;
    pub const CallGraph = stub_root.CallGraph;
    pub const CallGraphBuilder = stub_root.CallGraphBuilder;
    pub const buildCallGraph = stub_root.buildCallGraph;
};
pub const dependency = struct {
    pub const Module = stub_root.Module;
    pub const DependencyEdge = stub_root.DependencyEdge;
    pub const ImportType = stub_root.ImportType;
    pub const DependencyGraph = stub_root.DependencyGraph;
    pub const DependencyAnalyzer = stub_root.DependencyAnalyzer;
    pub const buildDependencyGraph = stub_root.buildDependencyGraph;
};

// --- Error & Enums ---

pub const ExploreError = error{ExploreDisabled};
pub const ExploreLevel = enum(u2) { quick = 0, medium = 1, thorough = 2, deep = 3 };
pub const OutputFormat = enum { human, json, compact, yaml };
pub const FileType = enum { source, header, test_file, documentation, config, data, binary, other };

// --- Config Types ---

pub const ExploreConfig = struct {
    level: ExploreLevel = .medium,
    max_files: usize = 10000,
    max_depth: usize = 20,
    timeout_ms: u64 = 60000,
    include_patterns: []const []const u8 = &.{},
    exclude_patterns: []const []const u8 = &.{},
    case_sensitive: bool = false,
    use_regex: bool = false,
    parallel_io: bool = true,
    worker_count: ?usize = null,
    follow_symlinks: bool = false,
    include_hidden: bool = false,
    file_size_limit_bytes: ?u64 = null,
    output_format: OutputFormat = .human,
    pub fn defaultForLevel(level: ExploreLevel) ExploreConfig {
        return .{ .level = level };
    }
};

pub const FileFilter = struct {
    extensions: ?[]const []const u8 = null,
    exclude_extensions: ?[]const []const u8 = null,
    min_size_bytes: ?u64 = null,
    max_size_bytes: ?u64 = null,
    modified_after: ?i128 = null,
    modified_before: ?i128 = null,
    pub fn matches(_: *const FileFilter, _: anytype) bool {
        return false;
    }
};

pub const SearchScope = struct {
    paths: []const []const u8 = &.{"."},
    recursive: bool = true,
    filter: ?FileFilter = null,
};

pub const SearchOptions = struct {
    scope: SearchScope = .{},
    patterns: []const []const u8 = &.{},
    config: ExploreConfig = .{},
};

// --- Result Types ---

pub const Match = struct {
    path: []const u8 = "",
    line: u32 = 0,
    content: []const u8 = "",
    context_before: []const u8 = "",
    context_after: []const u8 = "",
};
pub const MatchType = enum { exact, fuzzy, regex, semantic };
pub const ExplorationStats = struct { files_searched: u32 = 0, matches_found: u32 = 0, search_time_ms: i64 = 0 };

pub const ExploreResult = struct {
    allocator: std.mem.Allocator = undefined,
    matches: []Match = &.{},
    stats: ExplorationStats = .{},
    query: []const u8 = "",
    matches_found: usize = 0,
    files_scanned: usize = 0,
    duration_ms: i64 = 0,
    cancelled: bool = false,
    explore_error: ?ExploreError = null,
    error_message: ?[]const u8 = null,
    pub fn init(allocator: std.mem.Allocator, _: []const u8, _: ExploreLevel) ExploreResult {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *ExploreResult) void {}
    pub fn formatHuman(_: *ExploreResult, _: anytype) !void {}
    pub fn formatJSON(_: *ExploreResult, _: anytype) !void {}
};

// --- Query Types ---

pub const QueryIntent = enum { search, definition, usage, structure };
pub const ParsedQuery = struct { original: []const u8, intent: QueryIntent, terms: []const []const u8 };
pub const QueryUnderstanding = struct {
    pub fn parse(_: std.mem.Allocator, query_str: []const u8) ParsedQuery {
        return .{ .original = query_str, .intent = .search, .terms = &.{} };
    }
};

// --- Stub Impl Types ---

pub const FileVisitor = struct {
    pub fn init(_: std.mem.Allocator, _: ExploreConfig) FileVisitor {
        return .{};
    }
    pub fn deinit(_: *FileVisitor) void {}
};
pub const FileStats = struct { total_files: usize = 0, total_bytes: u64 = 0 };

pub const SearchPattern = struct { pattern: []const u8 = "", pattern_type: PatternType = .literal };
pub const PatternType = enum { literal, glob, regex };
pub const PatternCompiler = struct {
    pub fn init(_: std.mem.Allocator) PatternCompiler {
        return .{};
    }
    pub fn deinit(_: *PatternCompiler) void {}
};

pub const ExploreAgent = struct {
    allocator: std.mem.Allocator = undefined,
    config: ExploreConfig = .{},
    pub fn init(_: std.mem.Allocator, _: ExploreConfig) ExploreError!ExploreAgent {
        return ExploreError.ExploreDisabled;
    }
    pub fn deinit(_: *ExploreAgent) void {}
    pub fn explore(_: *ExploreAgent, _: []const u8, _: []const u8) ExploreError!ExploreResult {
        return ExploreError.ExploreDisabled;
    }
    pub fn exploreWithPatterns(_: *ExploreAgent, _: []const u8, _: []const []const u8) ExploreError!ExploreResult {
        return ExploreError.ExploreDisabled;
    }
    pub fn exploreNaturalLanguage(_: *ExploreAgent, _: []const u8, _: []const u8) ExploreError!ExploreResult {
        return ExploreError.ExploreDisabled;
    }
    pub fn cancel(_: *ExploreAgent) void {}
    pub fn isCancelled(_: *ExploreAgent) bool {
        return false;
    }
    pub fn getStats(_: *ExploreAgent) ExplorationStats {
        return .{};
    }
};

pub const AstNode = struct { node_type: AstNodeType = .unknown, name: []const u8 = "" };
pub const AstNodeType = enum { unknown, function, struct_decl, variable, import };
pub const ParsedFile = struct { path: []const u8 = "", nodes: []AstNode = &.{} };
pub const AstParser = struct {
    pub fn init(_: std.mem.Allocator) AstParser {
        return .{};
    }
    pub fn deinit(_: *AstParser) void {}
};

pub const ParallelExplorer = struct {
    pub fn init(_: std.mem.Allocator, _: ExploreConfig) ParallelExplorer {
        return .{};
    }
    pub fn deinit(_: *ParallelExplorer) void {}
};
pub const WorkItem = struct { path: []const u8 = "" };

pub const Function = struct { name: []const u8 = "", file: []const u8 = "", line: u32 = 0 };
pub const CallEdge = struct { caller: Function = .{}, callee: Function = .{} };
pub const CallGraph = struct {
    functions: []Function = &.{},
    edges: []CallEdge = &.{},
    pub fn deinit(_: *CallGraph) void {}
};
pub const CallGraphBuilder = struct {
    pub fn init(_: std.mem.Allocator) CallGraphBuilder {
        return .{};
    }
    pub fn deinit(_: *CallGraphBuilder) void {}
};

pub const Module = struct { name: []const u8 = "", path: []const u8 = "" };
pub const DependencyEdge = struct { from: Module = .{}, to: Module = .{}, import_type: ImportType = .standard };
pub const ImportType = enum { standard, package, builtin };
pub const DependencyGraph = struct {
    modules: []Module = &.{},
    edges: []DependencyEdge = &.{},
    pub fn deinit(_: *DependencyGraph) void {}
};
pub const DependencyAnalyzer = struct {
    pub fn init(_: std.mem.Allocator) DependencyAnalyzer {
        return .{};
    }
    pub fn deinit(_: *DependencyAnalyzer) void {}
};

// --- Free Functions ---

pub fn createDefaultAgent(_: std.mem.Allocator) ExploreAgent {
    return .{};
}
pub fn createQuickAgent(_: std.mem.Allocator) ExploreAgent {
    return .{};
}
pub fn createThoroughAgent(_: std.mem.Allocator) ExploreAgent {
    return .{};
}

pub fn parallelExplore(_: std.mem.Allocator, _: []const u8, _: ExploreConfig, _: []const u8) ExploreError!ExploreResult {
    return ExploreError.ExploreDisabled;
}
pub fn buildCallGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!CallGraph {
    return ExploreError.ExploreDisabled;
}
pub fn buildDependencyGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!DependencyGraph {
    return ExploreError.ExploreDisabled;
}

pub fn isEnabled() bool {
    return false;
}
