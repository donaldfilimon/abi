const std = @import("std");
const build_options = @import("build_options");

pub const config = @import("config.zig");
pub const results = @import("results.zig");
pub const fs = @import("fs.zig");
pub const search = @import("search.zig");
pub const agent = @import("agent.zig");
pub const query = @import("query.zig");
pub const ast = @import("ast.zig");
pub const parallel = @import("parallel.zig");
pub const callgraph = @import("callgraph.zig");
pub const dependency = @import("dependency.zig");

pub const ExploreConfig = config.ExploreConfig;
pub const ExploreLevel = config.ExploreLevel;
pub const OutputFormat = config.OutputFormat;
pub const FileType = config.FileType;
pub const FileFilter = config.FileFilter;
pub const SearchScope = config.SearchScope;
pub const SearchOptions = config.SearchOptions;

pub const ExploreResult = results.ExploreResult;
pub const Match = results.Match;
pub const MatchType = results.MatchType;
pub const ExploreError = results.ExploreError;
pub const ExplorationStats = results.ExplorationStats;

pub const FileVisitor = fs.FileVisitor;
pub const FileStats = fs.FileStats;

pub const SearchPattern = search.SearchPattern;
pub const PatternType = search.PatternType;
pub const PatternCompiler = search.PatternCompiler;

pub const ExploreAgent = agent.ExploreAgent;

pub const QueryIntent = query.QueryIntent;
pub const ParsedQuery = query.ParsedQuery;
pub const QueryUnderstanding = query.QueryUnderstanding;

pub const AstNode = ast.AstNode;
pub const AstNodeType = ast.AstNodeType;
pub const ParsedFile = ast.ParsedFile;
pub const AstParser = ast.AstParser;

pub const ParallelExplorer = parallel.ParallelExplorer;
pub const WorkItem = parallel.WorkItem;

pub const Function = callgraph.Function;
pub const CallEdge = callgraph.CallEdge;
pub const CallGraph = callgraph.CallGraph;
pub const CallGraphBuilder = callgraph.CallGraphBuilder;

pub const Module = dependency.Module;
pub const DependencyEdge = dependency.DependencyEdge;
pub const ImportType = dependency.ImportType;
pub const DependencyGraph = dependency.DependencyGraph;
pub const DependencyAnalyzer = dependency.DependencyAnalyzer;

pub fn createDefaultAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return agent.createDefaultAgent(allocator);
}

pub fn createQuickAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return agent.createQuickAgent(allocator);
}

pub fn createThoroughAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return agent.createThoroughAgent(allocator);
}

pub fn parallelExplore(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    explore_config: ExploreConfig,
    search_query: []const u8,
) !ExploreResult {
    return parallel.parallelExplore(allocator, root_path, explore_config, search_query);
}

pub fn buildCallGraph(allocator: std.mem.Allocator, file_paths: []const []const u8) !CallGraph {
    return callgraph.buildCallGraph(allocator, file_paths);
}

pub fn buildDependencyGraph(allocator: std.mem.Allocator, file_paths: []const []const u8) !DependencyGraph {
    return dependency.buildDependencyGraph(allocator, file_paths);
}

/// Check if explore features are enabled
pub fn isEnabled() bool {
    return build_options.enable_ai and build_options.enable_explore;
}
