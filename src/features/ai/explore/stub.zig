//! AI explore stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// ── Re-exported types ──────────────────────────────────────────────────────

pub const ExploreError = types.ExploreError;
pub const ExploreLevel = types.ExploreLevel;
pub const OutputFormat = types.OutputFormat;
pub const FileType = types.FileType;
pub const MatchType = types.MatchType;
pub const QueryIntent = types.QueryIntent;
pub const PatternType = types.PatternType;
pub const ImportType = types.ImportType;
pub const AstNodeType = types.AstNodeType;
pub const ExploreConfig = types.ExploreConfig;
pub const FileFilter = types.FileFilter;
pub const SearchScope = types.SearchScope;
pub const SearchOptions = types.SearchOptions;
pub const DiscoveryConfig = types.DiscoveryConfig;
pub const AdaptiveConfig = types.AdaptiveConfig;
pub const Match = types.Match;
pub const ExplorationStats = types.ExplorationStats;
pub const ExploreResult = types.ExploreResult;
pub const ParsedQuery = types.ParsedQuery;
pub const QueryUnderstanding = types.QueryUnderstanding;
pub const FileVisitor = types.FileVisitor;
pub const FileStats = types.FileStats;
pub const SearchPattern = types.SearchPattern;
pub const PatternCompiler = types.PatternCompiler;
pub const ExploreAgent = types.ExploreAgent;
pub const AstNode = types.AstNode;
pub const ParsedFile = types.ParsedFile;
pub const AstParser = types.AstParser;
pub const ParallelExplorer = types.ParallelExplorer;
pub const WorkItem = types.WorkItem;
pub const Function = types.Function;
pub const CallEdge = types.CallEdge;
pub const CallGraph = types.CallGraph;
pub const CallGraphBuilder = types.CallGraphBuilder;
pub const Module = types.Module;
pub const DependencyEdge = types.DependencyEdge;
pub const DependencyGraph = types.DependencyGraph;
pub const DependencyAnalyzer = types.DependencyAnalyzer;
pub const ModelFormat = types.ModelFormat;
pub const QuantizationType = types.QuantizationType;
pub const DiscoveredModel = types.DiscoveredModel;
pub const SystemCapabilities = types.SystemCapabilities;
pub const ModelRequirements = types.ModelRequirements;
pub const WarmupResult = types.WarmupResult;
pub const ModelDiscovery = types.ModelDiscovery;
pub const FileMetadata = types.FileMetadata;
pub const CodeSnippet = types.CodeSnippet;
pub const CodebaseAnswer = types.CodebaseAnswer;
pub const IndexResult = types.IndexResult;
pub const IndexStats = types.IndexStats;
pub const CodebaseIndex = types.CodebaseIndex;
pub const detectCapabilities = types.detectCapabilities;
pub const runWarmup = types.runWarmup;

// ── Submodule Namespace Re-exports ─────────────────────────────────────────

pub const config = struct {
    pub const ExploreConfig = types.ExploreConfig;
    pub const ExploreLevel = types.ExploreLevel;
    pub const OutputFormat = types.OutputFormat;
    pub const FileType = types.FileType;
    pub const FileFilter = types.FileFilter;
    pub const SearchScope = types.SearchScope;
    pub const SearchOptions = types.SearchOptions;
};
pub const results = struct {
    pub const ExploreResult = types.ExploreResult;
    pub const Match = types.Match;
    pub const MatchType = types.MatchType;
    pub const ExploreError = types.ExploreError;
    pub const ExplorationStats = types.ExplorationStats;
};
pub const fs = struct {
    pub const FileVisitor = types.FileVisitor;
    pub const FileStats = types.FileStats;
};
pub const search = struct {
    pub const SearchPattern = types.SearchPattern;
    pub const PatternType = types.PatternType;
    pub const PatternCompiler = types.PatternCompiler;
};
pub const agent = struct {
    pub const ExploreAgent = types.ExploreAgent;
    pub const createDefaultAgent = @import("stub.zig").createDefaultAgent;
    pub const createQuickAgent = @import("stub.zig").createQuickAgent;
    pub const createThoroughAgent = @import("stub.zig").createThoroughAgent;
};
pub const query = struct {
    pub const QueryIntent = types.QueryIntent;
    pub const ParsedQuery = types.ParsedQuery;
    pub const QueryUnderstanding = types.QueryUnderstanding;
};
pub const ast = struct {
    pub const AstNode = types.AstNode;
    pub const AstNodeType = types.AstNodeType;
    pub const ParsedFile = types.ParsedFile;
    pub const AstParser = types.AstParser;
};
pub const parallel = struct {
    pub const ParallelExplorer = types.ParallelExplorer;
    pub const WorkItem = types.WorkItem;
    pub const parallelExplore = @import("stub.zig").parallelExplore;
};
pub const callgraph = struct {
    pub const Function = types.Function;
    pub const CallEdge = types.CallEdge;
    pub const CallGraph = types.CallGraph;
    pub const CallGraphBuilder = types.CallGraphBuilder;
    pub const buildCallGraph = @import("stub.zig").buildCallGraph;
};
pub const dependency = struct {
    pub const Module = types.Module;
    pub const DependencyEdge = types.DependencyEdge;
    pub const ImportType = types.ImportType;
    pub const DependencyGraph = types.DependencyGraph;
    pub const ModuleDependency = types.DependencyGraph.ModuleDependency;
    pub const DependencyAnalyzer = types.DependencyAnalyzer;
    pub const buildDependencyGraph = @import("stub.zig").buildDependencyGraph;
};

// ── Free Functions ─────────────────────────────────────────────────────────

pub fn createDefaultAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}
pub fn createQuickAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}
pub fn createThoroughAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}
pub fn parallelExplore(_: std.mem.Allocator, _: []const u8, _: ExploreConfig, _: []const u8) ExploreError!ExploreResult {
    return ExploreError.FeatureDisabled;
}
pub fn buildCallGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!CallGraph {
    return ExploreError.FeatureDisabled;
}
pub fn buildDependencyGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!DependencyGraph {
    return ExploreError.FeatureDisabled;
}
pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
