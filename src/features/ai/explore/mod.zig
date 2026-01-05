const std = @import("std");

pub const config = @import("config.zig");
pub const results = @import("results.zig");
pub const fs = @import("fs.zig");
pub const search = @import("search.zig");
pub const agent = @import("agent.zig");
pub const query = @import("query.zig");
pub const ast = @import("ast.zig");

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

pub fn createDefaultAgent(allocator: std.mem.Allocator) ExploreAgent {
    return agent.createDefaultAgent(allocator);
}

pub fn createQuickAgent(allocator: std.mem.Allocator) ExploreAgent {
    return agent.createQuickAgent(allocator);
}

pub fn createThoroughAgent(allocator: std.mem.Allocator) ExploreAgent {
    return agent.createThoroughAgent(allocator);
}
