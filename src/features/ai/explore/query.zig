const std = @import("std");
const ExploreConfig = @import("config.zig").ExploreConfig;
const ExploreLevel = @import("config.zig").ExploreLevel;
const SearchPattern = @import("search.zig").SearchPattern;
const PatternCompiler = @import("search.zig").PatternCompiler;
const PatternType = @import("search.zig").PatternType;

pub const QueryIntent = enum {
    find_functions,
    find_types,
    find_tests,
    find_imports,
    find_comments,
    find_configs,
    find_docs,
    find_any,
    analyze_structure,
    list_files,
    count_occurrences,
    find_pattern,
    unknown,
};

pub const ParsedQuery = struct {
    original: []const u8,
    intent: QueryIntent,
    patterns: []const []const u8,
    target_paths: []const []const u8,
    file_extensions: []const []const u8,
    is_natural_language: bool,
    confidence: f32,
};

pub const QueryUnderstanding = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) QueryUnderstanding {
        return QueryUnderstanding{ .allocator = allocator };
    }

    pub fn deinit(_: *QueryUnderstanding) void {}

    pub fn parse(self: *QueryUnderstanding, query: []const u8) !ParsedQuery {
        var parsed = ParsedQuery{
            .original = query,
            .intent = .unknown,
            .patterns = &.{},
            .target_paths = &.{},
            .file_extensions = &.{},
            .is_natural_language = true,
            .confidence = 0.0,
        };

        const lower_query = try self.toLowercase(query);
        defer self.allocator.free(lower_query);

        errdefer self.freeParsedQuery(parsed);

        parsed.intent = self.classifyIntent(lower_query);
        parsed.patterns = try self.extractPatterns(self.allocator, lower_query, parsed.intent);
        parsed.target_paths = try self.extractTargetPaths(self.allocator, lower_query);
        parsed.file_extensions = try self.extractFileExtensions(self.allocator, lower_query);
        parsed.confidence = self.calculateConfidence(lower_query, parsed.intent);

        return parsed;
    }

    fn toLowercase(self: *QueryUnderstanding, text: []const u8) ![]const u8 {
        const result = try self.allocator.alloc(u8, text.len);
        for (text, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }

    fn classifyIntent(self: *QueryUnderstanding, query: []const u8) QueryIntent {
        _ = self;
        if (std.mem.indexOf(u8, query, "function") != null or
            std.mem.indexOf(u8, query, "fn ") != null or
            std.mem.indexOf(u8, query, "method") != null)
        {
            return .find_functions;
        }
        if (std.mem.indexOf(u8, query, "type") != null or
            std.mem.indexOf(u8, query, "struct") != null or
            std.mem.indexOf(u8, query, "enum") != null or
            std.mem.indexOf(u8, query, "interface") != null)
        {
            return .find_types;
        }
        if (std.mem.indexOf(u8, query, "test") != null or
            std.mem.indexOf(u8, query, "_test") != null)
        {
            return .find_tests;
        }
        if (std.mem.indexOf(u8, query, "import") != null or
            std.mem.indexOf(u8, query, "use ") != null or
            std.mem.indexOf(u8, query, "require") != null)
        {
            return .find_imports;
        }
        if (std.mem.indexOf(u8, query, "comment") != null or
            std.mem.indexOf(u8, query, "todo") != null or
            std.mem.indexOf(u8, query, "fixme") != null)
        {
            return .find_comments;
        }
        if (std.mem.indexOf(u8, query, "config") != null or
            std.mem.indexOf(u8, query, "setting") != null)
        {
            return .find_configs;
        }
        if (std.mem.indexOf(u8, query, "doc") != null or
            std.mem.indexOf(u8, query, "documentation") != null)
        {
            return .find_docs;
        }
        if (std.mem.indexOf(u8, query, "how many") != null or
            std.mem.indexOf(u8, query, "count") != null)
        {
            return .count_occurrences;
        }
        if (std.mem.indexOf(u8, query, "list") != null or
            std.mem.indexOf(u8, query, "show all") != null)
        {
            return .list_files;
        }
        if (std.mem.indexOf(u8, query, "where") != null or
            std.mem.indexOf(u8, query, "find") != null)
        {
            return .find_any;
        }
        if (std.mem.indexOf(u8, query, "structure") != null or
            std.mem.indexOf(u8, query, "analyze") != null)
        {
            return .analyze_structure;
        }
        return .find_any;
    }

    fn extractPatterns(self: *QueryUnderstanding, allocator: std.mem.Allocator, query: []const u8, intent: QueryIntent) ![]const []const u8 {
        var patterns = std.ArrayListUnmanaged([]const u8){};
        errdefer patterns.deinit(allocator);

        switch (intent) {
            .find_functions => {
                if (std.mem.indexOf(u8, query, "pub fn") != null) {
                    try patterns.append(allocator, "pub fn ");
                } else if (std.mem.indexOf(u8, query, "fn ") != null) {
                    try patterns.append(allocator, "fn ");
                }
                try self.extractKeywords(allocator, query, &patterns);
            },
            .find_types => {
                if (std.mem.indexOf(u8, query, "pub const") != null) {
                    try patterns.append(allocator, "pub const ");
                }
                if (std.mem.indexOf(u8, query, "pub struct") != null) {
                    try patterns.append(allocator, "pub struct ");
                }
                if (std.mem.indexOf(u8, query, "pub enum") != null) {
                    try patterns.append(allocator, "pub enum ");
                }
                try self.extractKeywords(allocator, query, &patterns);
            },
            .find_tests => {
                try patterns.append(allocator, "test ");
                try patterns.append(allocator, "_test");
            },
            .find_imports => {
                try patterns.append(allocator, "import ");
                try patterns.append(allocator, "use @");
            },
            .find_comments => {
                try patterns.append(allocator, "//");
                try patterns.append(allocator, "///");
                try patterns.append(allocator, "/*");
                if (std.mem.indexOf(u8, query, "todo") != null) {
                    try patterns.append(allocator, "TODO");
                }
                if (std.mem.indexOf(u8, query, "fixme") != null) {
                    try patterns.append(allocator, "FIXME");
                }
            },
            .find_configs => {
                try patterns.append(allocator, "pub const");
                try patterns.append(allocator, "const CONFIG");
            },
            .find_docs => {
                try patterns.append(allocator, "///");
                try patterns.append(allocator, "//!");
            },
            else => {
                try self.extractKeywords(allocator, query, &patterns);
            },
        }

        return patterns.toOwnedSlice(allocator);
    }

    fn extractKeywords(_: *QueryUnderstanding, allocator: std.mem.Allocator, query: []const u8, patterns: *std.ArrayListUnmanaged([]const u8)) !void {
        var tokens = std.mem.tokenizeAny(u8, query, " \t\n\r.,;:!?\"'()[]{}");
        const valid_keywords = [_][]const u8{ "handler", "controller", "service", "model", "view", "route", "api", "endpoint", "database", "cache", "config", "util", "helper", "parser", "builder", "factory", "singleton", "observer", "strategy", "adapter" };
        while (tokens.next()) |token| {
            if (token.len > 2 and token.len < 50) {
                for (valid_keywords) |kw| {
                    if (std.ascii.eqlIgnoreCase(token, kw)) {
                        try patterns.append(allocator, kw);
                        break;
                    }
                }
            }
        }
    }

    fn extractTargetPaths(_: *QueryUnderstanding, allocator: std.mem.Allocator, query: []const u8) ![]const []const u8 {
        var paths = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (paths.items) |path| {
                allocator.free(path);
            }
            paths.deinit(allocator);
        }

        const path_patterns = [_][]const u8{ "src/", "src/", "tests/", "docs/", "examples/", "lib/", "include/", "bin/", "tools/" };
        for (path_patterns) |pattern| {
            if (std.mem.indexOf(u8, query, pattern)) |idx| {
                var end = idx + pattern.len;
                while (end < query.len and (std.ascii.isAlphanumeric(query[end]) or query[end] == '/' or query[end] == '_' or query[end] == '-')) {
                    end += 1;
                }
                if (end > idx) {
                    const path = query[idx..end];
                    const duped = try allocator.dupe(u8, path);
                    paths.append(allocator, duped) catch |err| {
                        allocator.free(duped);
                        return err;
                    };
                }
            }
        }

        return paths.toOwnedSlice(allocator);
    }

    fn extractFileExtensions(_: *QueryUnderstanding, allocator: std.mem.Allocator, query: []const u8) ![]const []const u8 {
        var extensions = std.ArrayListUnmanaged([]const u8){};
        errdefer extensions.deinit(allocator);

        const ext_map = [_][2][]const u8{
            .{ "zig", ".zig" },
            .{ "rust", ".rs" },
            .{ "c file", ".c" },
            .{ "cpp", ".cpp" },
            .{ "python", ".py" },
            .{ "javascript", ".js" },
            .{ "typescript", ".ts" },
            .{ "go file", ".go" },
            .{ "java", ".java" },
            .{ "rust file", ".rs" },
            .{ "test", "_test.zig" },
            .{ "header", ".h" },
            .{ "markdown", ".md" },
        };

        for (ext_map) |mapping| {
            if (std.mem.indexOf(u8, query, mapping[0]) != null) {
                try extensions.append(allocator, mapping[1]);
            }
        }

        return extensions.toOwnedSlice(allocator);
    }

    fn calculateConfidence(_: *QueryUnderstanding, query: []const u8, intent: QueryIntent) f32 {
        var confidence: f32 = 0.5;

        if (intent != .unknown) confidence += 0.2;

        const natural_indicators = [_][]const u8{ "find", "where", "how", "list", "show", "get", "search" };
        for (natural_indicators) |indicator| {
            if (std.mem.indexOf(u8, query, indicator) != null) {
                confidence += 0.1;
                break;
            }
        }

        const code_indicators = [_][]const u8{ "function", "type", "class", "struct", "enum", "interface", "pub ", "fn ", "const" };
        for (code_indicators) |indicator| {
            if (std.mem.indexOf(u8, query, indicator) != null) {
                confidence += 0.1;
                break;
            }
        }

        return @min(confidence, 1.0);
    }

    pub fn suggestPatterns(self: *QueryUnderstanding, parsed: *const ParsedQuery) ![]SearchPattern {
        var patterns = std.ArrayListUnmanaged(SearchPattern){};
        var compiler = PatternCompiler.init(self.allocator);

        for (parsed.patterns) |pattern_str| {
            const ptype = if (std.mem.indexOfAny(u8, pattern_str, "*?[]") != null) PatternType.glob else PatternType.literal;
            const pattern = try compiler.compile(pattern_str, ptype, false);
            try patterns.append(self.allocator, pattern);
        }

        return patterns.toOwnedSlice(self.allocator);
    }

    /// Free all allocations associated with a parsed query
    /// Call this when done with a ParsedQuery returned from parse()
    pub fn freeParsedQuery(self: *QueryUnderstanding, parsed: ParsedQuery) void {
        // Free patterns array (strings are static literals, only free the array)
        if (parsed.patterns.len > 0) {
            self.allocator.free(parsed.patterns);
        }

        // Free target paths - these were duplicated with allocator.dupe()
        for (parsed.target_paths) |path| {
            self.allocator.free(path);
        }
        if (parsed.target_paths.len > 0) {
            self.allocator.free(parsed.target_paths);
        }

        // Free file extensions array (strings are static literals, only free the array)
        if (parsed.file_extensions.len > 0) {
            self.allocator.free(parsed.file_extensions);
        }
    }
};

test "query understanding patterns are case-insensitive" {
    const allocator = std.testing.allocator;

    var understander = QueryUnderstanding.init(allocator);
    defer understander.deinit();

    const parsed = try understander.parse("FIND PUB FN HANDLER");
    defer understander.freeParsedQuery(parsed);

    var has_pub_fn = false;
    var has_handler = false;
    for (parsed.patterns) |pattern| {
        if (std.mem.eql(u8, pattern, "pub fn ")) has_pub_fn = true;
        if (std.mem.eql(u8, pattern, "handler")) has_handler = true;
    }

    try std.testing.expect(has_pub_fn);
    try std.testing.expect(has_handler);
}
