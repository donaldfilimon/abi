const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const ExploreConfig = @import("config.zig").ExploreConfig;
const ExploreLevel = @import("config.zig").ExploreLevel;
const ExploreResult = @import("results.zig").ExploreResult;
const Match = @import("results.zig").Match;
const MatchType = @import("results.zig").MatchType;
const ExploreError = @import("results.zig").ExploreError;
const ExplorationStats = @import("results.zig").ExplorationStats;
const FileVisitor = @import("fs.zig").FileVisitor;
const FileStats = @import("fs.zig").FileStats;
const SearchPattern = @import("search.zig").SearchPattern;
const PatternCompiler = @import("search.zig").PatternCompiler;
const PatternType = @import("search.zig").PatternType;
const QueryUnderstanding = @import("query.zig").QueryUnderstanding;
const ParsedQuery = @import("query.zig").ParsedQuery;

pub const ExploreAgent = struct {
    allocator: std.mem.Allocator,
    config: ExploreConfig,
    compiler: PatternCompiler,
    stats: ExplorationStats,
    start_time: platform_time.Instant,
    cancelled: bool,
    cancellation_lock: sync.Mutex,
    io_backend: std.Io.Threaded,

    pub fn init(allocator: std.mem.Allocator, config: ExploreConfig) !ExploreAgent {
        return ExploreAgent{
            .allocator = allocator,
            .config = config,
            .compiler = PatternCompiler.init(allocator),
            .stats = ExplorationStats{},
            .start_time = platform_time.Instant.now() catch return error.TimerUnavailable,
            .cancelled = false,
            .cancellation_lock = sync.Mutex{},
            .io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty }),
        };
    }

    /// Clean up resources used by the explore agent
    pub fn deinit(self: *ExploreAgent) void {
        self.io_backend.deinit();
        // Reset stats
        self.stats = ExplorationStats{};
        self.cancelled = false;

        // The compiler is stateless, but we zero out the agent
        // for safety if it's accidentally reused
        self.* = undefined;
    }

    pub fn explore(self: *ExploreAgent, root_path: []const u8, query: []const u8) !ExploreResult {
        self.start_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        self.cancelled = false;

        var result = ExploreResult.init(self.allocator, query, self.config.level);
        errdefer result.deinit();

        var pattern = try self.compiler.compile(query, PatternType.literal, self.config.case_sensitive);
        defer pattern.deinit(self.allocator);

        var visitor = try FileVisitor.init(self.allocator, &self.config);
        defer visitor.deinit();

        visitor.visit(root_path) catch {
            result.explore_error = ExploreError.PathNotFound;
            result.error_message = try std.fmt.allocPrint(self.allocator, "Failed to access path: {s}", .{root_path});
            return result;
        };

        self.stats.files_discovered = visitor.getFileCount();
        result.files_scanned = visitor.getFileCount();

        for (visitor.getFiles()) |file_stat| {
            if (self.isCancelled()) {
                result.cancelled = true;
                break;
            }

            self.processFile(&result, &file_stat, &pattern);
            self.stats.files_processed += 1;
        }

        const end_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        const elapsed_ns = end_time.since(self.start_time);
        result.duration_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));
        self.stats.wall_time_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));

        return result;
    }

    pub fn exploreWithPatterns(self: *ExploreAgent, root_path: []const u8, patterns: []const []const u8) !ExploreResult {
        self.start_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        self.cancelled = false;

        var result = ExploreResult.init(self.allocator, "", self.config.level);
        errdefer result.deinit();

        var compiled_patterns = std.ArrayListUnmanaged(SearchPattern){};
        defer {
            for (compiled_patterns.items) |*p| {
                p.deinit(self.allocator);
            }
            compiled_patterns.deinit(self.allocator);
        }

        for (patterns) |pattern_str| {
            const ptype = if (self.config.use_regex) PatternType.regex else PatternType.literal;
            const pattern = try self.compiler.compile(pattern_str, ptype, self.config.case_sensitive);
            try compiled_patterns.append(self.allocator, pattern);
        }

        var visitor = try FileVisitor.init(self.allocator, &self.config);
        defer visitor.deinit();

        visitor.visit(root_path) catch {
            result.explore_error = ExploreError.PathNotFound;
            return result;
        };

        self.stats.files_discovered = visitor.getFileCount();

        for (visitor.getFiles()) |file_stat| {
            if (self.isCancelled()) {
                result.cancelled = true;
                break;
            }

            for (compiled_patterns.items) |*pattern| {
                self.processFile(&result, &file_stat, pattern);
            }
            self.stats.files_processed += 1;
        }

        const end_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        const elapsed_ns = end_time.since(self.start_time);
        result.duration_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));
        self.stats.wall_time_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));

        return result;
    }

    fn processFile(self: *ExploreAgent, result: *ExploreResult, file_stat: *const FileStats, pattern: *const SearchPattern) void {
        const basename = std.fs.path.basename(file_stat.path);
        const content = self.readFile(file_stat.path) catch {
            self.stats.errors += 1;
            return;
        };
        defer self.allocator.free(content);

        self.stats.total_bytes_read += content.len;

        const match_type = self.determineMatchType(file_stat.path);

        if (self.shouldMatch(basename, pattern)) {
            const relevance = self.calculateRelevance(content, pattern);
            const file_path_dup = self.allocator.dupe(u8, file_stat.path) catch {
                self.stats.errors += 1;
                return;
            };
            const line_content_dup = self.allocator.dupe(u8, "") catch {
                self.allocator.free(file_path_dup);
                self.stats.errors += 1;
                return;
            };
            const match_text_dup = self.allocator.dupe(u8, basename) catch {
                self.allocator.free(file_path_dup);
                self.allocator.free(line_content_dup);
                self.stats.errors += 1;
                return;
            };
            const ctx_before = self.allocator.dupe(u8, "") catch {
                self.allocator.free(file_path_dup);
                self.allocator.free(line_content_dup);
                self.allocator.free(match_text_dup);
                self.stats.errors += 1;
                return;
            };
            const ctx_after = self.allocator.dupe(u8, "") catch {
                self.allocator.free(file_path_dup);
                self.allocator.free(line_content_dup);
                self.allocator.free(match_text_dup);
                self.allocator.free(ctx_before);
                self.stats.errors += 1;
                return;
            };
            var match = Match{
                .file_path = file_path_dup,
                .line_number = 0,
                .line_content = line_content_dup,
                .match_type = match_type,
                .match_text = match_text_dup,
                .relevance_score = relevance,
                .context_before = ctx_before,
                .context_after = ctx_after,
            };
            result.addMatch(match) catch {
                match.deinit(self.allocator);
                self.stats.errors += 1;
                return;
            };
            self.stats.matches_found += 1;
        }

        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];
                if (self.shouldMatch(line, pattern)) {
                    const relevance = self.calculateRelevance(line, pattern);
                    if (relevance > 0.3) {
                        const context_before_raw = self.getContext(content, line_start, 3);
                        const context_after_raw = self.getContext(content, i + 1, 3);

                        const file_path_copy = self.allocator.dupe(u8, file_stat.path) catch {
                            self.stats.errors += 1;
                            continue;
                        };
                        const line_content = self.allocator.dupe(u8, line) catch {
                            self.allocator.free(file_path_copy);
                            self.stats.errors += 1;
                            continue;
                        };
                        const match_text = self.allocator.dupe(u8, line) catch {
                            self.allocator.free(file_path_copy);
                            self.allocator.free(line_content);
                            self.stats.errors += 1;
                            continue;
                        };
                        const context_before = self.allocator.dupe(u8, context_before_raw) catch {
                            self.allocator.free(file_path_copy);
                            self.allocator.free(line_content);
                            self.allocator.free(match_text);
                            self.stats.errors += 1;
                            continue;
                        };
                        const context_after = self.allocator.dupe(u8, context_after_raw) catch {
                            self.allocator.free(file_path_copy);
                            self.allocator.free(line_content);
                            self.allocator.free(match_text);
                            self.allocator.free(context_before);
                            self.stats.errors += 1;
                            continue;
                        };
                        var match = Match{
                            .file_path = file_path_copy,
                            .line_number = line_number,
                            .line_content = line_content,
                            .match_type = match_type,
                            .match_text = match_text,
                            .relevance_score = relevance,
                            .context_before = context_before,
                            .context_after = context_after,
                        };
                        result.addMatch(match) catch {
                            self.stats.errors += 1;
                            match.deinit(self.allocator);
                            continue;
                        };
                        self.stats.matches_found += 1;
                    }
                }
                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn readFile(self: *ExploreAgent, path: []const u8) ![]const u8 {
        const io = self.io_backend.io();
        const max_size = self.config.file_size_limit_bytes orelse (1024 * 1024);

        return std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(max_size)) catch {
            return error.FileNotFound;
        };
    }

    fn shouldMatch(self: *ExploreAgent, text: []const u8, pattern: *const SearchPattern) bool {
        _ = self;
        return std.mem.indexOf(u8, text, pattern.raw) != null;
    }

    /// Calculate a relevance score for a match based on multiple factors.
    /// Returns a score between 0.0 (irrelevant) and 1.0 (highly relevant).
    fn calculateRelevance(self: *ExploreAgent, content: []const u8, pattern: *const SearchPattern) f32 {
        _ = self;
        var score: f32 = 0.0;
        const pat = pattern.raw;

        if (pat.len == 0 or content.len == 0) return 0.0;

        // Count occurrences of pattern in content
        var count: usize = 0;
        var pos: usize = 0;
        while (std.mem.indexOfPos(u8, content, pos, pat)) |idx| {
            count += 1;
            pos = idx + 1;
        }

        if (count == 0) return 0.0;

        // Base score from occurrence count (diminishing returns)
        score += @min(0.3, @as(f32, @floatFromInt(count)) * 0.1);

        // Bonus for exact word boundary matches
        var word_matches: usize = 0;
        pos = 0;
        while (std.mem.indexOfPos(u8, content, pos, pat)) |idx| {
            const before_ok = idx == 0 or !std.ascii.isAlphanumeric(content[idx - 1]);
            const after_idx = idx + pat.len;
            const after_ok = after_idx >= content.len or !std.ascii.isAlphanumeric(content[after_idx]);
            if (before_ok and after_ok) {
                word_matches += 1;
            }
            pos = idx + 1;
        }
        score += @min(0.3, @as(f32, @floatFromInt(word_matches)) * 0.15);

        // Bonus for matches early in content (likely more important)
        if (std.mem.indexOf(u8, content, pat)) |first_pos| {
            if (first_pos < 100) {
                score += 0.15;
            } else if (first_pos < 500) {
                score += 0.1;
            } else if (first_pos < 1000) {
                score += 0.05;
            }
        }

        // Bonus for pattern density (matches per 100 chars)
        if (content.len > 0) {
            const density = @as(f32, @floatFromInt(count * 100)) / @as(f32, @floatFromInt(content.len));
            score += @min(0.2, density * 0.1);
        }

        // Bonus for matching common code patterns
        if (std.mem.indexOf(u8, content, "fn ") != null or
            std.mem.indexOf(u8, content, "pub fn ") != null or
            std.mem.indexOf(u8, content, "const ") != null or
            std.mem.indexOf(u8, content, "var ") != null)
        {
            score += 0.05;
        }

        return @min(1.0, score);
    }

    fn getContext(self: *ExploreAgent, content: []const u8, position: usize, lines: usize) []const u8 {
        _ = self;
        var start = position;
        var end = position;
        var line_count: usize = 0;

        while (start > 0 and line_count < lines) {
            start -= 1;
            if (content[start] == '\n') {
                line_count += 1;
            }
        }

        line_count = 0;
        while (end < content.len and line_count < lines) {
            if (content[end] == '\n') {
                line_count += 1;
            }
            end += 1;
        }

        return content[start..end];
    }

    fn determineMatchType(self: *ExploreAgent, path: []const u8) MatchType {
        _ = self;
        const basename = std.fs.path.basename(path);

        if (std.mem.startsWith(u8, basename, "test") or std.mem.endsWith(u8, basename, "_test.zig")) {
            return .test_case;
        }
        if (std.mem.startsWith(u8, basename, "test") or std.mem.endsWith(u8, basename, ".test")) {
            return .test_case;
        }

        if (std.mem.indexOf(u8, path, "test") != null) {
            return .test_case;
        }

        return .custom;
    }

    fn determineFileType(self: *ExploreAgent, path: []const u8) []const u8 {
        _ = self;
        const ext = std.fs.path.extension(path);

        const source_exts = std.ComptimeStringMap(void, .{
            .{ ".zig", 0 }, .{ ".c", 0 },  .{ ".cpp", 0 }, .{ ".h", 0 },
            .{ ".rs", 0 },  .{ ".go", 0 }, .{ ".py", 0 },  .{ ".js", 0 },
        });

        if (source_exts.has(ext)) return "source";

        const test_exts = std.ComptimeStringMap(void, .{
            .{ ".test", 0 }, .{ ".spec", 0 },
        });

        if (test_exts.has(ext)) return "test";

        return "other";
    }

    pub fn cancel(self: *ExploreAgent) void {
        self.cancellation_lock.lock();
        defer self.cancellation_lock.unlock();
        self.cancelled = true;
        self.stats.cancelled = true;
    }

    pub fn isCancelled(self: *ExploreAgent) bool {
        self.cancellation_lock.lock();
        defer self.cancellation_lock.unlock();
        return self.cancelled;
    }

    pub fn getStats(self: *ExploreAgent) ExplorationStats {
        return self.stats;
    }

    pub fn exploreNaturalLanguage(self: *ExploreAgent, root_path: []const u8, nl_query: []const u8) !ExploreResult {
        self.start_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        self.cancelled = false;

        var result = ExploreResult.init(self.allocator, nl_query, self.config.level);
        errdefer result.deinit();

        var query_understander = QueryUnderstanding.init(self.allocator);
        defer query_understander.deinit();

        const parsed = try query_understander.parse(nl_query);

        var compiled_patterns = std.ArrayListUnmanaged(SearchPattern){};
        defer {
            for (compiled_patterns.items) |*p| {
                p.deinit(self.allocator);
            }
            compiled_patterns.deinit(self.allocator);
        }

        if (parsed.patterns.len == 0) {
            const pattern = try self.compiler.compile(nl_query, PatternType.literal, self.config.case_sensitive);
            try compiled_patterns.append(self.allocator, pattern);
        } else {
            for (parsed.patterns) |pattern_str| {
                const ptype = if (self.config.use_regex or std.mem.indexOfAny(u8, pattern_str, "*?") != null) PatternType.glob else PatternType.literal;
                const pattern = try self.compiler.compile(pattern_str, ptype, self.config.case_sensitive);
                try compiled_patterns.append(self.allocator, pattern);
            }
        }

        var visitor = try FileVisitor.init(self.allocator, &self.config);
        defer visitor.deinit();

        visitor.visit(root_path) catch {
            result.explore_error = ExploreError.PathNotFound;
            const msg = try std.fmt.allocPrint(self.allocator, "Failed to access path: {s}", .{root_path});
            result.error_message = msg;
            return result;
        };

        self.stats.files_discovered = visitor.getFileCount();

        for (visitor.getFiles()) |file_stat| {
            if (self.isCancelled()) {
                result.cancelled = true;
                break;
            }

            const ext = std.fs.path.extension(file_stat.path);

            if (parsed.file_extensions.len > 0) {
                var found = false;
                for (parsed.file_extensions) |target_ext| {
                    if (std.mem.endsWith(u8, file_stat.path, target_ext) or std.mem.eql(u8, ext, target_ext)) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            }

            if (parsed.target_paths.len > 0) {
                var found = false;
                for (parsed.target_paths) |target_path| {
                    if (std.mem.indexOf(u8, file_stat.path, target_path) != null) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            }

            for (compiled_patterns.items) |*pattern| {
                self.processFile(&result, &file_stat, pattern);
            }
            self.stats.files_processed += 1;
        }

        const end_time = platform_time.Instant.now() catch return error.TimerUnavailable;
        const elapsed_ns = end_time.since(self.start_time);
        result.duration_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));
        self.stats.wall_time_ms = @intCast(@divTrunc(elapsed_ns, std.time.ns_per_ms));

        return result;
    }
};

pub fn createDefaultAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return ExploreAgent.init(allocator, ExploreConfig.defaultForLevel(.medium));
}

pub fn createQuickAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return ExploreAgent.init(allocator, ExploreConfig.defaultForLevel(.quick));
}

pub fn createThoroughAgent(allocator: std.mem.Allocator) !ExploreAgent {
    return ExploreAgent.init(allocator, ExploreConfig.defaultForLevel(.thorough));
}
