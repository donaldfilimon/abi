const std = @import("std");
const json = std.json;
const ExploreLevel = @import("config.zig").ExploreLevel;

pub const MatchType = enum {
    file_name,
    import_statement,
    function_definition,
    type_definition,
    variable_declaration,
    comment,
    doc_comment,
    test_case,
    config_key,
    url,
    error_pattern,
    custom,
};

pub const Match = struct {
    file_path: []const u8,
    line_number: usize,
    line_content: []const u8,
    match_type: MatchType,
    match_text: []const u8,
    relevance_score: f32,
    context_before: []const u8,
    context_after: []const u8,
    capture_groups: ?[][]const u8 = null,

    pub fn deinit(self: *Match, allocator: std.mem.Allocator) void {
        allocator.free(self.file_path);
        allocator.free(self.line_content);
        allocator.free(self.match_text);
        allocator.free(self.context_before);
        allocator.free(self.context_after);
        if (self.capture_groups) |groups| {
            for (groups) |g| {
                allocator.free(g);
            }
            allocator.free(groups);
        }
    }
};

pub const FileSummary = struct {
    path: []const u8,
    total_matches: usize,
    match_types: std.enums.EnumArray(MatchType, usize),
    relevance_score: f32,
    file_type: []const u8,
    size_bytes: u64,

    pub fn init(_: std.mem.Allocator, file_path: []const u8) FileSummary {
        return FileSummary{
            .path = file_path,
            .total_matches = 0,
            .match_types = std.enums.EnumArray(MatchType, usize).initFill(0),
            .relevance_score = 0.0,
            .file_type = "",
            .size_bytes = 0,
        };
    }
};

pub const ExploreError = error{
    PathNotFound,
    PermissionDenied,
    Timeout,
    Cancelled,
    TooManyFiles,
    InvalidPattern,
    InvalidQuery,
    MemoryAllocationFailed,
    InvalidConfiguration,
    UnsupportedOperation,
};

pub const ExploreResult = struct {
    allocator: std.mem.Allocator,
    query: []const u8,
    level: ExploreLevel,
    files_scanned: usize,
    matches_found: usize,
    duration_ms: u64,
    matches: std.ArrayListUnmanaged(Match),
    file_summaries: std.StringHashMapUnmanaged(FileSummary),
    explore_error: ?ExploreError,
    error_message: ?[]const u8,
    cancelled: bool,

    pub fn init(allocator: std.mem.Allocator, query: []const u8, level: ExploreLevel) ExploreResult {
        return ExploreResult{
            .allocator = allocator,
            .query = query,
            .level = level,
            .files_scanned = 0,
            .matches_found = 0,
            .duration_ms = 0,
            .matches = std.ArrayListUnmanaged(Match).empty,
            .file_summaries = std.StringHashMapUnmanaged(FileSummary){},
            .explore_error = null,
            .error_message = null,
            .cancelled = false,
        };
    }

    pub fn deinit(self: *ExploreResult) void {
        for (self.matches.items) |*match| {
            match.deinit(self.allocator);
        }
        self.matches.deinit(self.allocator);

        // file_summaries paths and file_types are not owned (they reference match paths)
        self.file_summaries.deinit(self.allocator);

        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }

    pub fn addMatch(self: *ExploreResult, match: Match) !void {
        try self.matches.append(self.allocator, match);
        self.matches_found += 1;

        const dir = std.Io.Dir.path.dirname(match.file_path) orelse "";
        if (self.file_summaries.getPtr(dir)) |summary| {
            summary.total_matches += 1;
            summary.match_types.set(match.match_type, summary.match_types.get(match.match_type) + 1);
        } else {
            var new_summary = FileSummary.init(self.allocator, dir);
            new_summary.total_matches = 1;
            new_summary.match_types.set(match.match_type, 1);
            try self.file_summaries.put(self.allocator, dir, new_summary);
        }
    }

    pub fn formatHuman(self: *ExploreResult, writer: anytype) !void {
        try writer.print("Exploration Results for: \"{s}\"\n", .{self.query});
        try writer.print("Level: {t}\n", .{self.level});
        try writer.print("Files Scanned: {d}\n", .{self.files_scanned});
        try writer.print("Matches Found: {d}\n", .{self.matches_found});
        try writer.print("Duration: {d}ms\n\n", .{self.duration_ms});

        if (self.explore_error) |err| {
            try writer.print("Error: {t}\n", .{err});
            if (self.error_message) |msg| {
                try writer.print("Details: {s}\n", .{msg});
            }
            return;
        }

        try writer.print("Top Matches:\n", .{});
        try writer.print("-------------\n", .{});

        const top_matches = self.getTopMatches(20);
        // Note: top_matches is a slice view into self.matches.items, not a separate allocation

        for (top_matches, 0..) |match, i| {
            try writer.print("{d}. {s}:{d}\n", .{ i + 1, match.file_path, match.line_number });
            try writer.print("   {s}\n", .{match.match_text});
            try writer.print("   Score: {d:.2}\n\n", .{match.relevance_score});
        }
    }

    pub fn formatJSON(self: *ExploreResult, writer: anytype) !void {
        try writer.print("{{", .{});
        try writer.print("\"query\":\"{s}\",", .{self.query});
        try writer.print("\"level\":\"{t}\",", .{self.level});
        try writer.print("\"files_scanned\":{d},", .{self.files_scanned});
        try writer.print("\"matches_found\":{d},", .{self.matches_found});
        try writer.print("\"duration_ms\":{d},", .{self.duration_ms});
        try writer.print("\"cancelled\":{},", .{self.cancelled});

        if (self.explore_error) |err| {
            try writer.print("\"error\":\"{t}\",", .{err});
        }

        try writer.print("\"matches\":[", .{});
        for (self.matches.items, 0..) |match, idx| {
            if (idx > 0) try writer.print(",", .{});
            try writer.print("{{", .{});
            try writer.print("\"file\":\"{s}\",", .{match.file_path});
            try writer.print("\"line\":{d},", .{match.line_number});
            try writer.print("\"type\":\"{t}\",", .{match.match_type});
            try writer.print("\"score\":{d:.2}", .{match.relevance_score});
            try writer.print("}}", .{});
        }
        try writer.print("]}}\n", .{});
    }

    fn getTopMatches(self: *ExploreResult, limit: usize) []Match {
        const count = @min(self.matches.items.len, limit);
        const sorted = self.matches.items[0..count];
        std.mem.sort(Match, sorted, {}, struct {
            fn less(_: void, a: Match, b: Match) bool {
                return a.relevance_score > b.relevance_score;
            }
        }.less);
        return sorted;
    }
};

pub const ExplorationStats = struct {
    files_discovered: usize = 0,
    files_processed: usize = 0,
    total_bytes_read: u64 = 0,
    matches_found: usize = 0,
    cpu_time_ms: u64 = 0,
    wall_time_ms: u64 = 0,
    cache_hits: usize = 0,
    cache_misses: usize = 0,
    errors: usize = 0,
    cancelled: bool = false,

    pub fn merge(self: *ExplorationStats, other: ExplorationStats) void {
        self.files_discovered += other.files_discovered;
        self.files_processed += other.files_processed;
        self.total_bytes_read += other.total_bytes_read;
        self.matches_found += other.matches_found;
        self.cpu_time_ms += other.cpu_time_ms;
        self.wall_time_ms = @max(self.wall_time_ms, other.wall_time_ms);
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.errors += other.errors;
        self.cancelled = self.cancelled or other.cancelled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
