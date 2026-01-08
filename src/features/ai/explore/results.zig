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
            .match_types = std.enums.EnumArray(MatchType, usize).init(0),
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
            .matches = std.ArrayListUnmanaged(Match){},
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

        var iterator = self.file_summaries.valueIterator();
        while (iterator.next()) |summary| {
            self.allocator.free(summary.path);
            self.allocator.free(summary.file_type);
        }
        self.file_summaries.deinit(self.allocator);

        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }

    pub fn addMatch(self: *ExploreResult, match: Match) !void {
        try self.matches.append(self.allocator, match);
        self.matches_found += 1;

        const dir = std.fs.path.dirname(match.file_path) orelse "";
        if (self.file_summaries.get(dir)) |*summary| {
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
        try writer.print("Level: {s}\n", .{@tagName(self.level)});
        try writer.print("Files Scanned: {d}\n", .{self.files_scanned});
        try writer.print("Matches Found: {d}\n", .{self.matches_found});
        try writer.print("Duration: {d}ms\n\n", .{self.duration_ms});

        if (self.explore_error) |err| {
            try writer.print("Error: {s}\n", .{@errorName(err)});
            if (self.error_message) |msg| {
                try writer.print("Details: {s}\n", .{msg});
            }
            return;
        }

        try writer.writeAll("Top Matches:\n");
        try writer.writeAll("-------------\n");

        const top_matches = self.getTopMatches(20);
        defer self.allocator.free(top_matches);

        for (top_matches, 0..) |match, i| {
            try writer.print("{d}. {s}:{d}\n", .{ i + 1, match.file_path, match.line_number });
            try writer.print("   {s}\n", .{match.match_text});
            try writer.print("   Score: {d:.2}\n\n", .{match.relevance_score});
        }
    }

    pub fn formatJSON(self: *ExploreResult, writer: anytype) !void {
        var obj = json.Object.init(self.allocator);
        defer obj.deinit();

        try obj.put("query", json.Value{ .string = self.query });
        try obj.put("level", json.Value{ .string = @tagName(self.level) });
        try obj.put("files_scanned", json.Value{ .integer = @as(i64, @intCast(self.files_scanned)) });
        try obj.put("matches_found", json.Value{ .integer = @as(i64, @intCast(self.matches_found)) });
        try obj.put("duration_ms", json.Value{ .integer = @as(i64, @intCast(self.duration_ms)) });
        try obj.put("cancelled", json.Value{ .bool = self.cancelled });

        if (self.explore_error) |err| {
            try obj.put("error", json.Value{ .string = @errorName(err) });
        }

        var matches_arr = json.Array.init(self.allocator);
        for (self.matches.items) |match| {
            var match_obj = json.Object.init(self.allocator);
            match_obj.put("file", json.Value{ .string = match.file_path }) catch {
                match_obj.deinit();
                continue;
            };
            match_obj.put("line", json.Value{ .integer = @as(i64, @intCast(match.line_number)) }) catch {
                match_obj.deinit();
                continue;
            };
            match_obj.put("type", json.Value{ .string = @tagName(match.match_type) }) catch {
                match_obj.deinit();
                continue;
            };
            match_obj.put("text", json.Value{ .string = match.match_text }) catch {
                match_obj.deinit();
                continue;
            };
            match_obj.put("score", json.Value{ .number = match.relevance_score }) catch {
                match_obj.deinit();
                continue;
            };
            matches_arr.append(json.Value{ .object = match_obj }) catch |err| {
                match_obj.deinit();
                return err;
            };
        }
        try obj.put("matches", json.Value{ .array = matches_arr });

        try json.stringify(json.Value{ .object = obj }, .{}, writer);
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
