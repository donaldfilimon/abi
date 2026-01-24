const std = @import("std");
const ExploreConfig = @import("config.zig").ExploreConfig;
const ExploreResult = @import("results.zig").ExploreResult;
const FileStats = @import("fs.zig").FileStats;
const SearchPattern = @import("search.zig").SearchPattern;
const PatternCompiler = @import("search.zig").PatternCompiler;
const fs = @import("fs.zig");

pub const WorkItem = struct {
    file_stat: FileStats,
    depth: usize,
};

pub const ParallelExplorer = struct {
    allocator: std.mem.Allocator,
    config: ExploreConfig,
    result: *ExploreResult,
    patterns: []const SearchPattern,
    should_cancel: std.atomic.Value(bool),
    processed: std.atomic.Value(usize),
    lock: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: ExploreConfig, result: *ExploreResult, patterns: []const SearchPattern) ParallelExplorer {
        return ParallelExplorer{
            .allocator = allocator,
            .config = config,
            .result = result,
            .patterns = patterns,
            .should_cancel = std.atomic.Value(bool).init(false),
            .processed = std.atomic.Value(usize).init(0),
            .lock = std.Thread.Mutex{},
        };
    }

    pub fn explore(self: *ParallelExplorer, files: []const FileStats) !void {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = self.config.worker_count orelse cpu_count;
        const effective_workers = @min(worker_count, files.len);

        if (effective_workers <= 1) {
            for (files) |file_stat| {
                if (self.should_cancel.load(.acquire)) break;
                try self.processFile(&file_stat);
                self.markProcessed();
            }
            return;
        }

        const chunks = try self.allocator.alloc([]const FileStats, effective_workers);
        defer self.allocator.free(chunks);

        const chunk_size = (files.len + effective_workers - 1) / effective_workers;

        for (chunks, 0..) |*chunk, i| {
            const start = i * chunk_size;
            const end = @min(start + chunk_size, files.len);
            if (start < end) {
                chunk.* = files[start..end];
            } else {
                chunk.* = &.{};
            }
        }

        const threads = try self.allocator.alloc(std.Thread, effective_workers);
        defer self.allocator.free(threads);

        var spawn_count: usize = 0;
        for (threads, chunks) |*thread, chunk| {
            if (chunk.len == 0) continue;

            thread.* = std.Thread.spawn(.{}, workerThread, .{ self, chunk }) catch |err| {
                std.debug.print("Failed to spawn worker thread: {}\n", .{err});
                // Process files on main thread as fallback
                for (chunk) |file_stat| {
                    self.lock.lock();
                    self.processFile(&file_stat) catch |process_err| {
                        std.log.warn("Failed to process file '{s}': {t}", .{
                            file_stat.path,
                            process_err,
                        });
                    };
                    self.lock.unlock();
                    self.markProcessed();
                }
                continue;
            };
            spawn_count += 1;
        }

        var thread_idx: usize = 0;
        while (thread_idx < spawn_count) {
            threads[thread_idx].join();
            thread_idx += 1;
        }
    }

    fn workerThread(self: *ParallelExplorer, files: []const FileStats) void {
        for (files) |file_stat| {
            if (self.should_cancel.load(.acquire)) return;

            self.lock.lock();
            self.processFile(&file_stat) catch |err| {
                std.log.warn("Failed to process file '{s}': {t}", .{ file_stat.path, err });
            };
            self.lock.unlock();

            self.markProcessed();
        }
    }

    fn processFile(self: *ParallelExplorer, file_stat: *const FileStats) !void {
        for (self.patterns) |*pattern| {
            const content = try fs.readFileContent(self.allocator, file_stat.path, self.config.file_size_limit_bytes);
            defer self.allocator.free(content);

            if (content.len == 0) continue;

            if (try self.shouldMatch(file_stat.path, content, pattern)) {
                try self.recordMatch(file_stat.path, content, pattern);
            }
        }
    }

    fn shouldMatch(self: *ParallelExplorer, file_path: []const u8, _: []const u8, _: *const SearchPattern) !bool {
        var extension = std.fs.path.extension(file_path);
        if (extension.len == 0) extension = file_path;

        var ext_match: bool = false;
        for (self.config.include_patterns) |pat| {
            if (try self.matchesExtension(extension, pat)) {
                ext_match = true;
                break;
            }
        }
        if (!ext_match and self.config.include_patterns.len == 0) {
            ext_match = true;
        }
        if (!ext_match) return false;

        return true;
    }

    fn matchesExtension(_: *ParallelExplorer, extension: []const u8, pattern: []const u8) !bool {
        if (std.mem.startsWith(u8, pattern, ".")) {
            return std.mem.eql(u8, extension, pattern);
        }
        if (std.mem.indexOf(u8, pattern, "*") != null) {
            return fs.matchesGlob(pattern, extension);
        }
        return false;
    }

    fn recordMatch(self: *ParallelExplorer, file_path: []const u8, content: []const u8, pattern: *const SearchPattern) !void {
        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];
                if (try self.matchesPattern(pattern, line)) {
                    try self.addMatch(file_path, line_number, line);
                }
                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn matchesPattern(_: *ParallelExplorer, pattern: *const SearchPattern, text: []const u8) !bool {
        const search = @import("search.zig");
        return search.match(pattern.*, text);
    }

    fn addMatch(self: *ParallelExplorer, file_path: []const u8, line_number: usize, line_content: []const u8) !void {
        const Match = @import("results.zig").Match;

        const match = Match{
            .file_path = try self.allocator.dupe(u8, file_path),
            .line_number = line_number,
            .line_content = try self.allocator.dupe(u8, line_content[0..@min(line_content.len, 100)]),
            .match_type = .custom,
            .match_text = try self.allocator.dupe(u8, line_content[0..@min(line_content.len, 100)]),
            .relevance_score = 0.5,
            .context_before = "",
            .context_after = "",
        };

        self.lock.lock();
        try self.result.matches.append(self.allocator, match);
        self.result.matches_found += 1;
        self.lock.unlock();
    }

    pub fn cancel(self: *ParallelExplorer) void {
        self.should_cancel.store(true, .release);
    }

    pub fn markProcessed(self: *ParallelExplorer) void {
        self.processed.fetchAdd(1, .release);
    }

    pub fn getProcessedCount(self: *ParallelExplorer) usize {
        return self.processed.load(.acquire);
    }
};

pub fn parallelExplore(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    config: ExploreConfig,
    query: []const u8,
) !ExploreResult {
    var result = ExploreResult.init(allocator, query, config.level);
    errdefer result.deinit();

    var patterns = std.ArrayListUnmanaged(SearchPattern){};
    defer {
        for (patterns.items) |*p| {
            p.deinit(allocator);
        }
        patterns.deinit(allocator);
    }

    var compiler = PatternCompiler.init(allocator);
    const pattern = try compiler.compile(query, .literal, config.case_sensitive);
    try patterns.append(allocator, pattern);

    var visitor = try fs.FileVisitor.init(allocator, &config);
    defer visitor.deinit();

    visitor.visit(root_path) catch {
        result.explore_error = ExploreResult.ExploreError.PathNotFound;
        const msg = try std.fmt.allocPrint(allocator, "Failed to access path: {s}", .{root_path});
        result.error_message = msg;
        return result;
    };

    const files = visitor.getFiles();

    var explorer = ParallelExplorer.init(allocator, config, &result, patterns.items);
    defer explorer.* = undefined;

    var timer = std.time.Timer.start() catch return error.TimerFailed;

    try explorer.explore(files);

    const elapsed_ns = timer.read();
    result.duration_ms = @divTrunc(@as(i128, @intCast(elapsed_ns)), std.time.ns_per_ms);

    return result;
}
