//! Benchmark Baseline Persistence Store
//!
//! Provides JSON-based baseline storage for benchmark regression detection:
//! - Git-friendly JSON format with sorted keys
//! - Branch-specific baseline storage
//! - Atomic file operations for consistency
//! - Automatic directory structure management
//!
//! ## Directory Structure
//!
//! ```
//! benchmarks/baselines/
//! ├── main/                    # Main branch baselines
//! │   ├── simd.json
//! │   ├── database.json
//! │   └── ...
//! ├── releases/                # Release tag baselines
//! │   ├── v1.0.0/
//! │   └── ...
//! └── branches/                # Feature branch baselines
//!     ├── feature-xyz/
//!     └── ...
//! ```
//!
//! ## Usage
//!
//! ```zig
//! var store = BaselineStore.init(allocator, "benchmarks/baselines");
//! defer store.deinit();
//!
//! // Save a baseline
//! try store.saveBaseline(.{
//!     .name = "vector_dot_128",
//!     .metric = "ops_per_sec",
//!     .value = 1500000.0,
//!     .unit = "ops/s",
//!     .timestamp = std.time.timestamp(),
//!     .git_commit = "abc123",
//!     .git_branch = "main",
//! });
//!
//! // Load and compare
//! if (try store.loadBaseline("vector_dot_128")) |baseline| {
//!     const change = try store.compareToBaseline(current_result);
//! }
//! ```

const std = @import("std");

/// A single benchmark result that can be persisted as a baseline
pub const BenchmarkResult = struct {
    /// Name of the benchmark (e.g., "vector_dot_128")
    name: []const u8,
    /// Metric being measured (e.g., "ops_per_sec", "latency_ns", "throughput_mbps")
    metric: []const u8,
    /// The measured value
    value: f64,
    /// Unit of measurement (e.g., "ops/s", "ns", "MB/s")
    unit: []const u8,
    /// Unix timestamp when the benchmark was run
    timestamp: i64,
    /// Git commit SHA (if available)
    git_commit: ?[]const u8 = null,
    /// Git branch name (if available)
    git_branch: ?[]const u8 = null,
    /// Category for grouping benchmarks
    category: ?[]const u8 = null,
    /// Statistical standard deviation (if available)
    std_dev: ?f64 = null,
    /// Number of samples/iterations
    sample_count: ?u64 = null,
    /// P99 latency in nanoseconds (if applicable)
    p99_ns: ?u64 = null,
    /// Memory allocated in bytes (if tracked)
    memory_bytes: ?u64 = null,

    /// Parse a BenchmarkResult from a JSON Value
    pub fn fromJson(allocator: std.mem.Allocator, json: std.json.Value) !BenchmarkResult {
        const obj = switch (json) {
            .object => |o| o,
            else => return error.InvalidJsonFormat,
        };

        // Required fields
        const name = blk: {
            const val = obj.get("name") orelse return error.MissingRequiredField;
            break :blk switch (val) {
                .string => |s| try allocator.dupe(u8, s),
                else => return error.InvalidFieldType,
            };
        };
        errdefer allocator.free(name);

        const metric = blk: {
            const val = obj.get("metric") orelse return error.MissingRequiredField;
            break :blk switch (val) {
                .string => |s| try allocator.dupe(u8, s),
                else => return error.InvalidFieldType,
            };
        };
        errdefer allocator.free(metric);

        const value = blk: {
            const val = obj.get("value") orelse return error.MissingRequiredField;
            break :blk switch (val) {
                .float => |f| f,
                .integer => |i| @as(f64, @floatFromInt(i)),
                else => return error.InvalidFieldType,
            };
        };

        const unit = blk: {
            const val = obj.get("unit") orelse return error.MissingRequiredField;
            break :blk switch (val) {
                .string => |s| try allocator.dupe(u8, s),
                else => return error.InvalidFieldType,
            };
        };
        errdefer allocator.free(unit);

        const timestamp = blk: {
            const val = obj.get("timestamp") orelse return error.MissingRequiredField;
            break :blk switch (val) {
                .integer => |i| i,
                else => return error.InvalidFieldType,
            };
        };

        // Optional fields
        const git_commit: ?[]const u8 = if (obj.get("git_commit")) |val| switch (val) {
            .string => |s| try allocator.dupe(u8, s),
            .null => null,
            else => return error.InvalidFieldType,
        } else null;
        errdefer if (git_commit) |gc| allocator.free(gc);

        const git_branch: ?[]const u8 = if (obj.get("git_branch")) |val| switch (val) {
            .string => |s| try allocator.dupe(u8, s),
            .null => null,
            else => return error.InvalidFieldType,
        } else null;
        errdefer if (git_branch) |gb| allocator.free(gb);

        const category: ?[]const u8 = if (obj.get("category")) |val| switch (val) {
            .string => |s| try allocator.dupe(u8, s),
            .null => null,
            else => return error.InvalidFieldType,
        } else null;
        errdefer if (category) |c| allocator.free(c);

        const std_dev: ?f64 = if (obj.get("std_dev")) |val| switch (val) {
            .float => |f| f,
            .integer => |i| @as(f64, @floatFromInt(i)),
            .null => null,
            else => return error.InvalidFieldType,
        } else null;

        const sample_count: ?u64 = if (obj.get("sample_count")) |val| switch (val) {
            .integer => |i| if (i >= 0) @as(u64, @intCast(i)) else null,
            .null => null,
            else => return error.InvalidFieldType,
        } else null;

        const p99_ns: ?u64 = if (obj.get("p99_ns")) |val| switch (val) {
            .integer => |i| if (i >= 0) @as(u64, @intCast(i)) else null,
            .null => null,
            else => return error.InvalidFieldType,
        } else null;

        const memory_bytes: ?u64 = if (obj.get("memory_bytes")) |val| switch (val) {
            .integer => |i| if (i >= 0) @as(u64, @intCast(i)) else null,
            .null => null,
            else => return error.InvalidFieldType,
        } else null;

        return .{
            .name = name,
            .metric = metric,
            .value = value,
            .unit = unit,
            .timestamp = timestamp,
            .git_commit = git_commit,
            .git_branch = git_branch,
            .category = category,
            .std_dev = std_dev,
            .sample_count = sample_count,
            .p99_ns = p99_ns,
            .memory_bytes = memory_bytes,
        };
    }

    /// Convert to JSON string
    pub fn toJsonString(self: BenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n");

        // Required fields (sorted alphabetically for git-friendly diffs)
        if (self.category) |cat| {
            try buf.appendSlice(allocator, "  \"category\": \"");
            try buf.appendSlice(allocator, cat);
            try buf.appendSlice(allocator, "\",\n");
        }

        if (self.git_branch) |branch| {
            try buf.appendSlice(allocator, "  \"git_branch\": \"");
            try buf.appendSlice(allocator, branch);
            try buf.appendSlice(allocator, "\",\n");
        }

        if (self.git_commit) |commit| {
            try buf.appendSlice(allocator, "  \"git_commit\": \"");
            try buf.appendSlice(allocator, commit);
            try buf.appendSlice(allocator, "\",\n");
        }

        if (self.memory_bytes) |mem| {
            const mem_str = try std.fmt.allocPrint(allocator, "  \"memory_bytes\": {d},\n", .{mem});
            defer allocator.free(mem_str);
            try buf.appendSlice(allocator, mem_str);
        }

        try buf.appendSlice(allocator, "  \"metric\": \"");
        try buf.appendSlice(allocator, self.metric);
        try buf.appendSlice(allocator, "\",\n");

        try buf.appendSlice(allocator, "  \"name\": \"");
        try buf.appendSlice(allocator, self.name);
        try buf.appendSlice(allocator, "\",\n");

        if (self.p99_ns) |p99| {
            const p99_str = try std.fmt.allocPrint(allocator, "  \"p99_ns\": {d},\n", .{p99});
            defer allocator.free(p99_str);
            try buf.appendSlice(allocator, p99_str);
        }

        if (self.sample_count) |count| {
            const count_str = try std.fmt.allocPrint(allocator, "  \"sample_count\": {d},\n", .{count});
            defer allocator.free(count_str);
            try buf.appendSlice(allocator, count_str);
        }

        if (self.std_dev) |sd| {
            const sd_str = try std.fmt.allocPrint(allocator, "  \"std_dev\": {d:.6},\n", .{sd});
            defer allocator.free(sd_str);
            try buf.appendSlice(allocator, sd_str);
        }

        const ts_str = try std.fmt.allocPrint(allocator, "  \"timestamp\": {d},\n", .{self.timestamp});
        defer allocator.free(ts_str);
        try buf.appendSlice(allocator, ts_str);

        try buf.appendSlice(allocator, "  \"unit\": \"");
        try buf.appendSlice(allocator, self.unit);
        try buf.appendSlice(allocator, "\",\n");

        const val_str = try std.fmt.allocPrint(allocator, "  \"value\": {d:.6}\n", .{self.value});
        defer allocator.free(val_str);
        try buf.appendSlice(allocator, val_str);

        try buf.appendSlice(allocator, "}");

        return buf.toOwnedSlice(allocator);
    }

    /// Free allocated memory for this result
    pub fn deinit(self: *BenchmarkResult, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.metric);
        allocator.free(self.unit);
        if (self.git_commit) |gc| allocator.free(gc);
        if (self.git_branch) |gb| allocator.free(gb);
        if (self.category) |c| allocator.free(c);
    }
};

/// Baseline store for persisting and loading benchmark baselines
pub const BaselineStore = struct {
    allocator: std.mem.Allocator,
    baselines_dir: []const u8,
    /// Cached baselines (lazy-loaded)
    cache: std.StringHashMapUnmanaged(BenchmarkResult),
    /// Whether cache has been populated
    cache_loaded: bool,

    /// Initialize a new baseline store
    pub fn init(allocator: std.mem.Allocator, baselines_dir: []const u8) BaselineStore {
        return .{
            .allocator = allocator,
            .baselines_dir = baselines_dir,
            .cache = .{},
            .cache_loaded = false,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *BaselineStore) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var result = entry.value_ptr.*;
            result.deinit(self.allocator);
        }
        self.cache.deinit(self.allocator);
    }

    /// Get the file path for a baseline
    fn getBaselineFilePath(
        self: *BaselineStore,
        benchmark_name: []const u8,
        branch: ?[]const u8,
    ) ![]u8 {
        var path_buf = std.ArrayListUnmanaged(u8){};
        errdefer path_buf.deinit(self.allocator);

        try path_buf.appendSlice(self.allocator, self.baselines_dir);
        try path_buf.appendSlice(self.allocator, "/");

        if (branch) |b| {
            if (std.mem.eql(u8, b, "main") or std.mem.eql(u8, b, "master")) {
                try path_buf.appendSlice(self.allocator, "main/");
            } else if (std.mem.startsWith(u8, b, "v") or std.mem.startsWith(u8, b, "release")) {
                try path_buf.appendSlice(self.allocator, "releases/");
                try path_buf.appendSlice(self.allocator, b);
                try path_buf.appendSlice(self.allocator, "/");
            } else {
                try path_buf.appendSlice(self.allocator, "branches/");
                // Sanitize branch name for filesystem
                for (b) |c| {
                    if (c == '/' or c == '\\' or c == ':' or c == '*' or c == '?' or c == '"' or c == '<' or c == '>' or c == '|') {
                        try path_buf.append(self.allocator, '_');
                    } else {
                        try path_buf.append(self.allocator, c);
                    }
                }
                try path_buf.appendSlice(self.allocator, "/");
            }
        } else {
            try path_buf.appendSlice(self.allocator, "main/");
        }

        // Sanitize benchmark name for filename
        for (benchmark_name) |c| {
            if (c == '/' or c == '\\' or c == ':' or c == '*' or c == '?' or c == '"' or c == '<' or c == '>' or c == '|' or c == ' ') {
                try path_buf.append(self.allocator, '_');
            } else {
                try path_buf.append(self.allocator, c);
            }
        }
        try path_buf.appendSlice(self.allocator, ".json");

        return path_buf.toOwnedSlice(self.allocator);
    }

    /// Load baseline for a specific benchmark
    pub fn loadBaseline(self: *BaselineStore, benchmark_name: []const u8) !?BenchmarkResult {
        return self.loadBaselineForBranch(benchmark_name, null);
    }

    /// Load baseline for a specific benchmark and branch
    pub fn loadBaselineForBranch(
        self: *BaselineStore,
        benchmark_name: []const u8,
        branch: ?[]const u8,
    ) !?BenchmarkResult {
        const file_path = try self.getBaselineFilePath(benchmark_name, branch);
        defer self.allocator.free(file_path);

        // Read file content
        const content = self.readFileContent(file_path) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        defer self.allocator.free(content);

        // Parse JSON
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch {
            return error.InvalidJsonFormat;
        };
        defer parsed.deinit();

        return try BenchmarkResult.fromJson(self.allocator, parsed.value);
    }

    /// Read file content using Zig 0.16 I/O
    fn readFileContent(self: *BaselineStore, path: []const u8) ![]u8 {
        // For simplicity, use synchronous file reading
        // In a real implementation, you might want async I/O
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return error.IoError,
        };
        defer file.close();

        const stat = file.stat() catch return error.IoError;
        if (stat.size > 10 * 1024 * 1024) return error.FileTooLarge; // 10MB limit

        const content = self.allocator.alloc(u8, stat.size) catch return error.OutOfMemory;
        errdefer self.allocator.free(content);

        const bytes_read = file.readAll(content) catch return error.IoError;
        if (bytes_read != stat.size) {
            self.allocator.free(content);
            return error.IoError;
        }

        return content;
    }

    /// Save a new baseline
    pub fn saveBaseline(self: *BaselineStore, result: BenchmarkResult) !void {
        const file_path = try self.getBaselineFilePath(result.name, result.git_branch);
        defer self.allocator.free(file_path);

        // Ensure directory exists
        const dir_end = std.mem.lastIndexOf(u8, file_path, "/") orelse return error.InvalidPath;
        const dir_path = file_path[0..dir_end];

        std.fs.cwd().makePath(dir_path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return error.IoError,
        };

        // Generate JSON content
        const json_content = try result.toJsonString(self.allocator);
        defer self.allocator.free(json_content);

        // Write to file atomically (write to temp, then rename)
        var tmp_path_buf: [512]u8 = undefined;
        const tmp_path = std.fmt.bufPrint(&tmp_path_buf, "{s}.tmp", .{file_path}) catch return error.PathTooLong;

        const file = std.fs.cwd().createFile(tmp_path, .{ .truncate = true }) catch return error.IoError;

        file.writeAll(json_content) catch {
            file.close();
            std.fs.cwd().deleteFile(tmp_path) catch {};
            return error.IoError;
        };
        file.close();

        // Atomic rename
        std.fs.cwd().rename(tmp_path, file_path) catch {
            std.fs.cwd().deleteFile(tmp_path) catch {};
            return error.IoError;
        };
    }

    /// Compare result against baseline, returns percentage change
    /// Positive value means improvement (higher is better for throughput metrics)
    /// Negative value means regression
    pub fn compareToBaseline(self: *BaselineStore, result: BenchmarkResult) !?f64 {
        const baseline = try self.loadBaselineForBranch(result.name, result.git_branch) orelse {
            // Try main branch as fallback
            if (result.git_branch != null and
                !std.mem.eql(u8, result.git_branch.?, "main") and
                !std.mem.eql(u8, result.git_branch.?, "master"))
            {
                const main_baseline = try self.loadBaselineForBranch(result.name, "main") orelse return null;
                defer {
                    var b = main_baseline;
                    b.deinit(self.allocator);
                }
                return calculatePercentChange(main_baseline.value, result.value, result.metric);
            }
            return null;
        };
        defer {
            var b = baseline;
            b.deinit(self.allocator);
        }

        return calculatePercentChange(baseline.value, result.value, result.metric);
    }

    /// Load all baselines for a branch
    pub fn loadAllBaselines(
        self: *BaselineStore,
        branch: []const u8,
        allocator: std.mem.Allocator,
    ) ![]BenchmarkResult {
        var results = std.ArrayListUnmanaged(BenchmarkResult){};
        errdefer {
            for (results.items) |*r| r.deinit(allocator);
            results.deinit(allocator);
        }

        // Determine directory path
        var dir_path_buf = std.ArrayListUnmanaged(u8){};
        defer dir_path_buf.deinit(self.allocator);

        try dir_path_buf.appendSlice(self.allocator, self.baselines_dir);
        try dir_path_buf.appendSlice(self.allocator, "/");

        if (std.mem.eql(u8, branch, "main") or std.mem.eql(u8, branch, "master")) {
            try dir_path_buf.appendSlice(self.allocator, "main");
        } else if (std.mem.startsWith(u8, branch, "v") or std.mem.startsWith(u8, branch, "release")) {
            try dir_path_buf.appendSlice(self.allocator, "releases/");
            try dir_path_buf.appendSlice(self.allocator, branch);
        } else {
            try dir_path_buf.appendSlice(self.allocator, "branches/");
            for (branch) |c| {
                if (c == '/' or c == '\\' or c == ':' or c == '*' or c == '?' or c == '"' or c == '<' or c == '>' or c == '|') {
                    try dir_path_buf.append(self.allocator, '_');
                } else {
                    try dir_path_buf.append(self.allocator, c);
                }
            }
        }

        const dir_path = dir_path_buf.items;

        // Open directory and iterate
        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return results.toOwnedSlice(allocator),
            else => return error.IoError,
        };
        defer dir.close();

        var iter = dir.iterate();
        while (iter.next() catch return error.IoError) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

            // Build full path
            var full_path_buf = std.ArrayListUnmanaged(u8){};
            defer full_path_buf.deinit(self.allocator);

            try full_path_buf.appendSlice(self.allocator, dir_path);
            try full_path_buf.appendSlice(self.allocator, "/");
            try full_path_buf.appendSlice(self.allocator, entry.name);

            // Read and parse
            const content = self.readFileContent(full_path_buf.items) catch continue;
            defer self.allocator.free(content);

            const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch continue;
            defer parsed.deinit();

            const result = BenchmarkResult.fromJson(allocator, parsed.value) catch continue;
            try results.append(allocator, result);
        }

        return results.toOwnedSlice(allocator);
    }

    /// Save multiple baselines at once
    pub fn saveAllBaselines(self: *BaselineStore, results: []const BenchmarkResult) !void {
        for (results) |result| {
            try self.saveBaseline(result);
        }
    }

    /// Delete a baseline
    pub fn deleteBaseline(self: *BaselineStore, benchmark_name: []const u8, branch: ?[]const u8) !void {
        const file_path = try self.getBaselineFilePath(benchmark_name, branch);
        defer self.allocator.free(file_path);

        std.fs.cwd().deleteFile(file_path) catch |err| switch (err) {
            error.FileNotFound => {}, // Already deleted, no-op
            else => return error.IoError,
        };
    }
};

/// Calculate percentage change between baseline and current value
/// For throughput metrics (ops/s, MB/s), higher is better
/// For latency metrics (ns, ms), lower is better
fn calculatePercentChange(baseline: f64, current: f64, metric: []const u8) f64 {
    if (baseline == 0) return 0;

    const change = ((current - baseline) / baseline) * 100.0;

    // For latency metrics, invert the sign (lower is better)
    if (std.mem.indexOf(u8, metric, "latency") != null or
        std.mem.indexOf(u8, metric, "_ns") != null or
        std.mem.indexOf(u8, metric, "_ms") != null or
        std.mem.indexOf(u8, metric, "time") != null)
    {
        return -change;
    }

    return change;
}

// ============================================================================
// Tests
// ============================================================================

test "BenchmarkResult JSON roundtrip" {
    const allocator = std.testing.allocator;

    const original = BenchmarkResult{
        .name = "test_benchmark",
        .metric = "ops_per_sec",
        .value = 1500000.5,
        .unit = "ops/s",
        .timestamp = 1706000000,
        .git_commit = "abc123def",
        .git_branch = "main",
        .category = "simd",
        .std_dev = 1500.25,
        .sample_count = 1000,
        .p99_ns = 750,
        .memory_bytes = 4096,
    };

    const json = try original.toJsonString(allocator);
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    var result = try BenchmarkResult.fromJson(allocator, parsed.value);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("test_benchmark", result.name);
    try std.testing.expectEqualStrings("ops_per_sec", result.metric);
    try std.testing.expectApproxEqAbs(@as(f64, 1500000.5), result.value, 0.01);
    try std.testing.expectEqualStrings("ops/s", result.unit);
    try std.testing.expectEqual(@as(i64, 1706000000), result.timestamp);
    try std.testing.expectEqualStrings("abc123def", result.git_commit.?);
    try std.testing.expectEqualStrings("main", result.git_branch.?);
}

test "calculatePercentChange throughput" {
    // Throughput: higher is better
    // 10% improvement
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), calculatePercentChange(100, 110, "ops_per_sec"), 0.01);
    // 10% regression
    try std.testing.expectApproxEqAbs(@as(f64, -10.0), calculatePercentChange(100, 90, "ops_per_sec"), 0.01);
}

test "calculatePercentChange latency" {
    // Latency: lower is better (inverted)
    // 10% faster (lower latency = improvement = positive change)
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), calculatePercentChange(100, 90, "latency_ns"), 0.01);
    // 10% slower (higher latency = regression = negative change)
    try std.testing.expectApproxEqAbs(@as(f64, -10.0), calculatePercentChange(100, 110, "latency_ns"), 0.01);
}

test "BaselineStore path generation" {
    const allocator = std.testing.allocator;

    var store = BaselineStore.init(allocator, "benchmarks/baselines");
    defer store.deinit();

    // Main branch
    const main_path = try store.getBaselineFilePath("vector_dot", "main");
    defer allocator.free(main_path);
    try std.testing.expectEqualStrings("benchmarks/baselines/main/vector_dot.json", main_path);

    // Feature branch
    const feature_path = try store.getBaselineFilePath("vector_dot", "feature/simd-opt");
    defer allocator.free(feature_path);
    try std.testing.expectEqualStrings("benchmarks/baselines/branches/feature_simd-opt/vector_dot.json", feature_path);

    // Release branch
    const release_path = try store.getBaselineFilePath("vector_dot", "v1.0.0");
    defer allocator.free(release_path);
    try std.testing.expectEqualStrings("benchmarks/baselines/releases/v1.0.0/vector_dot.json", release_path);
}
