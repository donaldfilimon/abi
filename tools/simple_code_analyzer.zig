//! Simple Code Quality Analyzer
//!
//! A simplified version of the code quality analyzer that focuses on basic metrics
//! and avoids complex data structures that cause compilation issues.

const std = @import("std");

/// Simple Code Quality Metrics
pub const SimpleMetrics = struct {
    lines_of_code: u32 = 0,
    function_count: u32 = 0,
    struct_count: u32 = 0,
    comment_lines: u32 = 0,
    complexity_score: u32 = 0,
    simd_function_count: u32 = 0,

    pub fn format(
        self: SimpleMetrics,
        comptime fmt: []const u8,
        options: anytype,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Simple Code Quality Metrics:\n", .{});
        try writer.print("  Lines of Code: {}\n", .{self.lines_of_code});
        try writer.print("  Functions: {}\n", .{self.function_count});
        try writer.print("  SIMD Functions: {}\n", .{self.simd_function_count});
        try writer.print("  Structs: {}\n", .{self.struct_count});
        try writer.print("  Comment Lines: {}\n", .{self.comment_lines});
        try writer.print("  Complexity Score: {}\n", .{self.complexity_score});
    }
};

/// Simple Code Analyzer
pub const SimpleCodeAnalyzer = struct {
    allocator: std.mem.Allocator,
    metrics: SimpleMetrics,

    pub fn init(allocator: std.mem.Allocator) !*SimpleCodeAnalyzer {
        const self = try allocator.create(SimpleCodeAnalyzer);
        self.* = .{
            .allocator = allocator,
            .metrics = .{},
        };
        return self;
    }

    pub fn deinit(self: *SimpleCodeAnalyzer) void {
        self.allocator.destroy(self);
    }

    /// Analyze a Zig source file
    pub fn analyzeFile(self: *SimpleCodeAnalyzer, file_path: []const u8) !void {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size > 1024 * 1024) {
            std.log.warn("File {s} is too large, skipping", .{file_path});
            return;
        }

        const content = try self.allocator.alloc(u8, @as(usize, @intCast(stat.size)));
        errdefer self.allocator.free(content);
        defer self.allocator.free(content);

        const bytes_read = try file.read(content);
        if (bytes_read != stat.size) {
            std.log.warn("Could not read entire file {s}", .{file_path});
            return;
        }

        try self.analyzeContent(content);
    }

    /// Analyze source code content
    fn analyzeContent(self: *SimpleCodeAnalyzer, content: []const u8) !void {
        var lines = std.mem.splitScalar(u8, content, '\n');

        // State variables for SIMD detection
        var in_function = false;
        var simd_detected_this_function = false;

        while (lines.next()) |line| {
            self.metrics.lines_of_code += 1;

            // Count functions
            if (std.mem.indexOf(u8, line, "fn ") != null) {
                self.metrics.function_count += 1;

                // End previous function's SIMD detection
                if (in_function and simd_detected_this_function) {
                    self.metrics.simd_function_count += 1;
                }

                // Start new function
                in_function = true;
                simd_detected_this_function = false;
            }

            // Count structs
            if (std.mem.indexOf(u8, line, "struct ") != null) {
                self.metrics.struct_count += 1;
            }

            // Count comments
            if (std.mem.indexOf(u8, line, "//") != null) {
                self.metrics.comment_lines += 1;
            }

            // Simple complexity scoring – ignore SIMD vector operations
            if ((std.mem.indexOf(u8, line, "if ") != null or
                std.mem.indexOf(u8, line, "while ") != null or
                std.mem.indexOf(u8, line, "for ") != null) and
                std.mem.indexOf(u8, line, "@Vector") == null)
            {
                self.metrics.complexity_score += 1;
            }

            // Detect SIMD usage inside a function
            if (in_function and std.mem.indexOf(u8, line, "@Vector") != null) {
                simd_detected_this_function = true;
            }
        }

        // Handle last function's SIMD detection
        if (in_function and simd_detected_this_function) {
            self.metrics.simd_function_count += 1;
        }
    }

    /// Generate simple report
    pub fn generateReport(self: *SimpleCodeAnalyzer, allocator: std.mem.Allocator) ![]const u8 {
        var report = std.array_list.Managed(u8).init(allocator);
        errdefer report.deinit();

        try report.appendSlice("Simple Code Quality Analysis Report\n");
        try report.appendSlice("================================\n\n");

        var buf: [1024]u8 = undefined;
        const metrics_str = try std.fmt.bufPrint(&buf, "{any}\n", .{self.metrics});
        try report.appendSlice(metrics_str);

        return report.toOwnedSlice();
    }
};

/// Main function for command-line usage
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var analyzer = try SimpleCodeAnalyzer.init(allocator);
    defer analyzer.deinit();

    std.log.info("🔍 Simple Code Quality Analyzer", .{});
    std.log.info("================================", .{});

    // Analyze source files
    const source_dirs = [_][]const u8{ "src", "tests", "tools" };

    for (source_dirs) |dir| {
        std.log.info("Analyzing directory: {s}", .{dir});
        try analyzeDirectory(analyzer, dir);
    }

    // Generate and display report
    const report = try analyzer.generateReport(allocator);
    defer allocator.free(report);

    std.log.info("\n{s}", .{report});

    // Write report to file
    const report_file = try std.fs.cwd().createFile("simple_code_quality_report.txt", .{});
    defer report_file.close();
    try report_file.writeAll(report);

    std.log.info("📊 Code quality report exported to: simple_code_quality_report.txt", .{});
}

/// Analyze all Zig files in a directory
fn analyzeDirectory(analyzer: *SimpleCodeAnalyzer, dir_path: []const u8) !void {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
        std.log.warn("Could not open directory {s}: {}", .{ dir_path, err });
        return;
    };
    defer dir.close();

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".zig")) {
            const file_path = try std.fmt.allocPrint(analyzer.allocator, "{s}/{s}", .{ dir_path, entry.name });
            defer analyzer.allocator.free(file_path);

            try analyzer.analyzeFile(file_path);
        } else if (entry.kind == .directory) {
            const subdir_path = try std.fmt.allocPrint(analyzer.allocator, "{s}/{s}", .{ dir_path, entry.name });
            defer analyzer.allocator.free(subdir_path);

            try analyzeDirectory(analyzer, subdir_path);
        }
    }
}
