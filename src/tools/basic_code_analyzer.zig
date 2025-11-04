//! Basic Code Quality Analyzer
//!
//! A minimal code quality analyzer that prints results to console

const std = @import("std");

/// Basic Code Quality Metrics
pub const BasicMetrics = struct {
    lines_of_code: u32 = 0,
    function_count: u32 = 0,
    struct_count: u32 = 0,
    comment_lines: u32 = 0,
    complexity_score: u32 = 0,

    pub fn print(self: BasicMetrics) void {
        std.log.info("Basic Code Quality Metrics:", .{});
        std.log.info("  Lines of Code: {}", .{self.lines_of_code});
        std.log.info("  Functions: {}", .{self.function_count});
        std.log.info("  Structs: {}", .{self.struct_count});
        std.log.info("  Comment Lines: {}", .{self.comment_lines});
        std.log.info("  Complexity Score: {}", .{self.complexity_score});
    }
};

/// Basic Code Analyzer
pub const BasicCodeAnalyzer = struct {
    allocator: std.mem.Allocator,
    metrics: BasicMetrics,

    pub fn init(allocator: std.mem.Allocator) !*BasicCodeAnalyzer {
        const self = try allocator.create(BasicCodeAnalyzer);
        self.* = .{
            .allocator = allocator,
            .metrics = .{},
        };
        return self;
    }

    pub fn deinit(self: *BasicCodeAnalyzer) void {
        self.allocator.destroy(self);
    }

    /// Analyze a Zig source file
    pub fn analyzeFile(self: *BasicCodeAnalyzer, file_path: []const u8) !void {
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
    fn analyzeContent(self: *BasicCodeAnalyzer, content: []const u8) !void {
        var lines = std.mem.splitScalar(u8, content, '\n');

        while (lines.next()) |line| {
            self.metrics.lines_of_code += 1;

            // Count functions
            if (std.mem.indexOf(u8, line, "fn ") != null) {
                self.metrics.function_count += 1;
            }

            // Count structs
            if (std.mem.indexOf(u8, line, "struct ") != null) {
                self.metrics.struct_count += 1;
            }

            // Count comments
            if (std.mem.indexOf(u8, line, "//") != null) {
                self.metrics.comment_lines += 1;
            }

            // Simple complexity scoring
            if (std.mem.indexOf(u8, line, "if ") != null or
                std.mem.indexOf(u8, line, "while ") != null or
                std.mem.indexOf(u8, line, "for ") != null)
            {
                self.metrics.complexity_score += 1;
            }
        }
    }

    /// Print report to console
    pub fn printReport(self: *BasicCodeAnalyzer) void {
        self.metrics.print();
    }
};

/// Main function for command-line usage
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var analyzer = try BasicCodeAnalyzer.init(allocator);
    defer analyzer.deinit();

    std.log.info("🔍 Basic Code Quality Analyzer", .{});
    std.log.info("===============================", .{});

    // Analyze source files
    const source_dirs = [_][]const u8{ "src", "tests", "tools" };

    for (source_dirs) |dir| {
        std.log.info("Analyzing directory: {s}", .{dir});
        try analyzeDirectory(analyzer, dir);
    }

    // Print report
    analyzer.printReport();

    std.log.info("📊 Code quality analysis complete!", .{});
}

/// Analyze all Zig files in a directory
fn analyzeDirectory(analyzer: *BasicCodeAnalyzer, dir_path: []const u8) !void {
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
