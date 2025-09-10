//! Static Analysis Tool for WDBX-AI Codebase
//!
//! This tool performs static analysis on the codebase to identify:
//! - Code quality issues
//! - Performance bottlenecks
//! - Security concerns
//! - Style inconsistencies

const std = @import("std");
const print = std.debug.print;

/// Analysis result severity levels
const Severity = enum {
    info,
    warning,
    err,
    critical,

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .info => "INFO",
            .warning => "WARNING",
            .err => "ERROR",
            .critical => "CRITICAL",
        };
    }
};

/// Analysis finding
const Finding = struct {
    file: []const u8,
    line: usize,
    column: usize,
    severity: Severity,
    message: []const u8,
    rule: []const u8,

    pub fn format(self: Finding) void {
        print("{s}:{d}:{d}: {s}: {s} [{s}]\n", .{ self.file, self.line, self.column, self.severity.toString(), self.message, self.rule });
    }
};

/// Static analyzer
pub const StaticAnalyzer = struct {
    allocator: std.mem.Allocator,
    findings: std.ArrayList(Finding),
    config: AnalysisConfig,

    const Self = @This();

    pub const AnalysisConfig = struct {
        check_performance: bool = true,
        check_security: bool = true,
        check_style: bool = true,
        check_complexity: bool = true,
        max_function_length: usize = 100,
        max_cyclomatic_complexity: usize = 10,
        max_parameter_count: usize = 8,
    };

    pub fn init(allocator: std.mem.Allocator, config: AnalysisConfig) Self {
        return .{
            .allocator = allocator,
            .findings = std.ArrayList(Finding){},
            .config = config,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.findings.items) |finding| {
            self.allocator.free(finding.file);
            self.allocator.free(finding.message);
            self.allocator.free(finding.rule);
        }
        self.findings.deinit(self.allocator);
    }

    /// Analyze a single file
    pub fn analyzeFile(self: *Self, file_path: []const u8) !void {
        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            print("Error opening file {s}: {}\n", .{ file_path, err });
            return;
        };
        defer file.close();

        const max_bytes: usize = 10 * 1024 * 1024;
        const file_size_u64 = file.getEndPos() catch 0;
        const to_read_u64 = if (file_size_u64 > @as(u64, max_bytes)) @as(u64, max_bytes) else file_size_u64;
        const to_read: usize = @intCast(to_read_u64);

        const content = try self.allocator.alloc(u8, to_read);
        errdefer self.allocator.free(content);
        _ = file.readAll(content) catch |err| {
            print("Error reading file {s}: {}\n", .{ file_path, err });
            return;
        };

        defer self.allocator.free(content);
        try self.analyzeContent(file_path, content);
    }

    /// Analyze file content
    fn analyzeContent(self: *Self, file_path: []const u8, content: []const u8) !void {
        var line_number: usize = 1;
        var lines = std.mem.splitSequence(u8, content, "\n");

        while (lines.next()) |line| {
            defer line_number += 1;

            if (self.config.check_style) {
                try self.checkStyleIssues(file_path, line_number, line);
            }

            if (self.config.check_security) {
                try self.checkSecurityIssues(file_path, line_number, line);
            }

            if (self.config.check_performance) {
                try self.checkPerformanceIssues(file_path, line_number, line);
            }
        }

        if (self.config.check_complexity) {
            try self.checkComplexity(file_path, content);
        }
    }

    /// Check for style issues
    fn checkStyleIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for trailing whitespace
        if (line.len > 0 and (line[line.len - 1] == ' ' or line[line.len - 1] == '\t')) {
            try self.addFinding(file_path, line_number, 1, .warning, "Trailing whitespace found", "style.trailing_whitespace");
        }

        // Check for long lines
        if (line.len > 120) {
            try self.addFinding(file_path, line_number, 1, .warning, "Line exceeds 120 characters", "style.line_length");
        }

        // Check for inconsistent indentation
        if (std.mem.indexOf(u8, line, "\t") != null and std.mem.indexOf(u8, line, "    ") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Mixed tabs and spaces", "style.indentation");
        }

        // Check for TODO/FIXME comments
        if (std.mem.indexOf(u8, line, "TODO") != null or std.mem.indexOf(u8, line, "FIXME") != null) {
            try self.addFinding(file_path, line_number, 1, .info, "TODO/FIXME comment found", "style.todo_comment");
        }
    }

    /// Check for security issues
    fn checkSecurityIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for potential buffer overflow risks
        if (std.mem.indexOf(u8, line, "@memcpy") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Manual memory copy - ensure bounds checking", "security.memory_safety");
        }

        // Check for unsafe pointer operations
        if (std.mem.indexOf(u8, line, "@ptrCast") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Unsafe pointer cast - verify type safety", "security.pointer_safety");
        }

        // Check for hardcoded secrets/passwords
        if (std.mem.indexOf(u8, line, "password") != null or
            std.mem.indexOf(u8, line, "secret") != null or
            std.mem.indexOf(u8, line, "token") != null)
        {
            if (std.mem.indexOf(u8, line, "=") != null) {
                try self.addFinding(file_path, line_number, 1, .err, "Potential hardcoded credential", "security.hardcoded_secrets");
            }
        }
    }

    /// Check for performance issues
    fn checkPerformanceIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for inefficient string operations
        if (std.mem.indexOf(u8, line, "std.fmt.allocPrint") != null) {
            try self.addFinding(file_path, line_number, 1, .info, "Consider using stack-allocated buffer for formatting", "performance.allocation");
        }

        // Check for unnecessary allocations in loops
        if (std.mem.indexOf(u8, line, "for") != null and
            std.mem.indexOf(u8, line, "alloc") != null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Potential allocation in loop - consider pre-allocating", "performance.loop_allocation");
        }

        // Check for expensive operations in hot paths
        if (std.mem.indexOf(u8, line, "std.fmt.parseInt") != null or
            std.mem.indexOf(u8, line, "std.fmt.parseFloat") != null)
        {
            try self.addFinding(file_path, line_number, 1, .info, "String parsing in potential hot path", "performance.string_parsing");
        }
    }

    /// Check code complexity
    fn checkComplexity(self: *Self, file_path: []const u8, content: []const u8) !void {
        var function_start: ?usize = null;
        var function_name: ?[]const u8 = null;
        var brace_count: i32 = 0;
        var line_count: usize = 0;
        var current_line: usize = 1;

        var lines = std.mem.splitSequence(u8, content, "\n");
        while (lines.next()) |line| {
            defer current_line += 1;

            const trimmed = std.mem.trim(u8, line, " \t");

            // Function detection
            if (std.mem.startsWith(u8, trimmed, "pub fn ") or
                std.mem.startsWith(u8, trimmed, "fn "))
            {
                function_start = current_line;
                // Extract function name
                const fn_start = std.mem.indexOf(u8, trimmed, "fn ").? + 3;
                const fn_end = std.mem.indexOf(u8, trimmed[fn_start..], "(").? + fn_start;
                function_name = trimmed[fn_start..fn_end];
                line_count = 0;
                brace_count = 0;
            }

            if (function_start) |start| {
                line_count += 1;

                // Count braces
                for (line) |char| {
                    if (char == '{') brace_count += 1;
                    if (char == '}') brace_count -= 1;
                }

                // Check if function ended
                if (brace_count <= 0 and function_name != null) {
                    if (line_count > self.config.max_function_length) {
                        const msg = try std.fmt.allocPrint(self.allocator, "Function '{s}' is {d} lines (max: {d})", .{ function_name.?, line_count, self.config.max_function_length });
                        defer self.allocator.free(msg);
                        try self.addFinding(file_path, start, 1, .warning, msg, "complexity.function_length");
                    }
                    function_start = null;
                    function_name = null;
                }
            }
        }
    }

    /// Add a finding to the results
    fn addFinding(self: *Self, file_path: []const u8, line: usize, column: usize, severity: Severity, message: []const u8, rule: []const u8) !void {
        const finding = Finding{
            .file = try self.allocator.dupe(u8, file_path),
            .line = line,
            .column = column,
            .severity = severity,
            .message = try self.allocator.dupe(u8, message),
            .rule = try self.allocator.dupe(u8, rule),
        };
        try self.findings.append(self.allocator, finding);
    }

    /// Generate analysis report
    pub fn generateReport(self: *Self) !void {
        print("=== WDBX Static Analysis Report ===\n\n", .{});

        var counts = [_]usize{0} ** 4;
        for (self.findings.items) |finding| {
            counts[@intFromEnum(finding.severity)] += 1;
        }

        print("Summary:\n", .{});
        print("  INFO: {d}\n", .{counts[@intFromEnum(Severity.info)]});
        print("  WARNING: {d}\n", .{counts[@intFromEnum(Severity.warning)]});
        print("  ERROR: {d}\n", .{counts[@intFromEnum(Severity.err)]});
        print("  CRITICAL: {d}\n", .{counts[@intFromEnum(Severity.critical)]});
        print("\n", .{});

        if (self.findings.items.len == 0) {
            print("No issues found! 🎉\n", .{});
            return;
        }

        print("Findings:\n", .{});
        for (self.findings.items) |finding| {
            finding.format();
        }
    }

    /// Analyze entire directory recursively
    pub fn analyzeDirectory(self: *Self, dir_path: []const u8) !void {
        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
            print("Error opening directory {s}: {}\n", .{ dir_path, err });
            return;
        };
        defer dir.close();

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".zig")) {
                const full_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, entry.name });
                defer self.allocator.free(full_path);
                try self.analyzeFile(full_path);
            } else if (entry.kind == .directory and !std.mem.eql(u8, entry.name, ".git") and
                !std.mem.eql(u8, entry.name, "zig-cache") and !std.mem.eql(u8, entry.name, "zig-out"))
            {
                const sub_dir = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, entry.name });
                defer self.allocator.free(sub_dir);
                try self.analyzeDirectory(sub_dir);
            }
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = StaticAnalyzer.AnalysisConfig{};
    var analyzer = StaticAnalyzer.init(allocator, config);
    defer analyzer.deinit();

    print("Running static analysis on WDBX codebase...\n\n", .{});

    try analyzer.analyzeDirectory("src");
    try analyzer.generateReport();
}
