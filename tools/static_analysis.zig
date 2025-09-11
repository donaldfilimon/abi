//! Enhanced Static Analysis Tool for WDBX-AI Codebase
//!
//! This tool performs comprehensive static analysis on the codebase to identify:
//! - Code quality issues with severity classification
//! - Performance bottlenecks and optimization opportunities
//! - Security concerns with modern threat detection
//! - Style inconsistencies and best practices
//! - SIMD optimization opportunities
//! - Memory management patterns
//! - Concurrency and thread safety issues
//! - Technical debt metrics

const std = @import("std");
const builtin = @import("builtin");
const print = std.debug.print;

/// Enhanced analysis result severity levels with priority scoring
const Severity = enum(u8) {
    info = 0,
    suggestion = 1,
    warning = 2,
    err = 3,
    critical = 4,

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .info => "INFO",
            .suggestion => "SUGGESTION",
            .warning => "WARNING",
            .err => "ERROR",
            .critical => "CRITICAL",
        };
    }

    pub fn getColor(self: Severity) []const u8 {
        return switch (self) {
            .info => "\x1b[36m", // Cyan
            .suggestion => "\x1b[32m", // Green
            .warning => "\x1b[33m", // Yellow
            .err => "\x1b[31m", // Red
            .critical => "\x1b[35m", // Magenta
        };
    }

    pub fn getResetColor() []const u8 {
        return "\x1b[0m";
    }
};

/// Enhanced analysis finding with context and suggestions
const Finding = struct {
    file: []const u8,
    line: usize,
    column: usize,
    severity: Severity,
    message: []const u8,
    rule: []const u8,
    context: []const u8,
    suggestion: []const u8,
    confidence: f32, // 0.0 - 1.0 confidence score

    pub fn format(self: Finding, enable_colors: bool) void {
        const color = if (enable_colors) self.severity.getColor() else "";
        const reset = if (enable_colors) Severity.getResetColor() else "";

        print("{s}:{d}:{d}: {s}{s}{s}: {s} [{s}] (confidence: {d:.2})\n", .{ self.file, self.line, self.column, color, self.severity.toString(), reset, self.message, self.rule, self.confidence });

        if (self.context.len > 0) {
            print("  Context: {s}\n", .{self.context});
        }
        if (self.suggestion.len > 0) {
            print("  {s}Suggestion{s}: {s}\n", .{ color, reset, self.suggestion });
        }
    }

    pub fn getScore(self: Finding) u32 {
        return (@as(u32, @intFromEnum(self.severity)) * 100) + @as(u32, @intFromFloat(self.confidence * 100));
    }
};

/// Enhanced configuration with feature flags
const AnalysisConfig = struct {
    check_performance: bool = true,
    check_security: bool = true,
    check_style: bool = true,
    check_complexity: bool = true,
    check_simd_opportunities: bool = true,
    check_memory_patterns: bool = true,
    check_concurrency: bool = true,
    check_error_handling: bool = true,

    // Thresholds
    max_function_length: usize = 100,
    max_cyclomatic_complexity: usize = 10,
    max_parameter_count: usize = 8,
    max_nesting_depth: usize = 5,
    min_confidence_threshold: f32 = 0.7,

    // Output options
    enable_colors: bool = true,
    enable_suggestions: bool = true,
    output_format: OutputFormat = .text,

    const OutputFormat = enum {
        text,
        json,
        sarif, // Static Analysis Results Interchange Format
        junit,
    };

    pub fn fromEnv(allocator: std.mem.Allocator) !AnalysisConfig {
        var config = AnalysisConfig{};

        if (std.process.getEnvVarOwned(allocator, "STATIC_ANALYSIS_COLORS")) |val| {
            defer allocator.free(val);
            config.enable_colors = std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1");
        } else |_| {}

        if (std.process.getEnvVarOwned(allocator, "STATIC_ANALYSIS_FORMAT")) |val| {
            defer allocator.free(val);
            config.output_format = std.meta.stringToEnum(OutputFormat, val) orelse .text;
        } else |_| {}

        return config;
    }
};

const SuppressEntry = struct { dir: []const u8, rule: []const u8 };

/// Enhanced static analyzer with comprehensive analysis capabilities
pub const StaticAnalyzer = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    findings: std.ArrayListUnmanaged(Finding),
    config: AnalysisConfig,
    allowed_secret_substrings: []const []const u8 = &.{},
    suppressed: []const SuppressEntry = &.{},

    // Analysis state
    total_lines: usize,
    total_files: usize,
    simd_opportunities: usize,
    performance_issues: usize,
    security_issues: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: AnalysisConfig) Self {
        var self: Self = .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .findings = .{},
            .config = config,
            .total_lines = 0,
            .total_files = 0,
            .simd_opportunities = 0,
            .performance_issues = 0,
            .security_issues = 0,
        };
        loadOverrides(&self) catch {};
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.findings.items) |finding| {
            self.allocator.free(finding.file);
            self.allocator.free(finding.message);
            self.allocator.free(finding.rule);
            self.allocator.free(finding.context);
            self.allocator.free(finding.suggestion);
        }
        self.findings.deinit(self.allocator);
        self.arena.deinit();
    }

    pub fn analyzeFile(self: *Self, file_path: []const u8) !void {
        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            print("Error opening file {s}: {}\n", .{ file_path, err });
            return;
        };
        defer file.close();

        // Read file with efficient memory management
        const st = try file.stat();
        const max_bytes: usize = 10 * 1024 * 1024;
        const file_size_u64: u64 = st.size;
        const max_bytes_u64: u64 = @intCast(max_bytes);
        const to_read_u64: u64 = if (file_size_u64 > max_bytes_u64) max_bytes_u64 else file_size_u64;
        const to_read: usize = @intCast(to_read_u64);

        const arena_allocator = self.arena.allocator();
        const buf = try arena_allocator.alloc(u8, to_read);
        const n = try file.readAll(buf);
        const content = buf[0..n];

        try self.analyzeContent(file_path, content);
        self.total_files += 1;
    }

    fn analyzeContent(self: *Self, file_path: []const u8, content: []const u8) !void {
        var line_number: usize = 1;
        var lines = std.mem.splitSequence(u8, content, "\n");

        // First pass: line-by-line analysis
        while (lines.next()) |line| {
            defer line_number += 1;
            self.total_lines += 1;

            // Run all enabled checks
            if (self.config.check_style) {
                try self.checkStyleIssues(file_path, line_number, line);
            }
            if (self.config.check_security) {
                try self.checkSecurityIssues(file_path, line_number, line);
            }
            if (self.config.check_performance) {
                try self.checkPerformanceIssues(file_path, line_number, line);
            }
            if (self.config.check_simd_opportunities) {
                try self.checkSIMDOpportunities(file_path, line_number, line);
            }
            if (self.config.check_memory_patterns) {
                try self.checkMemoryPatterns(file_path, line_number, line);
            }
            if (self.config.check_concurrency) {
                try self.checkConcurrencyIssues(file_path, line_number, line);
            }
            if (self.config.check_error_handling) {
                try self.checkErrorHandling(file_path, line_number, line);
            }
        }

        // Second pass: structural analysis
        if (self.config.check_complexity) {
            try self.checkComplexity(file_path, content);
        }
    }

    fn checkStyleIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for trailing whitespace
        if (line.len > 0 and (line[line.len - 1] == ' ' or line[line.len - 1] == '\t')) {
            try self.addFinding(file_path, line_number, 1, .warning, "Trailing whitespace found", "style.trailing_whitespace", line, "Remove trailing whitespace", 0.9);
        }

        // Check for long lines with context-aware thresholds
        const max_line_length: usize = if (std.mem.indexOf(u8, line, "//") != null) 140 else 120;
        if (line.len > max_line_length) {
            const suggestion = if (std.mem.indexOf(u8, line, "try std.fmt.allocPrint") != null)
                "Consider using multiline string formatting or breaking up the format string"
            else if (std.mem.indexOf(u8, line, "const") != null)
                "Consider breaking the declaration across multiple lines"
            else
                "Consider breaking this line for better readability";

            try self.addFinding(file_path, line_number, 1, .warning, "Line exceeds recommended length", "style.line_length", line, suggestion, 0.8);
        }

        // Check for inconsistent indentation
        if (std.mem.indexOf(u8, line, "\t") != null and std.mem.indexOf(u8, line, "    ") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Mixed tabs and spaces in indentation", "style.indentation", line, "Use consistent indentation (spaces are recommended)", 0.95);
        }

        // Check for TODO/FIXME comments with priority
        if (std.mem.indexOf(u8, line, "TODO") != null) {
            try self.addFinding(file_path, line_number, 1, .info, "TODO comment found", "style.todo_comment", line, "Consider creating an issue or addressing this TODO", 0.7);
        }
        if (std.mem.indexOf(u8, line, "FIXME") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "FIXME comment found", "style.fixme_comment", line, "This indicates a known issue that should be addressed", 0.8);
        }
        if (std.mem.indexOf(u8, line, "HACK") != null) {
            try self.addFinding(file_path, line_number, 1, .suggestion, "HACK comment found", "style.hack_comment", line, "Consider refactoring this workaround into a proper solution", 0.85);
        }

        // Check for magic numbers
        if (std.mem.indexOf(u8, line, "= 42") != null or std.mem.indexOf(u8, line, "== 42") != null) {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Magic number detected", "style.magic_numbers", line, "Consider defining this as a named constant", 0.75);
        }
    }

    fn checkSecurityIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        self.security_issues += 1;

        // Check for unsafe memory operations
        if (std.mem.indexOf(u8, line, "@memcpy") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Manual memory copy - ensure bounds checking", "security.memory_safety", line, "Use std.mem.copy or ensure source and destination sizes are validated", 0.85);
        }

        if (std.mem.indexOf(u8, line, "@ptrCast") != null) {
            try self.addFinding(file_path, line_number, 1, .warning, "Unsafe pointer cast - verify type safety", "security.pointer_safety", line, "Ensure the cast is safe and consider using @alignCast if needed", 0.8);
        }

        // Enhanced secret detection (tuned to reduce false positives)
        const secret_patterns = [_][]const u8{ "password", "secret", "token", "api_key", "apikey", "private_key", "credentials" };

        for (secret_patterns) |pattern| {
            // Create a temporary buffer for lowercased line
            var line_lower_buf: [1024]u8 = undefined;
            const line_lower = if (line.len <= line_lower_buf.len)
                std.ascii.lowerString(line_lower_buf[0..line.len], line)
            else
                line; // Fallback to original if too long

            const has_pattern = std.mem.indexOf(u8, line_lower, pattern) != null;
            const is_assignment = std.mem.indexOf(u8, line, "=") != null;
            const is_comment = std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "//");
            const is_loglevel = std.mem.indexOf(u8, line_lower, "std.debug.print") != null or std.mem.indexOf(u8, line_lower, "std.log.") != null;
            const is_header_construction = std.mem.indexOf(u8, line_lower, "authorization") != null or std.mem.indexOf(u8, line_lower, "bearer ") != null;
            const is_metadata_field = std.mem.indexOf(u8, line_lower, ".author") != null or std.mem.indexOf(u8, line_lower, ".description") != null;
            var is_allowed = false;
            if (has_pattern and self.allowed_secret_substrings.len > 0) {
                for (self.allowed_secret_substrings) |allowed| {
                    if (std.mem.indexOf(u8, line_lower, allowed) != null) {
                        is_allowed = true;
                        break;
                    }
                }
            }
            if (has_pattern and is_assignment and !is_comment and !is_loglevel and !is_header_construction and !is_metadata_field and !is_allowed) {
                const severity: Severity = if (std.mem.indexOf(u8, line, "\"") != null) .critical else .err;
                try self.addFinding(file_path, line_number, 1, severity, "Potential hardcoded credential", "security.hardcoded_secrets", line, "Use environment variables or secure configuration management", 0.9);
                break;
            }
        }

        // Check for SQL injection vulnerabilities
        if (std.mem.indexOf(u8, line, "SELECT") != null and
            std.mem.indexOf(u8, line, "++") != null)
        {
            try self.addFinding(file_path, line_number, 1, .critical, "Potential SQL injection vulnerability", "security.sql_injection", line, "Use parameterized queries or prepared statements", 0.9);
        }

        // Check for buffer overflow risks
        if (std.mem.indexOf(u8, line, "readAll") != null and
            std.mem.indexOf(u8, line, "1024 * 1024 * 1024") != null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Large buffer allocation", "security.buffer_overflow", line, "Consider using streaming or chunked reading for large files", 0.75);
        }

        // Check for unsafe random number generation
        if (std.mem.indexOf(u8, line, "std.rand.DefaultPrng") != null and
            std.mem.indexOf(u8, line, "cryptographic") == null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Non-cryptographic random number generator", "security.weak_random", line, "Use std.crypto.random for security-sensitive operations", 0.7);
        }
    }

    fn checkPerformanceIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        self.performance_issues += 1;

        // Check for inefficient string operations
        if (std.mem.indexOf(u8, line, "std.fmt.allocPrint") != null and
            std.mem.indexOf(u8, line, "for") != null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "String allocation in loop", "performance.string_allocation", line, "Consider using a buffer or StringBuilder pattern", 0.85);
        }

        // Check for unnecessary allocations
        if (std.mem.indexOf(u8, line, "alloc") != null and
            std.mem.indexOf(u8, line, "for") != null and
            std.mem.indexOf(u8, line, "defer") == null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Potential allocation in loop without defer", "performance.loop_allocation", line, "Pre-allocate buffers outside the loop or ensure proper cleanup", 0.8);
        }

        // Check for expensive operations in hot paths
        const expensive_ops = [_][]const u8{ "std.fmt.parseInt", "std.fmt.parseFloat", "std.json.parse", "std.ArrayList.append" };

        for (expensive_ops) |op| {
            if (std.mem.indexOf(u8, line, op) != null and
                std.mem.indexOf(u8, line, "for") != null)
            {
                try self.addFinding(file_path, line_number, 1, .suggestion, "Expensive operation in potential hot path", "performance.expensive_ops", line, "Consider optimizing or caching the result", 0.7);
                break;
            }
        }

        // Check for missing inline annotations
        if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "fn ") and
            line.len < 50 and
            std.mem.indexOf(u8, line, "inline") == null and
            std.mem.indexOf(u8, line, "pub") == null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Small function could benefit from inline annotation", "performance.missing_inline", line, "Consider adding 'inline' keyword for small, frequently called functions", 0.6);
        }

        // Check for arena allocator opportunities
        if (std.mem.indexOf(u8, line, "allocator.alloc") != null and
            std.mem.indexOf(u8, line, "defer") != null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Consider using arena allocator for bulk allocations", "performance.arena_opportunity", line, "Use ArenaAllocator for related allocations that can be freed together", 0.65);
        }
    }

    fn checkSIMDOpportunities(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Vector operations that could benefit from SIMD
        const vector_patterns = [_][]const u8{ "for (a, b)", "for (vector", "dot_product", "normalize", "magnitude" };

        for (vector_patterns) |pattern| {
            if (std.mem.indexOf(u8, line, pattern) != null and
                std.mem.indexOf(u8, line, "@Vector") == null)
            {
                self.simd_opportunities += 1;
                try self.addFinding(file_path, line_number, 1, .suggestion, "Potential SIMD optimization opportunity", "performance.simd_opportunity", line, "Consider using @Vector for SIMD-optimized operations", 0.7);
                break;
            }
        }

        // Math operations on arrays
        if ((std.mem.indexOf(u8, line, "* ") != null or std.mem.indexOf(u8, line, "+ ") != null) and
            std.mem.indexOf(u8, line, "[]f32") != null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Array math operations could use SIMD", "performance.simd_math", line, "Use SIMD vectors for parallel math operations on f32 arrays", 0.75);
        }
    }

    fn checkMemoryPatterns(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for missing defer statements
        if (std.mem.indexOf(u8, line, ".alloc(") != null and
            line_number < self.total_lines)
        {
            // This is a simplified check - in reality, we'd need lookahead
            try self.addFinding(file_path, line_number, 1, .suggestion, "Ensure allocation has corresponding defer free()", "memory.missing_defer", line, "Add 'defer allocator.free(variable)' after allocation", 0.6);
        }

        // Check for memory leaks in error paths
        if (std.mem.indexOf(u8, line, "try ") != null and
            std.mem.indexOf(u8, line, "alloc") != null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Ensure allocations are freed on error paths", "memory.error_path_leak", line, "Use errdefer for cleanup on error paths", 0.65);
        }

        // Check for double-free vulnerabilities
        if (std.mem.indexOf(u8, line, "free(") != null and
            std.mem.indexOf(u8, line, "if") != null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Conditional free() - check for double-free", "memory.double_free", line, "Ensure variables are set to null after freeing", 0.8);
        }
    }

    fn checkConcurrencyIssues(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for shared mutable state without synchronization
        if (std.mem.indexOf(u8, line, "var ") != null and
            std.mem.indexOf(u8, line, "shared") != null and
            std.mem.indexOf(u8, line, "atomic") == null and
            std.mem.indexOf(u8, line, "Mutex") == null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Shared mutable state without synchronization", "concurrency.race_condition", line, "Use atomic operations or mutexes for thread-safe access", 0.85);
        }

        // Check for atomic usage
        if (std.mem.indexOf(u8, line, "std.atomic") != null) {
            try self.addFinding(file_path, line_number, 1, .info, "Atomic operation found", "concurrency.atomic_usage", line, "Good use of atomic operations for thread safety", 0.9);
        }

        // Check for potential deadlocks
        if (std.mem.indexOf(u8, line, "lock()") != null and
            std.mem.indexOf(u8, line, "defer") == null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Lock without defer unlock", "concurrency.deadlock_risk", line, "Use 'defer mutex.unlock()' to prevent deadlocks", 0.9);
        }
    }

    fn checkErrorHandling(self: *Self, file_path: []const u8, line_number: usize, line: []const u8) !void {
        // Check for ignored errors
        if (std.mem.indexOf(u8, line, "_ = ") != null and
            std.mem.indexOf(u8, line, "try ") != null)
        {
            try self.addFinding(file_path, line_number, 1, .warning, "Error result explicitly ignored", "error_handling.ignored_error", line, "Consider handling the error or documenting why it's safe to ignore", 0.8);
        }

        // Check for catch-all error handling
        if (std.mem.indexOf(u8, line, "catch |err|") != null and
            std.mem.indexOf(u8, line, "switch") == null)
        {
            try self.addFinding(file_path, line_number, 1, .suggestion, "Generic error handling", "error_handling.generic_catch", line, "Consider specific error handling for different error types", 0.7);
        }

        // Check for proper error propagation
        if (std.mem.indexOf(u8, line, "return err") != null) {
            try self.addFinding(file_path, line_number, 1, .info, "Good error propagation", "error_handling.propagation", line, "Proper error propagation pattern", 0.9);
        }
    }

    fn checkComplexity(self: *Self, file_path: []const u8, content: []const u8) !void {
        var function_start: ?usize = null;
        var function_name: ?[]const u8 = null;
        var brace_count: i32 = 0;
        var line_count: usize = 0;
        var current_line: usize = 1;
        var nesting_depth: usize = 0;
        var max_nesting: usize = 0;
        var cyclomatic_complexity: usize = 1; // Base complexity

        var lines = std.mem.splitSequence(u8, content, "\n");
        while (lines.next()) |line| {
            defer current_line += 1;

            const trimmed = std.mem.trim(u8, line, " \t");

            // Function detection
            if (std.mem.startsWith(u8, trimmed, "pub fn ") or
                std.mem.startsWith(u8, trimmed, "fn "))
            {
                // If we have a previous function, analyze it
                if (function_start != null and function_name != null) {
                    try self.analyzeFunctionComplexity(file_path, function_start.?, function_name.?, line_count, max_nesting, cyclomatic_complexity);
                }

                function_start = current_line;
                const fn_start = std.mem.indexOf(u8, trimmed, "fn ").? + 3;
                const fn_end = std.mem.indexOf(u8, trimmed[fn_start..], "(").? + fn_start;
                function_name = trimmed[fn_start..fn_end];
                line_count = 0;
                brace_count = 0;
                nesting_depth = 0;
                max_nesting = 0;
                cyclomatic_complexity = 1;
            }

            if (function_start) |_| {
                line_count += 1;

                // Count braces and track nesting
                for (line) |char| {
                    switch (char) {
                        '{' => {
                            brace_count += 1;
                            nesting_depth += 1;
                            max_nesting = @max(max_nesting, nesting_depth);
                        },
                        '}' => {
                            brace_count -= 1;
                            if (nesting_depth > 0) nesting_depth -= 1;
                        },
                        else => {},
                    }
                }

                // Count complexity contributors
                const complexity_keywords = [_][]const u8{ "if ", "else if", "while", "for", "switch", "catch", "try", "defer" };
                for (complexity_keywords) |keyword| {
                    if (std.mem.indexOf(u8, line, keyword) != null) {
                        cyclomatic_complexity += 1;
                    }
                }

                // Check if function ended
                if (brace_count <= 0 and function_name != null) {
                    try self.analyzeFunctionComplexity(file_path, function_start.?, function_name.?, line_count, max_nesting, cyclomatic_complexity);
                    function_start = null;
                    function_name = null;
                }
            }
        }
    }

    fn analyzeFunctionComplexity(self: *Self, file_path: []const u8, start_line: usize, function_name: []const u8, line_count: usize, max_nesting: usize, cyclomatic_complexity: usize) !void {
        if (line_count > self.config.max_function_length) {
            const msg = try std.fmt.allocPrint(self.allocator, "Function '{s}' is {d} lines (max: {d})", .{ function_name, line_count, self.config.max_function_length });
            defer self.allocator.free(msg);
            try self.addFinding(file_path, start_line, 1, .warning, msg, "complexity.function_length", "", "Consider breaking this function into smaller functions", 0.8);
        }

        if (cyclomatic_complexity > self.config.max_cyclomatic_complexity) {
            const msg = try std.fmt.allocPrint(self.allocator, "Function '{s}' has cyclomatic complexity {d} (max: {d})", .{ function_name, cyclomatic_complexity, self.config.max_cyclomatic_complexity });
            defer self.allocator.free(msg);
            try self.addFinding(file_path, start_line, 1, .warning, msg, "complexity.cyclomatic", "", "Reduce branching by extracting methods or using lookup tables", 0.85);
        }

        if (max_nesting > self.config.max_nesting_depth) {
            const msg = try std.fmt.allocPrint(self.allocator, "Function '{s}' has nesting depth {d} (max: {d})", .{ function_name, max_nesting, self.config.max_nesting_depth });
            defer self.allocator.free(msg);
            try self.addFinding(file_path, start_line, 1, .suggestion, msg, "complexity.nesting", "", "Use early returns or extract nested logic into separate functions", 0.75);
        }
    }

    fn addFinding(self: *Self, file_path: []const u8, line: usize, column: usize, severity: Severity, message: []const u8, rule: []const u8, context: []const u8, suggestion: []const u8, confidence: f32) !void {
        if (confidence < self.config.min_confidence_threshold) return;
        if (self.suppressed.len > 0) {
            for (self.suppressed) |s| {
                if (std.mem.startsWith(u8, file_path, s.dir) and std.mem.eql(u8, rule, s.rule)) return;
            }
        }

        const finding = Finding{
            .file = try self.allocator.dupe(u8, file_path),
            .line = line,
            .column = column,
            .severity = severity,
            .message = try self.allocator.dupe(u8, message),
            .rule = try self.allocator.dupe(u8, rule),
            .context = try self.allocator.dupe(u8, context),
            .suggestion = try self.allocator.dupe(u8, suggestion),
            .confidence = confidence,
        };
        try self.findings.append(self.allocator, finding);
    }

    pub fn generateReport(self: *Self) !void {
        // Sort findings by severity and score
        std.mem.sort(Finding, self.findings.items, {}, struct {
            fn lessThan(_: void, a: Finding, b: Finding) bool {
                return a.getScore() > b.getScore();
            }
        }.lessThan);

        switch (self.config.output_format) {
            .text => try self.generateTextReport(),
            .json => try self.generateJsonReport(),
            .sarif => try self.generateSarifReport(),
            .junit => try self.generateJunitReport(),
        }
    }

    fn generateTextReport(self: *Self) !void {
        const color_reset = if (self.config.enable_colors) Severity.getResetColor() else "";
        const color_header = if (self.config.enable_colors) "\x1b[1;36m" else "";
        const color_stats = if (self.config.enable_colors) "\x1b[1;32m" else "";

        print("{s}=== Enhanced WDBX Static Analysis Report ==={s}\n\n", .{ color_header, color_reset });

        var counts = [_]usize{0} ** 5;
        for (self.findings.items) |finding| {
            counts[@intFromEnum(finding.severity)] += 1;
        }

        print("{s}Analysis Summary:{s}\n", .{ color_stats, color_reset });
        print("  Files Analyzed: {d}\n", .{self.total_files});
        print("  Lines of Code: {d}\n", .{self.total_lines});
        print("  SIMD Opportunities: {d}\n", .{self.simd_opportunities});
        print("  Performance Issues: {d}\n", .{self.performance_issues});
        print("  Security Issues: {d}\n", .{self.security_issues});
        print("\n", .{});

        print("Finding Counts by Severity:\n", .{});
        print("  INFO: {d}\n", .{counts[@intFromEnum(Severity.info)]});
        print("  SUGGESTION: {d}\n", .{counts[@intFromEnum(Severity.suggestion)]});
        print("  WARNING: {d}\n", .{counts[@intFromEnum(Severity.warning)]});
        print("  ERROR: {d}\n", .{counts[@intFromEnum(Severity.err)]});
        print("  CRITICAL: {d}\n", .{counts[@intFromEnum(Severity.critical)]});
        print("\n", .{});

        // Category summary (by rule prefix)
        var cat_style: usize = 0;
        var cat_security: usize = 0;
        var cat_performance: usize = 0;
        var cat_memory: usize = 0;
        var cat_concurrency: usize = 0;
        var cat_error: usize = 0;
        var cat_complexity: usize = 0;
        for (self.findings.items) |finding| {
            const rule = finding.rule;
            if (std.mem.startsWith(u8, rule, "style.")) {
                cat_style += 1;
            } else if (std.mem.startsWith(u8, rule, "security.")) {
                cat_security += 1;
            } else if (std.mem.startsWith(u8, rule, "performance.")) {
                cat_performance += 1;
            } else if (std.mem.startsWith(u8, rule, "memory.")) {
                cat_memory += 1;
            } else if (std.mem.startsWith(u8, rule, "concurrency.")) {
                cat_concurrency += 1;
            } else if (std.mem.startsWith(u8, rule, "error_handling.")) {
                cat_error += 1;
            } else if (std.mem.startsWith(u8, rule, "complexity.")) {
                cat_complexity += 1;
            }
        }
        print("Finding Counts by Category:\n", .{});
        print("  Style: {d}\n", .{cat_style});
        print("  Security: {d}\n", .{cat_security});
        print("  Performance: {d}\n", .{cat_performance});
        print("  Memory: {d}\n", .{cat_memory});
        print("  Concurrency: {d}\n", .{cat_concurrency});
        print("  Error Handling: {d}\n", .{cat_error});
        print("  Complexity: {d}\n", .{cat_complexity});
        print("\n", .{});

        if (self.findings.items.len == 0) {
            print("🎉 No issues found!\n", .{});
            return;
        }

        print("Findings (sorted by severity and confidence):\n", .{});
        for (self.findings.items) |finding| {
            finding.format(self.config.enable_colors);
        }

        try self.generateRecommendations();
    }

    fn generateJsonReport(self: *Self) !void {
        print("{{\"findings\":[", .{});
        for (self.findings.items, 0..) |finding, i| {
            if (i > 0) print(",", .{});
            print("{{\"file\":\"{s}\",\"line\":{d},\"severity\":\"{s}\",\"rule\":\"{s}\",\"message\":\"{s}\",\"confidence\":{d:.2}}}", .{ finding.file, finding.line, finding.severity.toString(), finding.rule, finding.message, finding.confidence });
        }
        print("],\"summary\":{{\"total_files\":{d},\"total_lines\":{d},\"simd_opportunities\":{d}}}}}\n", .{ self.total_files, self.total_lines, self.simd_opportunities });
    }

    fn generateSarifReport(self: *Self) !void {
        _ = self;
        print("SARIF report generation not yet implemented\n", .{});
    }

    fn generateJunitReport(self: *Self) !void {
        _ = self;
        print("JUnit report generation not yet implemented\n", .{});
    }

    fn generateRecommendations(self: *Self) !void {
        print("\n🔧 Actionable Recommendations:\n", .{});

        if (self.simd_opportunities > 0) {
            print("  • Consider implementing SIMD optimizations for {d} vector operations\n", .{self.simd_opportunities});
        }

        if (self.performance_issues > 5) {
            print("  • Review performance issues: focus on loop optimizations and memory allocation patterns\n", .{});
        }

        if (self.security_issues > 0) {
            print("  • Address security concerns: implement secure coding practices\n", .{});
        }

        const critical_count = blk: {
            var count: usize = 0;
            for (self.findings.items) |finding| {
                if (finding.severity == .critical) count += 1;
            }
            break :blk count;
        };

        if (critical_count > 0) {
            print("  • 🚨 {d} critical issues require immediate attention\n", .{critical_count});
        }
    }

    pub fn analyzeDirectory(self: *Self, dir_path: []const u8) !void {
        print("🔍 Analyzing directory: {s}\n", .{dir_path});

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

                print("  Analyzing: {s}\n", .{full_path});
                try self.analyzeFile(full_path);
            } else if (entry.kind == .directory and
                !std.mem.eql(u8, entry.name, ".git") and
                !std.mem.eql(u8, entry.name, "zig-cache") and
                !std.mem.eql(u8, entry.name, "zig-out"))
            {
                const sub_dir = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, entry.name });
                defer self.allocator.free(sub_dir);
                try self.analyzeDirectory(sub_dir);
            }
        }
    }
};

fn fileExists(path: []const u8) bool {
    _ = std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const st = try f.stat();
    const buf = try allocator.alloc(u8, @intCast(st.size));
    _ = try f.readAll(buf);
    return buf;
}

fn lowerStrings(arena: std.mem.Allocator, in: []const []const u8) ![]const []const u8 {
    var out = try arena.alloc([]const u8, in.len);
    for (in, 0..) |v, i| {
        const copy = try arena.alloc(u8, v.len);
        @memcpy(copy, v);
        _ = std.ascii.lowerString(copy, copy);
        out[i] = copy;
    }
    return out;
}

fn parseOverrides(arena: std.mem.Allocator, data: []const u8) ![]const []const u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, data, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root == .object) {
        if (root.object.get("allow_secret_substrings")) |ptr| {
            const arr = ptr.array.items;
            var out = try arena.alloc([]const u8, arr.len);
            for (arr, 0..) |item, i| out[i] = item.string;
            return try lowerStrings(arena, out);
        }
    }
    return &.{};
}

fn loadOverrides(self: *StaticAnalyzer) !void {
    const path = "tools/static_analysis.config.json";
    if (!fileExists(path)) return;
    const arena_alloc = self.arena.allocator();
    const bytes = try readFileAlloc(arena_alloc, path);
    self.allowed_secret_substrings = try parseOverrides(arena_alloc, bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, arena_alloc, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root == .object) {
        if (root.object.get("suppress")) |ptr| {
            const arr = ptr.array.items;
            var list = try arena_alloc.alloc(SuppressEntry, arr.len);
            var n: usize = 0;
            for (arr) |e| {
                if (e == .object) {
                    if (e.object.get("dir")) |d| if (e.object.get("rule")) |r| {
                        list[n] = .{ .dir = d.string, .rule = r.string };
                        n += 1;
                    };
                }
            }
            self.suppressed = list[0..n];
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try AnalysisConfig.fromEnv(allocator);
    var analyzer = StaticAnalyzer.init(allocator, config);
    defer analyzer.deinit();

    print("🚀 Running Enhanced Static Analysis on WDBX codebase...\n\n", .{});

    try analyzer.analyzeDirectory("src");
    try analyzer.analyzeDirectory("tests");
    try analyzer.analyzeDirectory("tools");
    try analyzer.analyzeDirectory("benchmarks");
    try analyzer.generateReport();
}
