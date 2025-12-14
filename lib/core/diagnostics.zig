//! Diagnostics and Error Reporting System
//!
//! This module provides a comprehensive diagnostics system for better
//! error reporting, context propagation, and debugging.

const std = @import("std");
const ArrayList = std.array_list.Managed;

const io = @import("io.zig");

/// Severity levels for diagnostic messages
pub const Severity = enum {
    debug,
    info,
    warning,
    err,
    fatal,

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warning => "WARNING",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }

    pub fn color(self: Severity) []const u8 {
        return switch (self) {
            .debug => "\x1b[36m", // Cyan
            .info => "\x1b[32m", // Green
            .warning => "\x1b[33m", // Yellow
            .err => "\x1b[31m", // Red
            .fatal => "\x1b[35m", // Magenta
        };
    }
};

/// Source location information
pub const SourceLocation = struct {
    file: []const u8,
    line: u32,
    column: u32,

    pub fn format(
        self: SourceLocation,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{d}:{d}", .{ self.file, self.line, self.column });
    }
};

/// Diagnostic message with context
pub const Diagnostic = struct {
    severity: Severity,
    message: []const u8,
    location: ?SourceLocation = null,
    context: ?[]const u8 = null,
    timestamp: i64,

    pub fn init(severity: Severity, message: []const u8) Diagnostic {
        return .{
            .severity = severity,
            .message = message,
            .timestamp = 0,
        };
    }

    pub fn withLocation(self: Diagnostic, location: SourceLocation) Diagnostic {
        var diag = self;
        diag.location = location;
        return diag;
    }

    pub fn withContext(self: Diagnostic, context: []const u8) Diagnostic {
        var diag = self;
        diag.context = context;
        return diag;
    }

    pub fn format(
        self: Diagnostic,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        // Color prefix
        try writer.writeAll(self.severity.color());
        try writer.writeAll(self.severity.toString());
        try writer.writeAll("\x1b[0m: ");

        // Location if available
        if (self.location) |loc| {
            try writer.print("{} - ", .{loc});
        }

        // Message
        try writer.writeAll(self.message);

        // Context if available
        if (self.context) |ctx| {
            try writer.print("\n  Context: {s}", .{ctx});
        }
    }
};

/// Diagnostic collector for aggregating messages
pub const DiagnosticCollector = struct {
    diagnostics: ArrayList(Diagnostic),
    max_errors: usize,
    error_count: usize,
    warning_count: usize,

    pub fn init(allocator: std.mem.Allocator) DiagnosticCollector {
        return .{
            .diagnostics = ArrayList(Diagnostic).init(allocator),
            .max_errors = 100,
            .error_count = 0,
            .warning_count = 0,
        };
    }

    pub fn deinit(self: *DiagnosticCollector) void {
        self.diagnostics.deinit();
    }

    pub fn add(self: *DiagnosticCollector, diagnostic: Diagnostic) !void {
        switch (diagnostic.severity) {
            .err, .fatal => {
                self.error_count += 1;
                if (self.error_count > self.max_errors) {
                    return error.TooManyErrors;
                }
            },
            .warning => self.warning_count += 1,
            else => {},
        }

        try self.diagnostics.append(diagnostic);
    }

    pub fn hasErrors(self: *const DiagnosticCollector) bool {
        return self.error_count > 0;
    }

    pub fn emit(self: *const DiagnosticCollector, writer: io.Writer) !void {
        for (self.diagnostics.items) |diag| {
            try writer.print("{}\n", .{diag});
        }

        if (self.error_count > 0 or self.warning_count > 0) {
            try writer.print("\n{d} error(s), {d} warning(s)\n", .{
                self.error_count,
                self.warning_count,
            });
        }
    }

    pub fn clear(self: *DiagnosticCollector) void {
        self.diagnostics.clearRetainingCapacity();
        self.error_count = 0;
        self.warning_count = 0;
    }
};

/// Error with context information
pub const ErrorContext = struct {
    err: anyerror,
    message: []const u8,
    location: ?SourceLocation = null,
    cause: ?*const ErrorContext = null,

    pub fn init(err: anyerror, message: []const u8) ErrorContext {
        return .{
            .err = err,
            .message = message,
        };
    }

    pub fn withLocation(self: ErrorContext, location: SourceLocation) ErrorContext {
        var ctx = self;
        ctx.location = location;
        return ctx;
    }

    pub fn withCause(self: ErrorContext, cause: *const ErrorContext) ErrorContext {
        var ctx = self;
        ctx.cause = cause;
        return ctx;
    }

    pub fn format(
        self: ErrorContext,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Error: {s} ({s})", .{ self.message, @errorName(self.err) });

        if (self.location) |loc| {
            try writer.print("\n  at {}", .{loc});
        }

        if (self.cause) |cause| {
            try writer.print("\nCaused by: {}", .{cause.*});
        }
    }
};

/// Macro for creating source location at compile time
pub inline fn here() SourceLocation {
    return .{
        .file = @src().file,
        .line = @src().line,
        .column = @src().column,
    };
}

test "Diagnostic: basic creation and formatting" {
    const testing = std.testing;

    const diag = Diagnostic.init(.err, "Test error message");
    try testing.expect(diag.severity == .err);
    try testing.expectEqualStrings("Test error message", diag.message);
}

test "Diagnostic: with location" {
    const testing = std.testing;

    const loc = SourceLocation{
        .file = "test.zig",
        .line = 42,
        .column = 10,
    };

    const diag = Diagnostic.init(.warning, "Test warning")
        .withLocation(loc);

    try testing.expect(diag.location != null);
    try testing.expectEqualStrings("test.zig", diag.location.?.file);
}

test "DiagnosticCollector: collecting and emitting" {
    const testing = std.testing;

    var collector = DiagnosticCollector.init(testing.allocator);
    defer collector.deinit();

    try collector.add(Diagnostic.init(.warning, "Warning 1"));
    try collector.add(Diagnostic.init(.err, "Error 1"));
    try collector.add(Diagnostic.init(.info, "Info 1"));

    try testing.expect(collector.hasErrors());
    try testing.expect(collector.error_count == 1);
    try testing.expect(collector.warning_count == 1);
}

test "ErrorContext: error chain" {
    const testing = std.testing;

    const root_cause = ErrorContext.init(error.FileNotFound, "Config file missing");
    const ctx = ErrorContext.init(error.InvalidConfiguration, "Failed to load config")
        .withCause(&root_cause);

    try testing.expect(ctx.cause != null);
    try testing.expectEqual(error.FileNotFound, ctx.cause.?.err);
}
