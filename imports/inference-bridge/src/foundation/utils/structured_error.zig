// ============================================================================
// ABI Framework - Structured Error Handling
// ============================================================================
//
// Adapted from abi-system-v2.0 error.zig module. Provides categorized errors
// with context propagation, structured logging, and composable error chains.
// Designed for production diagnostics without heap allocation on the error path.
//
// Changes from v2.0:
//   - Self-contained (no external utils dependency)
//   - Zig 0.16 compatible (no std.time.nanoTimestamp)
//   - Inline std.atomic.Value for Logger counter
//   - Uses @import("builtin") for debug detection
// ============================================================================

const std = @import("std");
const builtin = @import("builtin");

// --- Error Categories -------------------------------------------------------

pub const Category = enum(u8) {
    memory = 0,
    gpu = 1,
    simd = 2,
    io = 3,
    config = 4,
    runtime = 5,
    validation = 6,

    pub fn label(self: Category) []const u8 {
        return switch (self) {
            .memory => "MEMORY",
            .gpu => "GPU",
            .simd => "SIMD",
            .io => "IO",
            .config => "CONFIG",
            .runtime => "RUNTIME",
            .validation => "VALIDATION",
        };
    }
};

pub const Severity = enum(u8) {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    fatal = 5,

    pub fn label(self: Severity) []const u8 {
        return switch (self) {
            .trace => "TRACE",
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
            .fatal => "FATAL",
        };
    }

    pub fn ansiColor(self: Severity) []const u8 {
        return switch (self) {
            .trace => "\x1b[90m",
            .debug => "\x1b[36m",
            .info => "\x1b[32m",
            .warn => "\x1b[33m",
            .err => "\x1b[31m",
            .fatal => "\x1b[1;31m",
        };
    }
};

// --- Error Context ----------------------------------------------------------

/// Fixed-size error context that lives on the stack. No heap allocation
/// on the error path — critical for predictable latency.
pub const Context = struct {
    category: Category,
    severity: Severity,
    message: [256]u8 = undefined,
    message_len: u8 = 0,
    source_file: [128]u8 = undefined,
    source_file_len: u8 = 0,
    source_line: u32 = 0,
    timestamp_ns: i128 = 0,

    pub fn init(category: Category, severity: Severity, comptime fmt: []const u8, args: anytype) Context {
        var ctx = Context{
            .category = category,
            .severity = severity,
            // Absolute timestamps are not available in Zig 0.16 without the
            // I/O backend. Use relative timing via std.time.Instant if needed.
            .timestamp_ns = 0,
        };
        var writer = std.Io.Writer.fixed(&ctx.message);
        writer.print(fmt, args) catch {
            // Message truncated to fit fixed buffer — append "..." indicator
            if (writer.end + 3 <= ctx.message.len) {
                @memcpy(ctx.message[writer.end..][0..3], "...");
                writer.end += 3;
            }
        };
        ctx.message_len = @intCast(writer.end);
        return ctx;
    }

    pub fn withSource(self: *Context, file: []const u8, line: u32) *Context {
        const copy_len = @min(file.len, self.source_file.len);
        @memcpy(self.source_file[0..copy_len], file[0..copy_len]);
        self.source_file_len = @intCast(copy_len);
        self.source_line = line;
        return self;
    }

    pub fn getMessage(self: *const Context) []const u8 {
        return self.message[0..self.message_len];
    }

    pub fn format(self: *const Context, writer: anytype) !void {
        try writer.print("{s}{s}\x1b[0m [{s}] {s}", .{
            self.severity.ansiColor(),
            self.severity.label(),
            self.category.label(),
            self.getMessage(),
        });
        if (self.source_file_len > 0) {
            try writer.print(" ({s}:{d})", .{
                self.source_file[0..self.source_file_len],
                self.source_line,
            });
        }
        try writer.writeByte('\n');
    }
};

// --- Error Accumulator ------------------------------------------------------

/// Collects errors during a complex operation, enabling batch reporting.
/// Fixed capacity — no heap allocation.
pub fn ErrorAccumulator(comptime max_errors: usize) type {
    return struct {
        const Self = @This();

        errors: [max_errors]Context = undefined,
        count: usize = 0,
        overflow: bool = false,

        pub fn push(self: *Self, ctx: Context) void {
            if (self.count < max_errors) {
                self.errors[self.count] = ctx;
                self.count += 1;
            } else {
                self.overflow = true;
            }
        }

        pub fn pushMessage(self: *Self, category: Category, severity: Severity, comptime fmt: []const u8, args: anytype) void {
            self.push(Context.init(category, severity, fmt, args));
        }

        pub fn hasErrors(self: *const Self) bool {
            for (self.errors[0..self.count]) |*e| {
                if (@intFromEnum(e.severity) >= @intFromEnum(Severity.err)) return true;
            }
            return false;
        }

        pub fn dump(self: *const Self, writer: anytype) !void {
            for (self.errors[0..self.count]) |*e| {
                try e.format(writer);
            }
            if (self.overflow) {
                try writer.writeAll("\x1b[33mWARN\x1b[0m [SYSTEM] Error buffer overflow — some errors were dropped\n");
            }
        }

        pub fn clear(self: *Self) void {
            self.count = 0;
            self.overflow = false;
        }

        pub fn slice(self: *const Self) []const Context {
            return self.errors[0..self.count];
        }
    };
}

// --- Unified Error Set ------------------------------------------------------

pub const AbiError = error{
    OutOfMemory,
    BufferTooSmall,
    AlignmentViolation,
    PoolExhausted,
    InvalidSize,
    GpuNotAvailable,
    GpuOutOfMemory,
    GpuMapFailed,
    ShaderCompileFailed,
    SimdNotSupported,
    InvalidDimensions,
    DimensionMismatch,
    IoError,
    ConfigError,
    InvalidArgument,
    Overflow,
    Timeout,
};

// --- Convenience Constructors -----------------------------------------------

pub fn memoryError(comptime fmt: []const u8, args: anytype) Context {
    return Context.init(.memory, .err, fmt, args);
}

pub fn gpuError(comptime fmt: []const u8, args: anytype) Context {
    return Context.init(.gpu, .err, fmt, args);
}

pub fn simdError(comptime fmt: []const u8, args: anytype) Context {
    return Context.init(.simd, .err, fmt, args);
}

pub fn validationError(comptime fmt: []const u8, args: anytype) Context {
    return Context.init(.validation, .err, fmt, args);
}

pub fn info(category: Category, comptime fmt: []const u8, args: anytype) Context {
    return Context.init(category, .info, fmt, args);
}

// --- Logger -----------------------------------------------------------------

pub const Logger = struct {
    min_severity: Severity,
    category_filter: ?Category,
    total_logged: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn init(min_severity: Severity) Logger {
        return .{ .min_severity = min_severity, .category_filter = null };
    }

    pub fn log(self: *Logger, ctx: *const Context) void {
        if (@intFromEnum(ctx.severity) < @intFromEnum(self.min_severity)) return;
        if (self.category_filter) |filter| {
            if (ctx.category != filter) return;
        }
        // Format into stack buffer, then write to stderr via C fd
        var buf: [4096]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buf);
        ctx.format(&writer) catch return;
        const output = buf[0..writer.end];
        if (output.len > 0) {
            _ = std.c.write(2, output.ptr, output.len);
        }
        _ = self.total_logged.fetchAdd(1, .monotonic);
    }

    pub fn logMessage(self: *Logger, category: Category, severity: Severity, comptime fmt: []const u8, args: anytype) void {
        var ctx = Context.init(category, severity, fmt, args);
        self.log(&ctx);
    }
};

/// Global logger instance — configure severity at startup.
pub var global_logger: Logger = Logger.init(if (builtin.mode == .Debug) .debug else .warn);

// ─── Tests ──────────────────────────────────────────────────────────────────

test "Category labels" {
    try std.testing.expectEqualStrings("MEMORY", Category.memory.label());
    try std.testing.expectEqualStrings("GPU", Category.gpu.label());
    try std.testing.expectEqualStrings("IO", Category.io.label());
    try std.testing.expectEqualStrings("VALIDATION", Category.validation.label());
}

test "Severity labels and ordering" {
    try std.testing.expectEqualStrings("TRACE", Severity.trace.label());
    try std.testing.expectEqualStrings("ERROR", Severity.err.label());
    try std.testing.expectEqualStrings("FATAL", Severity.fatal.label());

    // Enum ordering: trace < debug < info < warn < err < fatal
    try std.testing.expect(@intFromEnum(Severity.trace) < @intFromEnum(Severity.err));
    try std.testing.expect(@intFromEnum(Severity.err) < @intFromEnum(Severity.fatal));
}

test "Context init and getMessage" {
    const ctx = Context.init(.memory, .err, "allocation failed: {d} bytes", .{@as(u64, 1024)});
    const msg = ctx.getMessage();
    try std.testing.expect(msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, msg, "1024") != null);
    try std.testing.expectEqual(Category.memory, ctx.category);
    try std.testing.expectEqual(Severity.err, ctx.severity);
}

test "Context withSource" {
    var ctx = Context.init(.gpu, .warn, "shader issue", .{});
    _ = ctx.withSource("src/gpu.zig", 42);
    try std.testing.expectEqual(@as(u32, 42), ctx.source_line);
    try std.testing.expect(ctx.source_file_len > 0);
}

test "Context format output" {
    const ctx = Context.init(.io, .info, "connected", .{});
    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try ctx.format(&writer);
    const output = buf[0..writer.end];
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "INFO") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "IO") != null);
}

test "ErrorAccumulator collects and reports" {
    var acc = ErrorAccumulator(4){};
    acc.pushMessage(.memory, .err, "alloc fail", .{});
    acc.pushMessage(.gpu, .warn, "shader warn", .{});

    try std.testing.expectEqual(@as(usize, 2), acc.count);
    try std.testing.expect(acc.hasErrors());
    try std.testing.expect(!acc.overflow);

    acc.clear();
    try std.testing.expectEqual(@as(usize, 0), acc.count);
    try std.testing.expect(!acc.hasErrors());
}

test "ErrorAccumulator overflow" {
    var acc = ErrorAccumulator(2){};
    acc.pushMessage(.memory, .info, "first", .{});
    acc.pushMessage(.gpu, .info, "second", .{});
    acc.pushMessage(.io, .info, "overflow", .{}); // should overflow
    try std.testing.expectEqual(@as(usize, 2), acc.count);
    try std.testing.expect(acc.overflow);
}

test "convenience constructors" {
    const e = memoryError("OOM at {d}", .{@as(usize, 0)});
    try std.testing.expectEqual(Category.memory, e.category);
    try std.testing.expectEqual(Severity.err, e.severity);

    const g = gpuError("device lost", .{});
    try std.testing.expectEqual(Category.gpu, g.category);

    const i = info(.config, "loaded", .{});
    try std.testing.expectEqual(Category.config, i.category);
    try std.testing.expectEqual(Severity.info, i.severity);
}

test "Logger filtering by severity" {
    var logger = Logger.init(.warn);
    // Debug < warn → should be filtered out
    var ctx = Context.init(.memory, .debug, "debug msg", .{});
    logger.log(&ctx); // should not crash, just skip
    try std.testing.expectEqual(@as(u64, 0), logger.total_logged.load(.acquire));
}

test {
    std.testing.refAllDecls(@This());
}
