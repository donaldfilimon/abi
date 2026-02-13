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
        var stream = std.io.fixedBufferStream(&ctx.message);
        stream.writer().print(fmt, args) catch {};
        ctx.message_len = @intCast(stream.pos);
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
        const stderr = std.io.getStdErr().writer();
        ctx.format(stderr) catch {};
        _ = self.total_logged.fetchAdd(1, .monotonic);
    }

    pub fn logMessage(self: *Logger, category: Category, severity: Severity, comptime fmt: []const u8, args: anytype) void {
        var ctx = Context.init(category, severity, fmt, args);
        self.log(&ctx);
    }
};

/// Global logger instance — configure severity at startup.
pub var global_logger: Logger = Logger.init(if (builtin.mode == .Debug) .debug else .warn);
