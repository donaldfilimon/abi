//! I/O Abstraction Layer
//!
//! This module provides a unified I/O abstraction layer to eliminate
//! direct stdout/stderr usage and enable better testing and composition.

const std = @import("std");

/// Writer abstraction that can be injected throughout the framework
pub const Writer = struct {
    context: *anyopaque,
    writeFn: *const fn (context: *anyopaque, bytes: []const u8) anyerror!usize,

    pub fn write(self: Writer, bytes: []const u8) !usize {
        return self.writeFn(self.context, bytes);
    }

    pub fn print(self: Writer, comptime fmt: []const u8, args: anytype) !void {
        var buf: [4096]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, fmt, args);
        _ = try self.write(text);
    }

    /// Create a Writer from any std.io.Writer
    pub fn fromAnyWriter(writer: anytype) Writer {
        const T = @TypeOf(writer);
        const ContextType = struct {
            writer: T,
        };

        const impl = struct {
            fn writeFn(ctx: *anyopaque, bytes: []const u8) anyerror!usize {
                const context: *ContextType = @ptrCast(@alignCast(ctx));
                return context.writer.write(bytes);
            }
        };

        const ctx = std.heap.page_allocator.create(ContextType) catch unreachable;
        ctx.* = .{ .writer = writer };

        return .{
            .context = ctx,
            .writeFn = impl.writeFn,
        };
    }

    /// Create a Writer that outputs to stdout
    pub fn stdout() Writer {
        return fromAnyWriter(std.io.getStdOut().writer());
    }

    /// Create a Writer that outputs to stderr
    pub fn stderr() Writer {
        return fromAnyWriter(std.io.getStdErr().writer());
    }

    /// Create a null Writer that discards all output
    pub fn null() Writer {
        const impl = struct {
            fn writeFn(_: *anyopaque, bytes: []const u8) anyerror!usize {
                return bytes.len;
            }
        };

        return .{
            .context = undefined,
            .writeFn = impl.writeFn,
        };
    }
};

/// Buffered writer for performance-critical paths
pub const BufferedWriter = struct {
    buffer: std.ArrayList(u8),
    backing_writer: Writer,

    pub fn init(allocator: std.mem.Allocator, backing_writer: Writer) BufferedWriter {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
            .backing_writer = backing_writer,
        };
    }

    pub fn deinit(self: *BufferedWriter) void {
        self.buffer.deinit();
    }

    pub fn write(self: *BufferedWriter, bytes: []const u8) !usize {
        try self.buffer.appendSlice(bytes);
        return bytes.len;
    }

    pub fn flush(self: *BufferedWriter) !void {
        _ = try self.backing_writer.write(self.buffer.items);
        self.buffer.clearRetainingCapacity();
    }

    pub fn writer(self: *BufferedWriter) Writer {
        const impl = struct {
            fn writeFn(ctx: *anyopaque, bytes: []const u8) anyerror!usize {
                const buf_writer: *BufferedWriter = @ptrCast(@alignCast(ctx));
                return buf_writer.write(bytes);
            }
        };

        return .{
            .context = self,
            .writeFn = impl.writeFn,
        };
    }
};

/// Test writer that captures output for testing
pub const TestWriter = struct {
    buffer: std.ArrayList(u8),

    pub fn init(allocator: std.mem.Allocator) TestWriter {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
        };
    }

    pub fn deinit(self: *TestWriter) void {
        self.buffer.deinit();
    }

    pub fn write(self: *TestWriter, bytes: []const u8) !usize {
        try self.buffer.appendSlice(bytes);
        return bytes.len;
    }

    pub fn getWritten(self: *TestWriter) []const u8 {
        return self.buffer.items;
    }

    pub fn clear(self: *TestWriter) void {
        self.buffer.clearRetainingCapacity();
    }

    pub fn writer(self: *TestWriter) Writer {
        const impl = struct {
            fn writeFn(ctx: *anyopaque, bytes: []const u8) anyerror!usize {
                const test_writer: *TestWriter = @ptrCast(@alignCast(ctx));
                return test_writer.write(bytes);
            }
        };

        return .{
            .context = self,
            .writeFn = impl.writeFn,
        };
    }
};

/// Output context for structured I/O throughout the framework
pub const OutputContext = struct {
    stdout: Writer,
    stderr: Writer,
    log_writer: ?Writer = null,

    pub fn default() OutputContext {
        return .{
            .stdout = Writer.stdout(),
            .stderr = Writer.stderr(),
        };
    }

    pub fn withLogWriter(stdout_writer: Writer, stderr_writer: Writer, log_writer: Writer) OutputContext {
        return .{
            .stdout = stdout_writer,
            .stderr = stderr_writer,
            .log_writer = log_writer,
        };
    }

    pub fn silent() OutputContext {
        return .{
            .stdout = Writer.null(),
            .stderr = Writer.null(),
        };
    }
};

test "Writer: basic write operations" {
    const testing = std.testing;

    var test_writer = TestWriter.init(testing.allocator);
    defer test_writer.deinit();

    const writer = test_writer.writer();
    _ = try writer.write("Hello, ");
    _ = try writer.write("World!");

    try testing.expectEqualStrings("Hello, World!", test_writer.getWritten());
}

test "Writer: formatted print" {
    const testing = std.testing;

    var test_writer = TestWriter.init(testing.allocator);
    defer test_writer.deinit();

    const writer = test_writer.writer();
    try writer.print("Number: {d}\n", .{42});

    try testing.expectEqualStrings("Number: 42\n", test_writer.getWritten());
}

test "BufferedWriter: buffering and flushing" {
    const testing = std.testing;

    var test_writer = TestWriter.init(testing.allocator);
    defer test_writer.deinit();

    var buf_writer = BufferedWriter.init(testing.allocator, test_writer.writer());
    defer buf_writer.deinit();

    _ = try buf_writer.write("Buffered ");
    try testing.expectEqualStrings("", test_writer.getWritten());

    _ = try buf_writer.write("output");
    try buf_writer.flush();

    try testing.expectEqualStrings("Buffered output", test_writer.getWritten());
}
