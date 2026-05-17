const std = @import("std");

pub const Status = enum {
    ready,
    busy,
    warning,
    disabled,
};

pub const Item = struct {
    label: []const u8,
    value: []const u8,
};

pub const State = struct {
    title: []const u8,
    status: Status = .ready,
    items: []const Item = &.{},
};

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    try output.print(allocator, "+------------------------------+\n", .{});
    try output.print(allocator, "| {s:<28} |\n", .{state.title});
    try output.print(allocator, "+------------------------------+\n", .{});
    try output.print(allocator, "status: {s}\n", .{statusText(state.status)});
    for (state.items) |item| {
        try output.print(allocator, "- {s}: {s}\n", .{ item.label, item.value });
    }
    try output.print(allocator, "\nCommands: abi help | abi agent train all | abi agent os dry-run <cmd>\n", .{});

    return output.toOwnedSlice(allocator);
}

pub fn initScreen() !void {
    std.debug.print("\x1b[?1049h\x1b[H", .{});
}

pub fn initScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049h\x1b[H");
}

pub fn clearScreen() !void {
    std.debug.print("\x1b[2J\x1b[H", .{});
}

pub fn clearScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[2J\x1b[H");
}

pub fn render(state: ScreenState) !void {
    std.debug.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    std.debug.print("Agents: abbey, aviva, abi | WDBX: in-memory training records\n", .{});
}

pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    try writer.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    try writer.writeAll("Agents: abbey, aviva, abi | WDBX: in-memory training records\n");
}

pub fn deinitScreen() void {
    std.debug.print("\x1b[?1049l", .{});
}

pub fn deinitScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049l");
}

test "dashboard requires a title" {
    try std.testing.expectError(error.InvalidTuiState, renderDashboard(std.testing.allocator, .{ .title = "" }));
}

test "dashboard renders status and items" {
    const rendered = try renderDashboard(std.testing.allocator, .{
        .title = "ABI",
        .status = .warning,
        .items = &.{.{ .label = "AI", .value = "safe" }},
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "status: warning") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "- AI: safe") != null);
}

test "writer render functions are testable" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }

        pub fn print(self: *@This(), comptime fmt: []const u8, args: anytype) !void {
            try self.buffer.print(self.allocator, fmt, args);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };

    try renderWriter(&writer, .{ .width = 80, .height = 24 });
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "80x24") != null);
}
