//! Inline CLI TUI editor command.
//!
//! Controls:
//! - Ctrl-S: save
//! - Ctrl-Q: quit (double press if unsaved)
//! - Arrows/Home/End/PageUp/PageDown: navigation
//! - Enter/Backspace/Delete/Tab: editing

const std = @import("std");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const tui = @import("../tui/mod.zig");

const EditorError = error{
    TerminalNotAttached,
};

const max_file_size = 8 * 1024 * 1024;
const gutter_cols: usize = 6;
const tab_width: usize = 4;
const message_ttl_ticks: u8 = 24;

pub const meta: command_mod.Meta = .{
    .name = "editor",
    .description = "Open an inline Cursor-like terminal text editor",
    .aliases = &.{"edit"},
    .subcommands = &.{"help"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (containsHelpArg(args)) {
        printHelp();
        return;
    }

    var file_path: ?[]const u8 = null;
    if (args.len > 0) {
        const arg = std.mem.sliceTo(args[0], 0);
        if (arg.len > 0 and arg[0] == '-') {
            std.debug.print("Unknown editor option: {s}\n", .{arg});
            printHelp();
            return error.InvalidArgument;
        }
        file_path = arg;
    }
    if (args.len > 1) {
        std.debug.print("editor accepts at most one file path.\n", .{});
        printHelp();
        return error.InvalidArgument;
    }

    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        std.debug.print("TUI editor is not supported on {s}\n", .{caps.platform_name});
        return;
    }

    var editor = try EditorState.init(ctx.allocator, ctx.io, file_path);
    defer editor.deinit();

    var terminal = tui.Terminal.init(ctx.allocator);
    defer terminal.deinit();

    try terminal.enter();
    defer terminal.exit() catch {};

    editor.attachTerminal(&terminal);

    while (true) {
        try editor.render();
        const ev = try terminal.readEvent();
        if (try editor.handleEvent(ev)) break;
    }
}

/// Checks if any argument in the provided slice is a help flag.
/// Recognizes `--help`, `-h`, or `help` as help arguments.
fn containsHelpArg(args: []const [:0]const u8) bool {
    for (args) |arg_z| {
        const arg = std.mem.sliceTo(arg_z, 0);
        if (arg) |a| {
            if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "help")) {
                return true;
            }
        }
    }
    return false;
}

/// Prints the editor command help text to stdout.
fn printHelp() void {
    const text =
        \\Usage: abi editor [file]
        \\
        \\Open a full-screen inline terminal text editor.
        \\
        \\Arguments:
        \\  file       Optional path to open (or create on save)
        \\
        \\Controls:
        \\  Ctrl-S     Save
        \\  Ctrl-Q     Quit (press twice if unsaved)
        \\  Arrows     Move cursor
        \\  Home/End   Move to line start/end
        \\  PageUp/Down Scroll by viewport
        \\  Enter      New line
        \\  Backspace  Delete backward
        \\  Delete     Delete forward
        \\  Tab        Insert spaces
        \\
        \\Examples:
        \\  abi editor
        \\  abi editor build.zig
        \\
    ;
    std.debug.print("{s}", .{text});
}

/// Represents a single line of text in the editor buffer.
/// Each line stores its content as a byte array.
const Line = struct {
    bytes: std.ArrayListUnmanaged(u8) = .empty,

    fn fromSlice(allocator: std.mem.Allocator, text: []const u8) !Line {
        var line: Line = .{};
        try line.bytes.appendSlice(allocator, text);
        return line;
    }

    fn deinit(self: *Line, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
        self.* = undefined;
    }
};

/// Manages the text content of the editor as a collection of lines.
/// Tracks modification state and provides operations for editing text.
const TextBuffer = struct {
    allocator: std.mem.Allocator,
    lines: std.ArrayListUnmanaged(Line) = .empty,
    modified: bool = false,

    fn initEmpty(allocator: std.mem.Allocator) !TextBuffer {
        var out: TextBuffer = .{ .allocator = allocator };
        try out.lines.append(allocator, .{});
        return out;
    }

    fn initFromSlice(allocator: std.mem.Allocator, text: []const u8) !TextBuffer {
        var out: TextBuffer = .{ .allocator = allocator };
        errdefer out.deinit();

        if (text.len == 0) {
            try out.lines.append(allocator, .{});
            return out;
        }

        var start: usize = 0;
        var idx: usize = 0;
        while (idx <= text.len) : (idx += 1) {
            if (idx == text.len or text[idx] == '\n') {
                var slice = text[start..idx];
                if (slice.len > 0 and slice[slice.len - 1] == '\r') {
                    slice = slice[0 .. slice.len - 1];
                }
                try out.lines.append(allocator, try Line.fromSlice(allocator, slice));
                start = idx + 1;
            }
        }

        if (out.lines.items.len == 0) {
            try out.lines.append(allocator, .{});
        }
        out.modified = false;
        return out;
    }

    fn loadFromPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !TextBuffer {
        const content = std.Io.Dir.cwd().readFileAlloc(
            io,
            path,
            allocator,
            .limited(max_file_size),
        ) catch |err| switch (err) {
            error.FileNotFound => return TextBuffer.initEmpty(allocator),
            else => return err,
        };
        defer allocator.free(content);
        return TextBuffer.initFromSlice(allocator, content);
    }

    fn deinit(self: *TextBuffer) void {
        for (self.lines.items) |*line_item| {
            line_item.deinit(self.allocator);
        }
        self.lines.deinit(self.allocator);
        self.* = undefined;
    }

    fn line(self: *TextBuffer, row: usize) *Line {
        return &self.lines.items[row];
    }

    fn lineLen(self: *TextBuffer, row: usize) usize {
        return self.lines.items[row].bytes.items.len;
    }

    fn clampCursor(self: *TextBuffer, row: *usize, col: *usize) !void {
        if (self.lines.items.len == 0) {
            try self.lines.append(self.allocator, .{});
        }
        if (row.* >= self.lines.items.len) row.* = self.lines.items.len - 1;
        const len = self.lineLen(row.*);
        if (col.* > len) col.* = len;
    }

    fn insertByte(self: *TextBuffer, row: usize, col: usize, ch: u8) !void {
        var line_ptr = self.line(row);
        const at = @min(col, line_ptr.bytes.items.len);
        try line_ptr.bytes.insert(self.allocator, at, ch);
        self.modified = true;
    }

    fn insertNewline(self: *TextBuffer, row: *usize, col: *usize) !void {
        var current = self.line(row.*);
        const split_at = @min(col.*, current.bytes.items.len);

        var new_line: Line = .{};
        errdefer new_line.deinit(self.allocator);
        try new_line.bytes.appendSlice(self.allocator, current.bytes.items[split_at..]);
        current.bytes.items.len = split_at;

        try self.lines.insert(self.allocator, row.* + 1, new_line);
        row.* += 1;
        col.* = 0;
        self.modified = true;
    }

    fn backspace(self: *TextBuffer, row: *usize, col: *usize) !void {
        if (col.* > 0) {
            var current = self.line(row.*);
            _ = current.bytes.orderedRemove(col.* - 1);
            col.* -= 1;
            self.modified = true;
            return;
        }

        if (row.* == 0) return;
        const prev_row = row.* - 1;
        const prev_len = self.lineLen(prev_row);
        var prev = self.line(prev_row);
        const current = self.line(row.*);
        try prev.bytes.appendSlice(self.allocator, current.bytes.items);

        var removed = self.lines.orderedRemove(row.*);
        removed.deinit(self.allocator);

        row.* = prev_row;
        col.* = prev_len;
        self.modified = true;
    }

    fn deleteForward(self: *TextBuffer, row: *usize, col: *usize) !void {
        var current = self.line(row.*);
        if (col.* < current.bytes.items.len) {
            _ = current.bytes.orderedRemove(col.*);
            self.modified = true;
            return;
        }

        if (row.* + 1 >= self.lines.items.len) return;
        try current.bytes.appendSlice(self.allocator, self.lines.items[row.* + 1].bytes.items);

        var removed = self.lines.orderedRemove(row.* + 1);
        removed.deinit(self.allocator);
        self.modified = true;
    }

    fn saveToPath(self: *TextBuffer, io: std.Io, path: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);

        for (self.lines.items, 0..) |line_item, idx| {
            try file.writeStreamingAll(io, line_item.bytes.items);
            if (idx + 1 < self.lines.items.len) {
                try file.writeStreamingAll(io, "\n");
            }
        }
        self.modified = false;
    }
};

/// Manages the complete editor state including buffer, cursor position,
/// viewport offsets, and UI rendering. Handles user input events and
/// coordinates between the text buffer and terminal display.
const EditorState = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    terminal: ?*tui.Terminal = null,
    buffer: TextBuffer,
    file_path: ?[]const u8 = null,
    cursor_row: usize = 0,
    cursor_col: usize = 0,
    row_offset: usize = 0,
    col_offset: usize = 0,
    quit_armed: bool = false,
    message_storage: [192]u8 = undefined,
    message_len: usize = 0,
    message_ticks: u8 = 0,

    fn init(allocator: std.mem.Allocator, io: std.Io, file_path: ?[]const u8) !EditorState {
        const buffer = if (file_path) |path|
            try TextBuffer.loadFromPath(allocator, io, path)
        else
            try TextBuffer.initEmpty(allocator);

        var out: EditorState = .{
            .allocator = allocator,
            .io = io,
            .buffer = buffer,
            .file_path = file_path,
        };
        out.setStatus(if (file_path == null) "Scratch buffer" else "File loaded");
        return out;
    }

    fn deinit(self: *EditorState) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    fn attachTerminal(self: *EditorState, term: *tui.Terminal) void {
        self.terminal = term;
    }

    fn render(self: *EditorState) EditorError!void {
        const term = self.terminal orelse return error.TerminalNotAttached;
        const size = term.size();

        try term.clear();
        try self.renderHeader(size);
        try self.renderBody(size);
        try self.renderStatusBar(size);
        try self.placeCursor(size);
        try term.flush();

        if (self.message_ticks > 0) self.message_ticks -= 1;
    }

    fn handleEvent(self: *EditorState, event: tui.Event) (EditorError || std.mem.Allocator.Error)!bool {
        const term = self.terminal orelse return error.TerminalNotAttached;
        const size = term.size();

        switch (event) {
            .mouse => return false,
            .key => |key| switch (key.code) {
                .ctrl_c => return true,
                .up => {
                    self.quit_armed = false;
                    if (self.cursor_row > 0) self.cursor_row -= 1;
                },
                .down => {
                    self.quit_armed = false;
                    if (self.cursor_row + 1 < self.buffer.lines.items.len) self.cursor_row += 1;
                },
                .left => {
                    self.quit_armed = false;
                    if (self.cursor_col > 0) {
                        self.cursor_col -= 1;
                    } else if (self.cursor_row > 0) {
                        self.cursor_row -= 1;
                        self.cursor_col = self.buffer.lineLen(self.cursor_row);
                    }
                },
                .right => {
                    self.quit_armed = false;
                    const len = self.buffer.lineLen(self.cursor_row);
                    if (self.cursor_col < len) {
                        self.cursor_col += 1;
                    } else if (self.cursor_row + 1 < self.buffer.lines.items.len) {
                        self.cursor_row += 1;
                        self.cursor_col = 0;
                    }
                },
                .home => {
                    self.quit_armed = false;
                    self.cursor_col = 0;
                },
                .end => {
                    self.quit_armed = false;
                    self.cursor_col = self.buffer.lineLen(self.cursor_row);
                },
                .page_up => {
                    self.quit_armed = false;
                    const page = contentRows(size.rows);
                    if (self.cursor_row > page) self.cursor_row -= page else self.cursor_row = 0;
                },
                .page_down => {
                    self.quit_armed = false;
                    const page = contentRows(size.rows);
                    const max_row = self.buffer.lines.items.len - 1;
                    self.cursor_row = @min(self.cursor_row + page, max_row);
                },
                .enter => {
                    self.quit_armed = false;
                    try self.buffer.insertNewline(&self.cursor_row, &self.cursor_col);
                },
                .tab => {
                    self.quit_armed = false;
                    for (0..tab_width) |_| {
                        try self.buffer.insertByte(self.cursor_row, self.cursor_col, ' ');
                        self.cursor_col += 1;
                    }
                },
                .backspace => {
                    self.quit_armed = false;
                    try self.buffer.backspace(&self.cursor_row, &self.cursor_col);
                },
                .delete => {
                    self.quit_armed = false;
                    try self.buffer.deleteForward(&self.cursor_row, &self.cursor_col);
                },
                .character => {
                    if (key.char) |ch| {
                        if (ch == 17) {
                            if (self.buffer.modified and !self.quit_armed) {
                                self.quit_armed = true;
                                self.setStatus("Unsaved changes. Press Ctrl-Q again to quit.");
                            } else {
                                return true;
                            }
                        } else if (ch == 19) {
                            self.quit_armed = false;
                            try self.save();
                        } else if (ch >= 32 and ch <= 126) {
                            self.quit_armed = false;
                            try self.buffer.insertByte(self.cursor_row, self.cursor_col, ch);
                            self.cursor_col += 1;
                        }
                    }
                },
                else => {},
            },
        }

        try self.buffer.clampCursor(&self.cursor_row, &self.cursor_col);
        self.ensureCursorVisible(size);
        return false;
    }

    fn renderHeader(self: *EditorState, size: tui.TerminalSize) EditorError!void {
        const term = self.terminal orelse return error.TerminalNotAttached;
        const file_name = self.file_path orelse "[scratch]";
        const dirty = if (self.buffer.modified) " *" else "";

        var header_buf: [256]u8 = undefined;
        const header = std.fmt.bufPrint(
            &header_buf,
            " ABI Editor  {s}{s} ",
            .{ file_name, dirty },
        ) catch " ABI Editor ";

        try term.moveTo(0, 0);
        try term.write("\x1b[1;36m");
        try writeClipped(term, header, size.cols);
        try term.write("\x1b[0m\x1b[K");
    }

    fn renderBody(self: *EditorState, size: tui.TerminalSize) EditorError!void {
        const term = self.terminal orelse return error.TerminalNotAttached;
        const rows = contentRows(size.rows);
        const max_text_cols = textCols(size.cols);

        for (0..rows) |view_row| {
            const screen_row: u16 = @intCast(view_row + 1);
            try term.moveTo(screen_row, 0);

            const line_index = self.row_offset + view_row;
            if (line_index < self.buffer.lines.items.len) {
                var ln_buf: [16]u8 = undefined;
                const ln = std.fmt.bufPrint(&ln_buf, "{d: >4} ", .{line_index + 1}) catch "     ";
                try term.write("\x1b[90m");
                try term.write(ln);
                try term.write("\x1b[0m\x1b[90m|\x1b[0m");

                if (max_text_cols > 0) {
                    const line = self.buffer.lines.items[line_index].bytes.items;
                    if (self.col_offset < line.len) {
                        const start = self.col_offset;
                        const end = @min(line.len, start + max_text_cols);
                        try term.write(line[start..end]);
                    }
                }
            } else {
                try term.write("\x1b[90m~\x1b[0m");
            }

            try term.write("\x1b[K");
        }
    }

    fn renderStatusBar(self: *EditorState, size: tui.TerminalSize) EditorError!void {
        const term = self.terminal orelse return error.TerminalNotAttached;
        if (size.rows == 0) return;

        const status_row = size.rows - 1;
        const file_name = self.file_path orelse "[scratch]";
        const dirty = if (self.buffer.modified) "*" else "";

        var left_buf: [220]u8 = undefined;
        const left = std.fmt.bufPrint(
            &left_buf,
            " {s}{s}  Ln {d}, Col {d} ",
            .{ file_name, dirty, self.cursor_row + 1, self.cursor_col + 1 },
        ) catch " editor ";

        const controls = "Ctrl-S save  Ctrl-Q quit";
        const msg = self.message();

        try term.moveTo(status_row, 0);
        try term.write("\x1b[7m");
        try writeClipped(term, left, size.cols);

        if (size.cols > 0) {
            const left_cols = @min(left.len, @as(usize, size.cols));
            if (left_cols < @as(usize, size.cols)) {
                try term.write(" ");
                try writeClipped(term, controls, size.cols - @as(u16, @intCast(left_cols + 1)));
            }
        }
        try term.write("\x1b[K\x1b[0m");

        if (size.rows >= 2 and msg.len > 0) {
            const msg_row = size.rows - 2;
            try term.moveTo(msg_row, 0);
            try term.write("\x1b[33m");
            try writeClipped(term, msg, size.cols);
            try term.write("\x1b[0m\x1b[K");
        }
    }

    fn placeCursor(self: *EditorState, size: tui.TerminalSize) EditorError!void {
        const term = self.terminal orelse return error.TerminalNotAttached;
        const rows = contentRows(size.rows);
        if (rows == 0) return;
        if (size.cols == 0) return;

        if (self.cursor_row < self.row_offset) return;
        const view_row = self.cursor_row - self.row_offset;
        if (view_row >= rows) return;

        const max_cols = textCols(size.cols);
        var col = gutter_cols;
        if (max_cols > 0 and self.cursor_col >= self.col_offset) {
            col += @min(self.cursor_col - self.col_offset, max_cols - 1);
        }
        if (col >= size.cols) col = size.cols - 1;

        try term.moveTo(@intCast(view_row + 1), @intCast(col));
    }

    fn ensureCursorVisible(self: *EditorState, size: tui.TerminalSize) void {
        const rows = @max(contentRows(size.rows), 1);
        const cols = @max(textCols(size.cols), 1);

        if (self.cursor_row < self.row_offset) {
            self.row_offset = self.cursor_row;
        } else if (self.cursor_row >= self.row_offset + rows) {
            self.row_offset = self.cursor_row - rows + 1;
        }

        if (self.cursor_col < self.col_offset) {
            self.col_offset = self.cursor_col;
        } else if (self.cursor_col >= self.col_offset + cols) {
            self.col_offset = self.cursor_col - cols + 1;
        }
    }

    fn save(self: *EditorState) !void {
        const path = self.file_path orelse {
            self.setStatus("No file path. Re-run as: abi editor <file>");
            return;
        };

        self.buffer.saveToPath(self.io, path) catch |err| {
            self.setStatusFmt("Save failed: {t}", .{err});
            return;
        };
        self.setStatus("Saved");
    }

    fn setStatus(self: *EditorState, text: []const u8) void {
        const n = @min(text.len, self.message_storage.len);
        @memcpy(self.message_storage[0..n], text[0..n]);
        self.message_len = n;
        self.message_ticks = message_ttl_ticks;
    }

    fn setStatusFmt(self: *EditorState, comptime fmt: []const u8, args: anytype) void {
        const text = std.fmt.bufPrint(&self.message_storage, fmt, args) catch "status";
        self.message_len = text.len;
        self.message_ticks = message_ttl_ticks;
    }

    fn message(self: *const EditorState) []const u8 {
        if (self.message_ticks == 0) return "";
        return self.message_storage[0..self.message_len];
    }
};

/// Calculates the number of content rows available for text display.
/// Subtracts header and status bar rows from the total terminal rows.
fn contentRows(total_rows: u16) usize {
    if (total_rows <= 2) return 0;
    return total_rows - 2;
}

/// Calculates the number of text columns available for content display.
/// Subtracts the gutter (line numbers) width from the total terminal columns.
fn textCols(total_cols: u16) usize {
    if (total_cols <= gutter_cols) return 0;
    return total_cols - gutter_cols;
}

/// Writes text to the terminal, clipping it to fit within the maximum column width.
/// If the text exceeds max_cols, only the first max_cols characters are written.
fn writeClipped(term: *tui.Terminal, text: []const u8, max_cols: u16) !void {
    const cols: usize = max_cols;
    if (cols == 0) return;
    const len = @min(text.len, cols);
    try term.write(text[0..len]);
}

test "text buffer newline/backspace keeps expected cursor semantics" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "abc");
    defer buf.deinit();

    var row: usize = 0;
    var col: usize = 3;
    try buf.insertNewline(&row, &col);
    try std.testing.expectEqual(@as(usize, 2), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 1), row);
    try std.testing.expectEqual(@as(usize, 0), col);

    try buf.backspace(&row, &col);
    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), row);
    try std.testing.expectEqual(@as(usize, 3), col);
}

test "EditorState.init without file path" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var editor = try EditorState.init(std.testing.allocator, io, null);
    defer editor.deinit();

    try std.testing.expect(editor.file_path == null);
    try std.testing.expectEqual(@as(usize, 1), editor.buffer.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), editor.buffer.lines.items[0].bytes.items.len);
    try std.testing.expectEqual(@as(usize, 0), editor.cursor_row);
    try std.testing.expectEqual(@as(usize, 0), editor.cursor_col);
    try std.testing.expect(!editor.buffer.modified);
}

test "EditorState.init with file path" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Create a temporary file
    const test_file = "test_editor_init.txt";
    var file = try std.Io.Dir.cwd().createFile(io, test_file, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, "hello\nworld");
    file.close(io);

    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var editor = try EditorState.init(std.testing.allocator, io, test_file);
    defer editor.deinit();

    try std.testing.expect(editor.file_path != null);
    try std.testing.expectEqualStrings(test_file, editor.file_path.?);
    try std.testing.expectEqual(@as(usize, 2), editor.buffer.lines.items.len);
    try std.testing.expectEqualStrings("hello", editor.buffer.lines.items[0].bytes.items);
    try std.testing.expectEqualStrings("world", editor.buffer.lines.items[1].bytes.items);
}

test "TextBuffer.loadFromPath file not found" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const non_existent = "non_existent_file_12345.txt";
    var buf = try TextBuffer.loadFromPath(std.testing.allocator, io, non_existent);
    defer buf.deinit();

    // Should return empty buffer when file not found
    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), buf.lines.items[0].bytes.items.len);
}

test "TextBuffer.loadFromPath with valid file" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_load.txt";
    var file = try std.Io.Dir.cwd().createFile(io, test_file, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, "line1\nline2\nline3");
    file.close(io);

    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var buf = try TextBuffer.loadFromPath(std.testing.allocator, io, test_file);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 3), buf.lines.items.len);
    try std.testing.expectEqualStrings("line1", buf.lines.items[0].bytes.items);
    try std.testing.expectEqualStrings("line2", buf.lines.items[1].bytes.items);
    try std.testing.expectEqualStrings("line3", buf.lines.items[2].bytes.items);
    try std.testing.expect(!buf.modified);
}

test "TextBuffer.loadFromPath with empty file" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_empty.txt";
    var file = try std.Io.Dir.cwd().createFile(io, test_file, .{ .truncate = true });
    defer file.close(io);
    file.close(io);

    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var buf = try TextBuffer.loadFromPath(std.testing.allocator, io, test_file);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), buf.lines.items[0].bytes.items.len);
}

test "TextBuffer.loadFromPath with CRLF line endings" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_crlf.txt";
    var file = try std.Io.Dir.cwd().createFile(io, test_file, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, "line1\r\nline2\r\n");
    file.close(io);

    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var buf = try TextBuffer.loadFromPath(std.testing.allocator, io, test_file);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 2), buf.lines.items.len);
    try std.testing.expectEqualStrings("line1", buf.lines.items[0].bytes.items);
    try std.testing.expectEqualStrings("line2", buf.lines.items[1].bytes.items);
}

test "TextBuffer.saveToPath creates and writes file" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_save.txt";
    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "save\nme\nplease");
    defer buf.deinit();

    buf.modified = true;
    try buf.saveToPath(io, test_file);
    try std.testing.expect(!buf.modified);

    // Verify file contents
    const content = try std.Io.Dir.cwd().readFileAlloc(io, test_file, std.testing.allocator, .{});
    defer std.testing.allocator.free(content);
    try std.testing.expectEqualStrings("save\nme\nplease", content);
}

test "TextBuffer.saveToPath overwrites existing file" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_overwrite.txt";
    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    // Create initial file
    var file = try std.Io.Dir.cwd().createFile(io, test_file, .{ .truncate = true });
    try file.writeStreamingAll(io, "old content");
    file.close(io);

    // Save new content
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "new content");
    defer buf.deinit();

    try buf.saveToPath(io, test_file);

    // Verify overwrite
    const content = try std.Io.Dir.cwd().readFileAlloc(io, test_file, std.testing.allocator, .{});
    defer std.testing.allocator.free(content);
    try std.testing.expectEqualStrings("new content", content);
}

test "TextBuffer.saveToPath with empty buffer" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const test_file = "test_empty_save.txt";
    defer std.Io.Dir.cwd().deleteFile(io, test_file) catch {};

    var buf = try TextBuffer.initEmpty(std.testing.allocator);
    defer buf.deinit();

    try buf.saveToPath(io, test_file);

    // Empty buffer should create file with no newline
    const content = try std.Io.Dir.cwd().readFileAlloc(io, test_file, std.testing.allocator, .{});
    defer std.testing.allocator.free(content);
    try std.testing.expectEqual(@as(usize, 0), content.len);
}

test "TextBuffer.clampCursor with empty buffer" {
    var buf = try TextBuffer.initEmpty(std.testing.allocator);
    defer buf.deinit();

    var row: usize = 10;
    var col: usize = 20;
    buf.clampCursor(&row, &col);

    try std.testing.expectEqual(@as(usize, 0), row);
    try std.testing.expectEqual(@as(usize, 0), col);
}

test "TextBuffer.clampCursor with row out of bounds" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2\nline3");
    defer buf.deinit();

    var row: usize = 100;
    var col: usize = 5;
    buf.clampCursor(&row, &col);

    try std.testing.expectEqual(@as(usize, 2), row); // Last valid row
    try std.testing.expectEqual(@as(usize, 5), col);
}

test "TextBuffer.clampCursor with col out of bounds" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "short\nvery long line here");
    defer buf.deinit();

    var row: usize = 1;
    var col: usize = 1000;
    buf.clampCursor(&row, &col);

    try std.testing.expectEqual(@as(usize, 1), row);
    try std.testing.expectEqual(@as(usize, 21), col); // Length of "very long line here"
}

test "TextBuffer.clampCursor with both out of bounds" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "a");
    defer buf.deinit();

    var row: usize = 50;
    var col: usize = 50;
    buf.clampCursor(&row, &col);

    try std.testing.expectEqual(@as(usize, 0), row);
    try std.testing.expectEqual(@as(usize, 1), col);
}

test "EditorState.render returns TerminalNotAttached when no terminal" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var editor = try EditorState.init(std.testing.allocator, io, null);
    defer editor.deinit();

    // Should error when terminal not attached
    try std.testing.expectError(error.TerminalNotAttached, editor.render());
}

test "EditorState.handleEvent returns TerminalNotAttached when no terminal" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var editor = try EditorState.init(std.testing.allocator, io, null);
    defer editor.deinit();

    const event = tui.Event{ .key = .{ .code = .up, .char = null } };
    try std.testing.expectError(error.TerminalNotAttached, editor.handleEvent(event));
}

test "TextBuffer.initEmpty creates single empty line" {
    var buf = try TextBuffer.initEmpty(std.testing.allocator);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), buf.lines.items[0].bytes.items.len);
    try std.testing.expect(!buf.modified);
}

test "TextBuffer.initFromSlice with empty string" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "");
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), buf.lines.items[0].bytes.items.len);
}

test "TextBuffer.initFromSlice with single line" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "single line");
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqualStrings("single line", buf.lines.items[0].bytes.items);
}

test "TextBuffer.initFromSlice with very long line" {
    const long_line = "a" ** 10000;
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, long_line);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 10000), buf.lines.items[0].bytes.items.len);
    try std.testing.expectEqualStrings(long_line, buf.lines.items[0].bytes.items);
}

test "TextBuffer.initFromSlice with multiple lines" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2\nline3\n");
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 4), buf.lines.items.len);
    try std.testing.expectEqualStrings("line1", buf.lines.items[0].bytes.items);
    try std.testing.expectEqualStrings("line2", buf.lines.items[1].bytes.items);
    try std.testing.expectEqualStrings("line3", buf.lines.items[2].bytes.items);
    try std.testing.expectEqualStrings("", buf.lines.items[3].bytes.items);
}

test "TextBuffer.insertByte modifies buffer" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "abc");
    defer buf.deinit();

    buf.modified = false;
    try buf.insertByte(0, 1, 'X');
    try std.testing.expect(buf.modified);
    try std.testing.expectEqualStrings("aXbc", buf.lines.items[0].bytes.items);
}

test "TextBuffer.insertByte at end of line" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "abc");
    defer buf.deinit();

    try buf.insertByte(0, 3, 'X');
    try std.testing.expectEqualStrings("abcX", buf.lines.items[0].bytes.items);
}

test "TextBuffer.insertByte beyond line length clamps" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "abc");
    defer buf.deinit();

    try buf.insertByte(0, 100, 'X');
    try std.testing.expectEqualStrings("abcX", buf.lines.items[0].bytes.items);
}

test "TextBuffer.insertNewline splits line correctly" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "hello world");
    defer buf.deinit();

    var row: usize = 0;
    var col: usize = 5;
    try buf.insertNewline(&row, &col);

    try std.testing.expectEqual(@as(usize, 2), buf.lines.items.len);
    try std.testing.expectEqualStrings("hello", buf.lines.items[0].bytes.items);
    try std.testing.expectEqualStrings(" world", buf.lines.items[1].bytes.items);
    try std.testing.expectEqual(@as(usize, 1), row);
    try std.testing.expectEqual(@as(usize, 0), col);
    try std.testing.expect(buf.modified);
}

test "TextBuffer.backspace at start of line merges with previous" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2");
    defer buf.deinit();

    var row: usize = 1;
    var col: usize = 0;
    try buf.backspace(&row, &col);

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqualStrings("line1line2", buf.lines.items[0].bytes.items);
    try std.testing.expectEqual(@as(usize, 0), row);
    try std.testing.expectEqual(@as(usize, 5), col);
}

test "TextBuffer.backspace at first line does nothing" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2");
    defer buf.deinit();

    var row: usize = 0;
    var col: usize = 0;
    try buf.backspace(&row, &col);

    try std.testing.expectEqual(@as(usize, 2), buf.lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), row);
    try std.testing.expectEqual(@as(usize, 0), col);
}

test "TextBuffer.deleteForward at end of line merges with next" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2");
    defer buf.deinit();

    var row: usize = 0;
    var col: usize = 5;
    try buf.deleteForward(&row, &col);

    try std.testing.expectEqual(@as(usize, 1), buf.lines.items.len);
    try std.testing.expectEqualStrings("line1line2", buf.lines.items[0].bytes.items);
}

test "TextBuffer.deleteForward at last line does nothing" {
    var buf = try TextBuffer.initFromSlice(std.testing.allocator, "line1\nline2");
    defer buf.deinit();

    var row: usize = 1;
    var col: usize = 5;
    try buf.deleteForward(&row, &col);

    try std.testing.expectEqual(@as(usize, 2), buf.lines.items.len);
    try std.testing.expectEqualStrings("line2", buf.lines.items[1].bytes.items);
}
