const std = @import("std");
const Abbey = @import("./main.zig").Abbey;
const Aviva = @import("./main.zig").Aviva;
const Abi = @import("./main.zig").Abi;
const Request = @import("./main.zig").Request;

pub const TuiError = error{
    TerminalError,
    InputError,
    CommandError,
    AllocationError,
    InvalidCommand,
    ParseError,
};

const gpa = std.heap.c_allocator;

pub const Command = union(enum) {
    help,
    check: []const u8,
    compute: []const usize,
    process: struct {
        text: []const u8,
        values: []const usize,
    },
    clear,
    exit,
};

pub const Term = struct {
    orig_term: ?std.os.termios = null,
    buf: [1024]u8 = undefined,
    stdin: std.fs.File,
    stdout: std.fs.File,

    pub fn init() TuiError!Term {
        var t = Term{
            .stdin = std.io.getStdIn(),
            .stdout = std.io.getStdOut(),
            .orig_term = null,
        };

        if (std.posix.isatty(0)) {
            const tio = std.os.tcgetattr(0) catch return TuiError.TerminalError;
            t.orig_term = tio;
            var raw = tio;
            raw.lflag &= ~(std.os.termiosFlags.ECHO | std.os.termiosFlags.ICANON);
            raw.c_cc[std.os.VMIN] = 1;
            raw.c_cc[std.os.VTIME] = 0;
            std.os.tcsetattr(0, std.os.TCSANOW, &raw) catch return TuiError.TerminalError;
        }

        try t.clearScreen();
        return t;
    }

    pub fn deinit(self: *Term) void {
        if (self.orig_term) |orig| {
            _ = std.os.tcsetattr(0, std.os.TCSANOW, &orig) catch {};
        }
        self.clearScreen() catch {};
    }

    pub fn clearScreen(self: Term) TuiError!void {
        const writer = self.stdout.writer();
        writer.writeAll("\x1B[2J\x1B[H") catch return TuiError.TerminalError;
    }

    pub fn readLine(self: *Term) TuiError!?[]const u8 {
        const writer = self.stdout.writer();
        writer.writeAll("> ") catch return TuiError.TerminalError;

        const reader = self.stdin.reader();
        const line = reader.readUntilDelimiterOrEof(&self.buf, '\n') catch return TuiError.InputError;
        return if (line) |l| std.mem.trimRight(u8, l, " \t\r\n") else null;
    }

    pub fn parseCommand(self: Term, line: []const u8) TuiError!Command {
        _ = self;
        var it = std.mem.tokenize(u8, line, " ");
        const cmd = it.next() orelse return TuiError.InvalidCommand;

        if (std.mem.eql(u8, cmd, "help")) {
            return Command.help;
        } else if (std.mem.eql(u8, cmd, "check")) {
            return Command{ .check = it.rest() };
        } else if (std.mem.eql(u8, cmd, "compute")) {
            var values = std.ArrayList(usize).init(gpa);
            defer values.deinit();

            while (it.next()) |num_str| {
                const num = std.fmt.parseInt(usize, num_str, 10) catch return TuiError.ParseError;
                values.append(num) catch return TuiError.AllocationError;
            }

            return Command{ .compute = values.toOwnedSlice() catch return TuiError.AllocationError };
        } else if (std.mem.eql(u8, cmd, "clear")) {
            return Command.clear;
        } else if (std.mem.eql(u8, cmd, "exit")) {
            return Command.exit;
        }

        return TuiError.InvalidCommand;
    }
};

pub fn run() !void {
    var term = try Term.init();
    defer term.deinit();

    try term.stdout.writer().print("Abi TUI (type 'help' for usage, 'exit' to quit)\n", .{});

    while (true) {
        const line = try term.readLine() orelse break;
        if (line.len == 0) continue;

        const cmd = term.parseCommand(line) catch |err| switch (err) {
            TuiError.InvalidCommand => {
                try term.stdout.writer().writeAll("Invalid command. Type 'help' for usage.\n");
                continue;
            },
            TuiError.ParseError => {
                try term.stdout.writer().writeAll("Error parsing command arguments.\n");
                continue;
            },
            else => |e| return e,
        };

        switch (cmd) {
            .help => {
                try term.stdout.writer().writeAll(
                    \\Available commands:
                    \\  help              Show this help message
                    \\  check <text>      Check text compliance
                    \\  compute <nums>    Compute sum of numbers
                    \\  clear            Clear the screen
                    \\  exit             Exit the program
                    \\
                );
            },
            .check => |text| {
                if (Abbey.isCompliant(text)) {
                    try term.stdout.writer().writeAll("Text is compliant.\n");
                } else {
                    try term.stdout.writer().writeAll("Text is NOT compliant.\n");
                }
            },
            .compute => |numbers| {
                const sum = Aviva.computeSum(numbers) catch |err| {
                    try term.stdout.writer().print("Error: {s}\n", .{@errorName(err)});
                    return;
                };
                try term.stdout.writer().print("Sum: {d}\n", .{sum});
            },
            .process => |_| try term.stdout.writer().writeAll("Process command not implemented.\n"),
            .clear => try term.clearScreen(),
            .exit => break,
        }
    }
}
