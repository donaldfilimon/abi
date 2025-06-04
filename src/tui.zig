const std = @import("std");
const Abbey = @import("./main.zig").Abbey;
const Aviva = @import("./main.zig").Aviva;
const Abi = @import("./main.zig").Abi;
const Request = @import("./main.zig").Request;

const gpa = std.heap.c_allocator;

pub const Term = struct {
    orig_term: ?std.os.termios = null,

    pub fn init() !Term {
        var t = Term{ .orig_term = null };
        if (std.posix.isatty(0)) {
            var tio = try std.os.tcgetattr(0);
            t.orig_term = tio;
            var raw = tio;
            raw.lflag &= ~(std.os.termiosFlags.ECHO | std.os.termiosFlags.ICANON);
            raw.c_cc[std.os.VMIN] = 1;
            raw.c_cc[std.os.VTIME] = 0;
            try std.os.tcsetattr(0, std.os.TCSANOW, &raw);
        }
        clearScreen();
        return t;
    }

    pub fn deinit(self: *Term) void {
        if (self.orig_term) |orig| {
            _ = std.os.tcsetattr(0, std.os.TCSANOW, &orig) catch {};
        }
        clearScreen();
    }
};

fn clearScreen() void {
    _ = std.os.write(1, "\x1b[2J\x1b[H");
}

fn execAndCapture(argv: []const []const u8) ![]u8 {
    var cp = std.ChildProcess.init(argv, gpa);
    cp.options.stdout_behavior = .Capture;
    try cp.spawn();
    const out = try cp.stdoutReader().readAllAlloc(gpa, 1 << 20);
    _ = cp.wait();
    return out;
}

pub fn run() !void {
    var term = try Term.init();
    defer term.deinit();

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

    try stdout.print("Abi TUI (type 'exit' to quit)\n", .{});

    var buf: [256]u8 = undefined;
    while (true) {
        try stdout.writeAll("> ");
        const line = (try stdin.readUntilDelimiterOrEof(&buf, '\n')) orelse break;
        const trimmed = std.mem.trimRight(u8, line, " \t\r\n");
        if (std.ascii.eqlIgnoreCase(trimmed, "exit")) break;
        try handleCommand(trimmed);
    }
}

fn handleCommand(line: []const u8) !void {
    var it = std.mem.tokenize(u8, line, " ");
    const cmd = it.next() orelse return;
    if (std.mem.eql(u8, cmd, "sum")) {
        var nums = std.ArrayList(usize).init(gpa);
        defer nums.deinit();
        while (it.next()) |t| {
            nums.append(std.fmt.parseInt(usize, t, 10) catch 0) catch {};
        }
        const total = Aviva.computeSum(nums.items);
        try std.io.getStdOut().writer().print("sum = {d}\n", .{total});
    } else if (std.mem.eql(u8, cmd, "check")) {
        const rest = line[cmd.len..];
        const text = std.mem.trimLeft(u8, rest, " ");
        const ok = Abbey.isCompliant(text);
        try std.io.getStdOut().writer().print("compliant: {s}\n", .{ if (ok) "yes" else "no" });
    } else if (std.mem.eql(u8, cmd, "abi")) {
        const text = it.next() orelse {
            try std.io.getStdOut().writer().print("usage: abi <text> <nums...>\n", .{});
            return;
        };
        var nums = std.ArrayList(usize).init(gpa);
        defer nums.deinit();
        while (it.next()) |t| {
            nums.append(std.fmt.parseInt(usize, t, 10) catch 0) catch {};
        }
        const req = Request{ .text = text, .values = nums.items };
        const res = Abi.process(req);
        try std.io.getStdOut().writer().print("{s}: {d}\n", .{ res.message, res.result });
    } else if (std.mem.eql(u8, cmd, "sh")) {
        var args_list = std.ArrayList([]const u8).init(gpa);
        defer args_list.deinit();
        while (it.next()) |t| {
            args_list.append(t) catch {};
        }
        const args = args_list.toOwnedSlice();
        defer gpa.free(args);
        const out = try execAndCapture(args);
        try std.io.getStdOut().writer().print("{s}\n", .{out});
        gpa.free(out);
    } else {
        try std.io.getStdOut().writer().print("unknown command\n", .{});
    }
}

