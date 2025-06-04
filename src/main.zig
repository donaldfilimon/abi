const std = @import("std");

pub const Request = struct {
    text: []const u8,
    values: []const usize,
};

pub const Response = struct {
    result: usize,
    message: []const u8,
};

/// Abbey persona: ensures simple ethical compliance
pub const Abbey = struct {
    pub fn isCompliant(text: []const u8) bool {
        // Very basic check for the word "bad"
        return std.mem.indexOf(u8, text, "bad") == null;
    }
};

/// Aviva persona: performs computation on provided values
pub const Aviva = struct {
    pub fn computeSum(values: []const usize) usize {
        var sum: usize = 0;
        for (values) |v| {
            sum += v;
        }
        return sum;
    }
};

/// Abi persona: orchestrates Abbey and Aviva
pub const Abi = struct {
    pub fn process(req: Request) Response {
        if (!Abbey.isCompliant(req.text)) {
            return Response{
                .result = 0,
                .message = "Ethics violation detected",
            };
        }
        const sum = Aviva.computeSum(req.values);
        return Response{
            .result = sum,
            .message = "Computation successful",
        };
    }
};

pub fn main() !void {
    var args = std.process.args();
    _ = args.next(); // exe name
    if (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "tui")) {
            const tui = @import("tui.zig");
            try tui.run();
            return;
        }
    }

    const req = Request{
        .text = "example input",
        .values = &[_]usize{ 1, 2, 3, 4 },
    };
    const res = Abi.process(req);
    const stdout = std.io.getStdOut().writer();
    try stdout.print("{s}: {d}\n", .{ res.message, res.result });
}

test "Abbey compliance" {
    try std.testing.expect(Abbey.isCompliant("good"));
    try std.testing.expect(!Abbey.isCompliant("bad"));
}

test "Aviva computeSum" {
    const vals = [_]usize{ 1, 2, 3 };
    try std.testing.expectEqual(@as(usize, 6), Aviva.computeSum(&vals));
}

test "Abi orchestrates personas" {
    const req = Request{ .text = "ok", .values = &[_]usize{ 1, 2 } };
    const res = Abi.process(req);
    try std.testing.expectEqual(@as(usize, 3), res.result);
    try std.testing.expectEqualStrings("Computation successful", res.message);
}

