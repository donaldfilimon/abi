//! zvim: Ultra-high-performance CLI with GPU acceleration, lock-free concurrency,
//! and platform-optimized implementations for Zig, Swift, and C++ development.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const perf = @import("performance.zig");
const gpu = @import("../zvim/gpu_renderer.zig");
const simd = @import("../zvim/simd_text.zig");
const lockfree = @import("lockfree.zig");
const platform = @import("platform.zig");

pub const Error = error{
    EmptyText,
    BlacklistedWord,
    TextTooLong,
    InvalidValues,
    ProcessingFailed,
};

pub const Request = struct {
    text: []const u8,
    values: []const usize,

    pub fn validate(self: Request) Error!void {
        if (self.text.len == 0) return Error.EmptyText;
        if (self.values.len == 0) return Error.InvalidValues;
        _ = try Abbey.checkCompliance(self.text);
    }
};

pub const Response = struct {
    result: usize,
    message: []const u8,
};

pub const ComplianceError = error{
    EmptyText,
    BlacklistedWord,
    TextTooLong,
};

/// Abbey persona: ensures simple ethical compliance
pub const Abbey = struct {
    const MAX_TEXT_LENGTH = 1000;
    const BLACKLISTED_WORDS = [_][]const u8{ "bad", "evil", "hate" };

    pub fn isCompliant(text: []const u8) bool {
        return checkCompliance(text) catch return false;
    }

    pub fn checkCompliance(text: []const u8) Error!bool {
        if (text.len == 0) return Error.EmptyText;
        if (text.len > MAX_TEXT_LENGTH) return Error.TextTooLong;

        // Check for blacklisted words
        for (BLACKLISTED_WORDS) |word| {
            if (std.mem.indexOf(u8, text, word) != null) {
                return Error.BlacklistedWord;
            }
        }
        return true;
    }
};

/// Aviva persona: performs computation on provided values
pub const Aviva = struct {
    pub fn computeSum(values: []const usize) Error!usize {
        if (values.len == 0) return Error.InvalidValues;
        var sum: usize = 0;
        for (values) |v| {
            sum = std.math.add(usize, sum, v) catch return Error.ProcessingFailed;
        }
        return sum;
    }
};

/// Abi persona: orchestrates Abbey and Aviva
pub const Abi = struct {
    pub fn process(req: Request) Error!Response {
        try req.validate();
        const sum = try Aviva.computeSum(req.values);
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
        } else if (std.mem.eql(u8, arg, "discord")) {
            const api = @import("discord/api.zig");
            const gw = @import("discord/gateway.zig");
            var gpa = std.heap.GeneralPurposeAllocator(.{}){};
            defer _ = gpa.deinit();
            const allocator = gpa.allocator();

            const token = std.process.getEnvVarOwned(allocator, "DISCORD_TOKEN") catch {
                std.log.err("DISCORD_TOKEN environment variable not set", .{});
                return;
            };
            defer allocator.free(token);

            const channel = args.next() orelse {
                std.log.err("channel id required", .{});
                return;
            };

            var bot = gw.DiscordBot.init(allocator, token);
            defer bot.deinit();
            // Non-blocking send using REST API
            try api.postMessage(allocator, token, channel, "Hello from Zig!");
            // Connect to gateway in blocking mode (example only)
            // try bot.connect();
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
    const res = try Abi.process(req);
    try std.testing.expectEqual(@as(usize, 3), res.result);
    try std.testing.expectEqualStrings("Computation successful", res.message);
}
