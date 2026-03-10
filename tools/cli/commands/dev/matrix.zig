//! CLI command: abi matrix
//!
//! A standalone TUI animation demonstrating native terminal rendering speeds.

const std = @import("std");
const command_mod = @import("../../command");
const context_mod = @import("../../framework/context");
const utils = @import("../../utils/mod.zig");
const tui = @import("../../terminal/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "matrix",
    .description = "Enter the dream state (Native Terminal Animation)",
    .subcommands = &.{"help"},
};

const Drop = struct {
    x: u16,
    y: i32,
    length: u16,
    speed: i32,
    chars: [64]u8,
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp(ctx.allocator);
        return;
    }

    var term = tui.Terminal.init(ctx.allocator);
    try term.enter();
    defer term.deinit();

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    const size = term.size();
    var drops = std.ArrayListUnmanaged(Drop).empty;
    defer drops.deinit(ctx.allocator);

    for (0..@as(usize, size.cols) / 2) |_| {
        try drops.append(ctx.allocator, initDrop(random, size));
    }

    var running = true;
    while (running) {
        // Handle input
        while (try term.pollEvent(0)) |event| {
            switch (event) {
                .key => |k| {
                    if (k.code == .escape or k.code == .ctrl_c or (k.code == .character and k.char != null and k.char.? == 'q')) {
                        running = false;
                    }
                },
                else => {},
            }
        }

        try term.clear();

        for (drops.items) |*drop| {
            drop.y += drop.speed;

            // Render drop
            if (drop.y > 0 and drop.y < size.rows) {
                try term.moveTo(@intCast(drop.y), drop.x);
                try term.write("\x1b[38;5;15m"); // Bright white head
                const ch_idx: usize = @intCast(@rem(drop.y, @as(i32, drop.length)));
                var buf = [_]u8{drop.chars[ch_idx]};
                try term.write(&buf);
            }

            // Render tail
            for (1..drop.length) |i| {
                const tail_y = drop.y - @as(i32, @intCast(i));
                if (tail_y >= 0 and tail_y < size.rows) {
                    try term.moveTo(@intCast(tail_y), drop.x);
                    if (i < 3) {
                        try term.write("\x1b[38;5;82m"); // Bright green
                    } else if (i < drop.length / 2) {
                        try term.write("\x1b[38;5;46m"); // Mid green
                    } else {
                        try term.write("\x1b[38;5;22m"); // Dark green
                    }
                    const ch_idx: usize = @intCast(@rem(tail_y, @as(i32, drop.length)));
                    var buf = [_]u8{drop.chars[ch_idx]};
                    try term.write(&buf);
                }
            }

            // Reset drop if completely off screen
            if (drop.y - @as(i32, drop.length) > size.rows) {
                drop.* = initDrop(random, size);
                drop.y = -@as(i32, drop.length); // Start offscreen
            }
        }

        try term.flush();
        @import("abi").services.shared.time.sleepMs(50);
    }
}

fn initDrop(random: std.Random, size: tui.TerminalSize) Drop {
    var drop: Drop = .{
        .x = random.intRangeAtMost(u16, 0, size.cols - 1),
        .y = random.intRangeAtMost(i32, -20, @intCast(size.rows)),
        .length = random.intRangeAtMost(u16, 5, 25),
        .speed = random.intRangeAtMost(i32, 1, 2),
        .chars = [_]u8{0} ** 64,
    };

    for (0..64) |i| {
        // Random ASCII characters
        drop.chars[i] = random.intRangeAtMost(u8, 33, 126);
    }

    return drop;
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi matrix", "")
        .description("Enter the native dream state animation.")
        .newline()
        .section("Options")
        .option(utils.help.common_options.help);

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
