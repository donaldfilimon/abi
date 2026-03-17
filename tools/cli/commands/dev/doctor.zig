//! Toolchain doctor command.
//!
//! Wraps the shared toolchain_doctor logic to provide diagnostic information
//! about the active Zig toolchain and environment.

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const toolchain = @import("toolchain_support");

pub const meta: command_mod.Meta = .{
    .name = "doctor",
    .description = "Inspect the active Zig toolchain and ABI environment",
};

pub fn run(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const issues = try toolchain.printDoctorReport(allocator, io);
    if (issues > 0) {
        std.debug.print("\nFAILED: toolchain doctor found {d} issue(s).\n", .{issues});
        return error.ExecutionFailed;
    }
}
