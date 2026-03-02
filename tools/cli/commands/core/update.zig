//! Self-update capability for the ABI Framework.
//!
//! Provides the ability to pull the latest changes from the git repository
//! and rebuild the CLI entirely natively using Zig.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const command_mod = @import("../../command.zig");
const utils = @import("../../utils/mod.zig");
const os = @import("abi").services.shared.os;

pub const meta: command_mod.Meta = .{
    .name = "update",
    .description = "Auto-update ABI framework from origin and recompile",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = args;
    const allocator = ctx.allocator;

    utils.output.printHeader("ABI Auto-Update");
    utils.output.printInfo("Fetching latest updates from git...", .{});

    // 1. Git pull
    const pull_cmd = try std.fmt.allocPrint(allocator, "git pull origin main", .{});
    defer allocator.free(pull_cmd);

    var pull_result = os.exec(allocator, pull_cmd) catch |err| {
        utils.output.printError("Failed to pull from git: {t}", .{err});
        return err;
    };
    defer pull_result.deinit();

    if (pull_result.exit_code != 0) {
        utils.output.printError("Git pull failed:\n{s}", .{pull_result.stderr});
        return error.UpdateFailed;
    }
    
    const pull_out = std.mem.trim(u8, pull_result.stdout, " \r\n");
    if (std.mem.eql(u8, pull_out, "Already up to date.")) {
        utils.output.printSuccess("ABI Framework is already up to date.", .{});
        return;
    }
    
    utils.output.printSuccess("Successfully pulled latest changes.", .{});

    // 2. Rebuild
    utils.output.printInfo("Recompiling ABI framework with Zig 0.16...", .{});
    
    // Using zig build
    const build_cmd = try std.fmt.allocPrint(allocator, "zig build install", .{});
    defer allocator.free(build_cmd);

    var build_result = os.exec(allocator, build_cmd) catch |err| {
        utils.output.printError("Failed to invoke zig build: {t}", .{err});
        return err;
    };
    defer build_result.deinit();

    if (build_result.exit_code != 0) {
        utils.output.printError("Compilation failed:\n{s}", .{build_result.stderr});
        return error.BuildFailed;
    }

    utils.output.printSuccess("ABI Framework successfully updated and compiled!", .{});
    utils.output.printInfo("Restart your terminal or run `abi version` to verify.", .{});
}
