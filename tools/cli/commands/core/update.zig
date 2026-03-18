//! Self-update capability for the ABI Framework.
//!
//! Provides the ability to pull the latest changes from the git repository
//! and rebuild the CLI entirely natively using Zig.

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const command_mod = @import("../../command.zig");
const utils = @import("../../utils/mod.zig");
const os = abi.foundation.os;

pub const meta: command_mod.Meta = .{
    .name = "update",
    .description = "Auto-update ABI framework from origin and recompile",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = args;
    const allocator = ctx.allocator;

    utils.output.printHeader("ABI Auto-Update");
    utils.output.printInfo("Fetching latest updates from git...", .{});

    // 1. Git pull — check exit code for failures
    var git_result = os.exec(allocator, "git pull origin main") catch |err| {
        utils.output.printError("Failed to spawn git pull: {}", .{err});
        return err;
    };
    defer git_result.deinit();

    if (!git_result.success()) {
        utils.output.printError("git pull failed with exit code {d}", .{git_result.exit_code});
        return error.GitPullFailed;
    }

    utils.output.printSuccess("Successfully pulled latest changes.", .{});

    // 2. Rebuild — check exit code for failures
    utils.output.printInfo("Recompiling ABI framework with Zig 0.16...", .{});

    var build_result = os.exec(allocator, "zig build install") catch |err| {
        utils.output.printError("Failed to spawn zig build: {}", .{err});
        return err;
    };
    defer build_result.deinit();

    if (!build_result.success()) {
        utils.output.printError("zig build install failed with exit code {d}", .{build_result.exit_code});
        return error.BuildFailed;
    }

    utils.output.printSuccess("ABI Framework successfully updated and compiled!", .{});
    utils.output.printInfo("Restart your terminal or run `abi version` to verify.", .{});
}
