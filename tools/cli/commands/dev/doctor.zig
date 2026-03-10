//! Toolchain doctor command.
//!
//! Wraps the toolchain_doctor logic to provide diagnostic information
//! about the active Zig toolchain and environment.

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const process_utils = utils.process;

pub const meta: command_mod.Meta = .{
    .name = "doctor",
    .description = "Inspect the active Zig toolchain and ABI environment",
};

/// Logic adapted from tools/scripts/toolchain_doctor.zig
const doctor_logic = struct {
    const util = struct {
        fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_size: usize) ![]u8 {
            return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_size));
        }

        fn trimSpace(s: []const u8) []const u8 {
            return std.mem.trim(u8, s, " \n\r\t");
        }

        fn commandExists(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) !bool {
            _ = allocator;
            const term = try process_utils.run(io, &[_][]const u8{ "which", cmd }, .ignore, .ignore);
            return switch (term) {
                .exited => |code| code == 0,
                else => false,
            };
        }

        const CommandResult = struct {
            output: []u8,
            exit_code: u32,
        };

        fn captureCommand(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) !CommandResult {
            const result = try std.process.run(allocator, io, .{
                .argv = &[_][]const u8{ "sh", "-c", cmd },
            });
            defer allocator.free(result.stderr);
            return .{
                .output = result.stdout,
                .exit_code = switch (result.term) {
                    .exited => |code| code,
                    else => 1,
                },
            };
        }

        fn fileExists(io: std.Io, path: []const u8) bool {
            std.Io.Dir.accessAbsolute(io, path, .{}) catch return false;
            return true;
        }
    };

    fn printEnvVar(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !void {
        const cmd = try std.fmt.allocPrint(allocator, "printf '%s' \"${s}\"", .{name});
        defer allocator.free(cmd);

        const result = try util.captureCommand(allocator, io, cmd);
        defer allocator.free(result.output);

        const value = util.trimSpace(result.output);
        if (value.len == 0) {
            std.debug.print("  {s}: (unset)\n", .{name});
        } else {
            std.debug.print("  {s}: {s}\n", .{ name, value });
        }
    }

    fn printCommandSummary(
        allocator: std.mem.Allocator,
        io: std.Io,
        label: []const u8,
        cmd: []const u8,
    ) !void {
        const result = util.captureCommand(allocator, io, cmd) catch {
            std.debug.print("  {s}: (unavailable)\n", .{label});
            return;
        };
        defer allocator.free(result.output);

        if (result.exit_code != 0) {
            std.debug.print("  {s}: (failed)\n", .{label});
            return;
        }

        const value = util.trimSpace(result.output);
        if (value.len == 0) {
            std.debug.print("  {s}: (empty)\n", .{label});
        } else {
            std.debug.print("  {s}: {s}\n", .{ label, value });
        }
    }

    fn printCommandFirstLine(
        allocator: std.mem.Allocator,
        io: std.Io,
        label: []const u8,
        cmd: []const u8,
    ) !void {
        const result = util.captureCommand(allocator, io, cmd) catch {
            std.debug.print("  {s}: (unavailable)\n", .{label});
            return;
        };
        defer allocator.free(result.output);

        if (result.exit_code != 0) {
            std.debug.print("  {s}: (failed)\n", .{label});
            return;
        }

        const trimmed = util.trimSpace(result.output);
        if (trimmed.len == 0) {
            std.debug.print("  {s}: (empty)\n", .{label});
            return;
        }

        var lines = std.mem.splitScalar(u8, trimmed, '\n');
        const first_line = lines.next() orelse trimmed;
        std.debug.print("  {s}: {s}\n", .{ label, first_line });
    }
};

pub fn run(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const expected_raw = doctor_logic.util.readFileAlloc(allocator, io, ".zigversion", 1024) catch {
        utils.output.printError(".zigversion not found in current directory.", .{});
        return error.FileNotFound;
    };
    defer allocator.free(expected_raw);
    const expected_version = doctor_logic.util.trimSpace(expected_raw);

    std.debug.print("ABI toolchain doctor\n", .{});
    std.debug.print("Pinned Zig (.zigversion): {s}\n\n", .{expected_version});

    if (!(try doctor_logic.util.commandExists(allocator, io, "zig"))) {
        std.debug.print("ERROR: no 'zig' binary found on PATH\n", .{});
        std.debug.print("Build bootstrap Zig via ./.zig-bootstrap/build.sh and ensure .zig-bootstrap/bin is on PATH.\n", .{});
        return error.ZigNotFound;
    }

    const active_path_res = try doctor_logic.util.captureCommand(allocator, io, "command -v zig");
    defer allocator.free(active_path_res.output);
    const active_zig = doctor_logic.util.trimSpace(active_path_res.output);

    const active_ver_res = try doctor_logic.util.captureCommand(allocator, io, "zig version");
    defer allocator.free(active_ver_res.output);
    const active_version = doctor_logic.util.trimSpace(active_ver_res.output);

    std.debug.print("Active zig:\n", .{});
    std.debug.print("  path:    {s}\n", .{active_zig});
    std.debug.print("  version: {s}\n\n", .{active_version});

    std.debug.print("Environment selectors:\n", .{});
    try doctor_logic.printEnvVar(allocator, io, "DEVELOPER_DIR");
    try doctor_logic.printEnvVar(allocator, io, "TOOLCHAINS");
    try doctor_logic.printEnvVar(allocator, io, "SDKROOT");
    std.debug.print("\n", .{});

    const builtin = @import("builtin");
    if (builtin.os.tag == .macos) {
        std.debug.print("Apple developer tools:\n", .{});
        try doctor_logic.printCommandSummary(allocator, io, "default xcode-select -p", "env -u DEVELOPER_DIR xcode-select -p");
        try doctor_logic.printCommandSummary(allocator, io, "xcrun --find clang", "xcrun --find clang");
        try doctor_logic.printCommandSummary(allocator, io, "xcrun --show-sdk-path", "xcrun --show-sdk-path");
        try doctor_logic.printCommandFirstLine(allocator, io, "clang --version", "clang --version");
        std.debug.print("\n", .{});
    }

    std.debug.print("All zig candidates on PATH (in precedence order):\n", .{});
    if (try doctor_logic.util.commandExists(allocator, io, "which")) {
        const which_res = try doctor_logic.util.captureCommand(allocator, io, "which -a zig");
        defer allocator.free(which_res.output);

        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(allocator);

        var lines = std.mem.splitScalar(u8, which_res.output, '\n');
        while (lines.next()) |line| {
            const trimmed = doctor_logic.util.trimSpace(line);
            if (trimmed.len == 0) continue;
            const gop = try seen.getOrPut(allocator, trimmed);
            if (gop.found_existing) continue;
            std.debug.print("  - {s}\n", .{trimmed});
        }
    } else {
        std.debug.print("  - (which unavailable; skipped)\n", .{});
    }
    std.debug.print("\n", .{});

    var issues: usize = 0;

    if (!std.mem.eql(u8, active_version, expected_version)) {
        std.debug.print("ISSUE: active zig version does not match .zigversion\n", .{});
        issues += 1;
    }

    if (issues == 0) {
        std.debug.print("OK: local Zig toolchain is deterministic and matches repository pin.\n", .{});
        return;
    }

    std.debug.print("\nFAILED: toolchain doctor found {d} issue(s).\n", .{issues});
}
