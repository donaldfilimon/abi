const std = @import("std");
const util = @import("util.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const expected_raw = util.readFileAlloc(allocator, io, ".zigversion", 1024) catch {
        std.debug.print("ERROR: .zigversion not found\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(expected_raw);
    const expected_version = util.trimSpace(expected_raw);

    std.debug.print("ABI toolchain doctor\n", .{});
    std.debug.print("Pinned Zig (.zigversion): {s}\n\n", .{expected_version});

    if (!(try util.commandExists(allocator, "zig"))) {
        std.debug.print("ERROR: no 'zig' binary found on PATH\n", .{});
        std.debug.print("Install via zvm and ensure ~/.zvm/bin is on PATH.\n", .{});
        std.process.exit(1);
    }

    const active_path_res = try util.captureCommand(allocator, "command -v zig");
    defer allocator.free(active_path_res.output);
    const active_zig = util.trimSpace(active_path_res.output);

    const active_ver_res = try util.captureCommand(allocator, "zig version");
    defer allocator.free(active_ver_res.output);
    const active_version = util.trimSpace(active_ver_res.output);

    std.debug.print("Active zig:\n", .{});
    std.debug.print("  path:    {s}\n", .{active_zig});
    std.debug.print("  version: {s}\n\n", .{active_version});

    std.debug.print("All zig candidates on PATH (in precedence order):\n", .{});
    if (try util.commandExists(allocator, "which")) {
        const which_res = try util.captureCommand(allocator, "which -a zig");
        defer allocator.free(which_res.output);

        var seen = std.StringHashMap(void).init(allocator);
        defer seen.deinit();

        var lines = std.mem.splitScalar(u8, which_res.output, '\n');
        while (lines.next()) |line| {
            const trimmed = util.trimSpace(line);
            if (trimmed.len == 0) continue;
            const gop = try seen.getOrPut(trimmed);
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

    const home_res = try util.captureCommand(allocator, "printf '%s' \"$HOME\"");
    defer allocator.free(home_res.output);
    const home = util.trimSpace(home_res.output);

    const zvm_zig = try std.fmt.allocPrint(allocator, "{s}/.zvm/bin/zig", .{home});
    defer allocator.free(zvm_zig);

    if (util.fileExists(io, zvm_zig) and !std.mem.eql(u8, active_zig, zvm_zig)) {
        std.debug.print("ISSUE: active zig is not the zvm-managed binary\n", .{});
        issues += 1;
    }

    if (issues == 0) {
        std.debug.print("OK: local Zig toolchain is deterministic and matches repository pin.\n", .{});
        return;
    }

    std.debug.print("\nSuggested fix:\n", .{});
    if (try util.commandExists(allocator, "zvm")) {
        std.debug.print("  1) zvm upgrade\n", .{});
        std.debug.print("  2) zvm install \"{s}\"\n", .{expected_version});
        std.debug.print("  3) zvm use \"{s}\"\n", .{expected_version});
        std.debug.print("  4) export PATH=\"$HOME/.zvm/bin:$PATH\"\n", .{});
        std.debug.print("  5) hash -r\n", .{});
        std.debug.print("  6) zig run tools/scripts/check_zig_version_consistency.zig\n", .{});
    } else {
        std.debug.print("  1) Install Zig {s} from https://ziglang.org/download/\n", .{expected_version});
        std.debug.print("  2) Put the Zig binary on PATH ahead of other installations\n", .{});
        std.debug.print("  3) hash -r\n", .{});
        std.debug.print("  4) zig run tools/scripts/check_zig_version_consistency.zig\n", .{});
    }

    std.debug.print("\nFAILED: toolchain doctor found {d} issue(s).\n", .{issues});
    std.process.exit(1);
}
