const std = @import("std");
const builtin = @import("builtin");
const util = @import("util.zig");

pub fn main(_: std.process.Init) !void {
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

    if (!(try util.commandExists(allocator, io, "zig"))) {
        std.debug.print("ERROR: no 'zig' binary found on PATH\n", .{});
        std.debug.print("Build CEL via ./.cel/build.sh and ensure .cel/bin is on PATH.\n", .{});
        std.process.exit(1);
    }

    const active_path_res = try util.captureCommand(allocator, io, "command -v zig");
    defer allocator.free(active_path_res.output);
    const active_zig = util.trimSpace(active_path_res.output);

    const active_ver_res = try util.captureCommand(allocator, io, "zig version");
    defer allocator.free(active_ver_res.output);
    const active_version = util.trimSpace(active_ver_res.output);

    std.debug.print("Active zig:\n", .{});
    std.debug.print("  path:    {s}\n", .{active_zig});
    std.debug.print("  version: {s}\n\n", .{active_version});

    std.debug.print("Environment selectors:\n", .{});
    try printEnvVar(allocator, io, "DEVELOPER_DIR");
    try printEnvVar(allocator, io, "TOOLCHAINS");
    try printEnvVar(allocator, io, "SDKROOT");
    std.debug.print("\n", .{});

    if (builtin.os.tag == .macos) {
        std.debug.print("Apple developer tools:\n", .{});
        try printCommandSummary(allocator, io, "default xcode-select -p", "env -u DEVELOPER_DIR xcode-select -p");
        try printCommandSummary(allocator, io, "xcrun --find clang", "xcrun --find clang");
        try printCommandSummary(allocator, io, "xcrun --show-sdk-path", "xcrun --show-sdk-path");
        try printCommandFirstLine(allocator, io, "clang --version", "clang --version");
        std.debug.print(
            "  hint: ABI's known-good override on this host is DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer\n",
            .{},
        );
        std.debug.print("\n", .{});
    }

    std.debug.print("All zig candidates on PATH (in precedence order):\n", .{});
    if (try util.commandExists(allocator, io, "which")) {
        const which_res = try util.captureCommand(allocator, io, "which -a zig");
        defer allocator.free(which_res.output);

        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(allocator);

        var lines = std.mem.splitScalar(u8, which_res.output, '\n');
        while (lines.next()) |line| {
            const trimmed = util.trimSpace(line);
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

    if (util.fileExists(io, ".cel/bin/zig") and
        !std.mem.endsWith(u8, active_zig, "/.cel/bin/zig") and
        !std.mem.eql(u8, active_zig, ".cel/bin/zig"))
    {
        std.debug.print("ISSUE: active zig is not the repo-local CEL binary\n", .{});
        issues += 1;
    } else {
        const home_res = try util.captureCommand(allocator, io, "printf '%s' \"$HOME\"");
        defer allocator.free(home_res.output);
        const home = util.trimSpace(home_res.output);

        const zvm_zig = try std.fmt.allocPrint(allocator, "{s}/.zvm/bin/zig", .{home});
        defer allocator.free(zvm_zig);

        if (util.fileExists(io, zvm_zig) and !std.mem.eql(u8, active_zig, zvm_zig)) {
            std.debug.print("ISSUE: active zig is not the zvm-managed binary\n", .{});
            issues += 1;
        }
    }

    // ── CEL toolchain check ────────────────────────────────────────────
    std.debug.print(".cel toolchain:\n", .{});
    const cel_zig_exists = util.fileExists(io, ".cel/bin/zig");
    if (cel_zig_exists) {
        const cel_ver_res = util.captureCommand(allocator, io, ".cel/bin/zig version") catch null;
        if (cel_ver_res) |res| {
            defer allocator.free(res.output);
            const cel_ver = util.trimSpace(res.output);
            std.debug.print("  .cel/bin/zig: {s}\n", .{cel_ver});

            if (std.mem.eql(u8, cel_ver, expected_version)) {
                std.debug.print("  Version match: YES\n", .{});
            } else {
                std.debug.print("  Version match: NO (expected {s})\n", .{expected_version});
            }
        }
    } else if (util.fileExists(io, ".cel/build.sh")) {
        std.debug.print("  .cel/bin/zig: NOT BUILT\n", .{});
        if (builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26) {
            std.debug.print("  ACTION: Run .cel/build.sh to build patched toolchain\n", .{});
        }
    } else {
        std.debug.print("  .cel: not present in this checkout\n", .{});
    }

    // Check if active zig is CEL
    if (std.mem.indexOf(u8, active_zig, ".cel/bin") != null) {
        std.debug.print("  Active zig source: .cel patched toolchain\n", .{});
    }
    if (util.fileExists(io, ".cel/bin/zls")) {
        const cel_zls_res = util.captureCommand(allocator, io, ".cel/bin/zls --version") catch null;
        if (cel_zls_res) |res| {
            defer allocator.free(res.output);
            std.debug.print("  .cel/bin/zls: {s}\n", .{util.trimSpace(res.output)});
        }
    } else if (cel_zig_exists) {
        std.debug.print("  .cel/bin/zls: NOT BUILT (run .cel/build.sh --zls-only)\n", .{});
    }
    std.debug.print("\n", .{});

    if (issues == 0) {
        std.debug.print("OK: local Zig toolchain is deterministic and matches repository pin.\n", .{});
        return;
    }

    std.debug.print("\nSuggested fix:\n", .{});

    // On blocked Darwin, recommend CEL first
    if (builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26) {
        std.debug.print("  Recommended (macOS 26+): Use the .cel patched toolchain\n", .{});
        std.debug.print("  1) ./.cel/build.sh\n", .{});
        std.debug.print("  2) eval \"$(./tools/scripts/use_cel.sh)\"\n", .{});
        std.debug.print("  3) zig build full-check\n", .{});
        std.debug.print("\n  Alternative: Use ZVM\n", .{});
    }

    if (try util.commandExists(allocator, io, "zvm")) {
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
