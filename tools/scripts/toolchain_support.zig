const std = @import("std");
const builtin = @import("builtin");

pub const SelectionSource = enum {
    none,
    abi_host_zig,
    zig_real,
    zig_env,
    cache,
    path,

    pub fn label(self: SelectionSource) []const u8 {
        return switch (self) {
            .none => "none",
            .abi_host_zig => "ABI_HOST_ZIG",
            .zig_real => "ZIG_REAL",
            .zig_env => "ZIG",
            .cache => "canonical cache",
            .path => "PATH",
        };
    }
};

pub const SelectionStatus = enum {
    ok,
    abi_host_zig_missing,
    abi_host_zig_mismatch,
    zig_real_missing,
    zig_missing,
    cache_stale,
    no_zig_found,
    unknown,
};

pub const Inspection = struct {
    expected_version: []u8,
    cache_root: []u8,
    cache_path: []u8,
    cache_exists: bool,
    cache_version: ?[]u8,
    cache_matches_expected: bool,
    selected_status: SelectionStatus,
    selected_source: SelectionSource,
    selected_env_name: ?[]u8,
    selected_path: ?[]u8,
    selected_version: ?[]u8,
    selected_matches_expected: bool,

    pub fn deinit(self: *Inspection, allocator: std.mem.Allocator) void {
        allocator.free(self.expected_version);
        allocator.free(self.cache_root);
        allocator.free(self.cache_path);
        if (self.cache_version) |value| allocator.free(value);
        if (self.selected_env_name) |value| allocator.free(value);
        if (self.selected_path) |value| allocator.free(value);
        if (self.selected_version) |value| allocator.free(value);
    }
};

pub fn inspect(allocator: std.mem.Allocator, io: std.Io) !Inspection {
    const result = try captureCommand(allocator, io, "bash tools/scripts/inspect_toolchain.sh");
    defer allocator.free(result.output);

    if (result.exit_code != 0) {
        std.debug.print("inspect_toolchain.sh failed (exit {d}):\n{s}\n", .{ result.exit_code, result.output });
        return error.ToolchainInspectionFailed;
    }

    var expected_version: ?[]u8 = null;
    errdefer if (expected_version) |value| allocator.free(value);
    var cache_root: ?[]u8 = null;
    errdefer if (cache_root) |value| allocator.free(value);
    var cache_path: ?[]u8 = null;
    errdefer if (cache_path) |value| allocator.free(value);
    var cache_exists = false;
    var cache_version: ?[]u8 = null;
    errdefer if (cache_version) |value| allocator.free(value);
    var cache_matches_expected = false;
    var selected_status: SelectionStatus = .unknown;
    var selected_source: SelectionSource = .none;
    var selected_env_name: ?[]u8 = null;
    errdefer if (selected_env_name) |value| allocator.free(value);
    var selected_path: ?[]u8 = null;
    errdefer if (selected_path) |value| allocator.free(value);
    var selected_version: ?[]u8 = null;
    errdefer if (selected_version) |value| allocator.free(value);
    var selected_matches_expected = false;

    var lines = std.mem.splitScalar(u8, result.output, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const eq_idx = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const key = line[0..eq_idx];
        const value = line[eq_idx + 1 ..];

        if (std.mem.eql(u8, key, "expected_version")) {
            expected_version = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "cache_root")) {
            cache_root = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "cache_path")) {
            cache_path = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, key, "cache_exists")) {
            cache_exists = parseBool(value);
        } else if (std.mem.eql(u8, key, "cache_version")) {
            cache_version = try dupeOptional(allocator, value);
        } else if (std.mem.eql(u8, key, "cache_matches_expected")) {
            cache_matches_expected = parseBool(value);
        } else if (std.mem.eql(u8, key, "selected_status")) {
            selected_status = parseStatus(value);
        } else if (std.mem.eql(u8, key, "selected_source")) {
            selected_source = parseSource(value);
        } else if (std.mem.eql(u8, key, "selected_env_name")) {
            selected_env_name = try dupeOptional(allocator, value);
        } else if (std.mem.eql(u8, key, "selected_path")) {
            selected_path = try dupeOptional(allocator, value);
        } else if (std.mem.eql(u8, key, "selected_version")) {
            selected_version = try dupeOptional(allocator, value);
        } else if (std.mem.eql(u8, key, "selected_matches_expected")) {
            selected_matches_expected = parseBool(value);
        }
    }

    return .{
        .expected_version = expected_version orelse return error.InvalidToolchainInspection,
        .cache_root = cache_root orelse return error.InvalidToolchainInspection,
        .cache_path = cache_path orelse return error.InvalidToolchainInspection,
        .cache_exists = cache_exists,
        .cache_version = cache_version,
        .cache_matches_expected = cache_matches_expected,
        .selected_status = selected_status,
        .selected_source = selected_source,
        .selected_env_name = selected_env_name,
        .selected_path = selected_path,
        .selected_version = selected_version,
        .selected_matches_expected = selected_matches_expected,
    };
}

pub fn printDoctorReport(allocator: std.mem.Allocator, io: std.Io) !usize {
    var inspection = try inspect(allocator, io);
    defer inspection.deinit(allocator);

    std.debug.print("ABI toolchain doctor\n", .{});
    std.debug.print("Pinned Zig (.zigversion): {s}\n\n", .{inspection.expected_version});

    std.debug.print("Active zig resolution:\n", .{});
    std.debug.print("  source:   {s}\n", .{inspection.selected_source.label()});
    std.debug.print("  path:     {s}\n", .{inspection.selected_path orelse "(unresolved)"});
    std.debug.print("  version:  {s}\n", .{inspection.selected_version orelse "(unresolved)"});
    std.debug.print("  matches:  {s}\n\n", .{boolLabel(inspection.selected_matches_expected)});

    std.debug.print("Canonical cached host-built Zig:\n", .{});
    std.debug.print("  path:     {s}\n", .{inspection.cache_path});
    std.debug.print("  exists:   {s}\n", .{boolLabel(inspection.cache_exists)});
    std.debug.print("  version:  {s}\n", .{inspection.cache_version orelse "(missing)"});
    std.debug.print("  matches:  {s}\n\n", .{boolLabel(inspection.cache_matches_expected)});

    std.debug.print("Environment selectors:\n", .{});
    try printEnvVar(allocator, io, "ABI_HOST_ZIG");
    try printEnvVar(allocator, io, "ABI_ZIG_SOURCE_DIR");
    try printEnvVar(allocator, io, "ABI_HOST_ZIG_CACHE_DIR");
    try printEnvVar(allocator, io, "ZIG_REAL");
    try printEnvVar(allocator, io, "ZIG");
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
        std.debug.print("\n", .{});
    }

    std.debug.print("All zig candidates on PATH (in precedence order):\n", .{});
    if (try commandExists(allocator, io, "which")) {
        const which_res = try captureCommand(allocator, io, "which -a zig");
        defer allocator.free(which_res.output);

        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(allocator);

        var lines = std.mem.splitScalar(u8, which_res.output, '\n');
        while (lines.next()) |line| {
            const trimmed = trimSpace(line);
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

    switch (inspection.selected_status) {
        .ok => {},
        .abi_host_zig_missing => {
            std.debug.print("ISSUE: ABI_HOST_ZIG points to a missing or non-executable binary\n", .{});
            issues += 1;
        },
        .abi_host_zig_mismatch => {
            std.debug.print("ISSUE: ABI_HOST_ZIG does not match .zigversion\n", .{});
            issues += 1;
        },
        .zig_real_missing => {
            std.debug.print("ISSUE: ZIG_REAL points to a missing or non-executable binary\n", .{});
            issues += 1;
        },
        .zig_missing => {
            std.debug.print("ISSUE: ZIG points to a missing or non-executable binary\n", .{});
            issues += 1;
        },
        .cache_stale => {
            std.debug.print("ISSUE: canonical cached host-built Zig is stale and must be rebuilt\n", .{});
            issues += 1;
        },
        .no_zig_found => {
            std.debug.print("ISSUE: no usable zig binary was resolved\n", .{});
            issues += 1;
        },
        .unknown => {
            std.debug.print("ISSUE: toolchain inspection returned an unknown status\n", .{});
            issues += 1;
        },
    }

    if (inspection.selected_status == .ok and !inspection.selected_matches_expected) {
        std.debug.print("ISSUE: active zig version does not match .zigversion\n", .{});
        issues += 1;
    }

    if (issues == 0) {
        std.debug.print("OK: active Zig resolution matches the repository pin.\n", .{});
        if (builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26) {
            std.debug.print(
                "NOTE: direct zig build gates on macOS 26+ should use the canonical cached host-built Zig or an explicit ABI_HOST_ZIG override.\n",
                .{},
            );
            if (!inspection.cache_exists) {
                try printRecommendedBootstrap(&inspection);
            }
        }
        return 0;
    }

    std.debug.print("\nRecommended next commands:\n", .{});
    try printRecommendedBootstrap(&inspection);
    std.debug.print("\nFAILED: toolchain doctor found {d} issue(s).\n", .{issues});
    return issues;
}

pub fn printRecommendedBootstrap(inspection: *const Inspection) !void {
    const cache_bin_dir = std.fs.path.dirname(inspection.cache_path) orelse inspection.cache_path;
    std.debug.print("  ./tools/scripts/bootstrap_host_zig.sh\n", .{});
    std.debug.print("  export PATH=\"{s}:$PATH\"\n", .{cache_bin_dir});
    std.debug.print("  hash -r\n", .{});
    std.debug.print("  zig build toolchain-doctor\n", .{});
}

fn dupeOptional(allocator: std.mem.Allocator, value: []const u8) !?[]u8 {
    if (value.len == 0) return null;
    const duplicated = try allocator.dupe(u8, value);
    return duplicated;
}

fn parseBool(value: []const u8) bool {
    return std.mem.eql(u8, value, "1") or std.mem.eql(u8, value, "true");
}

fn parseStatus(value: []const u8) SelectionStatus {
    if (std.mem.eql(u8, value, "ok")) return .ok;
    if (std.mem.eql(u8, value, "abi_host_zig_missing")) return .abi_host_zig_missing;
    if (std.mem.eql(u8, value, "abi_host_zig_mismatch")) return .abi_host_zig_mismatch;
    if (std.mem.eql(u8, value, "zig_real_missing")) return .zig_real_missing;
    if (std.mem.eql(u8, value, "zig_missing")) return .zig_missing;
    if (std.mem.eql(u8, value, "cache_stale")) return .cache_stale;
    if (std.mem.eql(u8, value, "no_zig_found")) return .no_zig_found;
    return .unknown;
}

fn parseSource(value: []const u8) SelectionSource {
    if (std.mem.eql(u8, value, "abi_host_zig")) return .abi_host_zig;
    if (std.mem.eql(u8, value, "zig_real")) return .zig_real;
    if (std.mem.eql(u8, value, "zig_env")) return .zig_env;
    if (std.mem.eql(u8, value, "cache")) return .cache;
    if (std.mem.eql(u8, value, "path")) return .path;
    return .none;
}

fn boolLabel(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

fn printEnvVar(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !void {
    const cmd = try std.fmt.allocPrint(allocator, "printf '%s' \"${s}\"", .{name});
    defer allocator.free(cmd);

    const result = try captureCommand(allocator, io, cmd);
    defer allocator.free(result.output);

    const value = trimSpace(result.output);
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
    const result = captureCommand(allocator, io, cmd) catch {
        std.debug.print("  {s}: (unavailable)\n", .{label});
        return;
    };
    defer allocator.free(result.output);

    if (result.exit_code != 0) {
        std.debug.print("  {s}: (failed)\n", .{label});
        return;
    }

    const value = trimSpace(result.output);
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
    const result = captureCommand(allocator, io, cmd) catch {
        std.debug.print("  {s}: (unavailable)\n", .{label});
        return;
    };
    defer allocator.free(result.output);

    if (result.exit_code != 0) {
        std.debug.print("  {s}: (failed)\n", .{label});
        return;
    }

    const trimmed = trimSpace(result.output);
    if (trimmed.len == 0) {
        std.debug.print("  {s}: (empty)\n", .{label});
        return;
    }

    var lines = std.mem.splitScalar(u8, trimmed, '\n');
    const first_line = lines.next() orelse trimmed;
    std.debug.print("  {s}: {s}\n", .{ label, first_line });
}

const CommandResult = struct {
    output: []u8,
    exit_code: i32,
};

fn trimSpace(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\r\n");
}

fn captureCommand(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) !CommandResult {
    const shell = if (builtin.os.tag == .windows) "cmd" else "sh";
    const shell_arg = if (builtin.os.tag == .windows) "/c" else "-c";

    // On macOS, relinked binaries may inherit a sanitized environment that
    // lacks /opt/homebrew/bin, /usr/local/bin on PATH and may even be missing
    // HOME.  Restore common tool locations and HOME so child shell scripts
    // (e.g. inspect_toolchain.sh) work correctly.
    // Note: plain ~ does not expand when HOME is unset, but ~username does.
    const env_fixup = if (comptime builtin.os.tag == .macos)
        "export PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\"; " ++
            "[ -z \"${HOME:-}\" ] && eval \"export HOME=~$(id -un)\"; "
    else
        "";
    const merged_cmd = try std.fmt.allocPrint(allocator, "{s}{s} 2>&1", .{ env_fixup, cmd });
    defer allocator.free(merged_cmd);

    const argv = [_][]const u8{ shell, shell_arg, merged_cmd };
    var child = try std.process.spawn(io, .{
        .argv = &argv,
        .stdout = .pipe,
    });

    const stdout = try readAllAlloc(io, child.stdout.?, allocator, 1024 * 1024);
    errdefer allocator.free(stdout);

    const term = try child.wait(io);
    const exit_code = switch (term) {
        .exited => |code| @as(i32, @intCast(code)),
        else => -1,
    };

    return .{
        .output = stdout,
        .exit_code = exit_code,
    };
}

fn commandExists(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !bool {
    const cmd = try std.fmt.allocPrint(allocator, "command -v {s} >/dev/null 2>&1", .{name});
    defer allocator.free(cmd);
    const result = try captureCommand(allocator, io, cmd);
    defer allocator.free(result.output);
    return result.exit_code == 0;
}

fn readAllAlloc(io: std.Io, file: std.Io.File, allocator: std.mem.Allocator, limit: usize) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var buffer: [4096]u8 = undefined;
    while (true) {
        const amt = file.readStreaming(io, &.{&buffer}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        try list.appendSlice(allocator, buffer[0..amt]);
        if (list.items.len > limit) return error.StreamTooLong;
    }

    return try list.toOwnedSlice(allocator);
}
