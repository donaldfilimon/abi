//! Zig Toolchain Management
//!
//! Build and install Zig and ZLS from master branch.
//!
//! Usage:
//!   abi toolchain install           Install both Zig and ZLS from master
//!   abi toolchain zig               Install Zig from master
//!   abi toolchain zls               Install ZLS from master
//!   abi toolchain status            Show installed versions
//!   abi toolchain update            Update to latest master
//!   abi toolchain path              Print install directory for shell config

const std = @import("std");
const builtin = @import("builtin");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
// libc import for environment and process access - required for Zig 0.16
const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("stdio.h");
});

// Wrapper functions for comptime children dispatch
fn wrapInstall(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runInstallBoth(allocator, &parser);
}
fn wrapZig(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runInstallZig(allocator, &parser);
}
fn wrapZls(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runInstallZls(allocator, &parser);
}
fn wrapStatus(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runStatusSubcommand(allocator, &parser);
}
fn wrapUpdate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runUpdateSubcommand(allocator, &parser);
}
fn wrapPath(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = ArgParser.init(allocator, args);
    try runPathSubcommand(allocator, &parser);
}

pub const meta: command_mod.Meta = .{
    .name = "toolchain",
    .description = "Build and install Zig/ZLS from master (install, update, status)",
    .subcommands = &.{ "install", "zig", "zls", "status", "update", "path", "help" },
    .children = &.{
        .{ .name = "install", .description = "Install both Zig and ZLS from master", .handler = wrapInstall },
        .{ .name = "zig", .description = "Install only Zig from master", .handler = wrapZig },
        .{ .name = "zls", .description = "Install only ZLS from master", .handler = wrapZls },
        .{ .name = "status", .description = "Show installed versions", .handler = wrapStatus },
        .{ .name = "update", .description = "Update to latest master", .handler = wrapUpdate },
        .{ .name = "path", .description = "Print install directory for shell config", .handler = wrapPath },
    },
};

const output = utils.output;
const ArgParser = utils.args.ArgParser;
const HelpBuilder = utils.help.HelpBuilder;
const common_options = utils.help.common_options;

/// Default installation directory
const default_install_dir = switch (builtin.os.tag) {
    .windows => "AppData/Local/abi/toolchain",
    else => ".local/abi/toolchain",
};

/// Zig repository URL
const zig_repo = "https://github.com/ziglang/zig.git";

/// ZLS repository URL
const zls_repo = "https://github.com/zigtools/zls.git";

const toolchain_subcommands = [_][]const u8{
    "install", "zig", "zls", "status", "update", "path", "help",
};

/// Run the toolchain command with the provided arguments.
/// Only reached when no child matches (help / unknown).
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        printHelp(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    // Unknown subcommand
    output.printError("Unknown toolchain command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &toolchain_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
}

fn runInstallBoth(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runInstall(allocator, parser, .both);
}

fn runInstallZig(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runInstall(allocator, parser, .zig_only);
}

fn runInstallZls(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runInstall(allocator, parser, .zls_only);
}

fn runStatusSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runStatus(allocator, parser);
}

fn runUpdateSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runUpdate(allocator, parser);
}

fn runPathSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    try runPath(allocator, parser);
}

const InstallTarget = enum { both, zig_only, zls_only };

fn runInstall(allocator: std.mem.Allocator, parser: *ArgParser, target: InstallTarget) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    const install_dir = parser.consumeOption(&[_][]const u8{ "--prefix", "-p" });
    const jobs = parser.consumeInt(u32, &[_][]const u8{ "--jobs", "-j" }, 0);
    const clean = parser.consumeFlag(&[_][]const u8{ "--clean", "-c" });

    const base_dir = try getInstallDir(allocator, install_dir);
    defer allocator.free(base_dir);

    output.printHeader("Zig Toolchain Installer");
    output.printKeyValue("Install directory", base_dir);
    var platform_buf: [32]u8 = undefined;
    const platform = std.fmt.bufPrint(&platform_buf, "{t}", .{builtin.os.tag}) catch "unknown";
    output.printKeyValue("Platform", platform);

    var arch_buf: [32]u8 = undefined;
    const arch = std.fmt.bufPrint(&arch_buf, "{t}", .{builtin.cpu.arch}) catch "unknown";
    output.printKeyValue("Architecture", arch);

    // Create directories
    try ensureDir(allocator, base_dir);

    switch (target) {
        .both => {
            try installZig(allocator, base_dir, jobs, clean);
            try installZls(allocator, base_dir, jobs, clean);
        },
        .zig_only => try installZig(allocator, base_dir, jobs, clean),
        .zls_only => try installZls(allocator, base_dir, jobs, clean),
    }

    output.println("", .{});
    output.printSuccess("Installation complete!", .{});
    output.println("", .{});
    printPathInstructions(base_dir);
}

fn installZig(allocator: std.mem.Allocator, base_dir: []const u8, jobs: u32, clean: bool) !void {
    output.printHeader("Installing Zig from master");

    const src_dir = try std.fs.path.join(allocator, &.{ base_dir, "src", "zig" });
    defer allocator.free(src_dir);

    const bin_dir = try std.fs.path.join(allocator, &.{ base_dir, "bin" });
    defer allocator.free(bin_dir);

    try ensureDir(allocator, bin_dir);

    // Clone or update repository
    if (try dirExists(allocator, src_dir)) {
        if (clean) {
            output.printInfo("Cleaning existing Zig source...", .{});
            try runShellCommand(allocator, &.{ "rm", "-rf", src_dir });
            try cloneRepo(allocator, zig_repo, src_dir);
        } else {
            output.printInfo("Updating existing Zig source...", .{});
            try gitPull(allocator, src_dir);
        }
    } else {
        try cloneRepo(allocator, zig_repo, src_dir);
    }

    // Build Zig
    output.printInfo("Building Zig (this may take a while)...", .{});

    // Zig uses cmake for bootstrap build
    const cmake_build_dir = try std.fs.path.join(allocator, &.{ src_dir, "build" });
    defer allocator.free(cmake_build_dir);

    try ensureDir(allocator, cmake_build_dir);

    // Configure with cmake
    output.printInfo("Configuring with CMake...", .{});
    const cmake_prefix = try std.fmt.allocPrint(allocator, "-DCMAKE_INSTALL_PREFIX={s}", .{base_dir});
    defer allocator.free(cmake_prefix);
    try runShellCommandInDir(allocator, cmake_build_dir, &.{
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Release",
        cmake_prefix,
    });

    // Build
    output.printInfo("Compiling...", .{});
    if (jobs > 0) {
        const jobs_str = try std.fmt.allocPrint(allocator, "{d}", .{jobs});
        defer allocator.free(jobs_str);
        try runShellCommandInDir(allocator, cmake_build_dir, &.{ "cmake", "--build", ".", "--parallel", jobs_str });
    } else {
        try runShellCommandInDir(allocator, cmake_build_dir, &.{ "cmake", "--build", ".", "--parallel" });
    }

    // Install
    output.printInfo("Installing...", .{});
    try runShellCommandInDir(allocator, cmake_build_dir, &.{ "cmake", "--install", "." });

    // Verify installation
    const zig_bin = try std.fs.path.join(allocator, &.{ bin_dir, zigBinaryName() });
    defer allocator.free(zig_bin);

    if (try fileExists(allocator, zig_bin)) {
        output.printSuccess("Zig installed successfully", .{});
        // Get version
        const version = try getCommandOutput(allocator, &.{ zig_bin, "version" });
        defer allocator.free(version);
        output.printKeyValue("Version", version);
    } else {
        output.printError("Zig binary not found after installation", .{});
    }
}

fn installZls(allocator: std.mem.Allocator, base_dir: []const u8, jobs: u32, clean: bool) !void {
    output.printHeader("Installing ZLS from master");

    const src_dir = try std.fs.path.join(allocator, &.{ base_dir, "src", "zls" });
    defer allocator.free(src_dir);

    const bin_dir = try std.fs.path.join(allocator, &.{ base_dir, "bin" });
    defer allocator.free(bin_dir);

    try ensureDir(allocator, bin_dir);

    // Clone or update repository
    if (try dirExists(allocator, src_dir)) {
        if (clean) {
            output.printInfo("Cleaning existing ZLS source...", .{});
            try runShellCommand(allocator, &.{ "rm", "-rf", src_dir });
            try cloneRepo(allocator, zls_repo, src_dir);
        } else {
            output.printInfo("Updating existing ZLS source...", .{});
            try gitPull(allocator, src_dir);
        }
    } else {
        try cloneRepo(allocator, zls_repo, src_dir);
    }

    // Build ZLS using Zig
    output.printInfo("Building ZLS...", .{});

    // First check if zig is available
    const zig_bin = try std.fs.path.join(allocator, &.{ bin_dir, zigBinaryName() });
    defer allocator.free(zig_bin);

    const zig_cmd = if (try fileExists(allocator, zig_bin)) zig_bin else "zig";

    // Build ZLS with optional parallel jobs
    if (jobs > 0) {
        try runShellCommandInDir(allocator, src_dir, &.{
            zig_cmd, "build", "-Doptimize=ReleaseFast",
        });
    } else {
        try runShellCommandInDir(allocator, src_dir, &.{
            zig_cmd, "build", "-Doptimize=ReleaseFast",
        });
    }

    // Install to prefix (copy binaries)
    const zls_build_bin = try std.fs.path.join(allocator, &.{ src_dir, "zig-out", "bin", zlsBinaryName() });
    defer allocator.free(zls_build_bin);

    if (comptime builtin.os.tag == .windows) {
        const copy_str = try std.fmt.allocPrint(allocator, "copy \"{s}\" \"{s}\"", .{ zls_build_bin, bin_dir });
        defer allocator.free(copy_str);
        const copy_cmd = try allocator.dupeZ(u8, copy_str);
        defer allocator.free(copy_cmd);
        _ = c.system(copy_cmd.ptr);
    } else {
        const copy_str = try std.fmt.allocPrint(allocator, "cp \"{s}\" \"{s}/\"", .{ zls_build_bin, bin_dir });
        defer allocator.free(copy_str);
        const copy_cmd = try allocator.dupeZ(u8, copy_str);
        defer allocator.free(copy_cmd);
        _ = c.system(copy_cmd.ptr);
    }

    // Verify installation
    const zls_bin = try std.fs.path.join(allocator, &.{ bin_dir, zlsBinaryName() });
    defer allocator.free(zls_bin);

    if (try fileExists(allocator, zls_bin)) {
        output.printSuccess("ZLS installed successfully", .{});
        // Get version
        const version = try getCommandOutput(allocator, &.{ zls_bin, "--version" });
        defer allocator.free(version);
        output.printKeyValue("Version", version);
    } else {
        output.printError("ZLS binary not found after installation", .{});
    }
}

fn runStatus(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    const install_dir = parser.consumeOption(&[_][]const u8{ "--prefix", "-p" });

    const base_dir = try getInstallDir(allocator, install_dir);
    defer allocator.free(base_dir);

    const bin_dir = try std.fs.path.join(allocator, &.{ base_dir, "bin" });
    defer allocator.free(bin_dir);

    output.printHeader("Toolchain Status");
    output.printKeyValue("Install directory", base_dir);

    // Check Zig
    const zig_bin = try std.fs.path.join(allocator, &.{ bin_dir, zigBinaryName() });
    defer allocator.free(zig_bin);

    if (try fileExists(allocator, zig_bin)) {
        const version = getCommandOutput(allocator, &.{ zig_bin, "version" }) catch "unknown";
        defer if (!std.mem.eql(u8, version, "unknown")) allocator.free(version);
        output.printKeyValue("Zig", version);
    } else {
        output.printKeyValue("Zig", "not installed");
    }

    // Check ZLS
    const zls_bin = try std.fs.path.join(allocator, &.{ bin_dir, zlsBinaryName() });
    defer allocator.free(zls_bin);

    if (try fileExists(allocator, zls_bin)) {
        const version = getCommandOutput(allocator, &.{ zls_bin, "--version" }) catch "unknown";
        defer if (!std.mem.eql(u8, version, "unknown")) allocator.free(version);
        output.printKeyValue("ZLS", version);
    } else {
        output.printKeyValue("ZLS", "not installed");
    }

    // Check system Zig for comparison
    output.println("", .{});
    output.printInfo("System Zig (if any):", .{});
    const sys_version = getCommandOutput(allocator, &.{ "zig", "version" }) catch "not found in PATH";
    defer if (!std.mem.eql(u8, sys_version, "not found in PATH")) allocator.free(sys_version);
    output.printKeyValue("System Zig", sys_version);
}

fn runUpdate(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    const install_dir = parser.consumeOption(&[_][]const u8{ "--prefix", "-p" });
    const jobs = parser.consumeInt(u32, &[_][]const u8{ "--jobs", "-j" }, 0);
    const zig_only = parser.consumeFlag(&[_][]const u8{"--zig"});
    const zls_only = parser.consumeFlag(&[_][]const u8{"--zls"});

    const base_dir = try getInstallDir(allocator, install_dir);
    defer allocator.free(base_dir);

    output.printHeader("Updating Toolchain");

    if (zig_only) {
        try installZig(allocator, base_dir, jobs, false);
    } else if (zls_only) {
        try installZls(allocator, base_dir, jobs, false);
    } else {
        try installZig(allocator, base_dir, jobs, false);
        try installZls(allocator, base_dir, jobs, false);
    }

    output.println("", .{});
    output.printSuccess("Update complete!", .{});
}

fn runPath(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    const install_dir = parser.consumeOption(&[_][]const u8{ "--prefix", "-p" });
    const shell = parser.consumeOption(&[_][]const u8{ "--shell", "-s" });

    const base_dir = try getInstallDir(allocator, install_dir);
    defer allocator.free(base_dir);

    const bin_dir = try std.fs.path.join(allocator, &.{ base_dir, "bin" });
    defer allocator.free(bin_dir);

    if (shell) |sh| {
        if (std.mem.eql(u8, sh, "bash") or std.mem.eql(u8, sh, "zsh")) {
            std.debug.print("export PATH=\"{s}:$PATH\"\n", .{bin_dir});
        } else if (std.mem.eql(u8, sh, "fish")) {
            std.debug.print("set -gx PATH {s} $PATH\n", .{bin_dir});
        } else if (std.mem.eql(u8, sh, "powershell") or std.mem.eql(u8, sh, "pwsh")) {
            std.debug.print("$env:PATH = \"{s};$env:PATH\"\n", .{bin_dir});
        } else if (std.mem.eql(u8, sh, "cmd")) {
            std.debug.print("set PATH={s};%PATH%\n", .{bin_dir});
        } else {
            std.debug.print("{s}\n", .{bin_dir});
        }
    } else {
        std.debug.print("{s}\n", .{bin_dir});
    }
}

// Helper functions

fn getInstallDir(allocator: std.mem.Allocator, override: ?[]const u8) ![]const u8 {
    if (override) |dir| {
        return allocator.dupe(u8, dir);
    }

    const home = getEnvOwned(allocator, "HOME") orelse
        getEnvOwned(allocator, "USERPROFILE") orelse
        return error.HomeNotFound;
    defer allocator.free(home);

    return std.fs.path.join(allocator, &.{ home, default_install_dir });
}

/// Get environment variable (owned memory) - Zig 0.16 compatible.
fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) ?[]u8 {
    const name_z = allocator.dupeZ(u8, name) catch return null;
    defer allocator.free(name_z);

    const value_ptr = c.getenv(name_z.ptr);
    if (value_ptr) |ptr| {
        const value = std.mem.span(ptr);
        return allocator.dupe(u8, value) catch null;
    }
    return null;
}

fn ensureDir(allocator: std.mem.Allocator, path: []const u8) !void {
    // Use shell commands for recursive directory creation
    if (comptime builtin.os.tag == .windows) {
        runShellCommand(allocator, &.{ "cmd", "/c", "mkdir", path }) catch {};
    } else {
        runShellCommand(allocator, &.{ "mkdir", "-p", path }) catch {};
    }
}

fn dirExists(allocator: std.mem.Allocator, path: []const u8) !bool {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    dir.close(io);
    return true;
}

fn fileExists(allocator: std.mem.Allocator, path: []const u8) !bool {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    file.close(io);
    return true;
}

fn cloneRepo(allocator: std.mem.Allocator, repo: []const u8, dest: []const u8) !void {
    output.printInfo("Cloning {s}...", .{repo});
    try runShellCommand(allocator, &.{ "git", "clone", "--depth", "1", repo, dest });
}

fn gitPull(allocator: std.mem.Allocator, dir: []const u8) !void {
    try runShellCommandInDir(allocator, dir, &.{ "git", "fetch", "--depth", "1", "origin", "master" });
    try runShellCommandInDir(allocator, dir, &.{ "git", "reset", "--hard", "origin/master" });
}

fn runShellCommand(allocator: std.mem.Allocator, argv: []const []const u8) !void {
    // Build command string for system() - Zig 0.16 compatible
    var cmd_buf = std.ArrayListUnmanaged(u8).empty;
    defer cmd_buf.deinit(allocator);

    for (argv, 0..) |arg, i| {
        if (i > 0) try cmd_buf.append(allocator, ' ');
        // Quote arguments with spaces
        if (std.mem.indexOf(u8, arg, " ") != null) {
            try cmd_buf.append(allocator, '"');
            try cmd_buf.appendSlice(allocator, arg);
            try cmd_buf.append(allocator, '"');
        } else {
            try cmd_buf.appendSlice(allocator, arg);
        }
    }
    try cmd_buf.append(allocator, 0);

    const ret = c.system(@ptrCast(cmd_buf.items.ptr));
    if (ret != 0) return error.CommandFailed;
}

fn runShellCommandInDir(allocator: std.mem.Allocator, dir: []const u8, argv: []const []const u8) !void {
    // Build command with cd prefix - Zig 0.16 compatible
    var cmd_buf = std.ArrayListUnmanaged(u8).empty;
    defer cmd_buf.deinit(allocator);

    if (builtin.os.tag == .windows) {
        try cmd_buf.appendSlice(allocator, "cd /d \"");
    } else {
        try cmd_buf.appendSlice(allocator, "cd \"");
    }
    try cmd_buf.appendSlice(allocator, dir);
    try cmd_buf.appendSlice(allocator, "\" && ");

    for (argv, 0..) |arg, i| {
        if (i > 0) try cmd_buf.append(allocator, ' ');
        if (std.mem.indexOf(u8, arg, " ") != null) {
            try cmd_buf.append(allocator, '"');
            try cmd_buf.appendSlice(allocator, arg);
            try cmd_buf.append(allocator, '"');
        } else {
            try cmd_buf.appendSlice(allocator, arg);
        }
    }
    try cmd_buf.append(allocator, 0);

    const ret = c.system(@ptrCast(cmd_buf.items.ptr));
    if (ret != 0) return error.CommandFailed;
}

fn getCommandOutput(allocator: std.mem.Allocator, argv: []const []const u8) ![]const u8 {
    // For version checks, use popen - Zig 0.16 compatible
    var cmd_buf = std.ArrayListUnmanaged(u8).empty;
    defer cmd_buf.deinit(allocator);

    for (argv, 0..) |arg, i| {
        if (i > 0) try cmd_buf.append(allocator, ' ');
        if (std.mem.indexOf(u8, arg, " ") != null) {
            try cmd_buf.append(allocator, '"');
            try cmd_buf.appendSlice(allocator, arg);
            try cmd_buf.append(allocator, '"');
        } else {
            try cmd_buf.appendSlice(allocator, arg);
        }
    }
    try cmd_buf.append(allocator, 0);

    // Windows uses _popen/_pclose, POSIX uses popen/pclose
    const pipe = if (comptime builtin.os.tag == .windows)
        c._popen(@ptrCast(cmd_buf.items.ptr), "r")
    else
        c.popen(@ptrCast(cmd_buf.items.ptr), "r");

    if (pipe == null) return error.CommandFailed;
    defer _ = if (comptime builtin.os.tag == .windows) c._pclose(pipe) else c.pclose(pipe);

    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    var buf: [256]u8 = undefined;
    while (true) {
        const ptr = c.fgets(&buf, @intCast(buf.len), pipe);
        if (ptr == null) break;
        const len = std.mem.indexOf(u8, &buf, &[_]u8{0}) orelse buf.len;
        try result.appendSlice(allocator, buf[0..len]);
    }

    // Trim trailing newlines manually
    const cmd_output = result.toOwnedSlice(allocator) catch return error.OutOfMemory;
    var end = cmd_output.len;
    while (end > 0 and (cmd_output[end - 1] == '\r' or cmd_output[end - 1] == '\n')) {
        end -= 1;
    }
    if (end != cmd_output.len) {
        const final = try allocator.dupe(u8, cmd_output[0..end]);
        allocator.free(cmd_output);
        return final;
    }
    return cmd_output;
}

fn zigBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zig.exe" else "zig";
}

fn zlsBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zls.exe" else "zls";
}

fn printPathInstructions(base_dir: []const u8) void {
    output.printInfo("Add to your shell configuration:", .{});
    output.println("", .{});

    switch (builtin.os.tag) {
        .windows => {
            std.debug.print("  PowerShell:\n", .{});
            std.debug.print("    $env:PATH = \"{s}\\bin;$env:PATH\"\n", .{base_dir});
            std.debug.print("\n  Or run: abi toolchain path --shell powershell\n", .{});
        },
        else => {
            std.debug.print("  Bash/Zsh:\n", .{});
            std.debug.print("    export PATH=\"{s}/bin:$PATH\"\n", .{base_dir});
            std.debug.print("\n  Fish:\n", .{});
            std.debug.print("    set -gx PATH {s}/bin $PATH\n", .{base_dir});
            std.debug.print("\n  Or run: abi toolchain path --shell bash\n", .{});
        },
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi toolchain", "<command> [options]")
        .description("Build and install Zig and ZLS from master branch.")
        .section("Commands")
        .subcommand(.{ .name = "install", .description = "Install both Zig and ZLS from master" })
        .subcommand(.{ .name = "zig", .description = "Install only Zig from master" })
        .subcommand(.{ .name = "zls", .description = "Install only ZLS from master" })
        .subcommand(.{ .name = "status", .description = "Show installed versions" })
        .subcommand(.{ .name = "update", .description = "Update to latest master" })
        .subcommand(.{ .name = "path", .description = "Print install directory for shell config" })
        .newline()
        .section("Options")
        .option(.{ .short = "-p", .long = "--prefix", .arg = "DIR", .description = "Installation directory (default: ~/.local/abi/toolchain)" })
        .option(.{ .short = "-j", .long = "--jobs", .arg = "N", .description = "Number of parallel build jobs" })
        .option(.{ .short = "-c", .long = "--clean", .description = "Clean build (remove existing source)" })
        .option(.{ .short = "-s", .long = "--shell", .arg = "SHELL", .description = "Output PATH for shell (bash, zsh, fish, powershell)" })
        .option(common_options.help)
        .newline()
        .section("Update Options")
        .option(.{ .long = "--zig", .description = "Update only Zig" })
        .option(.{ .long = "--zls", .description = "Update only ZLS" })
        .newline()
        .section("Requirements")
        .example("git", "For cloning repositories")
        .example("cmake", "For building Zig (bootstrap)")
        .example("C compiler", "GCC, Clang, or MSVC for Zig bootstrap")
        .newline()
        .section("Examples")
        .example("abi toolchain install", "Install Zig and ZLS to default location")
        .example("abi toolchain install -p ~/zig-master -j 8", "Custom location, 8 parallel jobs")
        .example("abi toolchain zig --clean", "Fresh Zig install")
        .example("abi toolchain update --zls", "Update only ZLS")
        .example("abi toolchain status", "Check installed versions")
        .example("abi toolchain path --shell bash >> ~/.bashrc", "Add to shell config");

    builder.print();
}
