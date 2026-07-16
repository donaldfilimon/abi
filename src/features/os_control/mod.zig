const std = @import("std");
const builtin = @import("builtin");
const temp_path = @import("../../foundation/temp_path.zig");

pub const CommandIntent = enum {
    read_only,
    modify_files,
    network,
    system,
    unknown,
};

pub const Decision = enum {
    deny,
    allow_dry_run,
    allow_execute,
};

pub const TrustedCommandSpec = struct {
    name: []const u8,
    executable_path: []const u8,
};

pub const CommandRequest = struct {
    argv: []const []const u8,
    cwd: ?[]const u8 = null,
    intent: CommandIntent = .unknown,
    confirm_execution: bool = false,
};

pub const Policy = struct {
    workspace_root: []const u8,
    dry_run_only: bool = true,
    allow_execution: bool = false,
    require_confirmation: bool = true,
    allow_shell: bool = false,
    allowed_commands: []const []const u8 = &.{},
    denied_commands: []const []const u8 = &.{ "rm", "rmdir", "mv", "chmod", "chown", "sudo", "su", "kill", "pkill", "shutdown", "reboot", "launchctl" },
};

pub const CommandResult = struct {
    decision: Decision,
    exit_code: ?u8 = null,
    message: []const u8,
};

test {
    std.testing.refAllDecls(@This());
}

pub fn trustedCommandSpec(name: []const u8) ?TrustedCommandSpec {
    if (!std.mem.eql(u8, name, std.fs.path.basename(name))) return null;
    return switch (builtin.target.os.tag) {
        .macos => trustedCommandSpecForPaths(name, .{
            .true_path = "/usr/bin/true",
            .pwd_path = "/bin/pwd",
            .ls_path = "/bin/ls",
            .whoami_path = "/usr/bin/whoami",
            .date_path = "/bin/date",
        }),
        .linux => trustedCommandSpecForPaths(name, .{
            .true_path = "/usr/bin/true",
            .pwd_path = "/usr/bin/pwd",
            .ls_path = "/usr/bin/ls",
            .whoami_path = "/usr/bin/whoami",
            .date_path = "/usr/bin/date",
        }),
        else => null,
    };
}

/// Performs pure/static policy preflight only. This checks command identity,
/// command-specific arguments, lexical containment, and confirmation state but
/// does not access the filesystem. Callers must use `renderDryRun` or
/// `executeConfirmed` for filesystem-canonical authorization via
/// `prepareCommand`; a successful preflight is not final authorization.
pub fn validateCommand(request: CommandRequest, policy: Policy) CommandResult {
    if (policy.workspace_root.len == 0) return deny("workspace root is required");
    if (request.argv.len == 0 or request.argv[0].len == 0) return deny("command argv is required");
    if (!std.fs.path.isAbsolute(policy.workspace_root) or hasParentPathSegment(policy.workspace_root)) return deny("workspace root must be an absolute path without traversal");
    const cwd = request.cwd orelse return deny("an explicit command cwd is required");
    if (!std.fs.path.isAbsolute(cwd) or hasParentPathSegment(cwd)) return deny("command cwd must be an absolute path without traversal");
    if (!pathContained(cwd, policy.workspace_root)) return deny("command cwd is outside workspace");
    for (request.argv) |arg| {
        if (std.mem.indexOfScalar(u8, arg, 0) != null) return deny("command argv contains a NUL byte");
    }
    if (!policy.allow_shell and isShell(request.argv[0])) return deny("shell execution is disabled by policy");
    if (contains(request.argv[0], policy.denied_commands)) return deny("command is denied by policy");
    if (policy.allowed_commands.len == 0 or !containsBareCommand(request.argv[0], policy.allowed_commands)) return deny("command is not in the execution allow-list");
    const spec = trustedCommandSpec(request.argv[0]) orelse return deny("command has no trusted executable on this target");
    if (!argumentsAllowed(spec.name, request.argv[1..])) return deny("command arguments are not allowed by policy");

    if (policy.dry_run_only or !policy.allow_execution) return .{ .decision = .allow_dry_run, .message = "dry-run allowed; execution disabled by policy" };
    if (policy.require_confirmation and !request.confirm_execution) return deny("execution requires explicit confirmation");

    return .{ .decision = .allow_execute, .message = "execution allowed by explicit policy and confirmation" };
}

/// Applies static preflight plus opened-handle canonical cwd authorization,
/// then renders the original and resolved argv without spawning a child.
pub fn renderDryRun(allocator: std.mem.Allocator, io: std.Io, request: CommandRequest, policy: Policy) anyerror![]u8 {
    var prepared = try prepareCommand(allocator, io, request, policy);
    defer prepared.deinit();

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "dry-run: cwd=");
    try appendQuoted(&out, allocator, prepared.cwd);
    try out.appendSlice(allocator, " argv=[");
    try appendArgv(&out, allocator, request.argv);
    try out.appendSlice(allocator, "] resolved_argv=[");
    try appendArgv(&out, allocator, prepared.argv);
    try out.append(allocator, ']');
    return try out.toOwnedSlice(allocator);
}

/// Applies static preflight plus opened-handle canonical cwd authorization,
/// then spawns the trusted absolute executable with an empty environment and an
/// opened cwd directory handle.
pub fn executeConfirmed(allocator: std.mem.Allocator, io: std.Io, request: CommandRequest, policy: Policy) anyerror!CommandResult {
    var prepared = try prepareCommand(allocator, io, request, policy);
    defer prepared.deinit();
    if (prepared.decision != .allow_execute) return error.CommandDenied;

    var child_environ = std.process.Environ.Map.init(allocator);
    defer child_environ.deinit();

    var child = try std.process.spawn(io, .{
        .argv = prepared.argv,
        .cwd = .{ .dir = prepared.cwd_dir },
        .environ_map = &child_environ,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    const term = try child.wait(io);
    return switch (term) {
        .exited => |code| .{ .decision = .allow_execute, .exit_code = code, .message = "command exited" },
        else => .{ .decision = .allow_execute, .exit_code = 1, .message = "command terminated without exit code" },
    };
}

fn deny(message: []const u8) CommandResult {
    return .{ .decision = .deny, .message = message };
}

fn contains(command: []const u8, list: []const []const u8) bool {
    const base = std.fs.path.basename(command);
    for (list) |item| {
        if (std.mem.eql(u8, command, item) or std.mem.eql(u8, base, item)) return true;
    }
    return false;
}

fn containsBareCommand(command: []const u8, list: []const []const u8) bool {
    if (!std.mem.eql(u8, command, std.fs.path.basename(command))) return false;
    for (list) |item| {
        if (std.mem.eql(u8, command, item)) return true;
    }
    return false;
}

fn isShell(command: []const u8) bool {
    return contains(command, &.{ "sh", "bash", "zsh", "fish" });
}

fn pathContained(path: []const u8, workspace_root: []const u8) bool {
    if (!std.fs.path.isAbsolute(path)) return false;
    if (std.mem.eql(u8, path, workspace_root)) return true;
    if (!std.mem.startsWith(u8, path, workspace_root)) return false;
    return path.len > workspace_root.len and path[workspace_root.len] == std.fs.path.sep;
}

fn hasParentPathSegment(path: []const u8) bool {
    var parts = std.mem.splitScalar(u8, path, std.fs.path.sep);
    while (parts.next()) |part| {
        if (std.mem.eql(u8, part, "..")) return true;
    }
    return false;
}

const TrustedPaths = struct {
    true_path: []const u8,
    pwd_path: []const u8,
    ls_path: []const u8,
    whoami_path: []const u8,
    date_path: []const u8,
};

fn trustedCommandSpecForPaths(name: []const u8, paths: TrustedPaths) ?TrustedCommandSpec {
    if (std.mem.eql(u8, name, "true")) return .{ .name = "true", .executable_path = paths.true_path };
    if (std.mem.eql(u8, name, "pwd")) return .{ .name = "pwd", .executable_path = paths.pwd_path };
    if (std.mem.eql(u8, name, "ls")) return .{ .name = "ls", .executable_path = paths.ls_path };
    if (std.mem.eql(u8, name, "whoami")) return .{ .name = "whoami", .executable_path = paths.whoami_path };
    if (std.mem.eql(u8, name, "date")) return .{ .name = "date", .executable_path = paths.date_path };
    return null;
}

fn argumentsAllowed(command: []const u8, args: []const []const u8) bool {
    if (std.mem.eql(u8, command, "true") or
        std.mem.eql(u8, command, "pwd") or
        std.mem.eql(u8, command, "whoami") or
        std.mem.eql(u8, command, "date"))
    {
        return args.len == 0;
    }
    if (!std.mem.eql(u8, command, "ls")) return false;

    var options = true;
    for (args) |arg| {
        if (options and std.mem.eql(u8, arg, "--")) {
            options = false;
            continue;
        }
        if (options and arg.len > 1 and arg[0] == '-') {
            if (!isAllowedLsOption(arg)) return false;
            continue;
        }
        return false;
    }
    return true;
}

fn isAllowedLsOption(arg: []const u8) bool {
    if (arg.len < 2 or arg[0] != '-' or std.mem.eql(u8, arg, "--")) return false;
    for (arg[1..]) |flag| {
        if (std.mem.indexOfScalar(u8, "1AaFhHilRrSst", flag) == null) return false;
    }
    return true;
}

const PreparedCommand = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    decision: Decision,
    cwd: []u8,
    cwd_dir: std.Io.Dir,
    argv: []const []const u8,

    fn deinit(self: *PreparedCommand) void {
        self.cwd_dir.close(self.io);
        self.allocator.free(self.cwd);
        for (self.argv) |arg| self.allocator.free(arg);
        self.allocator.free(self.argv);
        self.* = undefined;
    }
};

/// Final authorization shared by dry-run and execution. Unlike
/// `validateCommand`, this opens workspace and cwd handles, resolves canonical
/// paths from those handles, and retains the cwd handle through child spawn.
fn prepareCommand(allocator: std.mem.Allocator, io: std.Io, request: CommandRequest, policy: Policy) !PreparedCommand {
    const decision = validateCommand(request, policy);
    if (decision.decision == .deny) return error.CommandDenied;

    const request_cwd = request.cwd orelse return error.CommandDenied;
    const workspace_dir = std.Io.Dir.openDirAbsolute(io, policy.workspace_root, .{}) catch return error.CommandDenied;
    defer workspace_dir.close(io);
    const cwd_dir = std.Io.Dir.openDirAbsolute(io, request_cwd, .{}) catch return error.CommandDenied;
    errdefer cwd_dir.close(io);

    const workspace_root = canonicalDirPath(allocator, io, workspace_dir) catch return error.CommandDenied;
    defer allocator.free(workspace_root);
    const cwd = canonicalDirPath(allocator, io, cwd_dir) catch return error.CommandDenied;
    errdefer allocator.free(cwd);
    if (!pathContained(cwd, workspace_root)) return error.CommandDenied;

    const spec = trustedCommandSpec(request.argv[0]) orelse return error.CommandDenied;
    if (!std.fs.path.isAbsolute(spec.executable_path)) return error.CommandDenied;
    const argv = try allocator.alloc([]const u8, request.argv.len);
    var initialized: usize = 0;
    errdefer {
        for (argv[0..initialized]) |arg| allocator.free(arg);
        allocator.free(argv);
    }
    argv[0] = try allocator.dupe(u8, spec.executable_path);
    initialized = 1;

    for (request.argv[1..], 1..) |arg, index| {
        argv[index] = try allocator.dupe(u8, arg);
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .io = io,
        .decision = decision.decision,
        .cwd = cwd,
        .cwd_dir = cwd_dir,
        .argv = argv,
    };
}

fn canonicalDirPath(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
    var resolved: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const len = try dir.realPath(io, &resolved);
    return try allocator.dupe(u8, resolved[0..len]);
}

fn appendArgv(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, argv: []const []const u8) !void {
    for (argv, 0..) |arg, index| {
        if (index != 0) try out.appendSlice(allocator, ", ");
        try appendQuoted(out, allocator, arg);
    }
}

fn appendQuoted(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    const hex = "0123456789abcdef";
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => {
                if (byte >= 0x20 and byte <= 0x7e) {
                    try out.append(allocator, byte);
                } else {
                    try out.appendSlice(allocator, "\\x");
                    try out.append(allocator, hex[byte >> 4]);
                    try out.append(allocator, hex[byte & 0x0f]);
                }
            },
        }
    }
    try out.append(allocator, '"');
}

test "policy denies dangerous commands by default" {
    const alloc = std.testing.allocator;
    const tmp = try temp_path.getTempDir(alloc);
    defer alloc.free(tmp);
    const workspace_root = try std.fmt.allocPrint(alloc, "{s}/work", .{tmp});
    defer alloc.free(workspace_root);
    const result = validateCommand(.{ .argv = &.{ "rm", "file" } }, .{ .workspace_root = workspace_root });
    try std.testing.expectEqual(Decision.deny, result.decision);
}

test "policy allows dry-run by default" {
    const alloc = std.testing.allocator;
    const tmp = try temp_path.getTempDir(alloc);
    defer alloc.free(tmp);
    const workspace_root = try std.fmt.allocPrint(alloc, "{s}/work", .{tmp});
    defer alloc.free(workspace_root);
    const result = validateCommand(.{
        .argv = &.{ "ls", "-la" },
        .cwd = workspace_root,
    }, .{
        .workspace_root = workspace_root,
        .allowed_commands = &.{"ls"},
    });
    const expected: Decision = if (trustedCommandSpec("ls") != null) .allow_dry_run else .deny;
    try std.testing.expectEqual(expected, result.decision);
}

test "execution requires allow-list and confirmation" {
    const alloc = std.testing.allocator;
    const tmp = try temp_path.getTempDir(alloc);
    defer alloc.free(tmp);
    const workspace_root = try std.fmt.allocPrint(alloc, "{s}/work", .{tmp});
    defer alloc.free(workspace_root);
    const policy = Policy{
        .workspace_root = workspace_root,
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"true"},
    };
    try std.testing.expectEqual(Decision.deny, validateCommand(.{ .argv = &.{"true"}, .cwd = workspace_root }, policy).decision);
    const confirmed = validateCommand(.{
        .argv = &.{"true"},
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy);
    const expected: Decision = if (trustedCommandSpec("true") != null) .allow_execute else .deny;
    try std.testing.expectEqual(expected, confirmed.decision);
}

test "execution allow-list rejects path-qualified binaries" {
    const alloc = std.testing.allocator;
    const tmp = try temp_path.getTempDir(alloc);
    defer alloc.free(tmp);
    const workspace_root = try std.fmt.allocPrint(alloc, "{s}/work", .{tmp});
    defer alloc.free(workspace_root);
    const path_qualified = try std.fmt.allocPrint(alloc, "{s}/ls", .{tmp});
    defer alloc.free(path_qualified);
    const policy = Policy{
        .workspace_root = workspace_root,
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"ls"},
    };
    try std.testing.expectEqual(Decision.deny, validateCommand(.{
        .argv = &.{path_qualified},
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy).decision);
    const bare_result = validateCommand(.{
        .argv = &.{"ls"},
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy);
    const expected: Decision = if (trustedCommandSpec("ls") != null) .allow_execute else .deny;
    try std.testing.expectEqual(expected, bare_result.decision);
}

test "ls rejects all filesystem operands" {
    const alloc = std.testing.allocator;
    const tmp = try temp_path.getTempDir(alloc);
    defer alloc.free(tmp);
    const workspace_root = try std.fmt.allocPrint(alloc, "{s}/work", .{tmp});
    defer alloc.free(workspace_root);
    const policy = Policy{
        .workspace_root = workspace_root,
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"ls"},
    };
    try std.testing.expectEqual(Decision.deny, validateCommand(.{
        .argv = &.{ "ls", "../outside" },
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy).decision);
    try std.testing.expectEqual(Decision.deny, validateCommand(.{
        .argv = &.{ "ls", "safe/../../outside" },
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy).decision);
    try std.testing.expectEqual(Decision.deny, validateCommand(.{
        .argv = &.{ "ls", "safe/path" },
        .cwd = workspace_root,
        .confirm_execution = true,
    }, policy).decision);
}

test "trusted command specs resolve absolute executables without PATH" {
    const expected_ls: ?[]const u8 = switch (builtin.target.os.tag) {
        .macos => "/bin/ls",
        .linux => "/usr/bin/ls",
        else => null,
    };
    const spec = trustedCommandSpec("ls");
    if (expected_ls) |expected| {
        try std.testing.expect(spec != null);
        try std.testing.expectEqualStrings("ls", spec.?.name);
        try std.testing.expectEqualStrings(expected, spec.?.executable_path);
        try std.testing.expect(std.fs.path.isAbsolute(spec.?.executable_path));
        try std.testing.expect(!std.mem.eql(u8, spec.?.name, spec.?.executable_path));
    } else {
        try std.testing.expect(spec == null);
    }
    try std.testing.expect(trustedCommandSpec("rm") == null);
    try std.testing.expect(trustedCommandSpec("/bin/ls") == null);
}

test "command-specific policy rejects unnecessary and mutating arguments" {
    const policy = Policy{
        .workspace_root = "/tmp/work",
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{ "true", "pwd", "ls", "whoami", "date" },
    };
    inline for (.{ "true", "pwd", "whoami" }) |name| {
        try std.testing.expectEqual(Decision.deny, validateCommand(.{
            .argv = &.{ name, "unexpected" },
            .cwd = "/tmp/work",
            .confirm_execution = true,
        }, policy).decision);
    }
    inline for (.{ "-s", "--set", "010100002026" }) |arg| {
        try std.testing.expectEqual(Decision.deny, validateCommand(.{
            .argv = &.{ "date", arg },
            .cwd = "/tmp/work",
            .confirm_execution = true,
        }, policy).decision);
    }
}

test "workspace policy rejects absolute parent traversal" {
    const policy = Policy{
        .workspace_root = "/tmp/work",
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"ls"},
    };
    try std.testing.expectEqual(Decision.deny, validateCommand(.{
        .argv = &.{ "ls", "/tmp/work/../outside" },
        .cwd = "/tmp/work",
        .confirm_execution = true,
    }, policy).decision);
}

test "dry-run applies policy and renders allowed argv" {
    if (trustedCommandSpec("ls") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const root = try temp_path.tempFilePath(allocator, "abi_os_control_render", "dir");
    defer allocator.free(root);
    try std.Io.Dir.createDirPath(.cwd(), std.testing.io, root);
    defer std.Io.Dir.deleteTree(.cwd(), std.testing.io, root) catch |err| {
        std.log.warn("os_control render test cleanup failed: {s}", .{@errorName(err)});
    };

    const policy = Policy{
        .workspace_root = root,
        .allowed_commands = &.{"ls"},
    };
    const rendered = try renderDryRun(allocator, std.testing.io, .{
        .argv = &.{ "ls", "-la" },
        .cwd = root,
    }, policy);
    defer allocator.free(rendered);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "argv=[\"ls\", \"-la\"]") != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, rendered, '\n') == null);

    try std.testing.expectError(error.CommandDenied, renderDryRun(allocator, std.testing.io, .{
        .argv = &.{"rm"},
        .cwd = root,
    }, policy));
}

test "appendQuoted escapes terminal control and ambiguous bytes" {
    const allocator = std.testing.allocator;
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);
    try appendQuoted(&out, allocator, "a b\n\x1b\\\"");
    try std.testing.expectEqualStrings("\"a b\\n\\x1b\\\\\\\"\"", out.items);
}

test "opened cwd handle canonicalization rejects symlink outside workspace" {
    if (builtin.target.os.tag != .macos and builtin.target.os.tag != .linux) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const root = try temp_path.tempFilePath(allocator, "abi_os_control_root", "dir");
    defer allocator.free(root);
    try std.Io.Dir.createDirPath(.cwd(), std.testing.io, root);
    defer std.Io.Dir.deleteTree(.cwd(), std.testing.io, root) catch |err| {
        std.log.warn("os_control root test cleanup failed: {s}", .{@errorName(err)});
    };

    const outside = try temp_path.tempFilePath(allocator, "abi_os_control_outside", "dir");
    defer allocator.free(outside);
    try std.Io.Dir.createDirPath(.cwd(), std.testing.io, outside);
    defer std.Io.Dir.deleteTree(.cwd(), std.testing.io, outside) catch |err| {
        std.log.warn("os_control outside test cleanup failed: {s}", .{@errorName(err)});
    };

    const link = try std.fmt.allocPrint(allocator, "{s}/escape", .{root});
    defer allocator.free(link);
    try std.Io.Dir.symLinkAbsolute(std.testing.io, outside, link, .{});

    const policy = Policy{
        .workspace_root = root,
        .allowed_commands = &.{"ls"},
    };
    const static_preflight = validateCommand(.{
        .argv = &.{"ls"},
        .cwd = link,
    }, policy);
    try std.testing.expectEqual(Decision.allow_dry_run, static_preflight.decision);
    try std.testing.expectError(error.CommandDenied, renderDryRun(allocator, std.testing.io, .{
        .argv = &.{"ls"},
        .cwd = link,
    }, policy));
}

test "confirmed true executes with retained cwd handle and empty environment" {
    if (builtin.target.os.tag != .macos and builtin.target.os.tag != .linux) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const root = try temp_path.tempFilePath(allocator, "abi_os_control_execute", "dir");
    defer allocator.free(root);
    try std.Io.Dir.createDirPath(.cwd(), std.testing.io, root);
    defer std.Io.Dir.deleteTree(.cwd(), std.testing.io, root) catch |err| {
        std.log.warn("os_control execute test cleanup failed: {s}", .{@errorName(err)});
    };

    const result = try executeConfirmed(allocator, std.testing.io, .{
        .argv = &.{"true"},
        .cwd = root,
        .confirm_execution = true,
    }, .{
        .workspace_root = root,
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"true"},
    });
    try std.testing.expectEqual(Decision.allow_execute, result.decision);
    try std.testing.expectEqual(@as(?u8, 0), result.exit_code);
}
