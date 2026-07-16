const std = @import("std");
const builtin = @import("builtin");

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

pub fn pathContained(path: []const u8, workspace_root: []const u8) bool {
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
