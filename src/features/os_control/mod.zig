const std = @import("std");

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

pub fn validateCommand(request: CommandRequest, policy: Policy) CommandResult {
    if (policy.workspace_root.len == 0) return deny("workspace root is required");
    if (request.argv.len == 0 or request.argv[0].len == 0) return deny("command argv is required");
    if (!policy.allow_shell and isShell(request.argv[0])) return deny("shell execution is disabled by policy");
    if (contains(request.argv[0], policy.denied_commands)) return deny("command is denied by policy");
    if (!pathsContained(request, policy.workspace_root)) return deny("command path is outside workspace");

    if (policy.dry_run_only or !policy.allow_execution) return .{ .decision = .allow_dry_run, .message = "dry-run allowed; execution disabled by policy" };
    if (policy.allowed_commands.len == 0 or !contains(request.argv[0], policy.allowed_commands)) return deny("command is not in the execution allow-list");
    if (policy.require_confirmation and !request.confirm_execution) return deny("execution requires explicit confirmation");

    return .{ .decision = .allow_execute, .message = "execution allowed by explicit policy and confirmation" };
}

pub fn renderDryRun(allocator: std.mem.Allocator, request: CommandRequest) ![]u8 {
    var len: usize = "dry-run:".len;
    for (request.argv) |arg| len += 1 + arg.len;

    const out = try allocator.alloc(u8, len);
    var index: usize = 0;
    @memcpy(out[index..][0.."dry-run:".len], "dry-run:");
    index += "dry-run:".len;
    for (request.argv) |arg| {
        out[index] = ' ';
        index += 1;
        @memcpy(out[index..][0..arg.len], arg);
        index += arg.len;
    }
    return out;
}

pub fn executeConfirmed(allocator: std.mem.Allocator, io: std.Io, request: CommandRequest, policy: Policy) !CommandResult {
    _ = allocator;
    const decision = validateCommand(request, policy);
    if (decision.decision != .allow_execute) return error.CommandDenied;

    var child = try std.process.spawn(io, .{
        .argv = request.argv,
        .cwd = if (request.cwd) |cwd| .{ .path = cwd } else .inherit,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    const term = try child.wait(io);
    return switch (term) {
        .exited => |code| .{ .decision = .allow_execute, .exit_code = code, .message = "command exited" },
        else => .{ .decision = .allow_execute, .exit_code = null, .message = "command terminated without exit code" },
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

fn isShell(command: []const u8) bool {
    return contains(command, &.{ "sh", "bash", "zsh", "fish" });
}

fn pathsContained(request: CommandRequest, workspace_root: []const u8) bool {
    if (request.cwd) |cwd| {
        if (!pathContained(cwd, workspace_root)) return false;
    }

    for (request.argv[1..]) |arg| {
        if (arg.len == 0) continue;
        if (std.mem.indexOfScalar(u8, arg, 0) != null) return false;
        if (std.fs.path.isAbsolute(arg) and !pathContained(arg, workspace_root)) return false;
    }
    return true;
}

fn pathContained(path: []const u8, workspace_root: []const u8) bool {
    if (!std.fs.path.isAbsolute(path)) return true;
    if (std.mem.eql(u8, path, workspace_root)) return true;
    if (!std.mem.startsWith(u8, path, workspace_root)) return false;
    return path.len > workspace_root.len and path[workspace_root.len] == std.fs.path.sep;
}

test "policy denies dangerous commands by default" {
    const result = validateCommand(.{ .argv = &.{ "rm", "file" } }, .{ .workspace_root = "/tmp/work" });
    try std.testing.expectEqual(Decision.deny, result.decision);
}

test "policy allows dry-run by default" {
    const result = validateCommand(.{ .argv = &.{ "ls", "src" } }, .{ .workspace_root = "/tmp/work" });
    try std.testing.expectEqual(Decision.allow_dry_run, result.decision);
}

test "execution requires allow-list and confirmation" {
    const policy = Policy{
        .workspace_root = "/tmp/work",
        .dry_run_only = false,
        .allow_execution = true,
        .allowed_commands = &.{"true"},
    };
    try std.testing.expectEqual(Decision.deny, validateCommand(.{ .argv = &.{"true"} }, policy).decision);
    try std.testing.expectEqual(Decision.allow_execute, validateCommand(.{ .argv = &.{"true"}, .confirm_execution = true }, policy).decision);
}
