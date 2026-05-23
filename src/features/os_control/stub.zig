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
    _ = request;
    _ = policy;
    return .{ .decision = .deny, .message = "os control feature is disabled" };
}

pub fn renderDryRun(allocator: std.mem.Allocator, request: CommandRequest) ![]u8 {
    _ = request;
    return allocator.dupe(u8, "dry-run: os control feature is disabled");
}

test {
    std.testing.refAllDecls(@This());
}

pub fn executeConfirmed(allocator: std.mem.Allocator, io: std.Io, request: CommandRequest, policy: Policy) !CommandResult {
    _ = allocator;
    _ = io;
    _ = request;
    _ = policy;
    return error.CommandDenied;
}
