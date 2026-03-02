//! Workflow-orchestration contract helpers for Ralph runtime.

const std = @import("std");
const cfg = @import("config.zig");

pub const ReplanTrigger = enum {
    invalid_assumption,
    scope_break,
    blocked_step,
    verification_fail,
    conflict_detected,

    pub fn label(self: ReplanTrigger) []const u8 {
        return switch (self) {
            .invalid_assumption => "invalid-assumption",
            .scope_break => "scope-break",
            .blocked_step => "blocked-step",
            .verification_fail => "verification-fail",
            .conflict_detected => "conflict-detected",
        };
    }
};

pub const ContractCheck = struct {
    passed: bool = true,
    warning_count: usize = 0,
};

pub const RuntimeMetrics = struct {
    workflow_contract_passed: bool = true,
    workflow_warning_count: usize = 0,
    replan_trigger_count: usize = 0,
    correction_count: usize = 0,
    lessons_appended: usize = 0,
};

pub const ReplanNote = struct {
    trigger: ReplanTrigger,
    impact: []const u8,
    plan_change: []const u8,
    verification_change: []const u8,
};

pub const CONTRACT_PROMPT =
    \\Workflow-Orchestration Contract (apply every iteration):
    \\1. For non-trivial work, maintain a checkable plan in tasks/todo.md.
    \\2. If verification fails or assumptions break, stop and re-plan immediately.
    \\3. Do not mark work complete without concrete verification evidence.
    \\4. Keep changes minimal and focused.
    \\5. When a correction fixes a prior failure, append a root-cause lesson to tasks/lessons.md.
;

pub fn contractPrompt() []const u8 {
    return CONTRACT_PROMPT;
}

pub fn inspectContractFiles(io: std.Io, todo_path: []const u8, lessons_path: []const u8) ContractCheck {
    var result = ContractCheck{};
    if (!fileExists(io, todo_path)) {
        result.passed = false;
        result.warning_count += 1;
    }
    if (!fileExists(io, lessons_path)) {
        result.passed = false;
        result.warning_count += 1;
    }
    return result;
}

pub fn classifyTrigger(
    verify_passed: bool,
    verify_command_rejected: bool,
    tool_failed: bool,
) ?ReplanTrigger {
    if (verify_passed) return null;
    if (verify_command_rejected) return .invalid_assumption;
    if (tool_failed) return .blocked_step;
    return .verification_fail;
}

pub fn isVerifyCommandRejected(stderr: []const u8) bool {
    return cfg.containsIgnoreCase(stderr, "gate command rejected");
}

pub fn appendReplanNote(
    allocator: std.mem.Allocator,
    io: std.Io,
    todo_path: []const u8,
    note: ReplanNote,
) !bool {
    if (!fileExists(io, todo_path)) return false;

    const existing = std.Io.Dir.cwd().readFileAlloc(
        io,
        todo_path,
        allocator,
        .limited(4 * 1024 * 1024),
    ) catch return false;
    defer allocator.free(existing);

    const block = try std.fmt.allocPrint(allocator,
        \\
        \\### Re-Plan Note ({d})
        \\- Trigger: {s}
        \\- Impact: {s}
        \\- Plan change: {s}
        \\- Verification change: {s}
        \\
    , .{
        nowEpochSeconds(),
        note.trigger.label(),
        note.impact,
        note.plan_change,
        note.verification_change,
    });
    defer allocator.free(block);

    const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ existing, block });
    defer allocator.free(combined);
    cfg.writeFile(allocator, io, todo_path, combined) catch return false;
    return true;
}

pub fn appendCorrectionLesson(
    allocator: std.mem.Allocator,
    io: std.Io,
    lessons_path: []const u8,
    trigger: ReplanTrigger,
    root_cause: []const u8,
    prevention_rule: []const u8,
) !bool {
    if (!fileExists(io, lessons_path)) return false;

    const existing = std.Io.Dir.cwd().readFileAlloc(
        io,
        lessons_path,
        allocator,
        .limited(4 * 1024 * 1024),
    ) catch return false;
    defer allocator.free(existing);

    const date = try currentDateUtc(allocator);
    defer allocator.free(date);

    const entry = try std.fmt.allocPrint(allocator,
        \\
        \\## {s} - Automated correction after {s}
        \\- Root cause: {s}
        \\- Prevention rule: {s}
        \\
    , .{
        date,
        trigger.label(),
        root_cause,
        prevention_rule,
    });
    defer allocator.free(entry);

    const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ existing, entry });
    defer allocator.free(combined);
    cfg.writeFile(allocator, io, lessons_path, combined) catch return false;
    return true;
}

fn fileExists(io: std.Io, path: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn currentDateUtc(allocator: std.mem.Allocator) ![]u8 {
    const now_i64 = nowEpochSeconds();
    const now_u64: u64 = @intCast(@max(now_i64, 0));
    const epoch_seconds = std.time.epoch.EpochSeconds{ .secs = now_u64 };
    const year_day = epoch_seconds.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_of_month: u8 = month_day.day_index + 1;
    return std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}",
        .{ year_day.year, month_day.month.numeric(), day_of_month },
    );
}

fn nowEpochSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

test {
    std.testing.refAllDecls(@This());
}
