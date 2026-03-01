//! Ralph run artifact writing.

const std = @import("std");
const cfg = @import("config.zig");
const workspace = @import("workspace.zig");
const verification = @import("verification.zig");

pub const RunReport = struct {
    run_id: []const u8,
    worktree: []const u8,
    backend: []const u8,
    fallback: []const u8,
    strict_backend: bool,
    model: []const u8,
    plugin: ?[]const u8,
    iterations: usize,
    passing_iterations: usize,
    last_gate_passed: bool,
    last_gate_exit: u8,
    started_at: i64,
    ended_at: i64,
    verify_log: []const u8,
    /// Duration of the run in seconds.
    duration_seconds: i64 = 0,
    /// Number of skills extracted during this run.
    skills_added: u64 = 0,
    /// Gate command used for verification.
    gate_command: []const u8 = "zig build verify-all",
    workflow_contract_passed: bool = true,
    workflow_warning_count: usize = 0,
    replan_trigger_count: usize = 0,
    correction_count: usize = 0,
    lessons_appended: usize = 0,
};

pub const WorkflowIteration = struct {
    contract_prompt_injected: bool = false,
    warning_count: usize = 0,
    trigger: ?[]const u8 = null,
    trigger_logged: bool = false,
    correction_applied: bool = false,
    lessons_appended: usize = 0,
    verify_attempts: usize = 0,
    verify_command_rejected: bool = false,
};

pub fn writeIterationArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    run_id: []const u8,
    iteration: usize,
    task: []const u8,
    response: []const u8,
    verify: ?verification.VerifyResult,
    workflow: ?WorkflowIteration,
    changed: bool,
    committed: bool,
) !void {
    const run_dir = try workspace.runDirPath(allocator, run_id);
    defer allocator.free(run_dir);

    const path = try std.fmt.allocPrint(
        allocator,
        "{s}/iterations/{d}.json",
        .{ run_dir, iteration },
    );
    defer allocator.free(path);

    const verify_json = if (verify) |v|
        try std.fmt.allocPrint(
            allocator,
            "{{\"passed\":{s},\"exit_code\":{d},\"command\":{f}}}",
            .{
                if (v.passed) "true" else "false",
                v.exit_code,
                std.json.fmt(v.command, .{}),
            },
        )
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(verify_json);

    const workflow_json = if (workflow) |wf| blk: {
        var wf_buffer = std.ArrayListUnmanaged(u8).empty;
        defer wf_buffer.deinit(allocator);
        var wf_writer: std.Io.Writer.Allocating = .fromArrayList(allocator, &wf_buffer);
        defer wf_buffer = wf_writer.toArrayList();
        try std.json.Stringify.value(wf, .{}, &wf_writer.writer);
        break :blk try wf_buffer.toOwnedSlice(allocator);
    } else try allocator.dupe(u8, "null");
    defer allocator.free(workflow_json);

    const json = try std.fmt.allocPrint(
        allocator,
        "{{\"iteration\":{d},\"timestamp\":{d},\"task\":{f},\"response\":{f},\"changed\":{s},\"committed\":{s},\"verify\":{s},\"workflow\":{s}}}",
        .{
            iteration,
            workspace.nowEpochSeconds(),
            std.json.fmt(task, .{}),
            std.json.fmt(response, .{}),
            if (changed) "true" else "false",
            if (committed) "true" else "false",
            verify_json,
            workflow_json,
        },
    );
    defer allocator.free(json);
    try cfg.writeFile(allocator, io, path, json);
}

pub fn appendVerifyLog(
    allocator: std.mem.Allocator,
    io: std.Io,
    run_id: []const u8,
    iteration: usize,
    verify: verification.VerifyResult,
) ![]u8 {
    const run_dir = try workspace.runDirPath(allocator, run_id);
    defer allocator.free(run_dir);

    const path = try std.fmt.allocPrint(allocator, "{s}/verify-all.log", .{run_dir});
    defer allocator.free(path);

    const previous = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(64 * 1024 * 1024),
    ) catch try allocator.dupe(u8, "");
    defer allocator.free(previous);

    const combined = try std.fmt.allocPrint(
        allocator,
        "{s}\n===== iteration {d} (exit={d}) =====\n{s}\n{s}\n",
        .{ previous, iteration, verify.exit_code, verify.stdout, verify.stderr },
    );
    defer allocator.free(combined);
    try cfg.writeFile(allocator, io, path, combined);
    return allocator.dupe(u8, path);
}

pub fn writeReport(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: RunReport,
) !void {
    const run_dir = try workspace.runDirPath(allocator, report.run_id);
    defer allocator.free(run_dir);

    const report_path = try std.fmt.allocPrint(allocator, "{s}/report.json", .{run_dir});
    defer allocator.free(report_path);

    var buffer: std.ArrayListUnmanaged(u8) = .empty;
    defer buffer.deinit(allocator);
    var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &buffer);
    defer buffer = aw.toArrayList();
    try std.json.Stringify.value(report, .{}, &aw.writer);
    try cfg.writeFile(allocator, io, report_path, buffer.items);
    try workspace.writeLatestRun(allocator, io, report.run_id, report_path);
}

test {
    std.testing.refAllDecls(@This());
}
