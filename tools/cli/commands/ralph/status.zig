//! ralph status — show runtime state, latest run, and persisted skills.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const workspace = @import("workspace.zig");
const skills_store = @import("skills_store.zig");

const ReportView = struct {
    run_id: []u8 = &.{},
    backend: []u8 = &.{},
    fallback: []u8 = &.{},
    model: []u8 = &.{},
    iterations: usize = 0,
    passing_iterations: usize = 0,
    gate_passed: bool = false,
    gate_exit: u8 = 1,
    workflow_contract_passed: bool = true,
    workflow_warning_count: usize = 0,
    replan_trigger_count: usize = 0,
    correction_count: usize = 0,
    lessons_appended: usize = 0,

    fn deinit(self: *ReportView, allocator: std.mem.Allocator) void {
        allocator.free(self.run_id);
        allocator.free(self.backend);
        allocator.free(self.fallback);
        allocator.free(self.model);
    }
};

pub fn runStatus(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    for (args) |arg| {
        if (utils.args.matchesAny(std.mem.sliceTo(arg, 0), &[_][]const u8{ "--help", "-h", "help" })) {
            utils.output.println(
                \\Usage: abi ralph status
                \\
                \\Show Ralph run status: latest run id, backend chain, lock state, and persisted skills.
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (!cfg.fileExists(io, cfg.STATE_FILE)) {
        utils.output.printWarning("No Ralph workspace found. Run 'abi ralph init' first.", .{});
        return;
    }

    const state = cfg.readState(allocator, io);
    const lock_exists = cfg.fileExists(io, cfg.LOCK_FILE);
    const lock_info = workspace.readLockInfo(allocator, io, 60 * 60 * 4);
    const skill_count = skills_store.countSkills(allocator, io);

    utils.output.printHeader("Ralph status");
    utils.output.printKeyValue("Loop lock", if (lock_exists) "present" else "idle");
    if (lock_exists and lock_info.started_at > 0) {
        utils.output.printKeyValueFmt("Lock started_at", "{d} (stale={s})", .{
            lock_info.started_at,
            if (lock_info.stale) "true" else "false",
        });
    }
    utils.output.printKeyValueFmt("Total runs", "{d}", .{state.runs});
    utils.output.printKeyValueFmt("State last_ts", "{d}", .{state.last_run_ts});
    utils.output.printKeyValue("State last_gate", if (state.last_gate_passed) "pass" else "fail");
    utils.output.printKeyValueFmt("Persisted skills", "{d} ({s})", .{ skill_count, cfg.SKILLS_FILE });

    const latest_report_path = workspace.latestReportPath(allocator, io);
    if (latest_report_path) |path| {
        defer allocator.free(path);
        utils.output.printKeyValue("Latest report", path);

        if (parseReport(allocator, io, path)) |report| {
            defer {
                var mutable = report;
                mutable.deinit(allocator);
            }
            utils.output.printKeyValue("Run id", report.run_id);
            utils.output.printKeyValue("Backend", report.backend);
            utils.output.printKeyValue("Fallback", report.fallback);
            utils.output.printKeyValue("Model", report.model);
            utils.output.printKeyValueFmt("Iterations", "{d}", .{report.iterations});
            utils.output.printKeyValueFmt("Passing iters", "{d}", .{report.passing_iterations});
            utils.output.printKeyValueFmt("Last gate", "{s} (exit={d})", .{
                if (report.gate_passed) "pass" else "fail",
                report.gate_exit,
            });
            utils.output.printKeyValue("Workflow contract", if (report.workflow_contract_passed) "pass" else "warn");
            utils.output.printKeyValueFmt("Workflow warnings", "{d}", .{report.workflow_warning_count});
            utils.output.printKeyValueFmt("Re-plan triggers", "{d}", .{report.replan_trigger_count});
            utils.output.printKeyValueFmt("Corrections", "{d}", .{report.correction_count});
            utils.output.printKeyValueFmt("Lessons appended", "{d}", .{report.lessons_appended});
        }
    } else {
        utils.output.printKeyValue("Latest report", "(none)");
    }
}

fn parseReport(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ?ReportView {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(512 * 1024),
    ) catch return null;
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, contents, .{}) catch return null;
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };

    const run_id = valueString(allocator, obj, "run_id");
    const backend = valueString(allocator, obj, "backend");
    const fallback = valueString(allocator, obj, "fallback");
    const model = valueString(allocator, obj, "model");
    const iterations = valueUsize(obj, "iterations");
    const passing_iterations = valueUsize(obj, "passing_iterations");
    const gate_passed = valueBool(obj, "last_gate_passed");
    const gate_exit = valueU8(obj, "last_gate_exit");
    const workflow_contract_passed = valueBoolOr(obj, "workflow_contract_passed", true);
    const workflow_warning_count = valueUsize(obj, "workflow_warning_count");
    const replan_trigger_count = valueUsize(obj, "replan_trigger_count");
    const correction_count = valueUsize(obj, "correction_count");
    const lessons_appended = valueUsize(obj, "lessons_appended");

    return .{
        .run_id = run_id,
        .backend = backend,
        .fallback = fallback,
        .model = model,
        .iterations = iterations,
        .passing_iterations = passing_iterations,
        .gate_passed = gate_passed,
        .gate_exit = gate_exit,
        .workflow_contract_passed = workflow_contract_passed,
        .workflow_warning_count = workflow_warning_count,
        .replan_trigger_count = replan_trigger_count,
        .correction_count = correction_count,
        .lessons_appended = lessons_appended,
    };
}

fn valueString(
    allocator: std.mem.Allocator,
    obj: std.json.ObjectMap,
    key: []const u8,
) []u8 {
    const s = if (obj.get(key)) |v| switch (v) {
        .string => |value| value,
        else => "",
    } else "";
    return allocator.dupe(u8, s) catch &.{};
}

fn valueUsize(obj: std.json.ObjectMap, key: []const u8) usize {
    return if (obj.get(key)) |v| switch (v) {
        .integer => |value| @intCast(@max(value, 0)),
        else => 0,
    } else 0;
}

fn valueU8(obj: std.json.ObjectMap, key: []const u8) u8 {
    return if (obj.get(key)) |v| switch (v) {
        .integer => |value| @intCast(@min(@max(value, 0), 255)),
        else => 0,
    } else 0;
}

fn valueBool(obj: std.json.ObjectMap, key: []const u8) bool {
    return if (obj.get(key)) |v| switch (v) {
        .bool => |b| b,
        else => false,
    } else false;
}

fn valueBoolOr(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    if (obj.get(key)) |v| {
        return switch (v) {
            .bool => |b| b,
            else => default,
        };
    }
    return default;
}

test {
    std.testing.refAllDecls(@This());
}
