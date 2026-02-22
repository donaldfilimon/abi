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
            std.debug.print(
                \\Usage: abi ralph status
                \\
                \\Show Ralph run status: latest run id, backend chain, lock state, and persisted skills.
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (!cfg.fileExists(io, cfg.STATE_FILE)) {
        std.debug.print("No Ralph workspace found. Run 'abi ralph init' first.\n", .{});
        return;
    }

    const state = cfg.readState(allocator, io);
    const lock_exists = cfg.fileExists(io, cfg.LOCK_FILE);
    const lock_info = workspace.readLockInfo(allocator, io, 60 * 60 * 4);
    const skill_count = skills_store.countSkills(allocator, io);

    std.debug.print("Ralph status\n", .{});
    std.debug.print("────────────\n", .{});
    std.debug.print("Loop lock:       {s}\n", .{if (lock_exists) "present" else "idle"});
    if (lock_exists and lock_info.started_at > 0) {
        std.debug.print("Lock started_at: {d} (stale={s})\n", .{
            lock_info.started_at,
            if (lock_info.stale) "true" else "false",
        });
    }
    std.debug.print("Total runs:      {d}\n", .{state.runs});
    std.debug.print("State last_ts:   {d}\n", .{state.last_run_ts});
    std.debug.print("State last_gate: {s}\n", .{if (state.last_gate_passed) "pass" else "fail"});
    std.debug.print("Persisted skills:{d} ({s})\n", .{ skill_count, cfg.SKILLS_FILE });

    const latest_report_path = workspace.latestReportPath(allocator, io);
    if (latest_report_path) |path| {
        defer allocator.free(path);
        std.debug.print("Latest report:   {s}\n", .{path});

        if (parseReport(allocator, io, path)) |report| {
            defer {
                var mutable = report;
                mutable.deinit(allocator);
            }
            std.debug.print("Run id:          {s}\n", .{report.run_id});
            std.debug.print("Backend:         {s}\n", .{report.backend});
            std.debug.print("Fallback:        {s}\n", .{report.fallback});
            std.debug.print("Model:           {s}\n", .{report.model});
            std.debug.print("Iterations:      {d}\n", .{report.iterations});
            std.debug.print("Passing iters:   {d}\n", .{report.passing_iterations});
            std.debug.print("Last gate:       {s} (exit={d})\n", .{
                if (report.gate_passed) "pass" else "fail",
                report.gate_exit,
            });
        }
    } else {
        std.debug.print("Latest report:   (none)\n", .{});
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

    return .{
        .run_id = run_id,
        .backend = backend,
        .fallback = fallback,
        .model = model,
        .iterations = iterations,
        .passing_iterations = passing_iterations,
        .gate_passed = gate_passed,
        .gate_exit = gate_exit,
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
