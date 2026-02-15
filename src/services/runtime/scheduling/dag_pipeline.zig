// ============================================================================
// ABI Framework — Pipeline Scheduler: DAG-Based Task Orchestration
// Adapted from abi-system-v2.0/scheduler.zig
// ============================================================================
//
// Models multi-stage inference pipelines as directed acyclic graphs (DAGs).
// Topological sort via Kahn's algorithm, bitmask dependency tracking.
// Self-contained — no external utility dependencies.
//
// Changes from v2.0:
//   - Replaced utils.Time.Stopwatch with std.time.Timer (Zig 0.16)
//   - Removed @import("utils") dependency
//   - All timing uses std.time.Timer.start() / .read()
// ============================================================================

const std = @import("std");
const time = @import("../../shared/time.zig");

// ─── Stage Status ────────────────────────────────────────────────────────────

pub const StageStatus = enum(u8) {
    pending,
    ready,
    running,
    completed,
    failed,
    skipped,
};

// ─── Stage Definition ────────────────────────────────────────────────────────

pub const max_stages = 64;
const max_name_len = 48;

pub const Stage = struct {
    name: [max_name_len]u8 = .{0} ** max_name_len,
    name_len: u8 = 0,
    id: u8 = 0,
    dependencies: u64 = 0,
    execute_fn: ?*const fn (ctx: ?*anyopaque) bool = null,
    context: ?*anyopaque = null,
    status: StageStatus = .pending,
    duration_ns: u64 = 0,
    category: Category = .compute,

    pub fn getName(self: *const Stage) []const u8 {
        return self.name[0..self.name_len];
    }
};

pub const Category = enum(u8) {
    input,
    preprocess,
    compute,
    routing,
    persona,
    postprocess,
    output,
    moderation,
};

// ─── Pipeline ────────────────────────────────────────────────────────────────

pub const Pipeline = struct {
    stages: [max_stages]Stage = undefined,
    count: u8 = 0,
    exec_order: [max_stages]u8 = undefined,
    exec_order_len: u8 = 0,
    completed_mask: u64 = 0,
    error_stage: ?u8 = null,

    fn stageBit(stage_id: u8) u64 {
        return @as(u64, 1) << @intCast(stage_id);
    }

    fn hasDependencyOn(mask: u64, stage_id: u8) bool {
        return (mask & stageBit(stage_id)) != 0;
    }

    fn startTimer() time.Timer {
        return time.Timer.start() catch time.Timer{
            .start_instant = .{ .nanos = 0 },
        };
    }

    pub fn init() Pipeline {
        return .{};
    }

    pub fn addStage(self: *Pipeline, name: []const u8, category: Category) !u8 {
        if (self.count >= max_stages) return error.PipelineFull;

        const id = self.count;
        var stage = Stage{};
        stage.id = id;
        stage.category = category;

        const copy_len = @min(name.len, max_name_len);
        @memcpy(stage.name[0..copy_len], name[0..copy_len]);
        stage.name_len = @intCast(copy_len);

        self.stages[id] = stage;
        self.count += 1;
        return id;
    }

    pub fn addDependency(self: *Pipeline, stage: u8, dependency: u8) !void {
        if (stage >= self.count or dependency >= self.count) return error.InvalidStage;
        if (stage == dependency) return error.SelfDependency;
        self.stages[stage].dependencies |= stageBit(dependency);
    }

    pub fn bindExecutor(self: *Pipeline, stage_id: u8, func: *const fn (ctx: ?*anyopaque) bool, ctx: ?*anyopaque) void {
        std.debug.assert(stage_id < self.count);
        self.stages[stage_id].execute_fn = func;
        self.stages[stage_id].context = ctx;
    }

    // ── Topological Sort (Kahn's Algorithm) ──────────────────

    pub fn sort(self: *Pipeline) !void {
        if (self.count == 0) return;

        var in_degree: [max_stages]u8 = .{0} ** max_stages;
        for (0..self.count) |i| {
            var deps = self.stages[i].dependencies;
            while (deps != 0) {
                in_degree[i] += 1;
                deps &= deps - 1;
            }
        }

        var queue: [max_stages]u8 = undefined;
        var q_head: u8 = 0;
        var q_tail: u8 = 0;

        for (0..self.count) |i| {
            if (in_degree[i] == 0) {
                queue[q_tail] = @intCast(i);
                q_tail += 1;
            }
        }

        self.exec_order_len = 0;

        while (q_head < q_tail) {
            const stage_id = queue[q_head];
            q_head += 1;

            self.exec_order[self.exec_order_len] = stage_id;
            self.exec_order_len += 1;

            for (0..self.count) |j| {
                if (hasDependencyOn(self.stages[j].dependencies, stage_id)) {
                    in_degree[j] -= 1;
                    if (in_degree[j] == 0) {
                        queue[q_tail] = @intCast(j);
                        q_tail += 1;
                    }
                }
            }
        }

        if (self.exec_order_len != self.count) return error.CyclicDependency;
    }

    // ── Execution ────────────────────────────────────────────

    pub fn execute(self: *Pipeline) !PipelineResult {
        if (self.exec_order_len == 0) try self.sort();

        self.completed_mask = 0;
        self.error_stage = null;

        var stages_run: u8 = 0;
        var stages_skipped: u8 = 0;
        var stages_failed: u8 = 0;

        var pipeline_timer = startTimer();

        for (0..self.exec_order_len) |order_idx| {
            const stage_id = self.exec_order[order_idx];
            var stage = &self.stages[stage_id];

            if (stage.dependencies & ~self.completed_mask != 0) {
                stage.status = .skipped;
                stages_skipped += 1;
                continue;
            }

            stage.status = .running;
            var stage_timer = startTimer();

            const success = if (stage.execute_fn) |func|
                func(stage.context)
            else
                true;

            stage.duration_ns = stage_timer.read();

            if (success) {
                stage.status = .completed;
                self.completed_mask |= stageBit(stage_id);
                stages_run += 1;
            } else {
                stage.status = .failed;
                self.error_stage = stage_id;
                stages_failed += 1;
            }
        }

        return PipelineResult{
            .stages_total = self.count,
            .stages_run = stages_run,
            .stages_skipped = stages_skipped,
            .stages_failed = stages_failed,
            .total_ns = pipeline_timer.read(),
            .success = stages_failed == 0,
            .error_stage = self.error_stage,
        };
    }

    pub fn reset(self: *Pipeline) void {
        self.completed_mask = 0;
        self.error_stage = null;
        self.exec_order_len = 0;
        for (0..self.count) |i| {
            self.stages[i].status = .pending;
            self.stages[i].duration_ns = 0;
        }
    }

    // ── Queries ──────────────────────────────────────────────

    pub fn getStage(self: *const Pipeline, id: u8) *const Stage {
        std.debug.assert(id < self.count);
        return &self.stages[id];
    }

    pub fn findStage(self: *const Pipeline, name: []const u8) ?u8 {
        for (0..self.count) |i| {
            if (std.mem.eql(u8, self.stages[i].getName(), name)) return @intCast(i);
        }
        return null;
    }

    pub fn roots(self: *const Pipeline) [max_stages]u8 {
        var result: [max_stages]u8 = undefined;
        var idx: u8 = 0;
        for (0..self.count) |i| {
            if (self.stages[i].dependencies == 0) {
                result[idx] = @intCast(i);
                idx += 1;
            }
        }
        if (idx < max_stages) result[idx] = 0xFF;
        return result;
    }

    pub fn sinks(self: *const Pipeline) [max_stages]u8 {
        var has_dependents: u64 = 0;
        for (0..self.count) |i| {
            has_dependents |= self.stages[i].dependencies;
        }

        var result: [max_stages]u8 = undefined;
        var idx: u8 = 0;
        for (0..self.count) |i| {
            if ((has_dependents & stageBit(@intCast(i))) == 0) {
                result[idx] = @intCast(i);
                idx += 1;
            }
        }
        if (idx < max_stages) result[idx] = 0xFF;
        return result;
    }

    pub fn report(self: *const Pipeline, writer: anytype) !void {
        try writer.writeAll("\n  Pipeline Execution Report\n");
        try writer.writeAll("  ========================================\n");

        for (0..self.exec_order_len) |order_idx| {
            const stage_id = self.exec_order[order_idx];
            const stage = &self.stages[stage_id];

            const status_str: []const u8 = switch (stage.status) {
                .pending => "pending",
                .ready => "ready",
                .running => "running",
                .completed => "done",
                .failed => "FAILED",
                .skipped => "skipped",
            };

            try writer.print("  [{d:>2}] {s:<20} {s:<14} {d}ns\n", .{
                stage_id,
                stage.getName(),
                status_str,
                stage.duration_ns,
            });
        }

        try writer.writeAll("  ========================================\n\n");
    }
};

// ─── Pipeline Result ─────────────────────────────────────────────────────────

pub const PipelineResult = struct {
    stages_total: u8,
    stages_run: u8,
    stages_skipped: u8,
    stages_failed: u8,
    total_ns: u64,
    success: bool,
    error_stage: ?u8,
};

// ─── Pre-Built Pipeline Templates ────────────────────────────────────────────

pub fn createInferencePipeline() !Pipeline {
    var p = Pipeline.init();

    const input = try p.addStage("input", .input);
    const tokenize = try p.addStage("tokenize", .preprocess);
    const embed = try p.addStage("embed", .compute);
    const attend = try p.addStage("attend", .compute);
    const abi_route = try p.addStage("abi_route", .routing);
    const abbey_decode = try p.addStage("abbey_decode", .persona);
    const aviva_decode = try p.addStage("aviva_decode", .persona);
    const merge = try p.addStage("merge", .postprocess);
    const moderate = try p.addStage("moderate", .moderation);
    const output = try p.addStage("output", .output);

    try p.addDependency(tokenize, input);
    try p.addDependency(embed, tokenize);
    try p.addDependency(attend, embed);
    try p.addDependency(abi_route, attend);
    try p.addDependency(abbey_decode, abi_route);
    try p.addDependency(aviva_decode, abi_route);
    try p.addDependency(merge, abbey_decode);
    try p.addDependency(merge, aviva_decode);
    try p.addDependency(moderate, merge);
    try p.addDependency(output, moderate);

    try p.sort();
    return p;
}

test "Pipeline detects cyclic dependencies" {
    var pipeline = Pipeline.init();

    const a = try pipeline.addStage("a", .compute);
    const b = try pipeline.addStage("b", .compute);

    try pipeline.addDependency(a, b);
    try pipeline.addDependency(b, a);

    try std.testing.expectError(error.CyclicDependency, pipeline.sort());
}

test "Pipeline skips dependents when dependency fails" {
    var pipeline = Pipeline.init();

    const fail_stage_id = try pipeline.addStage("fail", .compute);
    const dependent_stage_id = try pipeline.addStage("dependent", .postprocess);
    try pipeline.addDependency(dependent_stage_id, fail_stage_id);

    const fail_exec = struct {
        fn run(_: ?*anyopaque) bool {
            return false;
        }
    }.run;
    const pass_exec = struct {
        fn run(_: ?*anyopaque) bool {
            return true;
        }
    }.run;

    pipeline.bindExecutor(fail_stage_id, fail_exec, null);
    pipeline.bindExecutor(dependent_stage_id, pass_exec, null);

    const result = try pipeline.execute();
    try std.testing.expect(!result.success);
    try std.testing.expectEqual(@as(?u8, fail_stage_id), result.error_stage);
    try std.testing.expectEqual(StageStatus.failed, pipeline.stages[fail_stage_id].status);
    try std.testing.expectEqual(StageStatus.skipped, pipeline.stages[dependent_stage_id].status);
}

test "Pipeline linear chain executes in order" {
    var pipeline = Pipeline.init();

    const a = try pipeline.addStage("input", .input);
    const b = try pipeline.addStage("compute", .compute);
    const c = try pipeline.addStage("output", .output);

    try pipeline.addDependency(b, a);
    try pipeline.addDependency(c, b);

    const pass_exec = struct {
        fn run(_: ?*anyopaque) bool {
            return true;
        }
    }.run;

    pipeline.bindExecutor(a, pass_exec, null);
    pipeline.bindExecutor(b, pass_exec, null);
    pipeline.bindExecutor(c, pass_exec, null);

    const result = try pipeline.execute();
    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u8, 3), result.stages_run);
    try std.testing.expectEqual(@as(u8, 0), result.stages_skipped);
    try std.testing.expectEqual(StageStatus.completed, pipeline.stages[a].status);
    try std.testing.expectEqual(StageStatus.completed, pipeline.stages[b].status);
    try std.testing.expectEqual(StageStatus.completed, pipeline.stages[c].status);
}

test "Pipeline empty pipeline executes successfully" {
    var pipeline = Pipeline.init();
    const result = try pipeline.execute();
    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u8, 0), result.stages_run);
}

test "Pipeline self-dependency rejected" {
    var pipeline = Pipeline.init();
    const a = try pipeline.addStage("self", .compute);
    try std.testing.expectError(error.SelfDependency, pipeline.addDependency(a, a));
}

test "Pipeline stage getName" {
    var pipeline = Pipeline.init();
    const id = try pipeline.addStage("test_stage", .preprocess);
    try std.testing.expectEqualStrings("test_stage", pipeline.stages[id].getName());
}

test "Pipeline addStage respects max_stages" {
    var pipeline = Pipeline.init();
    // Fill to capacity
    for (0..max_stages) |i| {
        _ = try pipeline.addStage("stage", if (i % 2 == 0) .compute else .input);
    }
    // Next add should fail
    try std.testing.expectError(error.PipelineFull, pipeline.addStage("overflow", .compute));
}

test "Pipeline stages without executors pass by default" {
    var pipeline = Pipeline.init();
    _ = try pipeline.addStage("auto_pass", .compute);
    const result = try pipeline.execute();
    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u8, 1), result.stages_run);
}
