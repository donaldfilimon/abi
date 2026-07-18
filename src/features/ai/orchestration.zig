//! General-purpose multi-agent orchestration: custom worker specs, scheduler
//! background submission, and browser-task planning (local dry-run; external MCP
//! delegation for real browser automation — no embedded headless browser).
const std = @import("std");
const scheduler_mod = @import("../../core/scheduler.zig");
const types = @import("types.zig");

pub const AgentConfig = types.AgentConfig;
pub const AgentResult = types.AgentResult;
pub const AgentTaskContext = types.AgentTaskContext;
pub const AgentProfile = types.AgentProfile;
pub const AgentToolHint = types.AgentToolHint;

pub const AgentWorkerSpec = struct {
    name: []const u8,
    instructions: []const u8,
    dry_run: bool = true,
    profile_override: ?AgentProfile = null,
    tool_hints: []const AgentToolHint = &.{},
};

pub const NamedAgentResult = struct {
    name: []const u8,
    result: AgentResult,
};

pub const CustomMultiAgentResult = struct {
    results: []NamedAgentResult,
    aggregated: []u8,
    task_ids: []u64,

    pub fn deinit(self: *CustomMultiAgentResult, allocator: std.mem.Allocator) void {
        for (self.results) |entry| {
            entry.result.deinit(allocator);
            allocator.free(entry.name);
        }
        allocator.free(self.results);
        allocator.free(self.aggregated);
        allocator.free(self.task_ids);
    }
};

pub const BackgroundAgentBatch = struct {
    allocator: std.mem.Allocator,
    contexts: []*AgentTaskContext,
    task_ids: []u64,

    pub fn deinit(self: *BackgroundAgentBatch) void {
        for (self.contexts) |ctx| {
            ctx.deinitResult();
            self.allocator.destroy(ctx);
        }
        self.allocator.free(self.contexts);
        self.allocator.free(self.task_ids);
    }
};

pub const BrowserOrchestrationPlan = struct {
    output: []u8,
    requires_review: bool,
    execute_requested: bool,

    pub fn deinit(self: BrowserOrchestrationPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

pub fn workerSpecToConfig(spec: AgentWorkerSpec) AgentConfig {
    return .{
        .name = spec.name,
        .instructions = spec.instructions,
        .dry_run = spec.dry_run,
        .profile_override = spec.profile_override,
        .tool_hints = spec.tool_hints,
    };
}

pub fn formatToolHints(allocator: std.mem.Allocator, hints: []const AgentToolHint) ![]u8 {
    if (hints.len == 0) return try allocator.dupe(u8, "none");
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    for (hints, 0..) |hint, i| {
        if (i > 0) try list.append(allocator, ',');
        try list.appendSlice(allocator, hint.label());
    }
    return try list.toOwnedSlice(allocator);
}

pub const max_worker_count: usize = 32;

pub fn parseToolHintsCsv(allocator: std.mem.Allocator, csv: []const u8) ![]AgentToolHint {
    if (csv.len == 0) return try allocator.alloc(AgentToolHint, 0);
    var list = std.ArrayListUnmanaged(AgentToolHint).empty;
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (trimmed.len == 0) continue;
        const hint = types.AgentToolHint.parse(trimmed) orelse return error.InvalidAgentToolHint;
        try list.append(allocator, hint);
    }
    return try list.toOwnedSlice(allocator);
}

/// Parse `name|instructions|hints` worker definitions separated by `|`
/// between fields and `;` between workers. Hints are comma-separated.
pub fn parseWorkerSpecs(allocator: std.mem.Allocator, spec_text: []const u8) ![]AgentWorkerSpec {
    if (spec_text.len == 0) return error.InvalidWorkerSpec;
    var workers = std.ArrayListUnmanaged(AgentWorkerSpec).empty;
    errdefer {
        for (workers.items) |w| {
            allocator.free(w.name);
            allocator.free(w.instructions);
            allocator.free(w.tool_hints);
        }
        workers.deinit(allocator);
    }
    var worker_it = std.mem.splitScalar(u8, spec_text, ';');
    while (worker_it.next()) |segment| {
        const trimmed = std.mem.trim(u8, segment, " \t");
        if (trimmed.len == 0) continue;
        var fields = std.mem.splitScalar(u8, trimmed, '|');
        const name = std.mem.trim(u8, fields.next() orelse return error.InvalidWorkerSpec, " \t");
        const instructions = std.mem.trim(u8, fields.next() orelse return error.InvalidWorkerSpec, " \t");
        const hints_csv = std.mem.trim(u8, fields.next() orelse "", " \t");
        if (name.len == 0 or instructions.len == 0) return error.InvalidWorkerSpec;
        var owned_hints: ?[]AgentToolHint = try parseToolHintsCsv(allocator, hints_csv);
        errdefer if (owned_hints) |hints| allocator.free(hints);
        var owned_name: ?[]u8 = try allocator.dupe(u8, name);
        errdefer if (owned_name) |value| allocator.free(value);
        var owned_instructions: ?[]u8 = try allocator.dupe(u8, instructions);
        errdefer if (owned_instructions) |value| allocator.free(value);
        try workers.append(allocator, .{
            .name = owned_name.?,
            .instructions = owned_instructions.?,
            .tool_hints = owned_hints.?,
        });
        owned_name = null;
        owned_instructions = null;
        owned_hints = null;
    }
    if (workers.items.len == 0) return error.InvalidWorkerSpec;
    if (workers.items.len > max_worker_count) return error.InvalidWorkerSpec;
    return try workers.toOwnedSlice(allocator);
}

pub fn freeWorkerSpecs(allocator: std.mem.Allocator, specs: []AgentWorkerSpec) void {
    for (specs) |spec| {
        allocator.free(spec.name);
        allocator.free(spec.instructions);
        allocator.free(spec.tool_hints);
    }
    allocator.free(specs);
}

pub fn defaultTrioSpecs() [3]AgentWorkerSpec {
    return .{
        .{ .name = "abbey", .instructions = "Analytical review and structured safety analysis.", .profile_override = .abbey, .tool_hints = &.{.plan} },
        .{ .name = "aviva", .instructions = "Creative exploration and alternative perspectives.", .profile_override = .aviva, .tool_hints = &.{.explore} },
        .{ .name = "abi", .instructions = "Concise action-oriented execution plan.", .profile_override = .abi, .tool_hints = &.{.plan} },
    };
}

pub fn submitAgentsBackground(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    base_name: []const u8,
    specs: []const AgentWorkerSpec,
    input: []const u8,
    submitFn: *const fn (*scheduler_mod.Scheduler, []const u8, *AgentTaskContext) anyerror!u64,
) !BackgroundAgentBatch {
    if (specs.len == 0 or input.len == 0) return error.InvalidAgentConfig;
    if (specs.len > max_worker_count) return error.InvalidAgentConfig;

    var contexts = try allocator.alloc(*AgentTaskContext, specs.len);
    errdefer allocator.free(contexts);
    var task_ids = try allocator.alloc(u64, specs.len);
    errdefer allocator.free(task_ids);

    var submitted: usize = 0;
    errdefer {
        for (task_ids[0..submitted]) |task_id| {
            sched.cancel(task_id) catch |err| {
                std.log.err("failed to cancel partially submitted agent task {d}: {s}", .{ task_id, @errorName(err) });
            };
        }
        for (contexts[0..submitted]) |ctx| {
            ctx.deinitResult();
            allocator.destroy(ctx);
        }
    }

    for (specs, 0..) |spec, i| {
        const ctx = try allocator.create(AgentTaskContext);
        errdefer allocator.destroy(ctx);
        ctx.* = .{
            .allocator = allocator,
            .config = workerSpecToConfig(spec),
            .input = input,
        };
        const task_name = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ base_name, spec.name });
        defer allocator.free(task_name);
        task_ids[i] = try submitFn(sched, task_name, ctx);
        contexts[i] = ctx;
        submitted += 1;
    }

    return .{
        .allocator = allocator,
        .contexts = contexts,
        .task_ids = task_ids,
    };
}

pub fn collectBackgroundBatch(
    allocator: std.mem.Allocator,
    batch: *BackgroundAgentBatch,
    specs: []const AgentWorkerSpec,
) !CustomMultiAgentResult {
    if (specs.len != batch.contexts.len or specs.len != batch.task_ids.len) return error.InvalidAgentConfig;

    var results = try allocator.alloc(NamedAgentResult, specs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |entry| {
            entry.result.deinit(allocator);
            allocator.free(entry.name);
        }
        allocator.free(results);
    }

    var agg = std.ArrayListUnmanaged(u8).empty;
    errdefer agg.deinit(allocator);
    try agg.appendSlice(allocator, "=== CUSTOM MULTI-AGENT RESULTS ===\n");

    for (specs, 0..) |spec, i| {
        const ctx = batch.contexts[i];
        const agent_res = ctx.result orelse return error.MissingAgentResult;
        const owned_name = try allocator.dupe(u8, spec.name);
        results[i] = .{ .name = owned_name, .result = agent_res };
        ctx.result = null;
        filled += 1;

        const chunk = try std.fmt.allocPrint(allocator, "\n[{s}]\n{s}\n", .{ spec.name, agent_res.output });
        defer allocator.free(chunk);
        try agg.appendSlice(allocator, chunk);
    }
    try agg.appendSlice(allocator, "\n=== END ===");

    const aggregated = try agg.toOwnedSlice(allocator);
    errdefer allocator.free(aggregated);
    const task_ids_copy = try allocator.dupe(u64, batch.task_ids);

    return .{
        .results = results,
        .aggregated = aggregated,
        .task_ids = task_ids_copy,
    };
}

pub fn runCustomMultiAgentWithScheduler(
    allocator: std.mem.Allocator,
    sched: *scheduler_mod.Scheduler,
    base_name: []const u8,
    specs: []const AgentWorkerSpec,
    input: []const u8,
    submitFn: *const fn (*scheduler_mod.Scheduler, []const u8, *AgentTaskContext) anyerror!u64,
) !CustomMultiAgentResult {
    var batch = try submitAgentsBackground(allocator, sched, base_name, specs, input, submitFn);
    defer batch.deinit();
    try sched.runAll();
    return try collectBackgroundBatch(allocator, &batch, specs);
}

pub fn planBrowserOrchestration(
    allocator: std.mem.Allocator,
    task: []const u8,
    url: ?[]const u8,
    execute_confirmed: bool,
) !BrowserOrchestrationPlan {
    if (task.len == 0) return error.InvalidAgentConfig;
    const mode: []const u8 = if (execute_confirmed) "execute-requested" else "dry-run";
    const url_line = if (url) |u| try std.fmt.allocPrint(allocator, "target_url={s}\n", .{u}) else try allocator.dupe(u8, "");
    defer allocator.free(url_line);

    const output = try std.fmt.allocPrint(
        allocator,
        "orchestration=browser-local\nmode={s}\nreview_required=true\nembedded_browser=false\ndelegation_hint=external-mcp-playwright\npolicy=loopback-credentials-user-consent\ntool_hints_enforced=false\n{s}task={s}\nsteps:\n  1. record structured browser task spec locally\n  2. record tool_hints in agent output (constitution does not consume hints today)\n  3. recommended next step: delegate to external MCP Playwright peer (not performed by ABI)\n  4. {s}\n",
        .{
            mode,
            url_line,
            task,
            if (execute_confirmed)
                "execute path requires explicit --confirm and an external browser MCP; ABI does not launch a headless browser in-process"
            else
                "dry-run only — no navigation or credential access",
        },
    );
    return .{
        .output = output,
        .requires_review = true,
        .execute_requested = execute_confirmed,
    };
}

test "parse worker specs and tool hints" {
    const specs = try parseWorkerSpecs(std.testing.allocator, "scout|Explore pages|explore,browser;lead|Plan rollout|plan");
    defer freeWorkerSpecs(std.testing.allocator, specs);
    try std.testing.expectEqual(@as(usize, 2), specs.len);
    try std.testing.expectEqualStrings("scout", specs[0].name);
    try std.testing.expectEqual(@as(usize, 2), specs[0].tool_hints.len);
    try std.testing.expect(specs[0].tool_hints[1] == .browser);
}

test "parse worker specs without hints frees safely" {
    const specs = try parseWorkerSpecs(std.testing.allocator, "scout|Explore pages");
    defer freeWorkerSpecs(std.testing.allocator, specs);
    try std.testing.expectEqual(@as(usize, 1), specs.len);
    try std.testing.expectEqual(@as(usize, 0), specs[0].tool_hints.len);
}

test "parse worker specs rejects fan-out above max_worker_count" {
    var segments: std.ArrayListUnmanaged(u8) = .empty;
    defer segments.deinit(std.testing.allocator);
    for (0..max_worker_count + 1) |i| {
        if (i > 0) try segments.append(std.testing.allocator, ';');
        try segments.print(std.testing.allocator, "w{d}|instructions", .{i});
    }
    try std.testing.expectError(error.InvalidWorkerSpec, parseWorkerSpecs(std.testing.allocator, segments.items));
}

test "browser orchestration stays dry-run honest" {
    var plan = try planBrowserOrchestration(std.testing.allocator, "open docs", "https://example.com", false);
    defer plan.deinit(std.testing.allocator);
    try std.testing.expect(plan.requires_review);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "embedded_browser=false") != null);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "dry-run only") != null);
}

test "browser orchestration execute-confirmed stays honest" {
    var plan = try planBrowserOrchestration(std.testing.allocator, "navigate", null, true);
    defer plan.deinit(std.testing.allocator);
    try std.testing.expect(plan.execute_requested);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "execute-requested") != null);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "does not launch a headless browser") != null);
}

fn testAgentTask(_: ?*anyopaque) !void {}

fn submitFirstThenFail(sched: *scheduler_mod.Scheduler, name: []const u8, ctx: *AgentTaskContext) !u64 {
    if (sched.getTaskCount() != 0) return error.InjectedSubmitFailure;
    return sched.submit(name, .normal, testAgentTask, ctx);
}

test "partial background submission cancels queued tasks and frees contexts" {
    var sched = scheduler_mod.Scheduler.init(std.testing.allocator);
    defer sched.deinit();

    const specs = [_]AgentWorkerSpec{
        .{ .name = "first", .instructions = "first worker" },
        .{ .name = "second", .instructions = "second worker" },
    };
    try std.testing.expectError(
        error.InjectedSubmitFailure,
        submitAgentsBackground(std.testing.allocator, &sched, "test", &specs, "input", submitFirstThenFail),
    );
    try std.testing.expectEqual(@as(usize, 0), sched.getPendingCount());
    try std.testing.expectEqual(@as(usize, 1), sched.getCancelledCount());
    try sched.runAll();
}

test {
    std.testing.refAllDecls(@This());
}
