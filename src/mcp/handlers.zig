const std = @import("std");
const features = @import("abi").features;
const json_helpers = @import("json_helpers.zig");
const abi = @import("abi");

// Long-lived scheduler and WDBX store for the MCP server process.
// This lets MCP completion/training/query/stats tools observe the same in-process state.
var g_mcp_scheduler: ?abi.scheduler.Scheduler = null;
var g_scheduler_initialized = std.atomic.Value(bool).init(false);
var g_mcp_wdbx_store: ?abi.features.wdbx.Store = null;
var g_wdbx_initialized = std.atomic.Value(bool).init(false);
var g_wdbx_lock: abi.foundation.sync.SpinLock = .{};

fn ensureMcpScheduler() void {
    if (g_scheduler_initialized.load(.acquire)) return;
    if (g_scheduler_initialized.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) {
        g_mcp_scheduler = abi.scheduler.Scheduler.init(std.heap.page_allocator);
    }
}

pub fn getMcpScheduler() *abi.scheduler.Scheduler {
    ensureMcpScheduler();
    return &g_mcp_scheduler.?;
}

pub fn deinitMcpScheduler() void {
    if (g_mcp_scheduler) |*s| {
        s.deinit();
        g_mcp_scheduler = null;
    }
    g_scheduler_initialized.store(false, .release);
}

fn ensureMcpWdbxStore() void {
    if (g_wdbx_initialized.load(.acquire)) return;
    if (g_wdbx_initialized.cmpxchgStrong(false, true, .acq_rel, .acquire) == null) {
        g_mcp_wdbx_store = abi.features.wdbx.Store.init(std.heap.page_allocator);
    }
}

pub fn getMcpWdbxStore() *abi.features.wdbx.Store {
    ensureMcpWdbxStore();
    return &g_mcp_wdbx_store.?;
}

pub fn deinitMcpWdbxStore() void {
    g_wdbx_lock.lock();
    defer g_wdbx_lock.unlock();
    if (g_mcp_wdbx_store) |*store| {
        store.deinit();
        g_mcp_wdbx_store = null;
    }
    g_wdbx_initialized.store(false, .release);
}

pub fn handleInitializeJson(allocator: std.mem.Allocator, params: ?std.json.Value) ![]u8 {
    _ = params;
    return try allocator.dupe(u8,
        \\{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"abi-mcp","version":"0.2.0"}}
    );
}

const ToolDescriptor = struct {
    name: []const u8,
    description: []const u8,
    input_schema: []const u8, // pre-serialized JSON object (without outer {})
};

const tools: []const ToolDescriptor = &.{
    .{
        .name = "ai_run",
        .description = "Run AI inference with internal profile routing",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"}},\"required\":[\"input\"]}",
    },
    .{
        .name = "ai_complete",
        .description = "Run local model completion and record metadata in a transient WDBX store when available",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"},\"model\":{\"type\":\"string\"}},\"required\":[\"input\"]}",
    },
    .{
        .name = "ai_train",
        .description = "Train an agent profile",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"profile\":{\"type\":\"string\"},\"dataset\":{\"type\":\"string\"},\"format\":{\"type\":\"string\",\"enum\":[\"jsonl\",\"csv\",\"text\"]},\"artifact_dir\":{\"type\":\"string\"}},\"required\":[\"profile\",\"dataset\"]}",
    },
    .{
        .name = "wdbx_query",
        .description = "Query the vector store",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}",
    },
    .{
        .name = "scheduler_stats",
        .description = "Report live scheduler task counts and status for the MCP server",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "scheduler_info",
        .description = "Compatibility alias for scheduler_stats",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "connector_test",
        .description = "Exercise a connector through its deterministic local path",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"service\":{\"type\":\"string\",\"enum\":[\"openai\",\"anthropic\",\"discord\",\"twilio\",\"grok\"]},\"input\":{\"type\":\"string\"}},\"required\":[\"service\",\"input\"]}",
    },
    .{
        .name = "gpu_status",
        .description = "Report current GPU backend, acceleration mode, and capabilities",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "plugin_list",
        .description = "List bundled plugin metadata loaded through the plugin manager",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "wdbx_stats",
        .description = "Report statistics for a fresh transient WDBX store (kv, vectors, blocks, spatial)",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "plugin_run",
        .description = "Execute a registered plugin",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"input\":{\"type\":\"string\"}},\"required\":[\"name\"]}",
    },
};

pub fn handleToolsListJson(allocator: std.mem.Allocator) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"tools\":[");

    for (tools, 0..) |tool, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"name\":");
        try json_helpers.appendJsonString(&out, allocator, tool.name);
        try out.appendSlice(allocator, ",\"description\":");
        try json_helpers.appendJsonString(&out, allocator, tool.description);
        try out.appendSlice(allocator, ",\"inputSchema\":");
        try out.appendSlice(allocator, tool.input_schema);
        try out.appendSlice(allocator, "}");
    }

    try out.appendSlice(allocator, "]}");

    return try out.toOwnedSlice(allocator);
}

pub fn handleToolsCallJson(allocator: std.mem.Allocator, params: ?std.json.Value) ![]u8 {
    const p = params orelse return error.MissingParams;
    const params_obj = switch (p) {
        .object => |obj| obj,
        else => return error.MissingParams,
    };

    const name_val = params_obj.get("name") orelse return error.MissingToolName;
    const tool_name = switch (name_val) {
        .string => |s| s,
        else => return error.MissingToolName,
    };

    const ai_mod = features.ai;
    if (std.mem.eql(u8, tool_name, "ai_run")) {
        const args_obj = try toolArguments(params_obj);
        const input = try objectString(args_obj, "input", error.MissingInput);
        const response = try ai_mod.run(allocator, input);
        defer allocator.free(response);
        return try toolTextResult(allocator, response);
    }

    if (std.mem.eql(u8, tool_name, "ai_complete")) {
        const args_obj = try toolArguments(params_obj);
        const input = try objectString(args_obj, "input", error.MissingInput);
        const model = objectString(args_obj, "model", error.MissingModel) catch "abi-local";
        const text = try runLocalCompletion(allocator, input, model);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "ai_train")) {
        const args_obj = try toolArguments(params_obj);
        const profile = try objectString(args_obj, "profile", error.MissingProfile);
        const dataset = try objectString(args_obj, "dataset", error.MissingDataset);
        const artifact_dir = objectString(args_obj, "artifact_dir", error.MissingArtifactDir) catch "zig-cache/agents";

        var result = try ai_mod.train(allocator, .{
            .profile = profile,
            .dataset = .{ .path = dataset, .format = parseDatasetFormat(args_obj) },
            .artifact_dir = artifact_dir,
        });
        defer result.deinit(allocator);

        const status: []const u8 = if (result.accepted) "training accepted" else "training disabled";
        const text = try std.fmt.allocPrint(
            allocator,
            "{s} profile={s} dataset={s} records={d} backend={s}: {s}",
            .{ status, result.profile, result.dataset_path, result.records_stored, result.acceleration_backend, result.message },
        );
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "wdbx_query")) {
        const args_obj = try toolArguments(params_obj);
        const query = try objectString(args_obj, "query", error.MissingQuery);
        const text = try runLocalWdbxQuery(allocator, query);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "scheduler_stats") or std.mem.eql(u8, tool_name, "scheduler_info")) {
        const text = try schedulerStatsText(allocator);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "connector_test")) {
        const args_obj = try toolArguments(params_obj);
        const service = try objectString(args_obj, "service", error.MissingConnectorService);
        const input = try objectString(args_obj, "input", error.MissingInput);
        const text = try runConnectorTest(allocator, service, input);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "gpu_status")) {
        const status = features.gpu.detectBackend();
        const caps = features.gpu.backendCapabilitiesList();
        const text = try std.fmt.allocPrint(
            allocator,
            "backend={s} available={s} accelerated={s} capabilities={d} message={s}",
            .{ features.gpu.backendName(status.backend), if (status.available) "true" else "false", if (status.accelerated) "true" else "false", caps.len, status.message },
        );
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "wdbx_stats")) {
        var store = features.wdbx.Store.init(allocator);
        defer store.deinit();
        const s = store.stats();
        const dims_str = if (s.vector_dimensions) |d|
            std.fmt.allocPrint(allocator, "{d}", .{d}) catch "null"
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(dims_str);

        const text = try std.fmt.allocPrint(
            allocator,
            "kv={d} vectors={d} blocks={d} spatial={d} dims={s} backend={s}",
            .{ s.kv_entries, s.vectors, s.blocks, s.spatial_records, dims_str, features.gpu.backendName(s.acceleration.backend) },
        );
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "plugin_list")) {
        const text = try runPluginList(allocator);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "plugin_run")) {
        const args_obj = try toolArguments(params_obj);
        const name = try objectString(args_obj, "name", error.MissingPluginName);
        const input = objectString(args_obj, "input", error.MissingInput) catch "";

        var pm = abi.plugins.PluginManager.init(allocator);
        defer pm.deinit();

        try loadBundledPlugins(&pm);

        const output = try pm.run(allocator, name, input);
        defer allocator.free(output);

        return try toolTextResult(allocator, output);
    }

    return error.UnknownTool;
}

fn schedulerStatsText(allocator: std.mem.Allocator) ![]u8 {
    const sched = getMcpScheduler();
    const s = sched.stats();
    return try std.fmt.allocPrint(
        allocator,
        "scheduler running={d} pending={d} completed={d} failed={d} source=mcp-server",
        .{ s.running, s.pending, s.completed, s.failed },
    );
}

fn runConnectorTest(allocator: std.mem.Allocator, service: []const u8, input: []const u8) ![]u8 {
    const connectors = abi.connectors;

    if (std.mem.eql(u8, service, "openai")) {
        var client = connectors.openai.Client.init(allocator, .{ .api_key = "mcp-local-key", .base_url = "https://api.openai.com" });
        defer client.deinit();
        const messages = try buildUserMessages(allocator, input);
        defer allocator.free(messages);
        var response = try client.chatCompletion(allocator, "gpt-local", messages);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=openai status={d} body={s}", .{ response.status, response.body });
    }

    if (std.mem.eql(u8, service, "anthropic")) {
        var client = connectors.anthropic.Client.init(allocator, .{ .api_key = "mcp-local-key", .base_url = "https://api.anthropic.com" });
        defer client.deinit();
        var response = try client.message(allocator, "claude-local", input, 256);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=anthropic status={d} body={s}", .{ response.status, response.body });
    }

    if (std.mem.eql(u8, service, "discord")) {
        var bot = connectors.discord.Bot.init(allocator, .{ .token = "mcp-local-token", .client_id = "123456789012345678" });
        defer bot.deinit();
        try bot.connect();
        const body = try bot.sendMessage(allocator, "234567890123456789", input);
        defer allocator.free(body);
        return try std.fmt.allocPrint(allocator, "connector=discord status=200 body={s}", .{body});
    }

    if (std.mem.eql(u8, service, "twilio")) {
        var client = connectors.twilio.Client.init(allocator, connectors.twilio.TwilioConfig.local());
        defer client.deinit();
        var response = try client.handleConversationRelayEvent(allocator, .{
            .kind = .user_transcript,
            .conversation_id = "CA" ++ "0123456789abcdef0123456789abcdef",
            .customer_id = "+15551234567",
            .transcript = input,
        }, "ABI local relay acknowledged.");
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=twilio status=200 body={s}", .{response.text});
    }

    if (std.mem.eql(u8, service, "grok")) {
        var client = connectors.grok.Client.init(allocator, connectors.grok.grokConfig());
        defer client.deinit();
        const messages = try buildUserMessages(allocator, input);
        defer allocator.free(messages);
        var response = try client.chatCompletion(allocator, "grok-local", messages);
        defer response.deinit(allocator);
        return try std.fmt.allocPrint(allocator, "connector=grok status={d} body={s}", .{ response.status, response.body });
    }

    return error.UnknownConnector;
}

fn buildUserMessages(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "[{\"role\":\"user\",\"content\":");
    try abi.connectors.json.appendJsonString(&out, allocator, input);
    try out.appendSlice(allocator, "}]");
    return try out.toOwnedSlice(allocator);
}

fn loadBundledPlugins(pm: *abi.plugins.PluginManager) !void {
    _ = pm.loadPlugin("src/plugins/example-plugin") catch |err| switch (err) {
        error.AlreadyLoaded => {},
        else => return err,
    };
    _ = pm.loadPlugin("src/plugins/example-wdbx-plugin") catch |err| switch (err) {
        error.AlreadyLoaded => {},
        else => return err,
    };
}

fn runPluginList(allocator: std.mem.Allocator) ![]u8 {
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    try loadBundledPlugins(&pm);

    const list = try pm.listPlugins();
    defer allocator.free(list);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "plugins count={d}", .{list.len});
    for (list) |plugin| {
        try out.print(
            allocator,
            " name={s} version={s} target={s} entry={s} description={s};",
            .{ plugin.name, plugin.version, plugin.target_feature, plugin.entry_point, plugin.description },
        );
    }
    return try out.toOwnedSlice(allocator);
}

fn runLocalCompletion(allocator: std.mem.Allocator, input: []const u8, model: []const u8) ![]u8 {
    const ai_mod = features.ai;
    const wdbx = features.wdbx;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const sched = getMcpScheduler();
    var result = try ai_mod.completeWithScheduler(
        allocator,
        &store,
        sched,
        "complete:mcp",
        .{ .input = input, .model = model, .store_result = true },
    );
    defer result.deinit(allocator);
    const stats = store.stats();
    const persisted = result.query_vector_id != null and result.response_vector_id != null and result.block_id != null;

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(
        allocator,
        "model={s} profile={s} audit_passed={s} persisted={s} kv_entries={d} vectors={d} blocks={d}",
        .{ result.model, result.selected_profile.label(), if (result.audit.passed) "true" else "false", if (persisted) "true" else "false", stats.kv_entries, stats.vectors, stats.blocks },
    );
    if (result.query_vector_id) |qid| {
        try out.print(allocator, " query_vector_id={d} metadata_key=completion:{d}", .{ qid, qid });
    }
    if (result.response_vector_id) |rid| try out.print(allocator, " response_vector_id={d}", .{rid});
    if (result.block_id) |block_id| {
        const block_hex = std.fmt.bytesToHex(block_id, .lower);
        try out.print(allocator, " block_id={s}", .{&block_hex});
    }
    if (!persisted) try out.print(allocator, " wdbx_status={s}", .{stats.acceleration.message});
    try out.print(allocator, ": {s}", .{result.output});
    return try out.toOwnedSlice(allocator);
}

fn toolArguments(params_obj: std.json.ObjectMap) !std.json.ObjectMap {
    const args_val = params_obj.get("arguments") orelse return error.MissingArguments;
    return switch (args_val) {
        .object => |obj| obj,
        else => return error.MissingArguments,
    };
}

fn objectString(obj: std.json.ObjectMap, key: []const u8, missing_error: anyerror) ![]const u8 {
    const value = obj.get(key) orelse return missing_error;
    return switch (value) {
        .string => |s| s,
        else => missing_error,
    };
}

fn parseDatasetFormat(obj: std.json.ObjectMap) features.ai.DatasetFormat {
    const value = obj.get("format") orelse return .jsonl;
    const s = switch (value) {
        .string => |format| format,
        else => return .jsonl,
    };
    if (std.mem.eql(u8, s, "csv")) return .csv;
    if (std.mem.eql(u8, s, "text")) return .text;
    return .jsonl;
}

fn toolTextResult(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
    try json_helpers.appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}]}");
    return out.toOwnedSlice(allocator);
}

fn runLocalWdbxQuery(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    const wdbx = features.wdbx;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    const ai_mod = features.ai;
    const abbey_vec = [_]f32{ 0.92, 0.48, 0.25, 0.76 };
    const aviva_vec = [_]f32{ 0.34, 0.94, 0.82, 0.41 };
    const abi_vec = [_]f32{ 0.71, 0.69, 0.88, 0.97 };
    _ = store.putVector(&abbey_vec) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };
    _ = store.putVector(&aviva_vec) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };
    _ = store.putVector(&abi_vec) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };

    const query_vec = ai_mod.textEmbedding(query);
    const results = store.search(&query_vec, 1) catch |err| {
        if (ai_mod.isFeatureDisabled(err)) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };
    defer allocator.free(results);

    if (results.len == 0) return try allocator.dupe(u8, "wdbx query returned no local matches");
    return try std.fmt.allocPrint(
        allocator,
        "wdbx local match profile={s} vector_id={d} score={d:.3}",
        .{ localProfileForVector(results[0].id), results[0].id, results[0].score },
    );
}

fn localProfileForVector(id: u32) []const u8 {
    return switch (id) {
        1 => "abbey",
        2 => "aviva",
        3 => "abi",
        else => "unknown",
    };
}
