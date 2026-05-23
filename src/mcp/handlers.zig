const std = @import("std");
const features = @import("abi").features;
const json_helpers = @import("json_helpers.zig");

pub fn handleInitializeJson(allocator: std.mem.Allocator, params: ?std.json.Value) ![]u8 {
    _ = params;
    return try allocator.dupe(u8,
        \\{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"abi-mcp","version":"0.2.0"}}
    );
}

pub fn handleToolsListJson(allocator: std.mem.Allocator) ![]u8 {
    return try allocator.dupe(u8,
        \\{"tools":[{"name":"ai_run","description":"Run AI inference with profile routing","inputSchema":{"type":"object","properties":{"input":{"type":"string"},"profile":{"type":"string"}},"required":["input"]}},{"name":"ai_complete","description":"Run local model completion and persist WDBX metadata","inputSchema":{"type":"object","properties":{"input":{"type":"string"},"model":{"type":"string"}},"required":["input"]}},{"name":"ai_train","description":"Train an agent profile","inputSchema":{"type":"object","properties":{"profile":{"type":"string"},"dataset":{"type":"string"}},"required":["profile","dataset"]}},{"name":"wdbx_query","description":"Query the vector store","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}
    );
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

        const text = try std.fmt.allocPrint(
            allocator,
            "training accepted profile={s} dataset={s} records={d} backend={s}: {s}",
            .{ result.profile, result.dataset_path, result.records_stored, result.acceleration_backend, result.message },
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

    return error.UnknownTool;
}

fn runLocalCompletion(allocator: std.mem.Allocator, input: []const u8, model: []const u8) ![]u8 {
    const ai_mod = features.ai;
    const wdbx = features.wdbx;
    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var result = try ai_mod.completeWithStore(allocator, &store, .{ .input = input, .model = model, .store_result = true });
    defer result.deinit(allocator);

    return try std.fmt.allocPrint(
        allocator,
        "model={s} profile={s} audit_passed={s} vectors={d} blocks={d}: {s}",
        .{ result.model, result.selected_profile.label(), if (result.audit.passed) "true" else "false", store.vectorCount(), store.blockCount(), result.output },
    );
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
    var out: std.ArrayList(u8) = .empty;
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

    const abbey_vec = [_]f32{ 0.92, 0.48, 0.25, 0.76 };
    const aviva_vec = [_]f32{ 0.34, 0.94, 0.82, 0.41 };
    const abi_vec = [_]f32{ 0.71, 0.69, 0.88, 0.97 };
    _ = store.putVector(&abbey_vec) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "FeatureDisabled")) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };
    _ = store.putVector(&aviva_vec) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "FeatureDisabled")) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };
    _ = store.putVector(&abi_vec) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "FeatureDisabled")) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
        return err;
    };

    const query_vec = features.ai.textEmbedding(query);
    const results = store.search(&query_vec, 1) catch |err| {
        if (std.mem.eql(u8, @errorName(err), "FeatureDisabled")) return try allocator.dupe(u8, "wdbx local match unavailable: wdbx feature is disabled for this build");
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
