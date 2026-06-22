const std = @import("std");
const features = @import("abi").features;
const json_helpers = @import("json_helpers.zig");
const ai_tools = @import("ai_tools.zig");
const connector_tools = @import("connector_tools.zig");
const plugin_tools = @import("plugin_tools.zig");
const middleware = @import("middleware.zig");
const abi = @import("abi");

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
    /// Declarative validation rules enforced by `middleware.validateArguments`
    /// before dispatch. Empty for no-argument tools.
    fields: []const middleware.FieldSpec = &.{},
};

const tools: []const ToolDescriptor = &.{
    .{
        .name = "ai_run",
        .description = "Run AI inference with internal profile routing",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"}},\"required\":[\"input\"]}",
        .fields = &.{
            .{ .name = "input", .required = true, .missing_error = error.MissingInput },
        },
    },
    .{
        .name = "ai_complete",
        .description = "Run local model completion and record metadata in the MCP WDBX store when available",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"},\"model\":{\"type\":\"string\"}},\"required\":[\"input\"]}",
        .fields = &.{
            .{ .name = "input", .required = true, .missing_error = error.MissingInput },
            .{ .name = "model", .required = false, .missing_error = error.MissingModel },
        },
    },
    .{
        .name = "ai_train",
        .description = "Train an agent profile",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"profile\":{\"type\":\"string\"},\"dataset\":{\"type\":\"string\"},\"format\":{\"type\":\"string\",\"enum\":[\"jsonl\",\"csv\",\"text\"]},\"artifact_dir\":{\"type\":\"string\"}},\"required\":[\"profile\",\"dataset\"]}",
        .fields = &.{
            .{ .name = "profile", .required = true, .kind = .identifier, .missing_error = error.MissingProfile },
            .{ .name = "dataset", .required = true, .kind = .file_path, .missing_error = error.MissingDataset },
            .{ .name = "format", .required = false, .kind = .enum_choice, .missing_error = error.InvalidDatasetFormat, .invalid_error = error.InvalidDatasetFormat, .choices = &.{ "jsonl", "csv", "text" } },
            .{ .name = "artifact_dir", .required = false, .kind = .file_path, .missing_error = error.MissingArtifactDir },
        },
    },
    .{
        .name = "wdbx_query",
        .description = "Query the vector store",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}",
        .fields = &.{
            .{ .name = "query", .required = true, .missing_error = error.MissingQuery },
        },
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
        .fields = &.{
            .{ .name = "service", .required = true, .kind = .enum_choice, .missing_error = error.MissingConnectorService, .invalid_error = error.UnknownConnector, .choices = &.{ "openai", "anthropic", "discord", "twilio", "grok" } },
            .{ .name = "input", .required = true, .missing_error = error.MissingInput },
        },
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
        .description = "Report statistics for the long-lived MCP WDBX store (kv, vectors, blocks, spatial)",
        .input_schema = "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
    },
    .{
        .name = "plugin_run",
        .description = "Execute a registered plugin",
        .input_schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"input\":{\"type\":\"string\"}},\"required\":[\"name\"]}",
        .fields = &.{
            .{ .name = "name", .required = true, .kind = .identifier, .missing_error = error.MissingPluginName },
            .{ .name = "input", .required = false, .missing_error = error.MissingInput },
        },
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

    // Enforce each tool's declarative argument policy before dispatch. The
    // descriptor's per-field `missing_error`/`invalid_error` mirror the handlers'
    // historical errors, so the frozen MCP error contract is preserved while NUL,
    // length, path-traversal, and enum checks now run uniformly. Unknown names
    // match no descriptor and fall through to the UnknownTool arm below.
    for (tools) |descriptor| {
        if (std.mem.eql(u8, tool_name, descriptor.name)) {
            try middleware.validateArguments(descriptor.fields, params_obj);
            break;
        }
    }

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
        const requested = objectString(args_obj, "model", error.MissingModel) catch features.ai.models.default_model;
        // Canonicalize catalog aliases (e.g. "fable-5" -> "claude-fable-5") so the
        // recorded model label is canonical; unknown ids pass through unchanged.
        const model = features.ai.models.canonical(requested);
        const text = try ai_tools.runLocalCompletion(allocator, input, model);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "ai_train")) {
        const args_obj = try toolArguments(params_obj);
        const profile = try objectString(args_obj, "profile", error.MissingProfile);
        const dataset = try objectString(args_obj, "dataset", error.MissingDataset);
        const artifact_dir = objectString(args_obj, "artifact_dir", error.MissingArtifactDir) catch "zig-cache/agents";

        const config = ai_mod.TrainingConfig{
            .profile = profile,
            .dataset = .{ .path = dataset, .format = parseDatasetFormat(args_obj) },
            .artifact_dir = artifact_dir,
        };

        const text = try ai_tools.runTraining(allocator, config);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "wdbx_query")) {
        const args_obj = try toolArguments(params_obj);
        const query = try objectString(args_obj, "query", error.MissingQuery);
        const text = try ai_tools.runLocalWdbxQuery(allocator, query);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "scheduler_stats") or std.mem.eql(u8, tool_name, "scheduler_info")) {
        const text = try ai_tools.schedulerStatsText(allocator);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "connector_test")) {
        const args_obj = try toolArguments(params_obj);
        const service = try objectString(args_obj, "service", error.MissingConnectorService);
        const input = try objectString(args_obj, "input", error.MissingInput);
        const text = try connector_tools.runConnectorTest(allocator, service, input);
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
        const text = try ai_tools.wdbxStatsText(allocator);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "plugin_list")) {
        const text = try plugin_tools.runPluginList(allocator);
        defer allocator.free(text);
        return try toolTextResult(allocator, text);
    }

    if (std.mem.eql(u8, tool_name, "plugin_run")) {
        const args_obj = try toolArguments(params_obj);
        const name = try objectString(args_obj, "name", error.MissingPluginName);
        const input = objectString(args_obj, "input", error.MissingInput) catch "";

        var pm = abi.plugins.PluginManager.init(allocator);
        defer pm.deinit();

        try plugin_tools.loadBundledPlugins(&pm);

        const output = try pm.run(allocator, name, input);
        defer allocator.free(output);

        return try toolTextResult(allocator, output);
    }

    return error.UnknownTool;
}

/// Maps any error that can surface from request dispatch to a stable,
/// non-leaking client message. Both the stdio and HTTP transports route through
/// this so neither exposes raw `@errorName` identifiers (the HTTP path formerly
/// interpolated `@errorName(err)` straight into the JSON-RPC error body).
pub fn errorMessage(err: anyerror) []const u8 {
    return switch (err) {
        error.ParseError, error.InvalidJsonFormat, error.RequestTooLarge => "Parse error",
        error.InvalidRequest => "Invalid Request",
        error.MethodNotFound, error.UnknownTool => "Method not found",
        error.MissingParams => "Missing params",
        error.MissingToolName => "Missing tool name",
        error.MissingArguments => "Missing arguments",
        error.MissingInput => "Missing input",
        error.MissingModel => "Missing model",
        error.MissingProfile => "Missing profile",
        error.MissingDataset => "Missing dataset",
        error.MissingQuery => "Missing query",
        error.MissingConnectorService => "Missing connector service",
        error.MissingPluginName => "Missing plugin name",
        error.MissingArtifactDir => "Missing artifact dir",
        error.UnknownConnector => "Invalid connector service",
        error.InvalidDatasetFormat => "Invalid dataset format",
        error.InvalidFieldValue => "Invalid field value",
        error.InvalidFieldEncoding => "Invalid field encoding",
        error.FieldTooLong => "Field too long",
        error.PathTraversal => "Invalid path",
        error.InvalidCompletionInput => "Invalid input",
        else => "Internal error",
    };
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

// --- Tests ---

// Pins the middleware↔catalog invariant the validation layer relies on: every
// field a tool validates (its `fields` policy) must also be advertised in that
// tool's `input_schema`. Without this, a validated-but-unadvertised field (or a
// renamed schema property) would drift silently — the middleware would reject an
// argument the published schema never mentioned.
test "every validated field is advertised in the tool input schema" {
    for (tools) |descriptor| {
        for (descriptor.fields) |field| {
            std.testing.expect(std.mem.indexOf(u8, descriptor.input_schema, field.name) != null) catch |err| {
                std.debug.print(
                    "tool '{s}' validates field '{s}' absent from its input_schema\n",
                    .{ descriptor.name, field.name },
                );
                return err;
            };
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
