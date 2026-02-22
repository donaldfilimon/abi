//! MCP bridge for ZLS (LSP) tooling.

const std = @import("std");
const lsp = @import("../lsp/mod.zig");
const Server = @import("server.zig").Server;

const max_doc_bytes = 4 * 1024 * 1024;

pub fn createZlsServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-zls", version);

    try server.addTool(.{
        .def = .{
            .name = "zls_request",
            .description = "Send an arbitrary LSP request to ZLS",
            .input_schema =
            \\{"type":"object","properties":{"method":{"type":"string","description":"LSP method name"},"params":{"description":"LSP params object/array"},"path":{"type":"string","description":"Optional file path to open before request"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["method"]}
            ,
        },
        .handler = handleRequest,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_hover",
            .description = "Get hover info for a symbol at a position",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"line":{"type":"integer","description":"0-based line"},"character":{"type":"integer","description":"0-based character"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path","line","character"]}
            ,
        },
        .handler = handleHover,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_completion",
            .description = "Get completion items at a position",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"line":{"type":"integer","description":"0-based line"},"character":{"type":"integer","description":"0-based character"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path","line","character"]}
            ,
        },
        .handler = handleCompletion,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_definition",
            .description = "Find definition locations for a symbol at a position",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"line":{"type":"integer","description":"0-based line"},"character":{"type":"integer","description":"0-based character"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path","line","character"]}
            ,
        },
        .handler = handleDefinition,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_references",
            .description = "Find references for a symbol at a position",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"line":{"type":"integer","description":"0-based line"},"character":{"type":"integer","description":"0-based character"},"include_declaration":{"type":"boolean","description":"Include declaration (default: true)"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path","line","character"]}
            ,
        },
        .handler = handleReferences,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_rename",
            .description = "Rename a symbol at a position",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"line":{"type":"integer","description":"0-based line"},"character":{"type":"integer","description":"0-based character"},"new_name":{"type":"string","description":"New symbol name"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path","line","character","new_name"]}
            ,
        },
        .handler = handleRename,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_format",
            .description = "Format a document and return text edits",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"tab_size":{"type":"integer","description":"Tab size (default: 4)"},"insert_spaces":{"type":"boolean","description":"Insert spaces (default: true)"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path"]}
            ,
        },
        .handler = handleFormat,
    });

    try server.addTool(.{
        .def = .{
            .name = "zls_diagnostics",
            .description = "Fetch diagnostics for a document (textDocument/diagnostic)",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path"},"text":{"type":"string","description":"Optional document text (overrides file read)"},"workspace_root":{"type":"string","description":"Workspace root override"},"zls_path":{"type":"string","description":"ZLS binary path"},"zig_exe_path":{"type":"string","description":"Zig compiler path"},"log_level":{"type":"string","description":"ZLS log level"},"enable_snippets":{"type":"boolean","description":"Enable snippets"}},"required":["path"]}
            ,
        },
        .handler = handleDiagnostics,
    });

    return server;
}

const ClientContext = struct {
    env: lsp.EnvConfig,
    io_backend: std.Io.Threaded,
    client: lsp.Client,

    pub fn deinit(self: *ClientContext) void {
        self.client.deinit();
        self.io_backend.deinit();
        self.env.deinit();
    }
};

fn initClientContext(
    allocator: std.mem.Allocator,
    params: std.json.ObjectMap,
) !ClientContext {
    var env = try lsp.loadConfigFromEnv(allocator);
    errdefer env.deinit();

    var cfg = env.config;
    applyConfigOverrides(&cfg, params);

    var io_backend = initIoBackend(allocator);
    errdefer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    errdefer client.deinit();

    return .{ .env = env, .io_backend = io_backend, .client = client };
}

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    const Result = @TypeOf(std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty }));
    if (@typeInfo(Result) == .error_union) {
        return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty }) catch @panic("I/O backend init failed");
    }
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

fn handleRequest(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    const method = getString(p, "method") orelse return error.InvalidParams;

    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    if (getString(p, "path")) |path| {
        _ = try openDocument(&ctx, path, getString(p, "text"));
    } else if (getString(p, "text") != null) {
        return error.InvalidParams;
    }

    var params_json: ?[]u8 = null;
    if (p.get("params")) |value| {
        params_json = try std.json.Stringify.valueAlloc(allocator, value, .{});
    }
    defer if (params_json) |owned| allocator.free(owned);

    const resp = try ctx.client.requestRaw(method, params_json);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleHover(allocator: std.mem.Allocator, params: ?std.json.ObjectMap, out: *std.ArrayListUnmanaged(u8)) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const pos = try parsePosition(p);
    const resp = try ctx.client.hover(uri, pos);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleCompletion(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const pos = try parsePosition(p);
    const resp = try ctx.client.completion(uri, pos);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleDefinition(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const pos = try parsePosition(p);
    const resp = try ctx.client.definition(uri, pos);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleReferences(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const pos = try parsePosition(p);
    const include_decl = getBool(p, "include_declaration") orelse true;
    const resp = try ctx.client.references(uri, pos, include_decl);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleRename(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const pos = try parsePosition(p);
    const new_name = getString(p, "new_name") orelse return error.InvalidParams;
    const resp = try ctx.client.rename(uri, pos, new_name);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleFormat(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const tab_size_raw = getInt(p, "tab_size") orelse 4;
    if (tab_size_raw < 0) return error.InvalidParams;
    const insert_spaces = getBool(p, "insert_spaces") orelse true;
    const options = lsp.types.FormattingOptions{
        .tabSize = @intCast(tab_size_raw),
        .insertSpaces = insert_spaces,
    };
    const resp = try ctx.client.formatting(uri, options);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn handleDiagnostics(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;
    var ctx = try initClientContext(allocator, p);
    defer ctx.deinit();

    const path = getString(p, "path") orelse return error.InvalidParams;
    const uri = try openDocument(&ctx, path, getString(p, "text"));
    defer allocator.free(uri);

    const resp = try ctx.client.diagnostics(uri);
    defer allocator.free(resp.json);
    try appendResponse(allocator, out, resp);
}

fn applyConfigOverrides(config: *lsp.Config, params: std.json.ObjectMap) void {
    if (getString(params, "zls_path")) |path| {
        config.zls_path = path;
    }
    if (getString(params, "zig_exe_path")) |path| {
        config.zig_exe_path = path;
    }
    if (getString(params, "workspace_root")) |path| {
        config.workspace_root = path;
    }
    if (getString(params, "root")) |path| {
        config.workspace_root = path;
    }
    if (getString(params, "log_level")) |level| {
        config.log_level = level;
    }
    if (getBool(params, "enable_snippets")) |flag| {
        config.enable_snippets = flag;
    }
}

fn openDocument(
    ctx: *ClientContext,
    path: []const u8,
    text_override: ?[]const u8,
) ![]u8 {
    const io = ctx.io_backend.io();
    const resolved = try lsp.resolvePath(ctx.env.allocator, io, ctx.client.workspaceRoot(), path);
    defer ctx.env.allocator.free(resolved);

    const uri = try lsp.pathToUri(ctx.env.allocator, resolved);

    var owned_text: ?[]u8 = null;
    const text = text_override orelse blk: {
        const contents = try std.Io.Dir.cwd().readFileAlloc(io, resolved, ctx.env.allocator, .limited(max_doc_bytes));
        owned_text = contents;
        break :blk contents;
    };
    defer if (owned_text) |owned| ctx.env.allocator.free(owned);

    try ctx.client.didOpen(.{
        .uri = uri,
        .text = text,
    });

    return uri;
}

fn parsePosition(params: std.json.ObjectMap) !lsp.types.Position {
    const line = getInt(params, "line") orelse return error.InvalidParams;
    const character = getInt(params, "character") orelse return error.InvalidParams;
    if (line < 0 or character < 0) return error.InvalidParams;
    return .{ .line = @intCast(line), .character = @intCast(character) };
}

fn appendResponse(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), resp: lsp.Response) !void {
    if (resp.is_error) {
        try out.appendSlice(allocator, "LSP error: ");
    }
    try out.appendSlice(allocator, resp.json);
}

fn getString(params: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = params.get(key) orelse return null;
    return if (value == .string) value.string else null;
}

fn getBool(params: std.json.ObjectMap, key: []const u8) ?bool {
    const value = params.get(key) orelse return null;
    return switch (value) {
        .bool => |v| v,
        else => null,
    };
}

fn getInt(params: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = params.get(key) orelse return null;
    return switch (value) {
        .integer => |v| v,
        else => null,
    };
}

test {
    std.testing.refAllDecls(@This());
}
