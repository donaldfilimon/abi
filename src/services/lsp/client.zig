//! ZLS LSP client (JSON-RPC over stdio).

const std = @import("std");
const config_mod = @import("../../core/config/mod.zig");
const jsonrpc = @import("jsonrpc.zig");
const types = @import("types.zig");
const zig_toolchain = @import("../shared/utils/zig_toolchain.zig");

pub const Config = config_mod.LspConfig;

pub const Response = struct {
    json: []u8,
    is_error: bool,
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
    child: std.process.Child,
    stdin_writer: std.Io.File.Writer,
    stdout_reader: std.Io.File.Reader,
    read_buf: [65536]u8,
    write_buf: [65536]u8,
    next_id: i64,
    max_payload_bytes: usize,
    root_path: []u8,
    root_uri: []u8,
    owned_zig_path: ?[]u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, io: std.Io, config: Config) !Self {
        const root_path = try resolveWorkspaceRoot(allocator, io, config.workspace_root);
        errdefer allocator.free(root_path);

        const root_uri = try pathToUri(allocator, root_path);
        errdefer allocator.free(root_uri);

        const argv = [_][]const u8{config.zls_path};
        var child = try std.process.spawn(io, .{
            .argv = &argv,
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
            .cwd = .{ .path = root_path },
        });
        errdefer child.kill(io);

        if (child.stdin == null or child.stdout == null) {
            child.kill(io);
            return error.MissingPipe;
        }

        var client = Self{
            .allocator = allocator,
            .io = io,
            .config = config,
            .child = child,
            .stdin_writer = undefined,
            .stdout_reader = undefined,
            .read_buf = undefined,
            .write_buf = undefined,
            .next_id = 1,
            .max_payload_bytes = 8 * 1024 * 1024,
            .root_path = root_path,
            .root_uri = root_uri,
            .owned_zig_path = null,
        };

        client.stdout_reader = child.stdout.?.reader(io, &client.read_buf);
        client.stdin_writer = child.stdin.?.writer(io, &client.write_buf);

        try client.initialize();
        return client;
    }

    pub fn deinit(self: *Self) void {
        self.shutdown() catch {};
        if (self.child.id != null) {
            self.child.kill(self.io);
        }
        self.allocator.free(self.root_path);
        self.allocator.free(self.root_uri);
        if (self.owned_zig_path) |owned| {
            self.allocator.free(owned);
        }
    }

    pub fn workspaceRoot(self: *const Self) []const u8 {
        return self.root_path;
    }

    pub fn didOpen(self: *Self, doc: types.TextDocumentItem) !void {
        const params = types.DidOpenTextDocumentParams{ .textDocument = doc };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        try self.notifyRaw("textDocument/didOpen", params_json);
    }

    pub fn hover(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return self.positionRequest("textDocument/hover", uri, pos);
    }

    pub fn completion(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return self.positionRequest("textDocument/completion", uri, pos);
    }

    pub fn definition(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return self.positionRequest("textDocument/definition", uri, pos);
    }

    pub fn references(self: *Self, uri: []const u8, pos: types.Position, include_decl: bool) !Response {
        const params = types.ReferencesParams{
            .textDocument = .{ .uri = uri },
            .position = pos,
            .context = .{ .includeDeclaration = include_decl },
        };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        return self.requestRaw("textDocument/references", params_json);
    }

    pub fn rename(
        self: *Self,
        uri: []const u8,
        pos: types.Position,
        new_name: []const u8,
    ) !Response {
        const params = types.RenameParams{
            .textDocument = .{ .uri = uri },
            .position = pos,
            .newName = new_name,
        };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        return self.requestRaw("textDocument/rename", params_json);
    }

    pub fn formatting(
        self: *Self,
        uri: []const u8,
        options: types.FormattingOptions,
    ) !Response {
        const params = types.DocumentFormattingParams{
            .textDocument = .{ .uri = uri },
            .options = options,
        };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        return self.requestRaw("textDocument/formatting", params_json);
    }

    pub fn diagnostics(self: *Self, uri: []const u8) !Response {
        const params = struct {
            textDocument: types.TextDocumentIdentifier,
        }{
            .textDocument = .{ .uri = uri },
        };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        return self.requestRaw("textDocument/diagnostic", params_json);
    }

    pub fn requestRaw(
        self: *Self,
        method: []const u8,
        params_json: ?[]const u8,
    ) !Response {
        const id = self.next_id;
        self.next_id += 1;

        const payload = try buildMessageJson(self.allocator, id, method, params_json);
        defer self.allocator.free(payload);

        try jsonrpc.writeMessage(&self.stdin_writer.interface, payload);
        try self.stdin_writer.flush();

        var read_count: usize = 0;
        while (true) {
            const msg_opt = try jsonrpc.readMessageAlloc(
                self.allocator,
                &self.stdout_reader.interface,
                self.max_payload_bytes,
            );
            if (msg_opt == null) return error.EndOfStream;
            const msg = msg_opt.?;
            defer self.allocator.free(msg);

            const parsed = std.json.parseFromSlice(
                std.json.Value,
                self.allocator,
                msg,
                .{},
            ) catch {
                continue;
            };
            defer parsed.deinit();

            if (parsed.value != .object) continue;
            const obj = parsed.value.object;
            const id_val = obj.get("id") orelse {
                read_count += 1;
                if (read_count > 256) return error.ResponseNotFound;
                continue;
            };

            if (!idMatches(id_val, id)) {
                read_count += 1;
                if (read_count > 256) return error.ResponseNotFound;
                continue;
            }

            if (obj.get("result")) |result_val| {
                const result_json = try std.json.Stringify.valueAlloc(
                    self.allocator,
                    result_val,
                    .{},
                );
                return .{ .json = result_json, .is_error = false };
            }
            if (obj.get("error")) |error_val| {
                const err_json = try std.json.Stringify.valueAlloc(
                    self.allocator,
                    error_val,
                    .{},
                );
                return .{ .json = err_json, .is_error = true };
            }

            const empty = try self.allocator.dupe(u8, "null");
            return .{ .json = empty, .is_error = true };
        }
    }

    pub fn notifyRaw(
        self: *Self,
        method: []const u8,
        params_json: ?[]const u8,
    ) !void {
        const payload = try buildMessageJson(self.allocator, null, method, params_json);
        defer self.allocator.free(payload);

        try jsonrpc.writeMessage(&self.stdin_writer.interface, payload);
        try self.stdin_writer.flush();
    }

    pub fn waitForNotification(
        self: *Self,
        method: []const u8,
        max_messages: usize,
    ) !?[]u8 {
        var count: usize = 0;
        while (count < max_messages) : (count += 1) {
            const msg_opt = try jsonrpc.readMessageAlloc(
                self.allocator,
                &self.stdout_reader.interface,
                self.max_payload_bytes,
            );
            if (msg_opt == null) return null;
            const msg = msg_opt.?;
            defer self.allocator.free(msg);

            const parsed = std.json.parseFromSlice(
                std.json.Value,
                self.allocator,
                msg,
                .{},
            ) catch {
                continue;
            };
            defer parsed.deinit();

            if (parsed.value != .object) continue;
            const obj = parsed.value.object;
            const method_val = obj.get("method") orelse continue;
            if (method_val != .string) continue;
            if (!std.mem.eql(u8, method_val.string, method)) continue;

            const params_val = obj.get("params") orelse std.json.Value{ .null = {} };
            return try std.json.Stringify.valueAlloc(self.allocator, params_val, .{});
        }
        return null;
    }

    fn positionRequest(self: *Self, method: []const u8, uri: []const u8, pos: types.Position) !Response {
        const params = types.TextDocumentPositionParams{
            .textDocument = .{ .uri = uri },
            .position = pos,
        };
        const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
        defer self.allocator.free(params_json);
        return self.requestRaw(method, params_json);
    }

    fn initialize(self: *Self) !void {
        const zig_path = try resolveZigPath(self.allocator, self.io, self.config);
        self.owned_zig_path = zig_path.owned;

        const root_name = std.fs.path.basename(self.root_path);

        const InitParams = struct {
            processId: ?i64 = null,
            rootUri: ?[]const u8 = null,
            rootPath: ?[]const u8 = null,
            workspaceFolders: []const struct { uri: []const u8, name: []const u8 } = &.{},
            capabilities: struct {} = .{},
            initializationOptions: struct {
                zig_exe_path: ?[]const u8 = null,
                enable_snippets: bool = true,
                log_level: []const u8 = "info",
            },
            clientInfo: struct {
                name: []const u8,
                version: []const u8,
            },
            trace: []const u8 = "off",
        };

        const init_params = InitParams{
            .rootUri = self.root_uri,
            .rootPath = self.root_path,
            .workspaceFolders = &.{.{ .uri = self.root_uri, .name = root_name }},
            .initializationOptions = .{
                .zig_exe_path = zig_path.path,
                .enable_snippets = self.config.enable_snippets,
                .log_level = self.config.log_level,
            },
            .clientInfo = .{
                .name = "abi-lsp",
                .version = "dev",
            },
        };

        const params_json = try std.json.Stringify.valueAlloc(self.allocator, init_params, .{});
        defer self.allocator.free(params_json);

        const resp = try self.requestRaw("initialize", params_json);
        defer self.allocator.free(resp.json);
        if (resp.is_error) return error.InitializeFailed;

        try self.notifyRaw("initialized", "{}");
    }

    fn shutdown(self: *Self) !void {
        const resp = self.requestRaw("shutdown", null) catch return;
        defer self.allocator.free(resp.json);
        self.notifyRaw("exit", null) catch {};
    }
};

fn buildMessageJson(
    allocator: std.mem.Allocator,
    id: ?i64,
    method: []const u8,
    params_json: ?[]const u8,
) ![]u8 {
    var writer = std.Io.Writer.Allocating.init(allocator);
    errdefer writer.deinit();

    var jw = std.json.Stringify{ .writer = &writer.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("jsonrpc");
    try jw.write("2.0");
    if (id) |value| {
        try jw.objectField("id");
        try jw.write(value);
    }
    try jw.objectField("method");
    try jw.write(method);
    if (params_json) |params| {
        try jw.objectField("params");
        try jw.beginWriteRaw();
        try writer.writer.writeAll(params);
        jw.endWriteRaw();
    }
    try jw.endObject();

    return writer.toOwnedSlice();
}

fn idMatches(id_val: std.json.Value, expected: i64) bool {
    return switch (id_val) {
        .integer => |v| v == expected,
        .number_string => |v| (std.fmt.parseInt(i64, v, 10) catch return false) == expected,
        .string => |v| (std.fmt.parseInt(i64, v, 10) catch return false) == expected,
        else => false,
    };
}

const ZigPathResult = struct {
    path: ?[]const u8,
    owned: ?[]u8,
};

fn resolveZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
) !ZigPathResult {
    if (config.zig_exe_path) |path| {
        return .{ .path = path, .owned = null };
    }

    const candidate = zig_toolchain.resolveExistingZvmMasterZigPath(allocator, io) catch return .{
        .path = null,
        .owned = null,
    };
    if (candidate == null) {
        return .{ .path = null, .owned = null };
    }

    return .{ .path = candidate.?, .owned = candidate.? };
}

pub fn resolveWorkspaceRoot(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_opt: ?[]const u8,
) ![]u8 {
    if (root_opt) |root| {
        return resolvePath(allocator, io, null, root);
    }
    return resolvePath(allocator, io, null, ".");
}

pub fn resolvePath(
    allocator: std.mem.Allocator,
    io: std.Io,
    base_opt: ?[]const u8,
    path: []const u8,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        return std.fs.path.resolve(allocator, &.{path});
    }

    if (base_opt) |base| {
        return std.fs.path.resolve(allocator, &.{ base, path });
    }

    const cwd_z = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd_z);
    return std.fs.path.resolve(allocator, &.{ cwd_z, path });
}

pub fn pathToUri(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, path, "file://")) {
        return allocator.dupe(u8, path);
    }

    var writer = std.Io.Writer.Allocating.init(allocator);
    errdefer writer.deinit();

    try writer.writer.writeAll("file://");

    var normalized = path;
    var tmp: ?[]u8 = null;
    if (comptime @import("builtin").os.tag == .windows) {
        tmp = try allocator.dupe(u8, path);
        for (tmp.?) |*c| {
            if (c.* == '\\') c.* = '/';
        }
        normalized = tmp.?;
    }
    defer if (tmp) |owned| allocator.free(owned);

    const component = std.Uri.Component{ .raw = normalized };
    try component.formatPath(&writer.writer);
    return writer.toOwnedSlice();
}

test {
    std.testing.refAllDecls(@This());
}
