//! ZLS LSP client (JSON-RPC over stdio).
//!
//! Decomposed into sub-modules:
//! - `transport.zig` — JSON-RPC send/receive, raw request/notification helpers
//! - `requests.zig` — hover, completion, definition, references, rename, formatting
//! - `notifications.zig` — didOpen, didClose, didChange

const std = @import("std");
const jsonrpc = @import("../jsonrpc.zig");
const types = @import("../types.zig");

pub const transport = @import("transport.zig");
pub const requests = @import("requests.zig");
pub const notifications = @import("notifications.zig");

// Inline LspConfig to avoid cross-directory import to ../../core/config/.
// Canonical definition: src/core/config/lsp.zig — keep in sync.
pub const Config = struct {
    zls_path: []const u8 = "zls",
    zig_exe_path: ?[]const u8 = null,
    workspace_root: ?[]const u8 = null,
    log_level: []const u8 = "info",
    enable_snippets: bool = true,

    pub fn defaults() Config {
        return .{};
    }
};

// Inline toolchain resolution to avoid cross-directory relative imports
// that break standalone compilation and ZLS analysis. The canonical
// implementation lives in src/foundation/utils/zig_toolchain.zig.
const zig_toolchain = struct {
    const builtin = @import("builtin");

    fn zigBinaryName() []const u8 {
        return if (builtin.os.tag == .windows) "zig.exe" else "zig";
    }

    pub fn zlsBinaryName() []const u8 {
        return if (builtin.os.tag == .windows) "zls.exe" else "zls";
    }

    pub fn resolveExistingPreferredZigPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        start_path: []const u8,
    ) !?[]u8 {
        _ = start_path;
        // ZVM master fallback
        const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
        const home = std.mem.span(home_ptr);
        const zvm_path = try std.fs.path.join(allocator, &.{ home, ".zvm", "master", zigBinaryName() });
        if (!fileExistsAbsolute(io, zvm_path)) {
            allocator.free(zvm_path);
            return null;
        }
        return zvm_path;
    }

    fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
        const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
        file.close(io);
        return true;
    }
};

pub const Response = transport.Response;

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
    owned_zls_path: ?[]u8,
    owned_zig_path: ?[]u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, io: std.Io, config: Config) !Self {
        const root_path = try resolveWorkspaceRoot(allocator, io, config.workspace_root);
        errdefer allocator.free(root_path);

        const root_uri = try pathToUri(allocator, root_path);
        errdefer allocator.free(root_uri);

        const zls_path = try resolveZlsPath(allocator, io, config, root_path);
        errdefer if (zls_path.owned) |owned| allocator.free(owned);

        const argv = [_][]const u8{zls_path.path};
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
            .owned_zls_path = zls_path.owned,
            .owned_zig_path = null,
        };

        client.stdout_reader = child.stdout.?.reader(io, &client.read_buf);
        client.stdin_writer = child.stdin.?.writer(io, &client.write_buf);

        try client.initialize();
        return client;
    }

    pub fn deinit(self: *Self) void {
        self.shutdown() catch |err| {
            std.log.debug("LSP shutdown failed (non-fatal): {t}", .{err});
        };
        if (self.child.id != null) {
            self.child.kill(self.io);
        }
        self.allocator.free(self.root_path);
        self.allocator.free(self.root_uri);
        if (self.owned_zls_path) |owned| {
            self.allocator.free(owned);
        }
        if (self.owned_zig_path) |owned| {
            self.allocator.free(owned);
        }
    }

    pub fn workspaceRoot(self: *const Self) []const u8 {
        return self.root_path;
    }

    // -- Notification methods (delegated to notifications.zig) --

    pub fn didOpen(self: *Self, doc: types.TextDocumentItem) !void {
        return notifications.didOpen(self, doc);
    }

    // -- Request methods (delegated to requests.zig) --

    pub fn hover(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return requests.hover(self, uri, pos);
    }

    pub fn completion(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return requests.completion(self, uri, pos);
    }

    pub fn definition(self: *Self, uri: []const u8, pos: types.Position) !Response {
        return requests.definition(self, uri, pos);
    }

    pub fn references(self: *Self, uri: []const u8, pos: types.Position, include_decl: bool) !Response {
        return requests.references(self, uri, pos, include_decl);
    }

    pub fn rename(
        self: *Self,
        uri: []const u8,
        pos: types.Position,
        new_name: []const u8,
    ) !Response {
        return requests.rename(self, uri, pos, new_name);
    }

    pub fn formatting(
        self: *Self,
        uri: []const u8,
        options: types.FormattingOptions,
    ) !Response {
        return requests.formatting(self, uri, options);
    }

    pub fn diagnostics(self: *Self, uri: []const u8) !Response {
        return requests.diagnostics(self, uri);
    }

    // -- Transport methods (delegated to transport.zig) --

    pub fn requestRaw(
        self: *Self,
        method: []const u8,
        params_json: ?[]const u8,
    ) !Response {
        return transport.requestRaw(
            self.allocator,
            &self.stdin_writer,
            &self.stdout_reader,
            &self.next_id,
            self.max_payload_bytes,
            method,
            params_json,
        );
    }

    pub fn notifyRaw(
        self: *Self,
        method: []const u8,
        params_json: ?[]const u8,
    ) !void {
        return transport.notifyRaw(
            self.allocator,
            &self.stdin_writer,
            method,
            params_json,
        );
    }

    pub fn waitForNotification(
        self: *Self,
        method: []const u8,
        max_messages: usize,
    ) !?[]u8 {
        return transport.waitForNotification(
            self.allocator,
            &self.stdout_reader,
            self.max_payload_bytes,
            method,
            max_messages,
        );
    }

    fn initialize(self: *Self) !void {
        const zig_path = try resolveZigPath(self.allocator, self.io, self.config, self.root_path);
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
        self.notifyRaw("exit", null) catch |err| {
            std.log.debug("LSP exit notification failed: {t}", .{err});
        };
    }
};

const ZigPathResult = struct {
    path: ?[]const u8,
    owned: ?[]u8,
};

const ZlsPathResult = struct {
    path: []const u8,
    owned: ?[]u8,
};

fn resolveZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
    root_path: []const u8,
) !ZigPathResult {
    if (config.zig_exe_path) |path| {
        return .{ .path = path, .owned = null };
    }

    if (try zig_toolchain.resolveExistingPreferredZigPath(allocator, io, root_path)) |candidate| {
        return .{ .path = candidate, .owned = candidate };
    }

    return .{ .path = null, .owned = null };
}

fn resolveZlsPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
    root_path: []const u8,
) !ZlsPathResult {
    if (!std.mem.eql(u8, config.zls_path, Config.defaults().zls_path)) {
        return .{ .path = config.zls_path, .owned = null };
    }

    _ = allocator;
    _ = io;
    _ = root_path;
    return .{ .path = config.zls_path, .owned = null };
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

fn createTestAbiRepo(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_root: []const u8,
    include_zig: bool,
    include_zls: bool,
) !void {
    const dirs = [_][]const u8{"src"};
    for (dirs) |sub| {
        const full = try std.fs.path.join(allocator, &.{ repo_root, sub });
        defer allocator.free(full);
        std.Io.Dir.createDirPath(.cwd(), io, full) catch |err| {
            std.log.warn("lsp: test repo directory creation failed: {}", .{err});
        };
    }

    const files = [_]struct { sub: []const u8, data: []const u8 }{
        .{ .sub = "build.zig", .data = "pub fn build(_: *anyopaque) void {}\n" },
        .{ .sub = "src/root.zig", .data = "pub const test_value = 1;\n" },
    };
    for (files) |entry| {
        const full = try std.fs.path.join(allocator, &.{ repo_root, entry.sub });
        defer allocator.free(full);
        const file = try std.Io.Dir.createFileAbsolute(io, full, .{});
        try file.writeStreamingAll(io, entry.data);
        file.close(io);
    }

    _ = include_zig;
    _ = include_zls;
}

fn testRepoRootPath(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd);
    return std.fs.path.join(allocator, &.{ cwd, ".zig-cache", "tmp", "lsp-test-repo" });
}

fn cleanupTestRepo(allocator: std.mem.Allocator, io: std.Io, repo_root: []const u8) void {
    std.Io.Dir.deleteTree(.cwd(), io, repo_root) catch |err| {
        std.log.warn("lsp: test repo cleanup failed: {}", .{err});
    };
    _ = allocator;
}

test "resolveZigPath prefers explicit override over fallback discovery" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const repo_root = try testRepoRootPath(std.testing.allocator, io);
    defer std.testing.allocator.free(repo_root);
    defer cleanupTestRepo(std.testing.allocator, io, repo_root);

    try createTestAbiRepo(std.testing.allocator, io, repo_root, true, false);

    const result = try resolveZigPath(std.testing.allocator, io, .{
        .zig_exe_path = "/override/zig",
    }, repo_root);

    try std.testing.expectEqualStrings("/override/zig", result.path.?);
    try std.testing.expect(result.owned == null);
}

test "resolveZlsPath falls back to PATH default when no override is set" {
    var io_backend = std.Io.Threaded.init(std.testing.allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const repo_root = try testRepoRootPath(std.testing.allocator, io);
    defer std.testing.allocator.free(repo_root);
    defer cleanupTestRepo(std.testing.allocator, io, repo_root);

    try createTestAbiRepo(std.testing.allocator, io, repo_root, false, false);

    const result = try resolveZlsPath(std.testing.allocator, io, Config.defaults(), repo_root);
    try std.testing.expectEqualStrings("zls", result.path);
    try std.testing.expect(result.owned == null);
}

test {
    std.testing.refAllDecls(@This());
}
