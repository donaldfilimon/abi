const std = @import("std");
const util = @import("util.zig");

const c = @cImport({
    @cInclude("stdlib.h");
});

const required_env = [_][]const u8{
    "DISCORD_BOT_TOKEN",
    "ABI_TEST_DISCORD_CHANNEL_ID",
    "ABI_TEST_DISCORD_WEBHOOK_URL",
    "OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "OLLAMA_HOST",
    "ABI_TEST_OLLAMA_MODEL",
    "ABI_TEST_GGUF_MODEL_PATH",
    "ABI_TEST_MODEL_SPEC",
};

const required_tools = [_][]const u8{
    "git",
    "cmake",
    "llvm-config",
};

const EnvOverlay = struct {
    allocator: std.mem.Allocator,
    values: std.StringHashMapUnmanaged([]const u8),

    fn init(allocator: std.mem.Allocator) EnvOverlay {
        return .{
            .allocator = allocator,
            .values = .empty,
        };
    }

    fn deinit(self: *EnvOverlay) void {
        var it = self.values.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.values.deinit(self.allocator);
    }

    fn get(self: *const EnvOverlay, key: []const u8) ?[]const u8 {
        return self.values.get(key);
    }

    fn put(self: *EnvOverlay, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        try self.values.put(self.allocator, key_copy, value_copy);
    }
};

fn getenvSlice(name: []const u8) ?[]const u8 {
    const ptr = c.getenv(name.ptr);
    if (ptr == null) return null;
    return std.mem.span(ptr);
}

fn getEnvValue(overlay: *const EnvOverlay, name: []const u8) ?[]const u8 {
    if (overlay.get(name)) |value| return value;

    var tmp: [128]u8 = undefined;
    if (name.len >= tmp.len) return null;
    @memcpy(tmp[0..name.len], name);
    tmp[name.len] = 0;
    return getenvSlice(tmp[0..name.len :0]);
}

fn loadEnvFile(overlay: *EnvOverlay, allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    const file_data = try util.readFileAlloc(allocator, io, path, 2 * 1024 * 1024);
    defer allocator.free(file_data);

    var lines = std.mem.splitScalar(u8, file_data, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r\n");
        if (line.len == 0 or line[0] == '#') continue;

        const eq_index = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const key = std.mem.trim(u8, line[0..eq_index], " \t");
        const value = std.mem.trim(u8, line[eq_index + 1 ..], " \t\"");
        if (key.len == 0) continue;

        try overlay.put(key, value);
    }
}

fn canReachUrl(allocator: std.mem.Allocator, url: []const u8) bool {
    const uri = std.Uri.parse(url) catch return false;

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();

    var client = std.http.Client{
        .allocator = allocator,
        .io = io_backend.io(),
    };
    defer client.deinit();

    var req = client.request(.GET, uri, .{}) catch return false;
    defer req.deinit();

    req.sendBodiless() catch return false;

    var header_buf: [4096]u8 = undefined;
    var response = req.receiveHead(&header_buf) catch return false;

    _ = response.head.status;
    return true;
}

fn modelSpecUrl(spec: []const u8) []const u8 {
    if (std.mem.startsWith(u8, spec, "http://") or std.mem.startsWith(u8, spec, "https://")) {
        return spec;
    }
    return "https://huggingface.co";
}

fn trimTrailingSlash(value: []const u8) []const u8 {
    var end = value.len;
    while (end > 0 and value[end - 1] == '/') : (end -= 1) {}
    return value[0..end];
}

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var args_iter = try std.process.Args.Iterator.initAllocator(init.args, allocator);
    defer args_iter.deinit();

    var args = std.ArrayListUnmanaged([]const u8).empty;
    defer args.deinit(allocator);

    _ = args_iter.next(); // executable path
    while (args_iter.next()) |arg| {
        try args.append(allocator, arg);
    }

    var env_file: ?[]const u8 = null;
    var json_out: []const u8 = "/tmp/abi-cli-full-preflight.json";

    var i: usize = 0;
    while (i < args.items.len) : (i += 1) {
        const arg = args.items[i];
        if (std.mem.eql(u8, arg, "--env-file")) {
            i += 1;
            if (i >= args.items.len) {
                std.debug.print("error: --env-file requires a path\n", .{});
                std.process.exit(1);
            }
            env_file = args.items[i];
        } else if (std.mem.eql(u8, arg, "--json-out")) {
            i += 1;
            if (i >= args.items.len) {
                std.debug.print("error: --json-out requires a path\n", .{});
                std.process.exit(1);
            }
            json_out = args.items[i];
        }
    }

    var overlay = EnvOverlay.init(allocator);
    defer overlay.deinit();

    if (env_file) |path| {
        loadEnvFile(&overlay, allocator, io, path) catch |err| {
            std.debug.print("ERROR: failed to read env file {s}: {t}\n", .{ path, err });
            std.process.exit(1);
        };
    }

    var missing_env = std.ArrayListUnmanaged([]const u8).empty;
    defer missing_env.deinit(allocator);

    var missing_tools = std.ArrayListUnmanaged([]const u8).empty;
    defer missing_tools.deinit(allocator);

    var failed_connectivity = std.ArrayListUnmanaged([]const u8).empty;
    defer failed_connectivity.deinit(allocator);

    for (required_env) |name| {
        const value = getEnvValue(&overlay, name);
        if (value == null or value.?.len == 0) {
            try missing_env.append(allocator, name);
        }
    }

    for (required_tools) |tool| {
        if (!(try util.commandExists(allocator, tool))) {
            try missing_tools.append(allocator, tool);
        }
    }

    const gguf_path = getEnvValue(&overlay, "ABI_TEST_GGUF_MODEL_PATH") orelse "";
    if (gguf_path.len > 0 and !util.fileExists(io, gguf_path)) {
        try missing_env.append(allocator, "file:ABI_TEST_GGUF_MODEL_PATH");
    }

    // Endpoint connectivity checks.
    if (!canReachUrl(allocator, "https://discord.com/api/v10")) {
        try failed_connectivity.append(allocator, "discord");
    }
    if (!canReachUrl(allocator, "https://api.openai.com/v1/models")) {
        try failed_connectivity.append(allocator, "openai");
    }
    if (!canReachUrl(allocator, "https://api.mistral.ai/v1/models")) {
        try failed_connectivity.append(allocator, "mistral");
    }
    if (!canReachUrl(allocator, "https://api.cohere.com/v1/models")) {
        try failed_connectivity.append(allocator, "cohere");
    }

    const ollama_host = getEnvValue(&overlay, "OLLAMA_HOST") orelse "";
    if (ollama_host.len > 0) {
        const ollama_url = try std.fmt.allocPrint(allocator, "{s}/api/tags", .{trimTrailingSlash(ollama_host)});
        defer allocator.free(ollama_url);
        if (!canReachUrl(allocator, ollama_url)) {
            try failed_connectivity.append(allocator, "ollama");
        }
    }

    const model_spec = getEnvValue(&overlay, "ABI_TEST_MODEL_SPEC") orelse "";
    if (model_spec.len > 0) {
        if (!canReachUrl(allocator, modelSpecUrl(model_spec))) {
            try failed_connectivity.append(allocator, "model-host");
        }
    }

    const ok = missing_env.items.len == 0 and missing_tools.items.len == 0 and failed_connectivity.items.len == 0;

    const report = .{
        .ok = ok,
        .missing_env = missing_env.items,
        .missing_tools = missing_tools.items,
        .failed_connectivity = failed_connectivity.items,
    };

    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();
    try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, &json_writer.writer);
    try json_writer.writer.writeByte('\n');
    const json_data = try json_writer.toOwnedSlice();
    defer allocator.free(json_data);

    var output_file = try std.Io.Dir.cwd().createFile(io, json_out, .{ .truncate = true });
    defer output_file.close(io);
    try output_file.writeStreamingAll(io, json_data);

    std.debug.print("CLI full preflight\n", .{});
    std.debug.print("  output: {s}\n", .{json_out});
    std.debug.print("  missing_env: {d}\n", .{missing_env.items.len});
    std.debug.print("  missing_tools: {d}\n", .{missing_tools.items.len});
    std.debug.print("  failed_connectivity: {d}\n", .{failed_connectivity.items.len});

    if (!ok) {
        if (missing_env.items.len > 0) {
            std.debug.print("Missing env prerequisites:\n", .{});
            for (missing_env.items) |name| std.debug.print("  - {s}\n", .{name});
        }
        if (missing_tools.items.len > 0) {
            std.debug.print("Missing tool prerequisites:\n", .{});
            for (missing_tools.items) |name| std.debug.print("  - {s}\n", .{name});
        }
        if (failed_connectivity.items.len > 0) {
            std.debug.print("Connectivity failures:\n", .{});
            for (failed_connectivity.items) |name| std.debug.print("  - {s}\n", .{name});
        }
        std.process.exit(1);
    }

    std.debug.print("OK: CLI full preflight passed\n", .{});
}
