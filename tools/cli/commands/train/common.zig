//! Shared utilities for the train subcommand modules.
//!
//! Re-exports common imports and provides parsing helpers, dataset utilities,
//! and file I/O helpers used across multiple train subcommands.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const tui = @import("../../tui/mod.zig");

pub const cli_io = utils.io_backend;
pub const gguf_writer = abi.ai.llm.io.gguf_writer;

pub fn parseOptimizer(val: []const u8) abi.ai.training.OptimizerType {
    if (std.mem.eql(u8, val, "sgd")) return .sgd;
    if (std.mem.eql(u8, val, "adam")) return .adam;
    if (std.mem.eql(u8, val, "adamw")) return .adamw;
    return .adamw; // default
}

pub fn parseLrSchedule(val: []const u8) abi.ai.training.LearningRateSchedule {
    if (std.mem.eql(u8, val, "constant")) return .constant;
    if (std.mem.eql(u8, val, "cosine")) return .cosine;
    if (std.mem.eql(u8, val, "warmup_cosine")) return .warmup_cosine;
    if (std.mem.eql(u8, val, "step")) return .step;
    if (std.mem.eql(u8, val, "polynomial")) return .polynomial;
    return .warmup_cosine; // default
}

pub const DatasetFormat = enum {
    tokenbin,
    text,
    jsonl,
};

pub const DatasetPath = struct {
    path: []const u8,
    owned: bool,
};

pub fn parseDatasetFormat(val: []const u8) DatasetFormat {
    if (std.mem.eql(u8, val, "text")) return .text;
    if (std.mem.eql(u8, val, "jsonl")) return .jsonl;
    return .tokenbin;
}

pub fn resolveDatasetPath(
    allocator: std.mem.Allocator,
    dataset_url: ?[]const u8,
    dataset_path: ?[]const u8,
    dataset_cache: ?[]const u8,
    max_bytes: usize,
) !DatasetPath {
    if (dataset_path) |path| {
        return .{ .path = path, .owned = false };
    }
    if (dataset_url == null) {
        return .{ .path = "", .owned = false };
    }

    const url = dataset_url.?;
    const cache_path = if (dataset_cache) |path|
        path
    else
        try defaultDatasetCachePath(allocator, url);
    const owned = dataset_cache == null;

    if (!fileExists(cache_path)) {
        try downloadToFile(allocator, url, cache_path, max_bytes);
    }

    return .{ .path = cache_path, .owned = owned };
}

pub fn defaultDatasetCachePath(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
    var name: []const u8 = "dataset.bin";
    if (std.mem.lastIndexOfScalar(u8, url, '/')) |idx| {
        const tail = url[idx + 1 ..];
        if (tail.len > 0) name = tail;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    std.Io.Dir.cwd().createDirPath(io, "datasets") catch {};

    return std.fs.path.join(allocator, &.{ "datasets", name });
}

pub fn downloadToFile(allocator: std.mem.Allocator, url: []const u8, path: []const u8, max_bytes: usize) !void {
    var client = try abi.shared.utils.async_http.AsyncHttpClient.init(allocator);
    defer client.deinit();

    var request = try abi.shared.utils.async_http.HttpRequest.init(allocator, .get, url);
    defer request.deinit();

    var response = try client.fetch(&request);
    defer response.deinit();

    if (!response.isSuccess()) {
        return error.DownloadFailed;
    }
    if (max_bytes > 0 and response.body.len > max_bytes) {
        return error.PayloadTooLarge;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, response.body);
}

pub fn loadTokensFromPath(
    allocator: std.mem.Allocator,
    format: DatasetFormat,
    path: []const u8,
    tokenizer: ?*abi.ai.llm.tokenizer.Tokenizer,
    max_tokens: usize,
) ![]u32 {
    switch (format) {
        .tokenbin => {
            var tokens = try abi.ai.database.readTokenBinFile(allocator, path);
            if (max_tokens > 0 and tokens.len > max_tokens) {
                const trimmed = try allocator.alloc(u32, max_tokens);
                @memcpy(trimmed, tokens[0..max_tokens]);
                allocator.free(tokens);
                tokens = trimmed;
            }
            return tokens;
        },
        .text => {
            const text = try readTextFile(allocator, path);
            defer allocator.free(text);
            if (tokenizer == null) return error.InvalidTokenizer;
            var tokens = try tokenizer.?.encode(allocator, text);
            if (max_tokens > 0 and tokens.len > max_tokens) {
                const trimmed = try allocator.alloc(u32, max_tokens);
                @memcpy(trimmed, tokens[0..max_tokens]);
                allocator.free(tokens);
                tokens = trimmed;
            }
            return tokens;
        },
        .jsonl => {
            const text = try readTextFile(allocator, path);
            defer allocator.free(text);
            if (tokenizer == null) return error.InvalidTokenizer;
            return try tokenizeJsonl(allocator, tokenizer.?, text, max_tokens);
        },
    }
}

pub fn readTextFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(256 * 1024 * 1024),
    );
}

pub fn tokenizeJsonl(
    allocator: std.mem.Allocator,
    tokenizer: *abi.ai.llm.tokenizer.Tokenizer,
    data: []const u8,
    max_tokens: usize,
) ![]u32 {
    var tokens = std.ArrayListUnmanaged(u32).empty;
    errdefer tokens.deinit(allocator);

    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var text = line;

        const parsed = std.json.parseFromSlice(
            struct {
                text: ?[]const u8 = null,
                instruction: ?[]const u8 = null,
                input: ?[]const u8 = null,
                output: ?[]const u8 = null,
            },
            allocator,
            line,
            .{},
        ) catch null;

        if (parsed) |p| {
            defer p.deinit();
            if (p.value.text) |t| {
                text = t;
            } else if (p.value.instruction != null or p.value.output != null) {
                var buf = std.ArrayListUnmanaged(u8).empty;
                defer buf.deinit(allocator);
                if (p.value.instruction) |instr| {
                    try buf.appendSlice(allocator, instr);
                }
                if (p.value.input) |inp| {
                    if (buf.items.len > 0) try buf.appendSlice(allocator, "\n");
                    try buf.appendSlice(allocator, inp);
                }
                if (p.value.output) |out| {
                    if (buf.items.len > 0) try buf.appendSlice(allocator, "\n");
                    try buf.appendSlice(allocator, out);
                }
                text = try buf.toOwnedSlice(allocator);
                defer allocator.free(@constCast(text));
            }
        }

        const line_tokens = try tokenizer.encode(allocator, text);
        defer allocator.free(line_tokens);
        try appendTokensWithLimit(allocator, &tokens, line_tokens, max_tokens);
        if (max_tokens > 0 and tokens.items.len >= max_tokens) break;
    }

    return tokens.toOwnedSlice(allocator);
}

pub fn appendTokensWithLimit(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    tokens: []const u32,
    max_tokens: usize,
) !void {
    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        if (max_tokens > 0 and out.items.len >= max_tokens) break;
        try out.append(allocator, tokens[idx]);
    }
}

pub fn clampTokens(tokens: []u32, vocab_size: u32) void {
    if (vocab_size == 0) return;
    const max_id = vocab_size - 1;
    for (tokens) |*t| {
        if (t.* > max_id) t.* = max_id;
    }
}

pub fn fileExists(path: []const u8) bool {
    var io_backend = cli_io.initIoBackend(std.heap.page_allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
