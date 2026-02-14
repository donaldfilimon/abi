const std = @import("std");

pub const FileMetadata = struct {
    path: []const u8 = "",
    size_bytes: usize = 0,
    line_count: usize = 0,
    chunk_count: usize = 0,
    last_indexed: i64 = 0,
};

pub const CodeSnippet = struct {
    file_path: []const u8 = "",
    start_line: usize = 0,
    end_line: usize = 0,
    content: []const u8 = "",
    relevance_score: f32 = 0,
};

pub const CodebaseAnswer = struct {
    allocator: std.mem.Allocator,
    snippets: std.ArrayListUnmanaged(CodeSnippet) = .{},
    summary: []const u8 = "",

    pub fn deinit(_: *CodebaseAnswer) void {}
};

pub const IndexResult = struct {
    files_indexed: usize = 0,
    total_chunks: usize = 0,
    total_lines: usize = 0,
    duration_ms: u64 = 0,
};

pub const IndexStats = struct {
    file_count: usize = 0,
    total_chunks: usize = 0,
    total_size_bytes: usize = 0,
    index_path: []const u8 = "",
};

pub const CodebaseIndex = struct {
    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: []const u8) error{AiDisabled}!Self {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn indexFile(_: *Self, _: []const u8, _: []const u8) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn query(_: *Self, _: []const u8) error{AiDisabled}!CodebaseAnswer {
        return error.AiDisabled;
    }

    pub fn getStats(_: *const Self) IndexStats {
        return .{};
    }
};
