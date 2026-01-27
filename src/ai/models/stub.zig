//! Model Management Module - Stub
//!
//! Provides stub implementations when AI feature is disabled.
//! All operations return `error.ModelsDisabled`.

const std = @import("std");

pub const Error = error{
    ModelsDisabled,
    ModelNotFound,
    DownloadFailed,
    InvalidModelId,
    CacheNotAccessible,
    NetworkError,
    FileSystemError,
    HuggingFaceError,
    OutOfMemory,
};

pub fn isEnabled() bool {
    return false;
}

// ============================================================================
// Manager Stub
// ============================================================================

pub const ManagerConfig = struct {
    cache_dir: ?[]const u8 = null,
};

pub const CachedModel = struct {
    path: []const u8,
    name: []const u8,
    size_bytes: u64,
    source_url: ?[]const u8 = null,
    downloaded_at: i64 = 0,
    checksum: ?[]const u8 = null,

    pub fn deinit(self: *CachedModel, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const Manager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ManagerConfig) Error!Manager {
        _ = config;
        _ = allocator;
        return error.ModelsDisabled;
    }

    pub fn deinit(self: *Manager) void {
        _ = self;
    }

    pub fn listModels(self: *Manager) []CachedModel {
        _ = self;
        return &.{};
    }

    pub fn getModel(self: *Manager, name: []const u8) Error!*CachedModel {
        _ = self;
        _ = name;
        return error.ModelsDisabled;
    }

    pub fn removeModel(self: *Manager, name: []const u8) Error!void {
        _ = self;
        _ = name;
        return error.ModelsDisabled;
    }

    pub fn getCacheDir(self: *Manager) []const u8 {
        _ = self;
        return "";
    }

    pub fn setCacheDir(self: *Manager, path: []const u8) Error!void {
        _ = self;
        _ = path;
        return error.ModelsDisabled;
    }
};

pub const manager = struct {
    pub const Manager = @This().Manager;
    pub const ManagerConfig = @This().ManagerConfig;
    pub const CachedModel = @This().CachedModel;
};

// ============================================================================
// Downloader Stub
// ============================================================================

pub const DownloadConfig = struct {
    output_path: ?[]const u8 = null,
    progress_callback: ?*const fn (DownloadProgress) void = null,
    resume_download: bool = true,
};

pub const DownloadProgress = struct {
    total_bytes: u64,
    downloaded_bytes: u64,
    speed_bytes_per_sec: u64,
    eta_seconds: ?u32,
};

pub const DownloadError = error{
    DownloadFailed,
    NetworkError,
    FileSystemError,
    ModelsDisabled,
};

pub const Downloader = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Downloader {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Downloader) void {
        _ = self;
    }

    pub fn download(self: *Downloader, url: []const u8, config: DownloadConfig) DownloadError![]const u8 {
        _ = self;
        _ = url;
        _ = config;
        return error.ModelsDisabled;
    }
};

pub const downloader = struct {
    pub const Downloader = @This().Downloader;
    pub const DownloadConfig = @This().DownloadConfig;
    pub const DownloadProgress = @This().DownloadProgress;
    pub const DownloadError = @This().DownloadError;
};

// ============================================================================
// HuggingFace Stub
// ============================================================================

pub const HuggingFaceModel = struct {
    id: []const u8,
    author: []const u8,
    downloads: u64,
    likes: u64,
    files: []const HuggingFaceFile,

    pub fn deinit(self: *HuggingFaceModel, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const HuggingFaceFile = struct {
    name: []const u8,
    size_bytes: u64,
    quantization: ?[]const u8,
};

pub const SearchResult = struct {
    models: []HuggingFaceModel,
    total_count: u64,

    pub fn deinit(self: *SearchResult, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const HuggingFaceClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, api_token: ?[]const u8) HuggingFaceClient {
        _ = api_token;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *HuggingFaceClient) void {
        _ = self;
    }

    pub fn search(self: *HuggingFaceClient, query: []const u8) Error!SearchResult {
        _ = self;
        _ = query;
        return error.ModelsDisabled;
    }

    pub fn getModel(self: *HuggingFaceClient, model_id: []const u8) Error!HuggingFaceModel {
        _ = self;
        _ = model_id;
        return error.ModelsDisabled;
    }

    pub fn resolveDownloadUrl(self: *HuggingFaceClient, model_id: []const u8, filename: []const u8) Error![]const u8 {
        _ = self;
        _ = model_id;
        _ = filename;
        return error.ModelsDisabled;
    }
};

pub const huggingface = struct {
    pub const HuggingFaceClient = @This().HuggingFaceClient;
    pub const HuggingFaceModel = @This().HuggingFaceModel;
    pub const HuggingFaceFile = @This().HuggingFaceFile;
    pub const SearchResult = @This().SearchResult;
};

// ============================================================================
// Tests
// ============================================================================

test "stub returns disabled error" {
    const manager_result = Manager.init(std.testing.allocator, .{});
    try std.testing.expectError(error.ModelsDisabled, manager_result);
}
