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
    auto_scan: bool = true,
    extensions: []const []const u8 = &.{ ".gguf", ".bin", ".safetensors" },
};

/// Model format enum (stub).
pub const ModelFormat = enum {
    gguf,
    safetensors,
    bin,
    pytorch,
    unknown,
};

/// Quantization type enum (stub).
pub const QuantizationType = enum {
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,
    f16,
    f32,
    unknown,
};

pub const CachedModel = struct {
    path: []const u8,
    name: []const u8,
    size_bytes: u64,
    format: ModelFormat = .unknown,
    quantization: ?QuantizationType = null,
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

    pub fn modelCount(self: *Manager) usize {
        _ = self;
        return 0;
    }

    pub fn totalCacheSize(self: *Manager) u64 {
        _ = self;
        return 0;
    }

    pub fn getModel(self: *Manager, name: []const u8) ?*CachedModel {
        _ = self;
        _ = name;
        return null;
    }

    pub fn addModel(self: *Manager, path: []const u8, size_bytes: u64, source_url: ?[]const u8) Error!*CachedModel {
        _ = self;
        _ = path;
        _ = size_bytes;
        _ = source_url;
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
    verify_checksum: bool = true,
    expected_checksum: ?[]const u8 = null,
};

pub const DownloadProgress = struct {
    total_bytes: u64,
    downloaded_bytes: u64,
    speed_bytes_per_sec: u64,
    eta_seconds: ?u32,
    percent: u8,
};

pub const DownloadError = error{
    DownloadFailed,
    NetworkError,
    FileSystemError,
    ModelsDisabled,
};

/// Download result containing path and metadata.
pub const DownloadResult = struct {
    /// Path to the downloaded file.
    path: []const u8,
    /// Total bytes downloaded.
    bytes_downloaded: u64,
    /// SHA256 checksum of the downloaded file (hex string).
    checksum: [64]u8,
    /// Whether the download was resumed.
    was_resumed: bool,
    /// Whether checksum was verified successfully.
    checksum_verified: bool,
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

    pub fn downloadWithIo(self: *Downloader, io: anytype, url: []const u8, config: DownloadConfig) DownloadError!DownloadResult {
        _ = self;
        _ = io;
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

/// Parsed model specification.
pub const ModelSpec = struct {
    /// Model ID (author/model-name).
    model_id: []const u8,
    /// Explicit filename if provided.
    filename: ?[]const u8,
    /// Quantization hint if provided (e.g., "Q4_K_M").
    quantization_hint: ?[]const u8,
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

    /// Build a filename from a quantization hint for a model.
    pub fn buildFilenameFromHint(self: *HuggingFaceClient, model_id: []const u8, quant_hint: []const u8) Error![]const u8 {
        _ = self;
        _ = model_id;
        _ = quant_hint;
        return error.ModelsDisabled;
    }

    /// Parse a model spec like "TheBloke/Model:Q4_K_M" into components.
    pub fn parseModelSpec(spec: []const u8) ModelSpec {
        return .{
            .model_id = spec,
            .filename = null,
            .quantization_hint = null,
        };
    }

    /// Get common quantization types and their descriptions.
    pub fn getQuantizationInfo() []const QuantInfo {
        return &.{
            .{ .name = "Q4_K_M", .bits = 4.5, .desc = "Medium quality" },
        };
    }

    /// Get list of popular GGUF model authors.
    pub fn getPopularAuthors() []const []const u8 {
        return &.{
            "TheBloke",
            "bartowski",
        };
    }
};

/// Quantization information.
pub const QuantInfo = struct {
    name: []const u8,
    bits: f32,
    desc: []const u8,
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
