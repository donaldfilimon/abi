//! Shared types for model management (used by both mod.zig and stub.zig).

const std = @import("std");
const discovery = @import("../explore/discovery.zig");

// Re-export from discovery
pub const ModelFormat = discovery.ModelFormat;
pub const QuantizationType = discovery.QuantizationType;

// Re-export from manager
pub const ManagerConfig = struct {
    cache_dir: ?[]const u8 = null,
    auto_scan: bool = true,
    extensions: []const []const u8 = &.{ ".gguf", ".mlx", ".bin", ".safetensors" },
};

pub const CachedModel = struct {
    path: []const u8,
    name: []const u8,
    size_bytes: u64,
    format: ModelFormat,
    quantization: ?QuantizationType,
    source_url: ?[]const u8,
    downloaded_at: i64,
    checksum: ?[]const u8,

    pub fn deinit(self: *CachedModel, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
        if (self.source_url) |url| allocator.free(url);
        if (self.checksum) |cs| allocator.free(cs);
    }
};

// Re-export from downloader
pub const DownloadConfig = struct {
    output_path: ?[]const u8 = null,
    progress_callback: ?*const fn (DownloadProgress) void = null,
    resume_download: bool = true,
    timeout_ms: u32 = 60_000,
    max_retries: u32 = 3,
    buffer_size: usize = 64 * 1024,
    verify_ssl: bool = true,
    verify_checksum: bool = true,
    expected_checksum: ?[]const u8 = null,
};

pub const DownloadProgress = struct {
    total_bytes: u64,
    downloaded_bytes: u64,
    speed_bytes_per_sec: u64,
    eta_seconds: ?u64,
    status: enum { downloading, verifying, complete, failed, paused },
};

pub const DownloadResult = struct {
    path: []const u8,
    bytes_downloaded: u64,
    checksum: [64]u8,
    was_resumed: bool,
    checksum_verified: bool,
};

pub const DownloadError = error{
    NetworkError,
    FileSystemError,
    ChecksumMismatch,
    TimeoutError,
    Cancelled,
    InvalidUrl,
    ServerError,
    ModelsDisabled,
};

// Re-export from huggingface
pub const HuggingFaceModel = struct {
    id: []const u8,
    author: []const u8,
    model_name: []const u8,
    downloads: u64,
    likes: u64,
    files: []HuggingFaceFile,
    tags: []const []const u8,

    pub fn deinit(self: *HuggingFaceModel, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.author);
        allocator.free(self.model_name);
        for (self.files) |*file| {
            file.deinit(allocator);
        }
        allocator.free(self.files);
        for (self.tags) |tag| {
            allocator.free(tag);
        }
        allocator.free(self.tags);
    }
};

pub const HuggingFaceFile = struct {
    filename: []const u8,
    size: u64,
    download_url: ?[]const u8,
    quantization: ?QuantizationType,

    pub fn deinit(self: *HuggingFaceFile, allocator: std.mem.Allocator) void {
        allocator.free(self.filename);
        if (self.download_url) |url| allocator.free(url);
    }
};

pub const SearchResult = struct {
    models: []HuggingFaceModel,
    total_count: u64,

    pub fn deinit(self: *SearchResult, allocator: std.mem.Allocator) void {
        for (self.models) |*model| {
            model.deinit(allocator);
        }
        allocator.free(self.models);
    }
};

pub const ModelSpec = struct {
    model_id: []const u8,
    filename: ?[]const u8 = null,
    quantization_hint: ?QuantizationType = null,
};

pub const QuantInfo = struct {
    name: []const u8,
    bits: f32,
    desc: []const u8,
};

// Re-export from registry
pub const ModelInfo = struct {
    name: []const u8,
    format: ModelFormat,
    quantization: ?QuantizationType,
    size_bytes: u64,
    path: []const u8,
    metadata: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn deinit(self: *ModelInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.path);
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit(allocator);
    }
};

// Error type
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

test {
    std.testing.refAllDecls(@This());
}
