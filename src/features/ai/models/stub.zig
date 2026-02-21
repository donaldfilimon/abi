//! Model Management stub — disabled at compile time.

const std = @import("std");

pub const Error = error{ ModelsDisabled, ModelNotFound, DownloadFailed, InvalidModelId, CacheNotAccessible, NetworkError, FileSystemError, HuggingFaceError, OutOfMemory };

pub fn isEnabled() bool {
    return false;
}

// --- Manager ---

pub const ManagerConfig = struct {
    cache_dir: ?[]const u8 = null,
    auto_scan: bool = true,
    extensions: []const []const u8 = &.{ ".gguf", ".mlx", ".bin", ".safetensors" },
};

pub const ModelFormat = enum { gguf, safetensors, bin, pytorch, unknown };
pub const QuantizationType = enum { q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32, unknown };

pub const CachedModel = struct {
    path: []const u8,
    name: []const u8,
    size_bytes: u64,
    format: ModelFormat = .unknown,
    quantization: ?QuantizationType = null,
    source_url: ?[]const u8 = null,
    downloaded_at: i64 = 0,
    checksum: ?[]const u8 = null,
    pub fn deinit(_: *CachedModel, _: std.mem.Allocator) void {}
};

pub const Manager = struct {
    allocator: std.mem.Allocator,
    pub fn init(_: std.mem.Allocator, _: ManagerConfig) Error!Manager {
        return error.ModelsDisabled;
    }
    pub fn deinit(_: *Manager) void {}
    pub fn listModels(_: *Manager) []CachedModel {
        return &.{};
    }
    pub fn modelCount(_: *Manager) usize {
        return 0;
    }
    pub fn totalCacheSize(_: *Manager) u64 {
        return 0;
    }
    pub fn getModel(_: *Manager, _: []const u8) ?*CachedModel {
        return null;
    }
    pub fn addModel(_: *Manager, _: []const u8, _: u64, _: ?[]const u8) Error!*CachedModel {
        return error.ModelsDisabled;
    }
    pub fn removeModel(_: *Manager, _: []const u8) Error!void {
        return error.ModelsDisabled;
    }
    pub fn getCacheDir(_: *Manager) []const u8 {
        return "";
    }
    pub fn setCacheDir(_: *Manager, _: []const u8) Error!void {
        return error.ModelsDisabled;
    }
    pub fn scanCacheDirWithIo(_: *Manager, _: anytype) Error!void {
        return error.ModelsDisabled;
    }
    pub fn scanCacheDir(_: *Manager) Error!void {
        return error.ModelsDisabled;
    }
};

pub const manager = struct {
    pub const Manager = @This().Manager;
    pub const ManagerConfig = @This().ManagerConfig;
    pub const CachedModel = @This().CachedModel;
};

// --- Downloader ---

pub const DownloadConfig = struct {
    output_path: ?[]const u8 = null,
    progress_callback: ?*const fn (DownloadProgress) void = null,
    resume_download: bool = true,
    verify_checksum: bool = true,
    expected_checksum: ?[]const u8 = null,
};

pub const DownloadProgress = struct { total_bytes: u64, downloaded_bytes: u64, speed_bytes_per_sec: u64, eta_seconds: ?u32, percent: u8 };
pub const DownloadError = error{ DownloadFailed, NetworkError, FileSystemError, ModelsDisabled };

pub const DownloadResult = struct {
    path: []const u8,
    bytes_downloaded: u64,
    checksum: [64]u8,
    was_resumed: bool,
    checksum_verified: bool,
};

pub const Downloader = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) Downloader {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *Downloader) void {}
    pub fn download(_: *Downloader, _: []const u8, _: DownloadConfig) DownloadError![]const u8 {
        return error.ModelsDisabled;
    }
    pub fn downloadWithIo(_: *Downloader, _: anytype, _: []const u8, _: DownloadConfig) DownloadError!DownloadResult {
        return error.ModelsDisabled;
    }
};

pub const downloader = struct {
    pub const Downloader = @This().Downloader;
    pub const DownloadConfig = @This().DownloadConfig;
    pub const DownloadProgress = @This().DownloadProgress;
    pub const DownloadError = @This().DownloadError;
};

// --- HuggingFace ---

pub const HuggingFaceModel = struct {
    id: []const u8,
    author: []const u8,
    downloads: u64,
    likes: u64,
    files: []const HuggingFaceFile,
    pub fn deinit(_: *HuggingFaceModel, _: std.mem.Allocator) void {}
};

pub const HuggingFaceFile = struct { name: []const u8, size_bytes: u64, quantization: ?[]const u8 };

pub const SearchResult = struct {
    models: []HuggingFaceModel,
    total_count: u64,
    pub fn deinit(_: *SearchResult, _: std.mem.Allocator) void {}
};

pub const ModelSpec = struct { model_id: []const u8, filename: ?[]const u8, quantization_hint: ?[]const u8 };

pub const HuggingFaceClient = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: ?[]const u8) HuggingFaceClient {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *HuggingFaceClient) void {}
    pub fn search(_: *HuggingFaceClient, _: []const u8) Error!SearchResult {
        return error.ModelsDisabled;
    }
    pub fn getModel(_: *HuggingFaceClient, _: []const u8) Error!HuggingFaceModel {
        return error.ModelsDisabled;
    }
    pub fn resolveDownloadUrl(_: *HuggingFaceClient, _: []const u8, _: []const u8) Error![]const u8 {
        return error.ModelsDisabled;
    }
    pub fn buildFilenameFromHint(_: *HuggingFaceClient, _: []const u8, _: []const u8) Error![]const u8 {
        return error.ModelsDisabled;
    }
    pub fn parseModelSpec(spec: []const u8) ModelSpec {
        return .{ .model_id = spec, .filename = null, .quantization_hint = null };
    }
    pub fn getQuantizationInfo() []const QuantInfo {
        return &.{.{ .name = "Q4_K_M", .bits = 4.5, .desc = "Medium quality" }};
    }
    pub fn getPopularAuthors() []const []const u8 {
        return &.{ "TheBloke", "bartowski" };
    }
};

pub const QuantInfo = struct { name: []const u8, bits: f32, desc: []const u8 };

pub const huggingface = struct {
    pub const HuggingFaceClient = @This().HuggingFaceClient;
    pub const HuggingFaceModel = @This().HuggingFaceModel;
    pub const HuggingFaceFile = @This().HuggingFaceFile;
    pub const SearchResult = @This().SearchResult;
};

// --- Registry (merged from stubs/model_registry.zig) ---

pub const ModelInfo = struct {
    name: []const u8 = "",
    parameters: u64 = 0,
    description: []const u8 = "",
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.ArrayListUnmanaged(ModelInfo) = .{},

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *ModelRegistry) void {
        self.models.deinit(self.allocator);
    }
    pub fn register(_: *ModelRegistry, _: ModelInfo) !void {
        return error.ModelsDisabled;
    }
    pub fn find(_: *ModelRegistry, _: []const u8) ?ModelInfo {
        return null;
    }
    pub fn count(self: *ModelRegistry) usize {
        return self.models.items.len;
    }
};

// --- Tests ---

test "stub returns disabled error" {
    const manager_result = Manager.init(std.testing.allocator, .{});
    try std.testing.expectError(error.ModelsDisabled, manager_result);
}
