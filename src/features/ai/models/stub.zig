//! Model Management stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const Error = types.Error;
pub const ManagerConfig = types.ManagerConfig;
pub const ModelFormat = types.ModelFormat;
pub const QuantizationType = types.QuantizationType;
pub const CachedModel = types.CachedModel;
pub const DownloadConfig = types.DownloadConfig;
pub const DownloadProgress = types.DownloadProgress;
pub const DownloadError = types.DownloadError;
pub const DownloadResult = types.DownloadResult;
pub const HuggingFaceModel = types.HuggingFaceModel;
pub const HuggingFaceFile = types.HuggingFaceFile;
pub const SearchResult = types.SearchResult;
pub const ModelSpec = types.ModelSpec;
pub const QuantInfo = types.QuantInfo;
pub const ModelInfo = types.ModelInfo;

pub fn isEnabled() bool {
    return false;
}

// --- Manager ---
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
    pub const Manager_ = Manager;
    pub const ManagerConfig_ = ManagerConfig;
    pub const CachedModel_ = CachedModel;
};

// --- Downloader ---
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
    pub const Downloader_ = Downloader;
    pub const DownloadConfig_ = DownloadConfig;
    pub const DownloadProgress_ = DownloadProgress;
    pub const DownloadError_ = DownloadError;
};

// --- HuggingFace ---
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

pub const huggingface = struct {
    pub const HuggingFaceClient_ = HuggingFaceClient;
    pub const HuggingFaceModel_ = HuggingFaceModel;
    pub const HuggingFaceFile_ = HuggingFaceFile;
    pub const SearchResult_ = SearchResult;
};

// --- Registry ---
pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.ArrayListUnmanaged(ModelInfo) = .empty,
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

test {
    std.testing.refAllDecls(@This());
}
