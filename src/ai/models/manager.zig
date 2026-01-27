//! Model Manager - Cache and catalog management
//!
//! Manages the local model cache, providing functionality to list, add, and remove
//! cached models. Integrates with the discovery system for model metadata.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const discovery = @import("../discovery.zig");

// libc import for environment access - required for Zig 0.16
const c = if (builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64)
    @cImport(@cInclude("stdlib.h"))
else
    struct {
        pub fn getenv(_: [*:0]const u8) ?[*:0]const u8 {
            return null;
        }
    };

/// Configuration for the model manager.
pub const ManagerConfig = struct {
    /// Custom cache directory (overrides default).
    cache_dir: ?[]const u8 = null,
    /// Whether to scan cache directory on init.
    auto_scan: bool = true,
    /// File extensions to recognize as models.
    extensions: []const []const u8 = &.{ ".gguf", ".bin", ".safetensors" },
};

/// Metadata for a cached model.
pub const CachedModel = struct {
    /// Full path to the model file.
    path: []const u8,
    /// Model name (derived from filename without extension).
    name: []const u8,
    /// File size in bytes.
    size_bytes: u64,
    /// Model format detected from extension.
    format: discovery.ModelFormat,
    /// Quantization type if detectable.
    quantization: ?discovery.QuantizationType,
    /// Source URL if downloaded.
    source_url: ?[]const u8,
    /// Download timestamp (Unix epoch).
    downloaded_at: i64,
    /// SHA256 checksum if available.
    checksum: ?[]const u8,

    /// Free resources associated with this model entry.
    pub fn deinit(self: *CachedModel, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
        if (self.source_url) |url| allocator.free(url);
        if (self.checksum) |cs| allocator.free(cs);
    }
};

/// Model cache manager.
pub const Manager = struct {
    allocator: std.mem.Allocator,
    config: ManagerConfig,
    cache_dir: []const u8,
    models: std.ArrayListUnmanaged(CachedModel),

    const Self = @This();

    /// Initialize the model manager.
    pub fn init(allocator: std.mem.Allocator, config: ManagerConfig) !Self {
        // Determine cache directory
        const cache_dir = if (config.cache_dir) |dir|
            try allocator.dupe(u8, dir)
        else
            try getDefaultCacheDir(allocator);

        var self = Self{
            .allocator = allocator,
            .config = config,
            .cache_dir = cache_dir,
            .models = .empty,
        };

        // Scan cache directory if requested
        if (config.auto_scan) {
            self.scanCacheDir() catch {}; // Best effort
        }

        return self;
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *Self) void {
        for (self.models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.models.deinit(self.allocator);
        self.allocator.free(self.cache_dir);
    }

    /// List all cached models.
    pub fn listModels(self: *Self) []CachedModel {
        return self.models.items;
    }

    /// Get a model by name.
    pub fn getModel(self: *Self, name: []const u8) ?*CachedModel {
        for (self.models.items) |*model| {
            if (std.mem.eql(u8, model.name, name)) {
                return model;
            }
        }
        return null;
    }

    /// Get a model by path.
    pub fn getModelByPath(self: *Self, path: []const u8) ?*CachedModel {
        for (self.models.items) |*model| {
            if (std.mem.eql(u8, model.path, path)) {
                return model;
            }
        }
        return null;
    }

    /// Add a model to the cache catalog.
    /// This registers an existing file, it doesn't copy it.
    pub fn addModel(self: *Self, path: []const u8, size_bytes: u64, source_url: ?[]const u8) !*CachedModel {
        // Check if already registered
        if (self.getModelByPath(path)) |existing| {
            return existing;
        }

        // Extract name from path
        const basename = std.fs.path.basename(path);
        const name = if (std.mem.lastIndexOf(u8, basename, ".")) |dot|
            basename[0..dot]
        else
            basename;

        // Detect format and quantization
        const ext = std.fs.path.extension(path);
        const format = discovery.ModelFormat.fromExtension(ext);
        const quantization = detectQuantizationFromName(name);

        // Create model entry
        const model = CachedModel{
            .path = try self.allocator.dupe(u8, path),
            .name = try self.allocator.dupe(u8, name),
            .size_bytes = size_bytes,
            .format = format,
            .quantization = quantization,
            .source_url = if (source_url) |url| try self.allocator.dupe(u8, url) else null,
            .downloaded_at = 0, // Timestamp would require I/O backend in Zig 0.16
            .checksum = null,
        };

        try self.models.append(self.allocator, model);
        return &self.models.items[self.models.items.len - 1];
    }

    /// Remove a model from the cache (deletes the file).
    pub fn removeModel(self: *Self, name: []const u8) !void {
        for (self.models.items, 0..) |*model, i| {
            if (std.mem.eql(u8, model.name, name)) {
                // Note: File deletion would require I/O backend in Zig 0.16
                // For now, just remove from catalog
                model.deinit(self.allocator);
                _ = self.models.orderedRemove(i);
                return;
            }
        }
        return error.FileNotFound;
    }

    /// Get the current cache directory.
    pub fn getCacheDir(self: *Self) []const u8 {
        return self.cache_dir;
    }

    /// Set a new cache directory.
    pub fn setCacheDir(self: *Self, path: []const u8) !void {
        self.allocator.free(self.cache_dir);
        self.cache_dir = try self.allocator.dupe(u8, path);

        // Clear and rescan
        for (self.models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.models.clearRetainingCapacity();

        if (self.config.auto_scan) {
            self.scanCacheDir() catch {};
        }
    }

    /// Scan cache directory for models.
    /// Note: In Zig 0.16, file system operations require I/O backend.
    /// This is a placeholder that requires external initialization.
    pub fn scanCacheDir(self: *Self) !void {
        // In Zig 0.16, directory scanning requires explicit I/O backend initialization.
        // This is typically done at the CLI layer. For library use, models should be
        // registered manually via addModel().
        _ = self;
    }

    /// Scan cache directory with I/O backend (Zig 0.16 compatible).
    pub fn scanCacheDirWithIo(self: *Self, io: std.Io) !void {
        var dir = std.Io.Dir.open(io, self.cache_dir, .{ .iterate = true }) catch return;
        defer dir.close(io);

        var iter = dir.iterate();
        while (true) {
            const entry = iter.next(io) catch break;
            if (entry == null) break;
            const e = entry.?;

            if (e.kind != .file) continue;

            // Check extension
            const ext = std.fs.path.extension(e.name);
            var is_model = false;
            for (self.config.extensions) |model_ext| {
                if (std.mem.eql(u8, ext, model_ext)) {
                    is_model = true;
                    break;
                }
            }
            if (!is_model) continue;

            // Build full path
            const full_path = std.fs.path.join(self.allocator, &.{ self.cache_dir, e.name }) catch continue;
            defer self.allocator.free(full_path);

            // Get file size (would need stat call)
            const size: u64 = 0; // Placeholder - actual size requires stat

            _ = self.addModel(full_path, size, null) catch continue;
        }
    }

    /// Get model count.
    pub fn modelCount(self: *Self) usize {
        return self.models.items.len;
    }

    /// Check if a model exists by name.
    pub fn hasModel(self: *Self, name: []const u8) bool {
        return self.getModel(name) != null;
    }

    /// Get total cache size in bytes.
    pub fn totalCacheSize(self: *Self) u64 {
        var total: u64 = 0;
        for (self.models.items) |model| {
            total += model.size_bytes;
        }
        return total;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the default cache directory for the current platform.
fn getDefaultCacheDir(allocator: std.mem.Allocator) ![]const u8 {
    // Try to get from environment first
    if (getEnv("ABI_MODEL_CACHE")) |cache_env| {
        return try allocator.dupe(u8, cache_env);
    }

    // Platform-specific defaults
    if (builtin.os.tag == .windows) {
        if (getEnv("LOCALAPPDATA")) |local_app_data| {
            return try std.fmt.allocPrint(allocator, "{s}\\abi\\models", .{local_app_data});
        }
        return try allocator.dupe(u8, "C:\\abi\\models");
    } else {
        // Unix-like (Linux, macOS)
        if (getEnv("HOME")) |home| {
            return try std.fmt.allocPrint(allocator, "{s}/.abi/models", .{home});
        }
        return try allocator.dupe(u8, "/tmp/abi/models");
    }
}

/// Get environment variable value (Zig 0.16 libc pattern).
fn getEnv(name: [:0]const u8) ?[]const u8 {
    if (builtin.target.os.tag == .freestanding or
        builtin.target.cpu.arch == .wasm32 or
        builtin.target.cpu.arch == .wasm64)
    {
        return null;
    }
    const value_ptr = c.getenv(name.ptr);
    if (value_ptr) |ptr| {
        return std.mem.span(ptr);
    }
    return null;
}

/// Detect quantization type from model name.
fn detectQuantizationFromName(name: []const u8) ?discovery.QuantizationType {
    const upper = blk: {
        var buf: [256]u8 = undefined;
        const len = @min(name.len, buf.len);
        for (name[0..len], 0..) |char, i| {
            buf[i] = std.ascii.toUpper(char);
        }
        break :blk buf[0..len];
    };

    // Check for common quantization patterns
    if (std.mem.indexOf(u8, upper, "Q4_K_M")) |_| return .q4_k_m;
    if (std.mem.indexOf(u8, upper, "Q4_K_S")) |_| return .q4_k_s;
    if (std.mem.indexOf(u8, upper, "Q5_K_M")) |_| return .q5_k_m;
    if (std.mem.indexOf(u8, upper, "Q5_K_S")) |_| return .q5_k_s;
    if (std.mem.indexOf(u8, upper, "Q6_K")) |_| return .q6_k;
    if (std.mem.indexOf(u8, upper, "Q8_0")) |_| return .q8_0;
    if (std.mem.indexOf(u8, upper, "Q8_1")) |_| return .q8_1;
    if (std.mem.indexOf(u8, upper, "Q5_0")) |_| return .q5_0;
    if (std.mem.indexOf(u8, upper, "Q5_1")) |_| return .q5_1;
    if (std.mem.indexOf(u8, upper, "Q4_0")) |_| return .q4_0;
    if (std.mem.indexOf(u8, upper, "Q4_1")) |_| return .q4_1;
    if (std.mem.indexOf(u8, upper, "Q3_K_M")) |_| return .q3_k_m;
    if (std.mem.indexOf(u8, upper, "Q3_K_S")) |_| return .q3_k_s;
    if (std.mem.indexOf(u8, upper, "Q2_K")) |_| return .q2_k;
    if (std.mem.indexOf(u8, upper, "IQ2_XXS")) |_| return .iq2_xxs;
    if (std.mem.indexOf(u8, upper, "IQ2_XS")) |_| return .iq2_xs;
    if (std.mem.indexOf(u8, upper, "IQ3_XXS")) |_| return .iq3_xxs;
    if (std.mem.indexOf(u8, upper, "F16")) |_| return .f16;
    if (std.mem.indexOf(u8, upper, "F32")) |_| return .f32;

    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "manager init and deinit" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var manager = try Manager.init(std.testing.allocator, .{ .auto_scan = false });
    defer manager.deinit();

    try std.testing.expect(manager.modelCount() == 0);
}

test "add and get model" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var manager = try Manager.init(std.testing.allocator, .{ .auto_scan = false });
    defer manager.deinit();

    _ = try manager.addModel("/path/to/llama-7b.Q4_K_M.gguf", 4_000_000_000, null);

    try std.testing.expectEqual(@as(usize, 1), manager.modelCount());

    const model = manager.getModel("llama-7b.Q4_K_M");
    try std.testing.expect(model != null);
    try std.testing.expectEqual(discovery.QuantizationType.q4_k_m, model.?.quantization.?);
}

test "detect quantization from name" {
    try std.testing.expectEqual(discovery.QuantizationType.q4_k_m, detectQuantizationFromName("llama-2-7b.Q4_K_M"));
    try std.testing.expectEqual(discovery.QuantizationType.q5_k_s, detectQuantizationFromName("mistral-7b-q5_k_s"));
    try std.testing.expectEqual(discovery.QuantizationType.q8_0, detectQuantizationFromName("model-Q8_0"));
    try std.testing.expect(detectQuantizationFromName("model-unknown") == null);
}

test "total cache size" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    var manager = try Manager.init(std.testing.allocator, .{ .auto_scan = false });
    defer manager.deinit();

    _ = try manager.addModel("/path/model1.gguf", 1000, null);
    _ = try manager.addModel("/path/model2.gguf", 2000, null);

    try std.testing.expectEqual(@as(u64, 3000), manager.totalCacheSize());
}
