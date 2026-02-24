//! HuggingFace Integration - Search and download models from HuggingFace Hub
//!
//! Provides functionality to search for models, list available files,
//! and resolve download URLs for GGUF models on HuggingFace.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const web_enabled = @hasDecl(build_options, "enable_web") and build_options.enable_web;
const web_client = if (web_enabled) @import("../../web/client.zig") else @as(?void, null);

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

/// HuggingFace model information.
pub const HuggingFaceModel = struct {
    /// Model ID (e.g., "TheBloke/Llama-2-7B-GGUF").
    id: []const u8,
    /// Model author/organization.
    author: []const u8,
    /// Model name (without author).
    model_name: []const u8,
    /// Number of downloads.
    downloads: u64,
    /// Number of likes.
    likes: u64,
    /// Available files in the repository.
    files: []HuggingFaceFile,
    /// Tags associated with the model.
    tags: []const []const u8,

    /// Free resources.
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

/// Information about a file in a HuggingFace repository.
pub const HuggingFaceFile = struct {
    /// Filename.
    name: []const u8,
    /// File size in bytes.
    size_bytes: u64,
    /// Detected quantization type (for GGUF files).
    quantization: ?[]const u8,
    /// SHA256 hash if available.
    sha256: ?[]const u8,

    /// Free resources.
    pub fn deinit(self: *HuggingFaceFile, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.quantization) |q| allocator.free(q);
        if (self.sha256) |h| allocator.free(h);
    }
};

/// Search result containing multiple models.
pub const SearchResult = struct {
    /// Found models.
    models: []HuggingFaceModel,
    /// Total count of matching models.
    total_count: u64,
    /// Search query used.
    query: []const u8,

    /// Free resources.
    pub fn deinit(self: *SearchResult, allocator: std.mem.Allocator) void {
        for (self.models) |*model| {
            model.deinit(allocator);
        }
        allocator.free(self.models);
        allocator.free(self.query);
    }
};

/// HuggingFace API client error types.
pub const HuggingFaceError = error{
    /// API request failed.
    ApiError,
    /// Model not found.
    ModelNotFound,
    /// File not found in model repository.
    FileNotFound,
    /// Invalid model identifier format.
    InvalidModelId,
    /// Authentication failed.
    AuthenticationFailed,
    /// Rate limited.
    RateLimited,
    /// Network error.
    NetworkError,
    /// JSON parsing error.
    ParseError,
    /// Out of memory.
    OutOfMemory,
};

/// HuggingFace API client.
pub const HuggingFaceClient = struct {
    allocator: std.mem.Allocator,
    api_token: ?[]const u8,
    base_url: []const u8,

    const Self = @This();
    const DEFAULT_BASE_URL = "https://huggingface.co";
    const API_BASE_URL = "https://huggingface.co/api";

    /// Initialize the HuggingFace client.
    pub fn init(allocator: std.mem.Allocator, api_token: ?[]const u8) Self {
        // Try to get token from environment if not provided
        const token = api_token orelse getEnv("ABI_HF_API_TOKEN") orelse
            getEnv("HF_API_TOKEN") orelse
            getEnv("HUGGING_FACE_HUB_TOKEN");

        return .{
            .allocator = allocator,
            .api_token = if (token) |t| allocator.dupe(u8, t) catch null else null,
            .base_url = DEFAULT_BASE_URL,
        };
    }

    /// Deinitialize the client.
    pub fn deinit(self: *Self) void {
        if (self.api_token) |token| {
            self.allocator.free(token);
        }
    }

    /// Search for models on HuggingFace.
    ///
    /// Searches for models matching the query. Results can be filtered by:
    /// - Model type (gguf, safetensors, etc.)
    /// - Author/organization
    /// - Tags
    pub fn search(self: *Self, query: []const u8, options: SearchOptions) HuggingFaceError!SearchResult {
        if (web_enabled) {
            // Build API URL with optional filters
            const gguf_filter: []const u8 = if (options.filter_gguf) "&filter=gguf" else "";

            const author_suffix = if (options.author) |author|
                std.fmt.allocPrint(self.allocator, "&author={s}", .{author}) catch
                    return HuggingFaceError.OutOfMemory
            else
                self.allocator.dupe(u8, "") catch return HuggingFaceError.OutOfMemory;
            defer self.allocator.free(author_suffix);

            const sort_param: []const u8 = if (options.sort_by_downloads) "&sort=downloads&direction=-1" else "";

            const url = std.fmt.allocPrint(
                self.allocator,
                "{s}/api/models?search={s}&limit={d}{s}{s}{s}",
                .{ self.base_url, query, options.limit, gguf_filter, author_suffix, sort_param },
            ) catch return HuggingFaceError.OutOfMemory;
            defer self.allocator.free(url);

            var http_client = web_client.HttpClient.init(self.allocator) catch
                return HuggingFaceError.NetworkError;
            defer http_client.deinit();

            const response = http_client.get(url) catch
                return HuggingFaceError.NetworkError;
            defer http_client.freeResponse(response);

            if (response.status != 200) return HuggingFaceError.ApiError;

            // The API returns a JSON array of model objects.
            // Full JSON parsing would populate SearchResult.models;
            // for now return the count heuristic from the response body.
            return SearchResult{
                .models = &.{},
                .total_count = if (response.body.len > 2) 1 else 0,
                .query = self.allocator.dupe(u8, query) catch return HuggingFaceError.OutOfMemory,
            };
        }

        // Fallback when web module is disabled
        return SearchResult{
            .models = &.{},
            .total_count = 0,
            .query = self.allocator.dupe(u8, query) catch return HuggingFaceError.OutOfMemory,
        };
    }

    /// Get detailed information about a specific model.
    pub fn getModel(self: *Self, model_id: []const u8) HuggingFaceError!HuggingFaceModel {
        // Validate model ID format (should be "author/model-name")
        if (std.mem.indexOf(u8, model_id, "/") == null) {
            return error.InvalidModelId;
        }

        // Note: Full implementation would make HTTP request to:
        // https://huggingface.co/api/models/<model_id>
        //
        // And list files from:
        // https://huggingface.co/api/models/<model_id>/tree/main

        _ = self;
        return error.ModelNotFound;
    }

    /// Resolve a download URL for a specific file in a model repository.
    ///
    /// Converts a model ID and filename into a direct download URL.
    /// Format: https://huggingface.co/<model_id>/resolve/main/<filename>
    pub fn resolveDownloadUrl(self: *Self, model_id: []const u8, filename: []const u8) ![]const u8 {
        // Validate inputs
        if (model_id.len == 0 or filename.len == 0) {
            return error.InvalidModelId;
        }

        // Build URL: https://huggingface.co/<model_id>/resolve/main/<filename>
        return try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}/resolve/main/{s}",
            .{ self.base_url, model_id, filename },
        );
    }

    /// Parse a model identifier with optional file specification.
    ///
    /// Supports formats:
    /// - "TheBloke/Llama-2-7B-GGUF" - Model ID only
    /// - "TheBloke/Llama-2-7B-GGUF:Q4_K_M" - Model ID with quantization hint
    /// - "TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf" - Model ID with filename
    pub fn parseModelSpec(spec: []const u8) ModelSpec {
        // Check for filename format (contains ".gguf" or similar)
        if (std.mem.indexOf(u8, spec, ".gguf")) |_| {
            // Has explicit filename - find the model ID
            const last_slash = std.mem.lastIndexOf(u8, spec, "/");
            if (last_slash) |idx| {
                // Check if this is model_id/filename or author/model/filename
                const before_last = spec[0..idx];
                if (std.mem.indexOf(u8, before_last, "/")) |_| {
                    // author/model/filename format
                    return .{
                        .model_id = before_last,
                        .filename = spec[idx + 1 ..],
                        .quantization_hint = null,
                    };
                }
            }
        }

        // Check for quantization hint format (contains ":")
        if (std.mem.indexOf(u8, spec, ":")) |colon_idx| {
            return .{
                .model_id = spec[0..colon_idx],
                .filename = null,
                .quantization_hint = spec[colon_idx + 1 ..],
            };
        }

        // Plain model ID
        return .{
            .model_id = spec,
            .filename = null,
            .quantization_hint = null,
        };
    }

    /// Build a filename from a quantization hint.
    ///
    /// Given a model ID and quantization type, constructs the expected filename.
    /// Example: "TheBloke/Llama-2-7B-GGUF" + "Q4_K_M" -> "llama-2-7b.Q4_K_M.gguf"
    pub fn buildFilenameFromHint(self: *Self, model_id: []const u8, quant_hint: []const u8) ![]const u8 {
        // Extract model name from ID
        const slash_idx = std.mem.indexOf(u8, model_id, "/") orelse return error.InvalidModelId;
        var model_name = model_id[slash_idx + 1 ..];

        // Remove common suffixes like "-GGUF"
        if (std.mem.endsWith(u8, model_name, "-GGUF")) {
            model_name = model_name[0 .. model_name.len - 5];
        }

        // Convert to lowercase and build filename
        var lower_name = try self.allocator.alloc(u8, model_name.len);
        defer self.allocator.free(lower_name);

        for (model_name, 0..) |char, i| {
            lower_name[i] = std.ascii.toLower(char);
        }

        return try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}.gguf",
            .{ lower_name, quant_hint },
        );
    }

    /// List popular GGUF model authors/organizations.
    pub fn getPopularAuthors() []const []const u8 {
        return &.{
            "TheBloke",
            "bartowski",
            "mradermacher",
            "MaziyarPanahi",
            "lmstudio-community",
            "QuantFactory",
        };
    }

    /// Get common quantization types and their descriptions.
    pub fn getQuantizationInfo() []const QuantInfo {
        return &.{
            .{ .name = "Q4_K_M", .bits = 4.5, .desc = "Medium quality, good balance of size and quality" },
            .{ .name = "Q4_K_S", .bits = 4.3, .desc = "Small, slightly lower quality than Q4_K_M" },
            .{ .name = "Q5_K_M", .bits = 5.5, .desc = "Higher quality, larger size" },
            .{ .name = "Q5_K_S", .bits = 5.3, .desc = "Medium-high quality" },
            .{ .name = "Q6_K", .bits = 6.5, .desc = "Very high quality, large size" },
            .{ .name = "Q8_0", .bits = 8.0, .desc = "Near-lossless quality, largest size" },
            .{ .name = "Q3_K_M", .bits = 3.5, .desc = "Lower quality, small size" },
            .{ .name = "Q2_K", .bits = 2.5, .desc = "Lowest quality, smallest size" },
            .{ .name = "IQ4_XS", .bits = 4.0, .desc = "Importance-weighted 4-bit" },
            .{ .name = "IQ3_XXS", .bits = 3.0, .desc = "Importance-weighted 3-bit" },
        };
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

/// Search options for model search.
pub const SearchOptions = struct {
    /// Maximum results to return.
    limit: u32 = 20,
    /// Filter by file type.
    filter_gguf: bool = true,
    /// Filter by author.
    author: ?[]const u8 = null,
    /// Sort by downloads.
    sort_by_downloads: bool = true,
};

/// Quantization information.
pub const QuantInfo = struct {
    name: []const u8,
    bits: f32,
    desc: []const u8,
};

// ============================================================================
// Helper Functions
// ============================================================================

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

// ============================================================================
// Tests
// ============================================================================

test "client init and deinit" {
    var client = HuggingFaceClient.init(std.testing.allocator, null);
    defer client.deinit();

    try std.testing.expectEqualStrings("https://huggingface.co", client.base_url);
}

test "parseModelSpec plain id" {
    const spec = HuggingFaceClient.parseModelSpec("TheBloke/Llama-2-7B-GGUF");
    try std.testing.expectEqualStrings("TheBloke/Llama-2-7B-GGUF", spec.model_id);
    try std.testing.expect(spec.filename == null);
    try std.testing.expect(spec.quantization_hint == null);
}

test "parseModelSpec with quantization" {
    const spec = HuggingFaceClient.parseModelSpec("TheBloke/Llama-2-7B-GGUF:Q4_K_M");
    try std.testing.expectEqualStrings("TheBloke/Llama-2-7B-GGUF", spec.model_id);
    try std.testing.expect(spec.filename == null);
    try std.testing.expectEqualStrings("Q4_K_M", spec.quantization_hint.?);
}

test "parseModelSpec with filename" {
    const spec = HuggingFaceClient.parseModelSpec("TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf");
    try std.testing.expectEqualStrings("TheBloke/Llama-2-7B-GGUF", spec.model_id);
    try std.testing.expectEqualStrings("llama-2-7b.Q4_K_M.gguf", spec.filename.?);
}

test "resolveDownloadUrl" {
    var client = HuggingFaceClient.init(std.testing.allocator, null);
    defer client.deinit();

    const url = try client.resolveDownloadUrl("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
    defer std.testing.allocator.free(url);

    try std.testing.expectEqualStrings(
        "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
        url,
    );
}

test "getPopularAuthors" {
    const authors = HuggingFaceClient.getPopularAuthors();
    try std.testing.expect(authors.len > 0);
    try std.testing.expectEqualStrings("TheBloke", authors[0]);
}

test "getQuantizationInfo" {
    const quants = HuggingFaceClient.getQuantizationInfo();
    try std.testing.expect(quants.len > 0);
    try std.testing.expectEqualStrings("Q4_K_M", quants[0].name);
}

test {
    std.testing.refAllDecls(@This());
}
