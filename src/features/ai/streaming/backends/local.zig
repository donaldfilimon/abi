//! Local LLM Backend for Streaming Inference
//!
//! Provides streaming inference using local GGUF models via the LLM engine.
//! Supports quantized models (Q4, Q5, Q8) with GPU acceleration.
//!
//! This backend adapts the LLM Engine's StreamingResponse to the streaming
//! API's TokenStream interface, enabling real-time token streaming through
//! the HTTP server.

const std = @import("std");
const mod = @import("mod.zig");
const llm = @import("../../llm/mod.zig");

/// Error set for local backend operations
pub const LocalBackendError = error{
    ModelNotLoaded,
    ModelLoadFailed,
    InferenceError,
    TokenizationFailed,
    OutOfMemory,
};

/// Local backend for GGUF model inference
///
/// Wraps the LLM Engine to provide streaming inference through the
/// backend router interface. Supports both owned and external Engine
/// references for flexible lifecycle management.
pub const LocalBackend = struct {
    allocator: std.mem.Allocator,
    engine: ?*llm.Engine,
    owns_engine: bool,
    model_loaded: bool,
    model_name: []const u8,
    model_path: ?[]const u8,
    inference_config: llm.InferenceConfig,

    const Self = @This();

    /// Initialize a new LocalBackend that creates and owns its own Engine
    pub fn init(allocator: std.mem.Allocator) !Self {
        // Create owned engine
        const engine = try allocator.create(llm.Engine);
        engine.* = llm.Engine.init(allocator, .{});

        return .{
            .allocator = allocator,
            .engine = engine,
            .owns_engine = true,
            .model_loaded = false,
            .model_name = "local-gguf",
            .model_path = null,
            .inference_config = .{},
        };
    }

    /// Initialize with an external Engine reference (for testing or shared engines)
    pub fn initWithEngine(allocator: std.mem.Allocator, engine: *llm.Engine) Self {
        return .{
            .allocator = allocator,
            .engine = engine,
            .owns_engine = false,
            .model_loaded = engine.loaded_model != null,
            .model_name = "local-gguf",
            .model_path = null,
            .inference_config = engine.config,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.owns_engine) {
            if (self.engine) |engine| {
                engine.deinit();
                self.allocator.destroy(engine);
            }
        }
        if (self.model_path) |path| {
            self.allocator.free(path);
        }
        self.* = undefined;
    }

    /// Load a GGUF model from the specified path
    ///
    /// This loads the model into the Engine, making it ready for inference.
    /// For large models, this may take several seconds.
    pub fn loadModel(self: *Self, path: []const u8) !void {
        const engine = self.engine orelse return LocalBackendError.ModelNotLoaded;

        // Store path for reference - dupe first since path may equal self.model_path
        const path_copy = try self.allocator.dupe(u8, path);

        if (self.model_path) |old_path| {
            self.allocator.free(old_path);
        }
        self.model_path = path_copy;
        {
            errdefer {
                self.model_path = null;
                self.allocator.free(path_copy);
            }
            // Load model into engine - map any error to ModelLoadFailed
            engine.loadModel(path_copy) catch {
                return LocalBackendError.ModelLoadFailed;
            };
        }

        self.model_loaded = true;
    }

    /// Start a streaming inference session
    ///
    /// Creates a LocalStreamState that produces tokens from the LLM Engine.
    /// If no model is loaded and a model_path is configured, attempts lazy loading.
    pub fn startStream(
        self: *Self,
        prompt: []const u8,
        config: mod.GenerationConfig,
    ) !LocalStreamState {
        const engine = self.engine orelse return LocalBackendError.ModelNotLoaded;

        // Lazy load model if path is set but model not loaded
        if (!self.model_loaded) {
            if (self.model_path) |path| {
                try self.loadModel(path);
            } else {
                return LocalBackendError.ModelNotLoaded;
            }
        }

        // Convert GenerationConfig to StreamingConfig
        const stream_config = llm.StreamingConfig{
            .max_tokens = config.max_tokens,
            .temperature = config.temperature,
            .top_k = config.top_k,
            .top_p = config.top_p,
        };

        // Create streaming response from engine
        const streaming_response = engine.createStreamingResponse(prompt, stream_config) catch |err| {
            return switch (err) {
                error.OutOfMemory => LocalBackendError.OutOfMemory,
                else => LocalBackendError.InferenceError,
            };
        };

        return LocalStreamState{
            .allocator = self.allocator,
            .streaming_response = streaming_response,
            .tokens_generated = 0,
            .is_complete = false,
        };
    }

    /// Generate complete response (non-streaming)
    pub fn generate(
        self: *Self,
        prompt: []const u8,
        config: mod.GenerationConfig,
    ) ![]u8 {
        const engine = self.engine orelse return LocalBackendError.ModelNotLoaded;

        // Lazy load model if path is set but model not loaded
        if (!self.model_loaded) {
            if (self.model_path) |path| {
                try self.loadModel(path);
            } else {
                return LocalBackendError.ModelNotLoaded;
            }
        }

        engine.config.max_new_tokens = config.max_tokens;
        engine.config.temperature = config.temperature;
        engine.config.top_p = config.top_p;
        engine.config.top_k = config.top_k;
        engine.config.repetition_penalty = config.repetition_penalty;

        return engine.generate(self.allocator, prompt) catch |err| switch (err) {
            error.OutOfMemory => LocalBackendError.OutOfMemory,
            error.TokenizationFailed => LocalBackendError.TokenizationFailed,
            else => LocalBackendError.InferenceError,
        };
    }

    /// Check if backend is available (has engine and optionally model)
    pub fn isAvailable(self: *Self) bool {
        return self.engine != null;
    }

    /// Check if a model is loaded and ready for inference
    pub fn isModelLoaded(self: *Self) bool {
        return self.model_loaded;
    }

    /// Get model information
    pub fn getModelInfo(self: Self) mod.ModelInfo {
        const supports_streaming = if (self.engine) |engine| engine.supportsStreaming() else true;
        return .{
            .name = self.model_name,
            .backend = .local,
            .max_tokens = self.inference_config.max_new_tokens,
            .supports_streaming = supports_streaming,
        };
    }

    /// Update inference configuration
    pub fn setConfig(self: *Self, config: llm.InferenceConfig) void {
        self.inference_config = config;
        if (self.engine) |engine| {
            engine.config = config;
        }
    }
};

/// State for local streaming inference
///
/// Wraps the LLM Engine's StreamingResponse and adapts its TokenEvent
/// to the streaming API's StreamToken format.
pub const LocalStreamState = struct {
    allocator: std.mem.Allocator,
    streaming_response: llm.StreamingResponse,
    tokens_generated: usize,
    is_complete: bool,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.streaming_response.deinit();
        self.* = undefined;
    }

    /// Get next token from the stream
    ///
    /// Returns null when the stream is exhausted or the generation is complete.
    /// The returned token text is allocated and must be freed by the caller.
    pub fn next(self: *Self, allocator: std.mem.Allocator) !?mod.StreamToken {
        if (self.is_complete) return null;

        // Get next token event from the LLM streaming response
        const event = self.streaming_response.next() catch |err| {
            self.is_complete = true;
            return switch (err) {
                error.OutOfMemory => LocalBackendError.OutOfMemory,
                error.Cancelled => null,
                else => LocalBackendError.InferenceError,
            };
        };

        if (event) |token_event| {
            self.tokens_generated += 1;

            if (token_event.is_final) {
                self.is_complete = true;
            }

            // Adapt TokenEvent to StreamToken
            return .{
                .text = if (token_event.text) |text|
                    try allocator.dupe(u8, text)
                else
                    try allocator.dupe(u8, ""),
                .id = token_event.token_id,
                .log_prob = null,
                .is_end = token_event.is_final,
                .index = self.tokens_generated - 1,
            };
        } else {
            // Stream exhausted
            self.is_complete = true;
            return null;
        }
    }

    /// Check if the stream is complete
    pub fn isComplete(self: *const Self) bool {
        return self.is_complete;
    }

    /// Get the number of tokens generated so far
    pub fn getTokenCount(self: *const Self) usize {
        return self.tokens_generated;
    }
};

// Tests
test "local backend initialization" {
    const allocator = std.testing.allocator;

    var backend = try LocalBackend.init(allocator);
    defer backend.deinit();

    try std.testing.expect(backend.isAvailable());
    try std.testing.expect(!backend.isModelLoaded());
    const info = backend.getModelInfo();
    try std.testing.expectEqualStrings("local-gguf", info.name);
    try std.testing.expect(info.supports_streaming);
}

test "local backend with external engine" {
    const allocator = std.testing.allocator;

    // Create external engine
    var engine = llm.Engine.init(allocator, .{});
    defer engine.deinit();

    // Initialize backend with external engine
    var backend = LocalBackend.initWithEngine(allocator, &engine);
    defer backend.deinit();

    try std.testing.expect(backend.isAvailable());
    try std.testing.expect(!backend.owns_engine);
}

test "local backend model path storage" {
    const allocator = std.testing.allocator;

    var backend = try LocalBackend.init(allocator);
    defer backend.deinit();

    // Store a path (won't actually load since model doesn't exist)
    if (backend.model_path) |old_path| {
        allocator.free(old_path);
    }
    backend.model_path = try allocator.dupe(u8, "/test/model.gguf");

    try std.testing.expectEqualStrings("/test/model.gguf", backend.model_path.?);
}

test "local stream state completion" {
    // This test verifies the state machine without requiring an actual model
    const allocator = std.testing.allocator;

    var backend = try LocalBackend.init(allocator);
    defer backend.deinit();

    // Without a model loaded, startStream should fail
    const result = backend.startStream("test", .{});
    try std.testing.expectError(LocalBackendError.ModelNotLoaded, result);
}
