//! Backend Router for Streaming Inference
//!
//! Routes streaming requests to appropriate backends:
//! - Local: GGUF model inference via LLM engine
//! - OpenAI: OpenAI API streaming
//! - Ollama: Local Ollama server
//! - Anthropic: Claude API streaming

const std = @import("std");
pub const local = @import("local.zig");
pub const external = @import("external.zig");

/// Available backend types
pub const BackendType = enum {
    local,
    openai,
    ollama,
    anthropic,

    pub fn fromString(s: []const u8) ?BackendType {
        const map = std.StaticStringMap(BackendType).initComptime(.{
            .{ "local", .local },
            .{ "openai", .openai },
            .{ "ollama", .ollama },
            .{ "anthropic", .anthropic },
        });
        return map.get(s);
    }

    pub fn toString(self: BackendType) []const u8 {
        return switch (self) {
            .local => "local",
            .openai => "openai",
            .ollama => "ollama",
            .anthropic => "anthropic",
        };
    }
};

/// Generation configuration
pub const GenerationConfig = struct {
    /// Maximum tokens to generate
    max_tokens: u32 = 1024,
    /// Sampling temperature (0.0 = deterministic, 1.0+ = creative)
    temperature: f32 = 0.7,
    /// Top-p nucleus sampling
    top_p: f32 = 0.9,
    /// Top-k sampling (0 = disabled)
    top_k: u32 = 0,
    /// Repetition penalty
    repetition_penalty: f32 = 1.0,
    /// Stop sequences
    stop_sequences: ?[]const []const u8 = null,
    /// Model name/ID (backend-specific)
    model: ?[]const u8 = null,
};

/// Token from streaming generation
pub const StreamToken = struct {
    /// Token text
    text: []const u8,
    /// Token ID (if available)
    id: ?u32 = null,
    /// Log probability (if available)
    log_prob: ?f32 = null,
    /// Is this the final token?
    is_end: bool = false,
    /// Token index in sequence
    index: usize = 0,
};

/// Token stream iterator
pub const TokenStream = struct {
    allocator: std.mem.Allocator,
    backend_type: BackendType,
    state: StreamState,
    buffer: std.ArrayListUnmanaged(u8),
    token_index: usize,

    const StreamState = union(enum) {
        local: local.LocalStreamState,
        external: external.ExternalStreamState,
    };

    pub fn init(allocator: std.mem.Allocator, backend_type: BackendType) TokenStream {
        return .{
            .allocator = allocator,
            .backend_type = backend_type,
            .state = undefined,
            .buffer = .empty,
            .token_index = 0,
        };
    }

    pub fn deinit(self: *TokenStream) void {
        self.buffer.deinit(self.allocator);
        switch (self.state) {
            .local => |*s| s.deinit(),
            .external => |*s| s.deinit(),
        }
        self.* = undefined;
    }

    /// Get next token from stream
    pub fn next(self: *TokenStream) !?StreamToken {
        var result = switch (self.state) {
            .local => |*s| try s.next(self.allocator),
            .external => |*s| try s.next(self.allocator),
        };

        if (result != null) {
            result.?.index = self.token_index;
            self.token_index += 1;
        }

        return result;
    }
};

/// Backend interface for streaming inference
pub const Backend = struct {
    allocator: std.mem.Allocator,
    backend_type: BackendType,
    impl: BackendImpl,

    const BackendImpl = union(enum) {
        local: local.LocalBackend,
        external: external.ExternalBackend,
    };

    pub fn init(allocator: std.mem.Allocator, backend_type: BackendType) !Backend {
        const impl: BackendImpl = switch (backend_type) {
            .local => .{ .local = try local.LocalBackend.init(allocator) },
            .openai => .{ .external = try external.ExternalBackend.init(allocator, .openai) },
            .ollama => .{ .external = try external.ExternalBackend.init(allocator, .ollama) },
            .anthropic => .{ .external = try external.ExternalBackend.init(allocator, .anthropic) },
        };

        return .{
            .allocator = allocator,
            .backend_type = backend_type,
            .impl = impl,
        };
    }

    pub fn deinit(self: *Backend) void {
        switch (self.impl) {
            .local => |*b| b.deinit(),
            .external => |*b| b.deinit(),
        }
        self.* = undefined;
    }

    /// Stream tokens from prompt
    pub fn streamTokens(self: *Backend, prompt: []const u8, config: GenerationConfig) !TokenStream {
        var stream = TokenStream.init(self.allocator, self.backend_type);
        errdefer stream.deinit();

        stream.state = switch (self.impl) {
            .local => |*b| .{ .local = try b.startStream(prompt, config) },
            .external => |*b| .{ .external = try b.startStream(prompt, config) },
        };

        return stream;
    }

    /// Generate complete response (non-streaming)
    pub fn generate(self: *Backend, prompt: []const u8, config: GenerationConfig) ![]u8 {
        return switch (self.impl) {
            .local => |*b| try b.generate(prompt, config),
            .external => |*b| try b.generate(prompt, config),
        };
    }

    /// Check if backend is available
    pub fn isAvailable(self: *Backend) bool {
        return switch (self.impl) {
            .local => |*b| b.isAvailable(),
            .external => |*b| b.isAvailable(),
        };
    }

    /// Get model information
    pub fn getModelInfo(self: *Backend) ModelInfo {
        return switch (self.impl) {
            .local => |b| b.getModelInfo(),
            .external => |b| b.getModelInfo(),
        };
    }
};

/// Model information
pub const ModelInfo = struct {
    name: []const u8,
    backend: BackendType,
    max_tokens: u32,
    supports_streaming: bool,
};

/// Backend router for managing multiple backends
pub const BackendRouter = struct {
    allocator: std.mem.Allocator,
    backends: std.EnumArray(BackendType, ?Backend),
    default_backend: BackendType,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        var backends = std.EnumArray(BackendType, ?Backend).initFill(null);

        // Initialize local backend by default
        backends.set(.local, try Backend.init(allocator, .local));

        return .{
            .allocator = allocator,
            .backends = backends,
            .default_backend = .local,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.backends.iterator();
        while (iter.next()) |entry| {
            if (entry.value.*) |*backend| {
                backend.deinit();
            }
        }
        self.* = undefined;
    }

    /// Get backend by type (lazy initialization)
    pub fn getBackend(self: *Self, backend_type: BackendType) !*Backend {
        if (self.backends.getPtr(backend_type).*) |*backend| {
            return backend;
        }

        // Initialize on demand
        self.backends.set(backend_type, try Backend.init(self.allocator, backend_type));
        return &self.backends.getPtr(backend_type).*.?;
    }

    /// Get default backend
    pub fn getDefaultBackend(self: *Self) !*Backend {
        return self.getBackend(self.default_backend);
    }

    /// Set default backend
    pub fn setDefaultBackend(self: *Self, backend_type: BackendType) void {
        self.default_backend = backend_type;
    }

    /// List available models as JSON
    pub fn listModelsJson(self: *Self, allocator: std.mem.Allocator) ![]u8 {
        var json = std.ArrayListUnmanaged(u8).empty;
        errdefer json.deinit(allocator);

        try json.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");

        var first = true;
        var iter = self.backends.iterator();
        while (iter.next()) |entry| {
            if (entry.value.*) |*backend| {
                if (!first) try json.append(allocator, ',');
                first = false;

                const info = backend.getModelInfo();
                try json.appendSlice(allocator, "{\"id\":\"");
                try json.appendSlice(allocator, info.name);
                try json.appendSlice(allocator, "\",\"object\":\"model\",\"owned_by\":\"");
                try json.appendSlice(allocator, info.backend.toString());
                try json.appendSlice(allocator, "\"}");
            }
        }

        try json.appendSlice(allocator, "]}");
        return json.toOwnedSlice(allocator);
    }
};

// Tests
test "backend type from string" {
    try std.testing.expectEqual(BackendType.local, BackendType.fromString("local").?);
    try std.testing.expectEqual(BackendType.openai, BackendType.fromString("openai").?);
    try std.testing.expect(BackendType.fromString("invalid") == null);
}

test "generation config defaults" {
    const config = GenerationConfig{};
    try std.testing.expectEqual(@as(u32, 1024), config.max_tokens);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), config.temperature, 0.001);
}

test "backend router initialization" {
    const allocator = std.testing.allocator;

    var router = try BackendRouter.init(allocator);
    defer router.deinit();

    const backend = try router.getBackend(.local);
    try std.testing.expectEqual(BackendType.local, backend.backend_type);
}
