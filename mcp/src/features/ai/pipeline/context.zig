//! Pipeline Context
//!
//! Mutable state that flows between pipeline steps. Each step reads from
//! and writes to the context, building up the final result incrementally.
//! All owned strings are heap-allocated via allocator.dupe().

const std = @import("std");
const types = @import("types.zig");
const BlockChain = types.BlockChain;
const BlockConfig = types.BlockConfig;
const RoutingWeights = types.RoutingWeights;
const ProfileTag = types.ProfileTag;

/// Mutable context threaded through each pipeline step.
pub const PipelineContext = struct {
    allocator: std.mem.Allocator,
    /// Original user input.
    input: []const u8,
    /// Session identifier for WDBX chain and modulation.
    session_id: []const u8,
    /// Unique ID for this pipeline execution.
    pipeline_id: u64,
    /// Retrieved context fragments from WDBX.
    context_fragments: std.ArrayListUnmanaged([]const u8),
    /// Rendered prompt after template interpolation.
    rendered_prompt: ?[]const u8 = null,
    /// Routing decision weights (set by route/modulate steps).
    routing_weights: ?RoutingWeights = null,
    /// Primary profile from routing.
    primary_profile: ?ProfileTag.ProfileType = null,
    /// Generated response from LLM.
    generated_response: ?[]const u8 = null,
    /// Whether constitution validation passed.
    validation_passed: bool = true,
    /// Block IDs created during this pipeline execution.
    block_ids: std.ArrayListUnmanaged(u64),
    /// Arbitrary string metadata (step-produced key-value pairs).
    metadata: std.StringHashMapUnmanaged([]const u8),
    /// The WDBX block chain for persistence.
    chain: ?*BlockChain = null,
    /// Current step index (incremented by executor).
    current_step: u16 = 0,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        input: []const u8,
        session_id: []const u8,
        pipeline_id: u64,
    ) !Self {
        return .{
            .allocator = allocator,
            .input = try allocator.dupe(u8, input),
            .session_id = try allocator.dupe(u8, session_id),
            .pipeline_id = pipeline_id,
            .context_fragments = .empty,
            .block_ids = .empty,
            .metadata = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.input);
        self.allocator.free(self.session_id);

        for (self.context_fragments.items) |frag| {
            self.allocator.free(frag);
        }
        self.context_fragments.deinit(self.allocator);

        if (self.rendered_prompt) |p| self.allocator.free(p);
        if (self.generated_response) |r| self.allocator.free(r);

        self.block_ids.deinit(self.allocator);

        var meta_it = self.metadata.iterator();
        while (meta_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit(self.allocator);
    }

    /// Add a retrieved context fragment (takes ownership via dupe).
    pub fn addFragment(self: *Self, fragment: []const u8) !void {
        const owned = try self.allocator.dupe(u8, fragment);
        try self.context_fragments.append(self.allocator, owned);
    }

    /// Set the rendered prompt (frees previous if any).
    pub fn setPrompt(self: *Self, prompt: []const u8) !void {
        if (self.rendered_prompt) |old| self.allocator.free(old);
        self.rendered_prompt = try self.allocator.dupe(u8, prompt);
    }

    /// Set the generated response (frees previous if any).
    pub fn setResponse(self: *Self, response: []const u8) !void {
        if (self.generated_response) |old| self.allocator.free(old);
        self.generated_response = try self.allocator.dupe(u8, response);
    }

    /// Record a block ID from a WDBX write.
    pub fn recordBlock(self: *Self, block_id: u64) !void {
        try self.block_ids.append(self.allocator, block_id);
    }

    /// Set a metadata key-value pair (takes ownership via dupe).
    pub fn setMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        const owned_value = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(owned_value);

        const maybe_old = try self.metadata.fetchPut(self.allocator, owned_key, owned_value);
        if (maybe_old) |old| {
            self.allocator.free(old.key);
            self.allocator.free(old.value);
        }
    }

    /// Get all context fragments concatenated (caller owns returned slice).
    pub fn joinFragments(self: *const Self, separator: []const u8) ![]const u8 {
        if (self.context_fragments.items.len == 0) return try self.allocator.dupe(u8, "");

        var total_len: usize = 0;
        for (self.context_fragments.items, 0..) |frag, i| {
            total_len += frag.len;
            if (i < self.context_fragments.items.len - 1) total_len += separator.len;
        }

        const result = try self.allocator.alloc(u8, total_len);
        var offset: usize = 0;
        for (self.context_fragments.items, 0..) |frag, i| {
            @memcpy(result[offset..][0..frag.len], frag);
            offset += frag.len;
            if (i < self.context_fragments.items.len - 1) {
                @memcpy(result[offset..][0..separator.len], separator);
                offset += separator.len;
            }
        }
        return result;
    }

    /// Generate a simple placeholder embedding from text (Fnv1a hash).
    pub fn hashEmbedding(text: []const u8) [4]f32 {
        if (text.len == 0) return .{ 0.0, 0.0, 0.0, 0.0 };
        const hash = std.hash.Fnv1a_32.hash(text);
        const f: f32 = @floatFromInt(hash);
        return .{
            @mod(f, 1000.0) / 1000.0,
            @mod(f * 1.618, 1000.0) / 1000.0,
            @mod(f * 2.236, 1000.0) / 1000.0,
            @mod(f * 3.142, 1000.0) / 1000.0,
        };
    }
};
