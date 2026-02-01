//! Trainable layer weights.

const std = @import("std");
const config_mod = @import("config.zig");
const utils = @import("utils.zig");

/// Trainable weights for a single transformer layer.
pub const TrainableLayerWeights = struct {
    allocator: std.mem.Allocator,

    // Attention weights
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,
    attn_norm: []f32,

    // FFN weights
    w_gate: []f32,
    w_up: []f32,
    w_down: []f32,
    ffn_norm: []f32,

    // Gradients
    d_w_q: []f32,
    d_w_k: []f32,
    d_w_v: []f32,
    d_w_o: []f32,
    d_attn_norm: []f32,
    d_w_gate: []f32,
    d_w_up: []f32,
    d_w_down: []f32,
    d_ffn_norm: []f32,

    pub fn init(allocator: std.mem.Allocator, config: config_mod.TrainableModelConfig) !TrainableLayerWeights {
        const head_dim = config.headDim();
        const kv_dim = config.num_kv_heads * head_dim;

        const w_q = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(w_q);
        const w_k = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(w_k);
        const w_v = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(w_v);
        const w_o = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(w_o);
        const attn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(attn_norm);

        const w_gate = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(w_gate);
        const w_up = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(w_up);
        const w_down = try allocator.alloc(f32, config.hidden_dim * config.intermediate_dim);
        errdefer allocator.free(w_down);
        const ffn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(ffn_norm);

        // Allocate gradients
        const d_w_q = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(d_w_q);
        const d_w_k = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(d_w_k);
        const d_w_v = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(d_w_v);
        const d_w_o = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(d_w_o);
        const d_attn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(d_attn_norm);

        const d_w_gate = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(d_w_gate);
        const d_w_up = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(d_w_up);
        const d_w_down = try allocator.alloc(f32, config.hidden_dim * config.intermediate_dim);
        errdefer allocator.free(d_w_down);
        const d_ffn_norm = try allocator.alloc(f32, config.hidden_dim);

        // Initialize weights
        utils.initializeXavier(w_q);
        utils.initializeXavier(w_k);
        utils.initializeXavier(w_v);
        utils.initializeXavier(w_o);
        @memset(attn_norm, 1.0);

        utils.initializeXavier(w_gate);
        utils.initializeXavier(w_up);
        utils.initializeXavier(w_down);
        @memset(ffn_norm, 1.0);

        // Zero gradients
        @memset(d_w_q, 0);
        @memset(d_w_k, 0);
        @memset(d_w_v, 0);
        @memset(d_w_o, 0);
        @memset(d_attn_norm, 0);
        @memset(d_w_gate, 0);
        @memset(d_w_up, 0);
        @memset(d_w_down, 0);
        @memset(d_ffn_norm, 0);

        return .{
            .allocator = allocator,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .attn_norm = attn_norm,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .ffn_norm = ffn_norm,
            .d_w_q = d_w_q,
            .d_w_k = d_w_k,
            .d_w_v = d_w_v,
            .d_w_o = d_w_o,
            .d_attn_norm = d_attn_norm,
            .d_w_gate = d_w_gate,
            .d_w_up = d_w_up,
            .d_w_down = d_w_down,
            .d_ffn_norm = d_ffn_norm,
        };
    }

    pub fn deinit(self: *TrainableLayerWeights) void {
        self.allocator.free(self.d_ffn_norm);
        self.allocator.free(self.d_w_down);
        self.allocator.free(self.d_w_up);
        self.allocator.free(self.d_w_gate);
        self.allocator.free(self.d_attn_norm);
        self.allocator.free(self.d_w_o);
        self.allocator.free(self.d_w_v);
        self.allocator.free(self.d_w_k);
        self.allocator.free(self.d_w_q);

        self.allocator.free(self.ffn_norm);
        self.allocator.free(self.w_down);
        self.allocator.free(self.w_up);
        self.allocator.free(self.w_gate);
        self.allocator.free(self.attn_norm);
        self.allocator.free(self.w_o);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_q);
        self.* = undefined;
    }

    pub fn zeroGradients(self: *TrainableLayerWeights) void {
        @memset(self.d_w_q, 0);
        @memset(self.d_w_k, 0);
        @memset(self.d_w_v, 0);
        @memset(self.d_w_o, 0);
        @memset(self.d_attn_norm, 0);
        @memset(self.d_w_gate, 0);
        @memset(self.d_w_up, 0);
        @memset(self.d_w_down, 0);
        @memset(self.d_ffn_norm, 0);
    }
};
