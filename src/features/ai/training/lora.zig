//! LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
//!
//! LoRA decomposes weight updates as: W' = W + BA
//! where B: [r, in] and A: [out, r], with r << min(in, out)
//!
//! This dramatically reduces trainable parameters while maintaining
//! model quality for fine-tuning tasks.
//!
//! Reference: https://arxiv.org/abs/2106.09685

const std = @import("std");

/// LoRA configuration.
pub const LoraConfig = struct {
    /// Rank of the low-rank decomposition
    rank: u32 = 8,
    /// Scaling factor (alpha / rank)
    alpha: f32 = 16.0,
    /// Dropout probability (0 = disabled)
    dropout: f32 = 0.0,
    /// Target modules (which weights to apply LoRA to)
    target_modules: TargetModules = .{},
    /// Whether to merge weights after training
    merge_weights: bool = true,
    /// Initialization scale for A matrix
    init_scale: f32 = 1.0,

    pub const TargetModules = struct {
        /// Apply to query projection
        q_proj: bool = true,
        /// Apply to key projection
        k_proj: bool = false,
        /// Apply to value projection
        v_proj: bool = true,
        /// Apply to output projection
        o_proj: bool = false,
        /// Apply to gate projection (FFN)
        gate_proj: bool = false,
        /// Apply to up projection (FFN)
        up_proj: bool = false,
        /// Apply to down projection (FFN)
        down_proj: bool = false,
    };

    /// Compute scaling factor.
    pub fn getScaling(self: LoraConfig) f32 {
        return self.alpha / @as(f32, @floatFromInt(self.rank));
    }
};

/// LoRA adapter for a single weight matrix.
/// Implements W' = W + scaling * B @ A
pub const LoraAdapter = struct {
    allocator: std.mem.Allocator,
    /// A matrix: [out_features, rank] - initialized to zero
    a: []f32,
    /// B matrix: [rank, in_features] - initialized from normal distribution
    b: []f32,
    /// Gradient for A
    d_a: []f32,
    /// Gradient for B
    d_b: []f32,
    /// Dimensions
    in_features: u32,
    out_features: u32,
    rank: u32,
    /// Scaling factor
    scaling: f32,
    /// Whether adapter is active
    active: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        in_features: u32,
        out_features: u32,
        config: LoraConfig,
    ) !LoraAdapter {
        const rank = config.rank;

        // A: [out_features, rank]
        const a = try allocator.alloc(f32, @as(usize, out_features) * rank);
        errdefer allocator.free(a);
        @memset(a, 0); // Initialize A to zero

        // B: [rank, in_features]
        const b = try allocator.alloc(f32, @as(usize, rank) * in_features);
        errdefer allocator.free(b);

        // Initialize B with small random values (Kaiming init)
        var rng = std.Random.DefaultPrng.init(@as(u64, in_features) *% @as(u64, out_features));
        const std_dev = config.init_scale / @sqrt(@as(f32, @floatFromInt(rank)));
        for (b) |*val| {
            val.* = rng.random().floatNorm(f32) * std_dev;
        }

        // Gradients
        const d_a = try allocator.alloc(f32, @as(usize, out_features) * rank);
        errdefer allocator.free(d_a);
        const d_b = try allocator.alloc(f32, @as(usize, rank) * in_features);
        errdefer allocator.free(d_b);

        @memset(d_a, 0);
        @memset(d_b, 0);

        return .{
            .allocator = allocator,
            .a = a,
            .b = b,
            .d_a = d_a,
            .d_b = d_b,
            .in_features = in_features,
            .out_features = out_features,
            .rank = rank,
            .scaling = config.getScaling(),
            .active = true,
        };
    }

    pub fn deinit(self: *LoraAdapter) void {
        self.allocator.free(self.d_b);
        self.allocator.free(self.d_a);
        self.allocator.free(self.b);
        self.allocator.free(self.a);
        self.* = undefined;
    }

    /// Forward pass: compute LoRA delta = scaling * x @ B^T @ A^T
    /// Returns the delta to add to the base weight output.
    pub fn forward(self: *const LoraAdapter, x: []const f32, out: []f32) void {
        if (!self.active) {
            @memset(out, 0);
            return;
        }

        const batch_size = @as(u32, @intCast(x.len / self.in_features));

        // Temporary for intermediate result: x @ B^T
        // x: [batch, in_features], B: [rank, in_features], B^T: [in_features, rank]
        // result: [batch, rank]
        var temp: [4096]f32 = undefined;
        const temp_slice = temp[0 .. batch_size * self.rank];

        // x @ B^T = [batch, rank]
        for (0..batch_size) |b| {
            for (0..self.rank) |r| {
                var sum: f32 = 0;
                for (0..self.in_features) |i| {
                    // B[r, i]
                    sum += x[b * self.in_features + i] * self.b[r * self.in_features + i];
                }
                temp_slice[b * self.rank + r] = sum;
            }
        }

        // temp @ A^T = [batch, out_features]
        // A: [out_features, rank], A^T: [rank, out_features]
        for (0..batch_size) |b| {
            for (0..self.out_features) |o| {
                var sum: f32 = 0;
                for (0..self.rank) |r| {
                    // A[o, r]
                    sum += temp_slice[b * self.rank + r] * self.a[o * self.rank + r];
                }
                out[b * self.out_features + o] = sum * self.scaling;
            }
        }
    }

    /// Backward pass: compute gradients for A and B.
    /// d_out: gradient from upstream [batch, out_features]
    /// x: input from forward [batch, in_features]
    pub fn backward(self: *LoraAdapter, d_out: []const f32, x: []const f32) void {
        if (!self.active) return;

        const batch_size = @as(u32, @intCast(x.len / self.in_features));

        // Compute intermediate: h = x @ B^T
        var h: [4096]f32 = undefined;
        const h_slice = h[0 .. batch_size * self.rank];

        for (0..batch_size) |b| {
            for (0..self.rank) |r| {
                var sum: f32 = 0;
                for (0..self.in_features) |i| {
                    sum += x[b * self.in_features + i] * self.b[r * self.in_features + i];
                }
                h_slice[b * self.rank + r] = sum;
            }
        }

        // d_A = scaling * d_out^T @ h
        // d_out: [batch, out_features], h: [batch, rank]
        // d_A: [out_features, rank]
        for (0..self.out_features) |o| {
            for (0..self.rank) |r| {
                var sum: f32 = 0;
                for (0..batch_size) |b| {
                    sum += d_out[b * self.out_features + o] * h_slice[b * self.rank + r];
                }
                self.d_a[o * self.rank + r] += sum * self.scaling;
            }
        }

        // d_h = d_out @ A (upstream gradient through A)
        var d_h: [4096]f32 = undefined;
        const d_h_slice = d_h[0 .. batch_size * self.rank];

        for (0..batch_size) |b| {
            for (0..self.rank) |r| {
                var sum: f32 = 0;
                for (0..self.out_features) |o| {
                    sum += d_out[b * self.out_features + o] * self.a[o * self.rank + r];
                }
                d_h_slice[b * self.rank + r] = sum * self.scaling;
            }
        }

        // d_B = d_h^T @ x
        // d_h: [batch, rank], x: [batch, in_features]
        // d_B: [rank, in_features]
        for (0..self.rank) |r| {
            for (0..self.in_features) |i| {
                var sum: f32 = 0;
                for (0..batch_size) |b| {
                    sum += d_h_slice[b * self.rank + r] * x[b * self.in_features + i];
                }
                self.d_b[r * self.in_features + i] += sum;
            }
        }
    }

    /// Zero gradients.
    pub fn zeroGrad(self: *LoraAdapter) void {
        @memset(self.d_a, 0);
        @memset(self.d_b, 0);
    }

    /// Merge LoRA weights into base weight matrix.
    /// W' = W + scaling * B^T @ A^T (transposed for typical weight layout)
    pub fn mergeInto(self: *const LoraAdapter, base_weights: []f32) void {
        // base_weights: [out_features, in_features]
        // delta = A @ B
        // A: [out_features, rank], B: [rank, in_features]
        // result: [out_features, in_features]
        for (0..self.out_features) |o| {
            for (0..self.in_features) |i| {
                var sum: f32 = 0;
                for (0..self.rank) |r| {
                    sum += self.a[o * self.rank + r] * self.b[r * self.in_features + i];
                }
                base_weights[o * self.in_features + i] += sum * self.scaling;
            }
        }
    }

    /// Number of trainable parameters.
    pub fn numParams(self: *const LoraAdapter) usize {
        return self.a.len + self.b.len;
    }

    /// Enable/disable adapter.
    pub fn setActive(self: *LoraAdapter, active: bool) void {
        self.active = active;
    }
};

/// LoRA layer adapters for a transformer layer.
pub const LoraLayerAdapters = struct {
    allocator: std.mem.Allocator,
    q_adapter: ?LoraAdapter,
    k_adapter: ?LoraAdapter,
    v_adapter: ?LoraAdapter,
    o_adapter: ?LoraAdapter,
    gate_adapter: ?LoraAdapter,
    up_adapter: ?LoraAdapter,
    down_adapter: ?LoraAdapter,

    pub fn init(
        allocator: std.mem.Allocator,
        hidden_dim: u32,
        kv_dim: u32,
        intermediate_dim: u32,
        config: LoraConfig,
    ) !LoraLayerAdapters {
        var self = LoraLayerAdapters{
            .allocator = allocator,
            .q_adapter = null,
            .k_adapter = null,
            .v_adapter = null,
            .o_adapter = null,
            .gate_adapter = null,
            .up_adapter = null,
            .down_adapter = null,
        };

        const targets = config.target_modules;

        if (targets.q_proj) {
            self.q_adapter = try LoraAdapter.init(allocator, hidden_dim, hidden_dim, config);
        }
        if (targets.k_proj) {
            self.k_adapter = try LoraAdapter.init(allocator, hidden_dim, kv_dim, config);
        }
        if (targets.v_proj) {
            self.v_adapter = try LoraAdapter.init(allocator, hidden_dim, kv_dim, config);
        }
        if (targets.o_proj) {
            self.o_adapter = try LoraAdapter.init(allocator, hidden_dim, hidden_dim, config);
        }
        if (targets.gate_proj) {
            self.gate_adapter = try LoraAdapter.init(allocator, hidden_dim, intermediate_dim, config);
        }
        if (targets.up_proj) {
            self.up_adapter = try LoraAdapter.init(allocator, hidden_dim, intermediate_dim, config);
        }
        if (targets.down_proj) {
            self.down_adapter = try LoraAdapter.init(allocator, intermediate_dim, hidden_dim, config);
        }

        return self;
    }

    pub fn deinit(self: *LoraLayerAdapters) void {
        if (self.down_adapter) |*a| a.deinit();
        if (self.up_adapter) |*a| a.deinit();
        if (self.gate_adapter) |*a| a.deinit();
        if (self.o_adapter) |*a| a.deinit();
        if (self.v_adapter) |*a| a.deinit();
        if (self.k_adapter) |*a| a.deinit();
        if (self.q_adapter) |*a| a.deinit();
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGrad(self: *LoraLayerAdapters) void {
        if (self.q_adapter) |*a| a.zeroGrad();
        if (self.k_adapter) |*a| a.zeroGrad();
        if (self.v_adapter) |*a| a.zeroGrad();
        if (self.o_adapter) |*a| a.zeroGrad();
        if (self.gate_adapter) |*a| a.zeroGrad();
        if (self.up_adapter) |*a| a.zeroGrad();
        if (self.down_adapter) |*a| a.zeroGrad();
    }

    /// Number of trainable parameters.
    pub fn numParams(self: *const LoraLayerAdapters) usize {
        var total: usize = 0;
        if (self.q_adapter) |*a| total += a.numParams();
        if (self.k_adapter) |*a| total += a.numParams();
        if (self.v_adapter) |*a| total += a.numParams();
        if (self.o_adapter) |*a| total += a.numParams();
        if (self.gate_adapter) |*a| total += a.numParams();
        if (self.up_adapter) |*a| total += a.numParams();
        if (self.down_adapter) |*a| total += a.numParams();
        return total;
    }
};

/// LoRA model wrapper.
/// Wraps a base model with LoRA adapters for all layers.
pub const LoraModel = struct {
    allocator: std.mem.Allocator,
    config: LoraConfig,
    layer_adapters: []LoraLayerAdapters,

    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        intermediate_dim: u32,
        config: LoraConfig,
    ) !LoraModel {
        const head_dim = hidden_dim / num_heads;
        const kv_dim = num_kv_heads * head_dim;

        const layer_adapters = try allocator.alloc(LoraLayerAdapters, num_layers);
        errdefer allocator.free(layer_adapters);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                layer_adapters[i].deinit();
            }
        }

        for (layer_adapters) |*adapter| {
            adapter.* = try LoraLayerAdapters.init(
                allocator,
                hidden_dim,
                kv_dim,
                intermediate_dim,
                config,
            );
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .layer_adapters = layer_adapters,
        };
    }

    pub fn deinit(self: *LoraModel) void {
        for (self.layer_adapters) |*adapter| {
            adapter.deinit();
        }
        self.allocator.free(self.layer_adapters);
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGrad(self: *LoraModel) void {
        for (self.layer_adapters) |*adapter| {
            adapter.zeroGrad();
        }
    }

    /// Number of trainable parameters.
    pub fn numParams(self: *const LoraModel) usize {
        var total: usize = 0;
        for (self.layer_adapters) |*adapter| {
            total += adapter.numParams();
        }
        return total;
    }

    /// Get layer adapter.
    pub fn getLayer(self: *LoraModel, layer_idx: usize) *LoraLayerAdapters {
        return &self.layer_adapters[layer_idx];
    }

    /// Merge all LoRA weights into base model.
    /// After merging, LoRA adapters can be discarded.
    pub fn mergeWeights(self: *const LoraModel, base_weights: anytype) void {
        // This would merge LoRA weights into the base model weights
        // The exact implementation depends on the base model structure
        _ = self;
        _ = base_weights;
    }

    /// Save LoRA weights to file.
    pub fn save(self: *const LoraModel, allocator: std.mem.Allocator, path: []const u8) !void {
        // Collect all LoRA parameters
        var total_params: usize = 0;
        for (self.layer_adapters) |adapter| {
            total_params += adapter.numParams();
        }

        const weights = try allocator.alloc(f32, total_params);
        defer allocator.free(weights);

        var offset: usize = 0;
        for (self.layer_adapters) |adapter| {
            if (adapter.q_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.k_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.v_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.o_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.gate_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.up_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
            if (adapter.down_adapter) |a| {
                @memcpy(weights[offset..][0..a.a.len], a.a);
                offset += a.a.len;
                @memcpy(weights[offset..][0..a.b.len], a.b);
                offset += a.b.len;
            }
        }

        _ = path;
        // Save to file (would use checkpoint system)
    }
};

test "lora adapter init" {
    const allocator = std.testing.allocator;

    const config = LoraConfig{
        .rank = 4,
        .alpha = 8.0,
    };

    var adapter = try LoraAdapter.init(allocator, 64, 64, config);
    defer adapter.deinit();

    try std.testing.expectEqual(@as(usize, 64 * 4), adapter.a.len);
    try std.testing.expectEqual(@as(usize, 4 * 64), adapter.b.len);
    try std.testing.expectEqual(@as(f32, 2.0), adapter.scaling);

    // A should be initialized to zero
    for (adapter.a) |val| {
        try std.testing.expectEqual(@as(f32, 0), val);
    }
}

test "lora adapter forward" {
    const allocator = std.testing.allocator;

    const config = LoraConfig{
        .rank = 2,
        .alpha = 2.0,
    };

    var adapter = try LoraAdapter.init(allocator, 4, 4, config);
    defer adapter.deinit();

    // Set known values for A (output transform)
    // A[out][rank] = identity-like
    @memset(adapter.a, 0);
    adapter.a[0] = 1.0; // A[0, 0]
    adapter.a[3] = 1.0; // A[1, 1]

    // Input
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;

    adapter.forward(&x, &out);

    // Output should be non-zero (B is randomly initialized)
    var has_nonzero = false;
    for (out) |val| {
        if (val != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "lora layer adapters" {
    const allocator = std.testing.allocator;

    const config = LoraConfig{
        .rank = 4,
        .alpha = 8.0,
        .target_modules = .{
            .q_proj = true,
            .v_proj = true,
            .k_proj = false,
        },
    };

    var layer = try LoraLayerAdapters.init(allocator, 64, 32, 128, config);
    defer layer.deinit();

    try std.testing.expect(layer.q_adapter != null);
    try std.testing.expect(layer.v_adapter != null);
    try std.testing.expect(layer.k_adapter == null);

    const params = layer.numParams();
    try std.testing.expect(params > 0);
}

test "lora model" {
    const allocator = std.testing.allocator;

    const config = LoraConfig{
        .rank = 4,
        .alpha = 8.0,
    };

    var model = try LoraModel.init(
        allocator,
        2, // layers
        64, // hidden_dim
        4, // num_heads
        4, // num_kv_heads
        128, // intermediate_dim
        config,
    );
    defer model.deinit();

    const params = model.numParams();
    try std.testing.expect(params > 0);

    // Zero gradients
    model.zeroGrad();
}
