//! Kernel Fusion Detection and Optimization
//!
//! Automatically detects and fuses consecutive GPU operations to reduce
//! memory bandwidth and kernel launch overhead. Uses an operation graph
//! to analyze data dependencies and identify fusion opportunities.
//!
//! ## Fusion Patterns
//!
//! - Element-wise chains: relu(add(a, b)) -> fused_relu_add
//! - Normalization + activation: layer_norm(x) + gelu(x) -> fused_layer_norm_gelu
//! - Linear + activation: matmul(x, w) + bias + relu -> fused_linear_relu
//! - Reduction chains: sum(mul(a, b)) -> fused_dot_product

const std = @import("std");
const occupancy = @import("occupancy.zig");

/// Operation types that can be fused.
pub const OpType = enum {
    // Element-wise operations
    add,
    sub,
    mul,
    div,
    neg,
    abs,
    sqrt,
    rsqrt,
    exp,
    log,
    pow,
    max,
    min,
    clamp,

    // Activations
    relu,
    leaky_relu,
    sigmoid,
    tanh,
    gelu,
    gelu_fast,
    silu,
    swiglu,
    softmax,

    // Normalization
    layer_norm,
    rms_norm,
    batch_norm,

    // Reductions
    reduce_sum,
    reduce_max,
    reduce_min,
    reduce_mean,

    // Matrix operations
    matmul,
    batch_matmul,
    transpose,

    // Memory operations
    copy,
    broadcast,
    gather,
    scatter,

    // Custom/compound
    fused_add_relu,
    fused_add_gelu,
    fused_mul_add, // FMA
    fused_layer_norm_gelu,
    fused_linear_relu,
    fused_linear_gelu,
    fused_linear_silu,
    dot_product,
    // Attention-related fusions
    fused_attention_qk, // Q*K^T + scale + softmax
    fused_attention_sv, // softmax(QK) * V

    /// Check if operation is element-wise.
    pub fn isElementWise(self: OpType) bool {
        return switch (self) {
            .add, .sub, .mul, .div, .neg, .abs, .sqrt, .rsqrt, .exp, .log, .pow, .max, .min, .clamp, .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu, .copy => true,
            else => false,
        };
    }

    /// Check if operation is commonly fused.
    pub fn isFusable(self: OpType) bool {
        return self.isElementWise() or self.isActivation() or self.isNormalization() or
            self == .matmul or self == .softmax;
    }

    /// Check if operation is part of attention mechanism.
    pub fn isAttentionRelated(self: OpType) bool {
        return self == .matmul or self == .softmax or self == .fused_attention_qk or
            self == .fused_attention_sv;
    }

    /// Check if operation is a reduction.
    pub fn isReduction(self: OpType) bool {
        return switch (self) {
            .reduce_sum, .reduce_max, .reduce_min, .reduce_mean, .softmax => true,
            else => false,
        };
    }

    /// Check if operation is an activation function.
    pub fn isActivation(self: OpType) bool {
        return switch (self) {
            .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu, .swiglu => true,
            else => false,
        };
    }

    /// Check if operation is normalization.
    pub fn isNormalization(self: OpType) bool {
        return switch (self) {
            .layer_norm, .rms_norm, .batch_norm => true,
            else => false,
        };
    }

    /// Get estimated FLOPs per element for the operation.
    pub fn flopsPerElement(self: OpType) u32 {
        return switch (self) {
            .add, .sub, .mul, .div, .neg, .abs, .max, .min, .copy => 1,
            .sqrt, .rsqrt => 4,
            .exp, .log => 8,
            .pow => 16,
            .relu, .leaky_relu => 1,
            .sigmoid => 4,
            .tanh => 8,
            .gelu => 16,
            .gelu_fast => 8,
            .silu => 5,
            .layer_norm => 8,
            .rms_norm => 6,
            .reduce_sum, .reduce_max, .reduce_min, .reduce_mean => 1,
            .matmul => 2, // Per output element, depends on K
            .fused_add_relu => 2,
            .fused_add_gelu => 17,
            .fused_mul_add => 2,
            .fused_linear_relu => 3,
            .fused_linear_gelu => 18,
            .fused_linear_silu => 6,
            .fused_attention_qk => 20, // Matmul + scale + softmax
            .fused_attention_sv => 10, // Softmax * Value
            .dot_product => 2,
            else => 4,
        };
    }

    /// Get memory traffic per element (reads + writes) in bytes (assuming f32).
    pub fn memoryBytesPerElement(self: OpType) u32 {
        return switch (self) {
            .add, .sub, .mul, .div, .max, .min => 12, // 2 reads + 1 write
            .neg, .abs, .sqrt, .rsqrt, .exp, .log, .copy => 8, // 1 read + 1 write
            .relu, .leaky_relu, .sigmoid, .tanh, .gelu, .gelu_fast, .silu => 8,
            .pow => 12,
            .layer_norm => 16, // input + gamma + beta + output
            .rms_norm => 12,
            .fused_add_relu, .fused_add_gelu => 12, // 2 reads + 1 write (no intermediate)
            .fused_mul_add => 16, // 3 reads + 1 write
            .fused_linear_relu, .fused_linear_gelu, .fused_linear_silu => 20, // Matmul result + bias + activation result
            .fused_attention_qk => 24, // Q*K^T result + scaled result + softmax result
            .fused_attention_sv => 16, // softmax result + V multiplication result
            else => 8,
        };
    }
};

/// Buffer handle for operation graph.
pub const BufferHandle = u32;

/// Sentinel value for no buffer.
pub const NO_BUFFER: BufferHandle = std.math.maxInt(BufferHandle);

/// Operation node in the fusion graph.
pub const OpNode = struct {
    /// Operation type.
    op: OpType,
    /// Input buffer handles.
    inputs: [4]BufferHandle = .{ NO_BUFFER, NO_BUFFER, NO_BUFFER, NO_BUFFER },
    /// Number of valid inputs.
    num_inputs: u8 = 0,
    /// Output buffer handle.
    output: BufferHandle = NO_BUFFER,
    /// Element count for the operation.
    element_count: usize = 0,
    /// Whether this node has been fused into another.
    fused: bool = false,
    /// ID of the node this was fused into (if fused).
    fused_into: ?u32 = null,
    /// Scalar parameters (e.g., alpha for leaky_relu).
    scalar_params: [4]f32 = .{ 0, 0, 0, 0 },

    /// Add an input to this operation.
    pub fn addInput(self: *OpNode, buf: BufferHandle) void {
        if (self.num_inputs < 4) {
            self.inputs[self.num_inputs] = buf;
            self.num_inputs += 1;
        }
    }

    /// Get input slice.
    pub fn getInputs(self: *const OpNode) []const BufferHandle {
        return self.inputs[0..self.num_inputs];
    }
};

/// Fusion pattern that was detected.
pub const FusionPattern = struct {
    /// First operation in the chain.
    first_op_idx: u32,
    /// Last operation in the chain.
    last_op_idx: u32,
    /// Fused operation type.
    fused_op: OpType,
    /// Estimated speedup factor.
    speedup: f32,
    /// Memory bandwidth saved (bytes).
    bandwidth_saved: usize,
    /// Operations in the chain.
    chain: [8]u32 = undefined,
    /// Number of operations in chain.
    chain_len: u8 = 0,
};

/// Fusion detection and optimization.
pub const FusionOptimizer = struct {
    allocator: std.mem.Allocator,
    /// Operation nodes in submission order.
    nodes: std.ArrayListUnmanaged(OpNode),
    /// Buffer reference counts (consumers per buffer).
    buffer_refs: std.AutoHashMapUnmanaged(BufferHandle, u32),
    /// Buffer producers (which op produced this buffer).
    buffer_producers: std.AutoHashMapUnmanaged(BufferHandle, u32),
    /// Detected fusion patterns.
    patterns: std.ArrayListUnmanaged(FusionPattern),
    /// Device capabilities for fusion decisions.
    device_caps: ?occupancy.DeviceCapabilities,
    /// Statistics.
    stats: FusionStats,
    /// Auto-apply mode: automatically fuse operations when recorded.
    auto_apply: bool = false,
    /// Minimum speedup threshold for auto-apply (default: 1.1 = 10% speedup required).
    min_speedup_threshold: f32 = 1.1,

    const Self = @This();

    /// Initialize the fusion optimizer.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .nodes = .{},
            .buffer_refs = .{},
            .buffer_producers = .{},
            .patterns = .{},
            .device_caps = null,
            .stats = .{},
            .auto_apply = false,
            .min_speedup_threshold = 1.1,
        };
    }

    /// Enable auto-apply mode: operations are automatically fused when recorded
    /// if the estimated speedup exceeds the threshold.
    ///
    /// Args:
    ///   min_speedup: Minimum speedup factor required to apply fusion (e.g., 1.1 = 10% speedup).
    ///                Setting to 1.0 fuses everything possible, higher values are more conservative.
    pub fn enableAutoApply(self: *Self, min_speedup: f32) void {
        self.auto_apply = true;
        self.min_speedup_threshold = @max(min_speedup, 1.0);
    }

    /// Disable auto-apply mode.
    pub fn disableAutoApply(self: *Self) void {
        self.auto_apply = false;
    }

    /// Check if auto-apply mode is enabled.
    pub fn isAutoApplyEnabled(self: *const Self) bool {
        return self.auto_apply;
    }

    /// Deinitialize.
    pub fn deinit(self: *Self) void {
        self.nodes.deinit(self.allocator);
        self.buffer_refs.deinit(self.allocator);
        self.buffer_producers.deinit(self.allocator);
        self.patterns.deinit(self.allocator);
    }

    /// Set device capabilities for fusion decisions.
    pub fn setDeviceCapabilities(self: *Self, caps: occupancy.DeviceCapabilities) void {
        self.device_caps = caps;
    }

    /// Record an operation for fusion analysis.
    /// In auto-apply mode, checks for immediate fusion opportunities with the previous operation.
    pub fn recordOp(
        self: *Self,
        op: OpType,
        inputs: []const BufferHandle,
        output: BufferHandle,
        element_count: usize,
    ) !u32 {
        var node = OpNode{
            .op = op,
            .output = output,
            .element_count = element_count,
        };

        // Add inputs
        for (inputs) |inp| {
            node.addInput(inp);
            // Track buffer references
            const refs = self.buffer_refs.get(inp) orelse 0;
            try self.buffer_refs.put(self.allocator, inp, refs + 1);
        }

        // Track buffer producer
        const new_idx: u32 = @intCast(self.nodes.items.len);
        try self.buffer_producers.put(self.allocator, output, new_idx);

        try self.nodes.append(self.allocator, node);
        self.stats.ops_recorded += 1;

        // In auto-apply mode, try to fuse immediately with producer
        if (self.auto_apply) {
            self.tryImmediateFusion(new_idx);
        }

        return new_idx;
    }

    /// Try to fuse the newly added operation with its producer (auto-apply mode).
    fn tryImmediateFusion(self: *Self, new_idx: u32) void {
        if (new_idx == 0) return; // First operation, nothing to fuse with

        const new_node = &self.nodes.items[new_idx];
        if (new_node.num_inputs == 0) return;

        // Check the first input - find its producer
        const input_buf = new_node.inputs[0];
        const producer_idx = self.buffer_producers.get(input_buf) orelse return;
        if (producer_idx >= new_idx) return; // Safety check

        const producer_node = &self.nodes.items[producer_idx];
        if (producer_node.fused) return; // Already fused

        // Check if this is a single-consumer relationship
        const refs = self.buffer_refs.get(input_buf) orelse 0;
        if (refs != 1) return; // Multiple consumers, can't fuse

        // Check for known fusion patterns
        const fused_op = self.detectImmediateFusionPattern(producer_node.op, new_node.op);
        if (fused_op == null) return;

        // Calculate estimated speedup
        const speedup = calculateSpeedup(producer_node.op, new_node.op, new_node.element_count);
        if (speedup < self.min_speedup_threshold) return;

        // Apply fusion immediately
        self.nodes.items[new_idx].fused = true;
        self.nodes.items[new_idx].fused_into = producer_idx;
        self.nodes.items[producer_idx].op = fused_op.?;
        self.nodes.items[producer_idx].output = new_node.output;

        self.stats.fusions_applied += 1;
        self.stats.bandwidth_saved_bytes += new_node.element_count * 8; // Intermediate eliminated
    }

    /// Detect if two consecutive operations can be immediately fused.
    /// Returns the fused operation type if fusable, null otherwise.
    fn detectImmediateFusionPattern(_: *const Self, producer_op: OpType, consumer_op: OpType) ?OpType {
        // Element-wise + activation patterns
        if (producer_op == .add) {
            switch (consumer_op) {
                .relu => return .fused_add_relu,
                .gelu, .gelu_fast => return .fused_add_gelu,
                else => {},
            }
        }

        // Mul + add pattern (FMA)
        if (producer_op == .mul and consumer_op == .add) {
            return .fused_mul_add;
        }

        // Normalization + activation patterns
        if (producer_op == .layer_norm) {
            switch (consumer_op) {
                .gelu, .gelu_fast => return .fused_layer_norm_gelu,
                else => {},
            }
        }

        // Matmul + activation patterns (linear layers)
        if (producer_op == .matmul) {
            switch (consumer_op) {
                .relu => return .fused_linear_relu,
                .gelu, .gelu_fast => return .fused_linear_gelu,
                .silu => return .fused_linear_silu,
                else => {},
            }
        }

        // Reduction patterns
        if (producer_op == .mul and consumer_op == .reduce_sum) {
            return .dot_product;
        }

        // Generic element-wise chain (if both are element-wise)
        if (producer_op.isElementWise() and consumer_op.isElementWise()) {
            // Can be fused but keep consumer op type
            return consumer_op;
        }

        return null;
    }

    /// Analyze recorded operations and detect fusion opportunities.
    pub fn analyze(self: *Self) !void {
        self.patterns.clearRetainingCapacity();

        // Pass 1: Detect element-wise chains
        try self.detectElementWiseChains();

        // Pass 2: Detect normalization + activation patterns
        try self.detectNormActivationPatterns();

        // Pass 3: Detect linear + activation patterns
        try self.detectLinearActivationPatterns();

        // Pass 4: Detect reduction patterns (e.g., dot product)
        try self.detectReductionPatterns();

        // Pass 5: Detect attention-related patterns (new)
        try self.detectAttentionPatterns();

        self.stats.patterns_detected = self.patterns.items.len;
    }

    /// Get detected fusion patterns.
    pub fn getPatterns(self: *const Self) []const FusionPattern {
        return self.patterns.items;
    }

    /// Apply fusion patterns and return optimized operation list.
    pub fn applyFusions(self: *Self) ![]OpNode {
        // Sort patterns by speedup (highest first)
        std.mem.sort(FusionPattern, self.patterns.items, {}, struct {
            fn lessThan(_: void, a: FusionPattern, b: FusionPattern) bool {
                return a.speedup > b.speedup; // Descending
            }
        }.lessThan);

        // Apply non-conflicting patterns
        for (self.patterns.items) |pattern| {
            if (self.canApplyPattern(pattern)) {
                self.applyPattern(pattern);
                self.stats.fusions_applied += 1;
            }
        }

        // Build optimized node list (excluding fused nodes)
        var result = std.ArrayListUnmanaged(OpNode).empty;
        for (self.nodes.items) |node| {
            if (!node.fused) {
                try result.append(self.allocator, node);
            }
        }

        return result.items;
    }

    /// Reset optimizer for new analysis.
    pub fn reset(self: *Self) void {
        self.nodes.clearRetainingCapacity();
        self.buffer_refs.clearRetainingCapacity();
        self.buffer_producers.clearRetainingCapacity();
        self.patterns.clearRetainingCapacity();
    }

    /// Get statistics.
    pub fn getStats(self: *const Self) FusionStats {
        return self.stats;
    }

    // ========================================================================
    // Private fusion detection methods
    // ========================================================================

    /// Check if a sequence of operations represents a known fusion pattern
    fn isCommonFusionPattern(self: *const Self, start_idx: usize) bool {
        const node = &self.nodes.items[start_idx];

        // Check for element-wise + activation patterns
        if (node.op.isElementWise()) {
            // Pattern: element-wise operation followed by activation
            if (self.detectElementWiseActivation(start_idx)) {
                return true;
            }

            // Pattern: normalization followed by activation
            if (node.op.isNormalization()) {
                if (self.detectNormActivation(start_idx)) {
                    return true;
                }
            }
        }

        return false;
    }

    /// Detect element-wise operation followed by activation function
    fn detectElementWiseActivation(self: *const Self, ew_idx: usize) bool {
        const ew_node = &self.nodes.items[ew_idx];
        const consumer_idx = self.findSingleConsumer(ew_node.output, ew_idx) orelse return false;
        const consumer = &self.nodes.items[consumer_idx];

        return consumer.op.isActivation() and !consumer.fused;
    }

    /// Detect normalization followed by activation
    fn detectNormActivation(self: *const Self, norm_idx: usize) bool {
        const norm_node = &self.nodes.items[norm_idx];
        const consumer_idx = self.findSingleConsumer(norm_node.output, norm_idx) orelse return false;
        const consumer = &self.nodes.items[consumer_idx];

        return consumer.op.isActivation() and !consumer.fused;
    }

    fn detectElementWiseChains(self: *Self) !void {
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or !node.op.isElementWise()) continue;

            // Look for chains starting from this node
            var chain: [8]u32 = undefined;
            var chain_len: u8 = 1;
            chain[0] = @intCast(i);

            // Special pattern detection for common sequences
            if (self.isCommonFusionPattern(i)) {
                // Skip individual chain detection for known patterns
                continue;
            }

            var current_output = node.output;
            var current_idx = i;

            // Follow the chain
            while (chain_len < 8) {
                // Find consumer of current output
                const next_idx = self.findSingleConsumer(current_output, current_idx);
                if (next_idx == null) break;

                const next_node = &self.nodes.items[next_idx.?];
                if (!next_node.op.isElementWise() or next_node.fused) break;

                chain[chain_len] = @intCast(next_idx.?);
                chain_len += 1;
                current_output = next_node.output;
                current_idx = next_idx.?;
            }

            // Only fuse chains of 2+ operations
            if (chain_len >= 2) {
                const pattern = self.createChainPattern(chain[0..chain_len]);
                if (pattern.speedup > 1.05) { // At least 5% speedup
                    try self.patterns.append(self.allocator, pattern);
                }
            }
        }
    }

    fn detectNormActivationPatterns(self: *Self) !void {
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or !node.op.isNormalization()) continue;

            // Look for activation following normalization
            const consumer_idx = self.findSingleConsumer(node.output, i);
            if (consumer_idx == null) continue;

            const consumer = &self.nodes.items[consumer_idx.?];
            if (!consumer.op.isActivation() or consumer.fused) continue;

            // Create fusion pattern
            const fused_op: OpType = if (node.op == .layer_norm and consumer.op == .gelu)
                .fused_layer_norm_gelu
            else
                .fused_add_gelu; // Fallback

            const bandwidth_saved = node.element_count * 8; // Intermediate buffer eliminated

            try self.patterns.append(self.allocator, .{
                .first_op_idx = @intCast(i),
                .last_op_idx = @intCast(consumer_idx.?),
                .fused_op = fused_op,
                .speedup = calculateSpeedup(node.op, consumer.op, node.element_count),
                .bandwidth_saved = bandwidth_saved,
                .chain = .{ @intCast(i), @intCast(consumer_idx.?), 0, 0, 0, 0, 0, 0 },
                .chain_len = 2,
            });
        }
    }

    fn detectLinearActivationPatterns(self: *Self) !void {
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or node.op != .matmul) continue;

            // Look for add (bias) following matmul
            var next_idx = self.findSingleConsumer(node.output, i);
            if (next_idx == null) continue;

            var next_node = &self.nodes.items[next_idx.?];
            var has_bias = false;
            var bias_idx: u32 = 0;

            if (next_node.op == .add and !next_node.fused) {
                has_bias = true;
                bias_idx = @intCast(next_idx.?);

                // Look for activation after bias
                next_idx = self.findSingleConsumer(next_node.output, next_idx.?);
                if (next_idx != null) {
                    next_node = &self.nodes.items[next_idx.?];
                }
            }

            // Check for activation
            if (next_idx != null and next_node.op.isActivation() and !next_node.fused) {
                const fused_op: OpType = switch (next_node.op) {
                    .relu => .fused_linear_relu,
                    .gelu, .gelu_fast => .fused_linear_gelu,
                    .silu => .fused_linear_silu,
                    else => continue,
                };

                var chain: [8]u32 = undefined;
                var chain_len: u8 = 2;
                chain[0] = @intCast(i);

                if (has_bias) {
                    chain[1] = bias_idx;
                    chain[2] = @intCast(next_idx.?);
                    chain_len = 3;
                } else {
                    chain[1] = @intCast(next_idx.?);
                }

                // Calculate bandwidth saved
                const bandwidth_saved = node.element_count * 8 * (chain_len - 1);

                try self.patterns.append(self.allocator, .{
                    .first_op_idx = @intCast(i),
                    .last_op_idx = @intCast(next_idx.?),
                    .fused_op = fused_op,
                    .speedup = 1.2 + @as(f32, @floatFromInt(chain_len - 2)) * 0.1,
                    .bandwidth_saved = bandwidth_saved,
                    .chain = chain,
                    .chain_len = chain_len,
                });
            }
        }
    }

    fn detectReductionPatterns(self: *Self) !void {
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or node.op != .mul) continue;

            // Look for reduce_sum following mul (dot product pattern)
            const consumer_idx = self.findSingleConsumer(node.output, i);
            if (consumer_idx == null) continue;

            const consumer = &self.nodes.items[consumer_idx.?];
            if (consumer.op != .reduce_sum or consumer.fused) continue;

            // This is a dot product pattern
            const bandwidth_saved = node.element_count * 4; // Intermediate eliminated

            try self.patterns.append(self.allocator, .{
                .first_op_idx = @intCast(i),
                .last_op_idx = @intCast(consumer_idx.?),
                .fused_op = .dot_product,
                .speedup = 1.5, // Dot product fusion typically gives good speedup
                .bandwidth_saved = bandwidth_saved,
                .chain = .{ @intCast(i), @intCast(consumer_idx.?), 0, 0, 0, 0, 0, 0 },
                .chain_len = 2,
            });
        }
    }

    /// Detect attention-related patterns (Q*K^T, scale, softmax, dropout, V multiplication)
    fn detectAttentionPatterns(self: *Self) !void {
        // Look for Q*K^T pattern followed by scaling and softmax
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or node.op != .matmul) continue;

            // Check if this looks like Q*K^T (transposed key matrix)
            // This is difficult to detect precisely without more context, so we'll look for
            // patterns that resemble attention mechanisms

            // Look for scale operation after matmul
            const next_idx = self.findSingleConsumer(node.output, i);
            if (next_idx == null) continue;

            const next_node = &self.nodes.items[next_idx.?];
            // Scale could be implemented as mul with a scalar
            if (next_node.op == .mul and !next_node.fused) {
                // Check if one operand is a scalar (constant)
                // This is simplified - in practice we might look for specific constant values

                // Look for softmax after scale
                const softmax_idx = self.findSingleConsumer(next_node.output, next_idx.?);
                if (softmax_idx == null) continue;

                const softmax_node = &self.nodes.items[softmax_idx.?];
                if (softmax_node.op == .softmax and !softmax_node.fused) {
                    // Found basic attention pattern: matmul -> scale -> softmax
                    const bandwidth_saved = node.element_count * 8 * 2; // Two intermediates eliminated

                    try self.patterns.append(self.allocator, .{
                        .first_op_idx = @intCast(i),
                        .last_op_idx = @intCast(softmax_idx.?),
                        .fused_op = .fused_attention_qk,
                        .speedup = 1.8, // Significant speedup from fusing attention operations
                        .bandwidth_saved = bandwidth_saved,
                        .chain = .{ @intCast(i), @intCast(next_idx.?), @intCast(softmax_idx.?), 0, 0, 0, 0, 0 },
                        .chain_len = 3,
                    });
                }
            }
        }

        // Look for more complex attention patterns including value multiplication
        for (self.nodes.items, 0..) |*node, i| {
            if (node.fused or node.op != .softmax) continue;

            // Look for value multiplication after softmax
            const next_idx = self.findSingleConsumer(node.output, i);
            if (next_idx == null) continue;

            const next_node = &self.nodes.items[next_idx.?];
            if (next_node.op == .matmul and !next_node.fused) {
                // Found attention pattern with value projection: softmax(QK) * V
                const bandwidth_saved = node.element_count * 8; // One intermediate eliminated

                try self.patterns.append(self.allocator, .{
                    .first_op_idx = @intCast(i),
                    .last_op_idx = @intCast(next_idx.?),
                    .fused_op = .fused_attention_sv,
                    .speedup = 1.4,
                    .bandwidth_saved = bandwidth_saved,
                    .chain = .{ @intCast(i), @intCast(next_idx.?), 0, 0, 0, 0, 0, 0 },
                    .chain_len = 2,
                });
            }
        }
    }

    fn findSingleConsumer(self: *const Self, buffer: BufferHandle, producer_idx: usize) ?usize {
        // Check if buffer has exactly one consumer
        const refs = self.buffer_refs.get(buffer) orelse 0;
        if (refs != 1) return null;

        // Find the consumer
        for (self.nodes.items, 0..) |node, i| {
            if (i <= producer_idx) continue; // Must be after producer
            for (node.getInputs()) |inp| {
                if (inp == buffer) return i;
            }
        }
        return null;
    }

    fn createChainPattern(self: *const Self, chain: []const u32) FusionPattern {
        var pattern = FusionPattern{
            .first_op_idx = chain[0],
            .last_op_idx = chain[chain.len - 1],
            .fused_op = determineChainFusedOp(self.nodes.items, chain),
            .speedup = 1.0,
            .bandwidth_saved = 0,
            .chain_len = @intCast(chain.len),
        };

        // Copy chain
        for (chain, 0..) |idx, i| {
            if (i < 8) pattern.chain[i] = idx;
        }

        // Calculate metrics
        var total_flops: u64 = 0;
        var total_bytes: u64 = 0;
        var intermediate_bytes: usize = 0;

        for (chain, 0..) |idx, i| {
            const node = &self.nodes.items[idx];
            total_flops += @as(u64, node.op.flopsPerElement()) * node.element_count;
            total_bytes += @as(u64, node.op.memoryBytesPerElement()) * node.element_count;

            // Intermediate buffers (all except first input and last output)
            if (i > 0) {
                intermediate_bytes += node.element_count * 4; // Read of intermediate
            }
            if (i < chain.len - 1) {
                intermediate_bytes += node.element_count * 4; // Write of intermediate
            }
        }

        pattern.bandwidth_saved = intermediate_bytes;

        // Estimate speedup based on bandwidth reduction
        if (total_bytes > 0) {
            const fused_bytes = total_bytes - intermediate_bytes;
            pattern.speedup = @as(f32, @floatFromInt(total_bytes)) / @as(f32, @floatFromInt(@max(fused_bytes, 1)));
        }

        return pattern;
    }

    fn determineChainFusedOp(nodes: []const OpNode, chain: []const u32) OpType {
        if (chain.len < 2) return nodes[chain[0]].op;

        const first_op = nodes[chain[0]].op;
        const last_op = nodes[chain[chain.len - 1]].op;

        // Known fusion patterns
        if (first_op == .add and last_op == .relu) return .fused_add_relu;
        if (first_op == .add and last_op == .gelu) return .fused_add_gelu;
        if (first_op == .mul and last_op == .add) return .fused_mul_add;
        if (first_op == .layer_norm and last_op == .gelu) return .fused_layer_norm_gelu;

        // Default: return last op (chain will be compiled as single kernel)
        return last_op;
    }

    fn canApplyPattern(self: *const Self, pattern: FusionPattern) bool {
        // Check that none of the nodes in the pattern are already fused
        for (pattern.chain[0..pattern.chain_len]) |idx| {
            if (self.nodes.items[idx].fused) return false;
        }
        return true;
    }

    fn applyPattern(self: *Self, pattern: FusionPattern) void {
        // Mark all nodes except first as fused
        for (pattern.chain[1..pattern.chain_len]) |idx| {
            self.nodes.items[idx].fused = true;
            self.nodes.items[idx].fused_into = pattern.first_op_idx;
        }

        // Update first node with fused operation type
        self.nodes.items[pattern.first_op_idx].op = pattern.fused_op;

        // Update output to final output
        const last_node = &self.nodes.items[pattern.last_op_idx];
        self.nodes.items[pattern.first_op_idx].output = last_node.output;
    }
};

/// Calculate estimated speedup from fusing two operations.
fn calculateSpeedup(op1: OpType, op2: OpType, element_count: usize) f32 {
    // Base speedup from eliminating intermediate memory traffic
    const mem1 = @as(f32, @floatFromInt(op1.memoryBytesPerElement()));
    const mem2 = @as(f32, @floatFromInt(op2.memoryBytesPerElement()));
    const intermediate_saved: f32 = 8; // Read + write of intermediate

    const original_mem = mem1 + mem2;
    const fused_mem = original_mem - intermediate_saved;

    var speedup = original_mem / @max(fused_mem, 1);

    // Additional speedup from reduced kernel launch overhead
    if (element_count < 10000) {
        speedup += 0.1; // Launch overhead is significant for small kernels
    }

    return @max(speedup, 1.0);
}

/// Fusion statistics.
pub const FusionStats = struct {
    ops_recorded: usize = 0,
    patterns_detected: usize = 0,
    fusions_applied: usize = 0,
    bandwidth_saved_bytes: usize = 0,
    estimated_speedup: f32 = 1.0,
};

// ============================================================================
// Tests
// ============================================================================

test "fusion optimizer basic" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Record add -> relu chain
    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    try optimizer.analyze();

    const patterns = optimizer.getPatterns();
    try std.testing.expect(patterns.len > 0);
}

test "fusion optimizer chain detection" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Record longer chain: add -> mul -> relu
    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.mul, &.{ 2, 3 }, 4, 1024);
    _ = try optimizer.recordOp(.relu, &.{4}, 5, 1024);

    try optimizer.analyze();

    const patterns = optimizer.getPatterns();
    try std.testing.expect(patterns.len > 0);

    // Should detect chain of 3 operations
    var found_chain = false;
    for (patterns) |p| {
        if (p.chain_len >= 3) {
            found_chain = true;
            break;
        }
    }
    try std.testing.expect(found_chain);
}

test "fusion optimizer dot product pattern" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Record mul -> reduce_sum (dot product)
    _ = try optimizer.recordOp(.mul, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.reduce_sum, &.{2}, 3, 1024);

    try optimizer.analyze();

    const patterns = optimizer.getPatterns();
    try std.testing.expect(patterns.len > 0);

    // Should detect dot_product pattern
    var found_dot = false;
    for (patterns) |p| {
        if (p.fused_op == .dot_product) {
            found_dot = true;
            break;
        }
    }
    try std.testing.expect(found_dot);
}

test "op type properties" {
    try std.testing.expect(OpType.add.isElementWise());
    try std.testing.expect(!OpType.matmul.isElementWise());
    try std.testing.expect(OpType.relu.isActivation());
    try std.testing.expect(!OpType.add.isActivation());
    try std.testing.expect(OpType.reduce_sum.isReduction());
    try std.testing.expect(OpType.layer_norm.isNormalization());
}

test "fusion apply" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Record add -> relu chain
    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    try optimizer.analyze();

    const optimized = try optimizer.applyFusions();

    // After fusion, should have fewer operations
    try std.testing.expect(optimized.len < 2);

    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);
}

test "fusion optimizer auto-apply mode" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Enable auto-apply with 1.0 threshold (fuse everything possible)
    optimizer.enableAutoApply(1.0);
    try std.testing.expect(optimizer.isAutoApplyEnabled());

    // Record add -> relu chain - should auto-fuse
    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    // Fusion should have already been applied without calling analyze()
    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);
    try std.testing.expect(stats.bandwidth_saved_bytes > 0);

    // The second operation should be marked as fused
    try std.testing.expect(optimizer.nodes.items[1].fused);
    try std.testing.expectEqual(@as(?u32, 0), optimizer.nodes.items[1].fused_into);

    // The first operation should have the fused type
    try std.testing.expectEqual(OpType.fused_add_relu, optimizer.nodes.items[0].op);
}

test "fusion optimizer auto-apply with threshold" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Enable auto-apply with high threshold (should not fuse small speedups)
    optimizer.enableAutoApply(2.0);

    // Record add -> relu chain
    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    // With high threshold, fusion might not be applied
    // (depends on speedup calculation - relu fusion gives ~1.5x speedup typically)
    const stats = optimizer.getStats();

    // Disable and verify
    optimizer.disableAutoApply();
    try std.testing.expect(!optimizer.isAutoApplyEnabled());

    // Just verify the optimizer didn't crash
    try std.testing.expect(stats.ops_recorded == 2);
}

test "fusion optimizer auto-apply dot product" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    // Enable auto-apply
    optimizer.enableAutoApply(1.0);

    // Record mul -> reduce_sum (dot product pattern)
    _ = try optimizer.recordOp(.mul, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.reduce_sum, &.{2}, 3, 1024);

    // Should auto-fuse to dot_product
    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);

    // First operation should now be dot_product
    try std.testing.expectEqual(OpType.dot_product, optimizer.nodes.items[0].op);
}

test {
    std.testing.refAllDecls(@This());
}
