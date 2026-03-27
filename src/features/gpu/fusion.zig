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

const types = @import("fusion/types.zig");
const detection = @import("fusion/detection.zig");

pub const OpType = types.OpType;
pub const BufferHandle = types.BufferHandle;
pub const NO_BUFFER = types.NO_BUFFER;
pub const OpNode = types.OpNode;
pub const FusionPattern = types.FusionPattern;
pub const FusionStats = types.FusionStats;

pub const FusionOptimizer = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(OpNode),
    buffer_refs: std.AutoHashMapUnmanaged(BufferHandle, u32),
    buffer_producers: std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: std.ArrayListUnmanaged(FusionPattern),
    device_caps: ?occupancy.DeviceCapabilities,
    stats: FusionStats,
    auto_apply: bool = false,
    min_speedup_threshold: f32 = 1.1,

    const Self = @This();

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

    pub fn enableAutoApply(self: *Self, min_speedup: f32) void {
        self.auto_apply = true;
        self.min_speedup_threshold = @max(min_speedup, 1.0);
    }

    pub fn disableAutoApply(self: *Self) void {
        self.auto_apply = false;
    }

    pub fn isAutoApplyEnabled(self: *const Self) bool {
        return self.auto_apply;
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit(self.allocator);
        self.buffer_refs.deinit(self.allocator);
        self.buffer_producers.deinit(self.allocator);
        self.patterns.deinit(self.allocator);
    }

    pub fn setDeviceCapabilities(self: *Self, caps: occupancy.DeviceCapabilities) void {
        self.device_caps = caps;
    }

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

        for (inputs) |inp| {
            node.addInput(inp);
            const refs = self.buffer_refs.get(inp) orelse 0;
            try self.buffer_refs.put(self.allocator, inp, refs + 1);
        }

        const new_idx: u32 = @intCast(self.nodes.items.len);
        try self.buffer_producers.put(self.allocator, output, new_idx);

        try self.nodes.append(self.allocator, node);
        self.stats.ops_recorded += 1;

        if (self.auto_apply) {
            self.tryImmediateFusion(new_idx);
        }

        return new_idx;
    }

    fn tryImmediateFusion(self: *Self, new_idx: u32) void {
        if (new_idx == 0) return;

        const new_node = &self.nodes.items[new_idx];
        if (new_node.num_inputs == 0) return;

        const input_buf = new_node.inputs[0];
        const producer_idx = self.buffer_producers.get(input_buf) orelse return;
        if (producer_idx >= new_idx) return;

        const producer_node = &self.nodes.items[producer_idx];
        if (producer_node.fused) return;

        const refs = self.buffer_refs.get(input_buf) orelse 0;
        if (refs != 1) return;

        const fused_op = detection.detectImmediateFusionPattern(producer_node.op, new_node.op);
        if (fused_op == null) return;

        const speedup = types.calculateSpeedup(producer_node.op, new_node.op, new_node.element_count);
        if (speedup < self.min_speedup_threshold) return;

        self.nodes.items[new_idx].fused = true;
        self.nodes.items[new_idx].fused_into = producer_idx;
        self.nodes.items[producer_idx].op = fused_op.?;
        self.nodes.items[producer_idx].output = new_node.output;

        self.stats.fusions_applied += 1;
        self.stats.bandwidth_saved_bytes += new_node.element_count * 8;
    }

    pub fn analyze(self: *Self) !void {
        self.patterns.clearRetainingCapacity();

        try detection.detectElementWiseChains(self.allocator, self.nodes.items, &self.buffer_refs, &self.patterns);
        try detection.detectNormActivationPatterns(self.allocator, self.nodes.items, &self.buffer_refs, &self.patterns);
        try detection.detectLinearActivationPatterns(self.allocator, self.nodes.items, &self.buffer_refs, &self.patterns);
        try detection.detectReductionPatterns(self.allocator, self.nodes.items, &self.buffer_refs, &self.patterns);
        try detection.detectAttentionPatterns(self.allocator, self.nodes.items, &self.buffer_refs, &self.patterns);

        self.stats.patterns_detected = self.patterns.items.len;
    }

    pub fn getPatterns(self: *const Self) []const FusionPattern {
        return self.patterns.items;
    }

    pub fn applyFusions(self: *Self) ![]OpNode {
        std.mem.sort(FusionPattern, self.patterns.items, {}, struct {
            fn lessThan(_: void, a: FusionPattern, b: FusionPattern) bool {
                return a.speedup > b.speedup;
            }
        }.lessThan);

        for (self.patterns.items) |pattern| {
            if (detection.canApplyPattern(self.nodes.items, pattern)) {
                detection.applyPattern(self.nodes.items, pattern);
                self.stats.fusions_applied += 1;
            }
        }

        var result = std.ArrayListUnmanaged(OpNode).empty;
        for (self.nodes.items) |node| {
            if (!node.fused) {
                try result.append(self.allocator, node);
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    pub fn reset(self: *Self) void {
        self.nodes.clearRetainingCapacity();
        self.buffer_refs.clearRetainingCapacity();
        self.buffer_producers.clearRetainingCapacity();
        self.patterns.clearRetainingCapacity();
    }

    pub fn getStats(self: *const Self) FusionStats {
        return self.stats;
    }
};

test "fusion optimizer basic" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

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

    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.mul, &.{ 2, 3 }, 4, 1024);
    _ = try optimizer.recordOp(.relu, &.{4}, 5, 1024);

    try optimizer.analyze();

    const patterns = optimizer.getPatterns();
    try std.testing.expect(patterns.len > 0);

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

    _ = try optimizer.recordOp(.mul, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.reduce_sum, &.{2}, 3, 1024);

    try optimizer.analyze();

    const patterns = optimizer.getPatterns();
    try std.testing.expect(patterns.len > 0);

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

    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    try optimizer.analyze();

    const optimized = try optimizer.applyFusions();

    try std.testing.expect(optimized.len < 2);

    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);
}

test "fusion optimizer auto-apply mode" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    optimizer.enableAutoApply(1.0);
    try std.testing.expect(optimizer.isAutoApplyEnabled());

    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);
    try std.testing.expect(stats.bandwidth_saved_bytes > 0);

    try std.testing.expect(optimizer.nodes.items[1].fused);
    try std.testing.expectEqual(@as(?u32, 0), optimizer.nodes.items[1].fused_into);

    try std.testing.expectEqual(OpType.fused_add_relu, optimizer.nodes.items[0].op);
}

test "fusion optimizer auto-apply with threshold" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    optimizer.enableAutoApply(2.0);

    _ = try optimizer.recordOp(.add, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.relu, &.{2}, 3, 1024);

    const stats = optimizer.getStats();

    optimizer.disableAutoApply();
    try std.testing.expect(!optimizer.isAutoApplyEnabled());

    try std.testing.expect(stats.ops_recorded == 2);
}

test "fusion optimizer auto-apply dot product" {
    const allocator = std.testing.allocator;
    var optimizer = FusionOptimizer.init(allocator);
    defer optimizer.deinit();

    optimizer.enableAutoApply(1.0);

    _ = try optimizer.recordOp(.mul, &.{ 0, 1 }, 2, 1024);
    _ = try optimizer.recordOp(.reduce_sum, &.{2}, 3, 1024);

    const stats = optimizer.getStats();
    try std.testing.expect(stats.fusions_applied > 0);

    try std.testing.expectEqual(OpType.dot_product, optimizer.nodes.items[0].op);
}

test {
    std.testing.refAllDecls(@This());
}
