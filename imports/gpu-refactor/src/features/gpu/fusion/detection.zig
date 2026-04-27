//! Fusion pattern detection algorithms.

const std = @import("std");
const types = @import("types.zig");
const OpType = types.OpType;
const OpNode = types.OpNode;
const BufferHandle = types.BufferHandle;
const FusionPattern = types.FusionPattern;
const calculateSpeedup = types.calculateSpeedup;

pub fn detectElementWiseChains(
    allocator: std.mem.Allocator,
    nodes: []OpNode,
    buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: *std.ArrayListUnmanaged(FusionPattern),
) !void {
    for (nodes, 0..) |*node, i| {
        if (node.fused or !node.op.isElementWise()) continue;

        var chain: [8]u32 = undefined;
        var chain_len: u8 = 1;
        chain[0] = @intCast(i);

        if (isCommonFusionPattern(nodes, buffer_refs, i)) {
            continue;
        }

        var current_output = node.output;
        var current_idx = i;

        while (chain_len < 8) {
            const next_idx = findSingleConsumer(nodes, buffer_refs, current_output, current_idx);
            if (next_idx == null) break;

            const next_node = &nodes[next_idx.?];
            if (!next_node.op.isElementWise() or next_node.fused) break;

            chain[chain_len] = @intCast(next_idx.?);
            chain_len += 1;
            current_output = next_node.output;
            current_idx = next_idx.?;
        }

        if (chain_len >= 2) {
            const pattern = createChainPattern(nodes, chain[0..chain_len]);
            if (pattern.speedup > 1.05) {
                try patterns.append(allocator, pattern);
            }
        }
    }
}

pub fn detectNormActivationPatterns(
    allocator: std.mem.Allocator,
    nodes: []OpNode,
    buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: *std.ArrayListUnmanaged(FusionPattern),
) !void {
    for (nodes, 0..) |*node, i| {
        if (node.fused or !node.op.isNormalization()) continue;

        const consumer_idx = findSingleConsumer(nodes, buffer_refs, node.output, i);
        if (consumer_idx == null) continue;

        const consumer = &nodes[consumer_idx.?];
        if (!consumer.op.isActivation() or consumer.fused) continue;

        const fused_op: OpType = if (node.op == .layer_norm and consumer.op == .gelu)
            .fused_layer_norm_gelu
        else
            .fused_add_gelu;

        const bandwidth_saved = node.element_count * 8;

        try patterns.append(allocator, .{
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

pub fn detectLinearActivationPatterns(
    allocator: std.mem.Allocator,
    nodes: []OpNode,
    buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: *std.ArrayListUnmanaged(FusionPattern),
) !void {
    for (nodes, 0..) |*node, i| {
        if (node.fused or node.op != .matmul) continue;

        var next_idx = findSingleConsumer(nodes, buffer_refs, node.output, i);
        if (next_idx == null) continue;

        var next_node = &nodes[next_idx.?];
        var has_bias = false;
        var bias_idx: u32 = 0;

        if (next_node.op == .add and !next_node.fused) {
            has_bias = true;
            bias_idx = @intCast(next_idx.?);

            next_idx = findSingleConsumer(nodes, buffer_refs, next_node.output, next_idx.?);
            if (next_idx != null) {
                next_node = &nodes[next_idx.?];
            }
        }

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

            const bandwidth_saved = node.element_count * 8 * (chain_len - 1);

            try patterns.append(allocator, .{
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

pub fn detectReductionPatterns(
    allocator: std.mem.Allocator,
    nodes: []OpNode,
    buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: *std.ArrayListUnmanaged(FusionPattern),
) !void {
    for (nodes, 0..) |*node, i| {
        if (node.fused or node.op != .mul) continue;

        const consumer_idx = findSingleConsumer(nodes, buffer_refs, node.output, i);
        if (consumer_idx == null) continue;

        const consumer = &nodes[consumer_idx.?];
        if (consumer.op != .reduce_sum or consumer.fused) continue;

        const bandwidth_saved = node.element_count * 4;

        try patterns.append(allocator, .{
            .first_op_idx = @intCast(i),
            .last_op_idx = @intCast(consumer_idx.?),
            .fused_op = .dot_product,
            .speedup = 1.5,
            .bandwidth_saved = bandwidth_saved,
            .chain = .{ @intCast(i), @intCast(consumer_idx.?), 0, 0, 0, 0, 0, 0 },
            .chain_len = 2,
        });
    }
}

pub fn detectAttentionPatterns(
    allocator: std.mem.Allocator,
    nodes: []OpNode,
    buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32),
    patterns: *std.ArrayListUnmanaged(FusionPattern),
) !void {
    for (nodes, 0..) |*node, i| {
        if (node.fused or node.op != .matmul) continue;

        const next_idx = findSingleConsumer(nodes, buffer_refs, node.output, i);
        if (next_idx == null) continue;

        const next_node = &nodes[next_idx.?];
        if (next_node.op == .mul and !next_node.fused) {
            const softmax_idx = findSingleConsumer(nodes, buffer_refs, next_node.output, next_idx.?);
            if (softmax_idx == null) continue;

            const softmax_node = &nodes[softmax_idx.?];
            if (softmax_node.op == .softmax and !softmax_node.fused) {
                const bandwidth_saved = node.element_count * 8 * 2;

                try patterns.append(allocator, .{
                    .first_op_idx = @intCast(i),
                    .last_op_idx = @intCast(softmax_idx.?),
                    .fused_op = .fused_attention_qk,
                    .speedup = 1.8,
                    .bandwidth_saved = bandwidth_saved,
                    .chain = .{ @intCast(i), @intCast(next_idx.?), @intCast(softmax_idx.?), 0, 0, 0, 0, 0 },
                    .chain_len = 3,
                });
            }
        }
    }

    for (nodes, 0..) |*node, i| {
        if (node.fused or node.op != .softmax) continue;

        const next_idx = findSingleConsumer(nodes, buffer_refs, node.output, i);
        if (next_idx == null) continue;

        const next_node = &nodes[next_idx.?];
        if (next_node.op == .matmul and !next_node.fused) {
            const bandwidth_saved = node.element_count * 8;

            try patterns.append(allocator, .{
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

pub fn detectImmediateFusionPattern(producer_op: OpType, consumer_op: OpType) ?OpType {
    if (producer_op == .add) {
        switch (consumer_op) {
            .relu => return .fused_add_relu,
            .gelu, .gelu_fast => return .fused_add_gelu,
            else => {},
        }
    }

    if (producer_op == .mul and consumer_op == .add) {
        return .fused_mul_add;
    }

    if (producer_op == .layer_norm) {
        switch (consumer_op) {
            .gelu, .gelu_fast => return .fused_layer_norm_gelu,
            else => {},
        }
    }

    if (producer_op == .matmul) {
        switch (consumer_op) {
            .relu => return .fused_linear_relu,
            .gelu, .gelu_fast => return .fused_linear_gelu,
            .silu => return .fused_linear_silu,
            else => {},
        }
    }

    if (producer_op == .mul and consumer_op == .reduce_sum) {
        return .dot_product;
    }

    if (producer_op.isElementWise() and consumer_op.isElementWise()) {
        return consumer_op;
    }

    return null;
}

pub fn findSingleConsumer(nodes: []const OpNode, buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32), buffer: BufferHandle, producer_idx: usize) ?usize {
    const refs = buffer_refs.get(buffer) orelse 0;
    if (refs != 1) return null;

    for (nodes, 0..) |node, i| {
        if (i <= producer_idx) continue;
        for (node.getInputs()) |inp| {
            if (inp == buffer) return i;
        }
    }
    return null;
}

pub fn canApplyPattern(nodes: []const OpNode, pattern: FusionPattern) bool {
    for (pattern.chain[0..pattern.chain_len]) |idx| {
        if (nodes[idx].fused) return false;
    }
    return true;
}

pub fn applyPattern(nodes: []OpNode, pattern: FusionPattern) void {
    for (pattern.chain[1..pattern.chain_len]) |idx| {
        nodes[idx].fused = true;
        nodes[idx].fused_into = pattern.first_op_idx;
    }

    nodes[pattern.first_op_idx].op = pattern.fused_op;

    const last_node = &nodes[pattern.last_op_idx];
    nodes[pattern.first_op_idx].output = last_node.output;
}

fn isCommonFusionPattern(nodes: []const OpNode, buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32), start_idx: usize) bool {
    const node = &nodes[start_idx];

    if (node.op.isElementWise()) {
        if (detectElementWiseActivation(nodes, buffer_refs, start_idx)) {
            return true;
        }

        if (node.op.isNormalization()) {
            if (detectNormActivation(nodes, buffer_refs, start_idx)) {
                return true;
            }
        }
    }

    return false;
}

fn detectElementWiseActivation(nodes: []const OpNode, buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32), ew_idx: usize) bool {
    const ew_node = &nodes[ew_idx];
    const consumer_idx = findSingleConsumer(nodes, buffer_refs, ew_node.output, ew_idx) orelse return false;
    const consumer = &nodes[consumer_idx];
    return consumer.op.isActivation() and !consumer.fused;
}

fn detectNormActivation(nodes: []const OpNode, buffer_refs: *const std.AutoHashMapUnmanaged(BufferHandle, u32), norm_idx: usize) bool {
    const norm_node = &nodes[norm_idx];
    const consumer_idx = findSingleConsumer(nodes, buffer_refs, norm_node.output, norm_idx) orelse return false;
    const consumer = &nodes[consumer_idx];
    return consumer.op.isActivation() and !consumer.fused;
}

fn createChainPattern(nodes: []const OpNode, chain: []const u32) FusionPattern {
    var pattern = FusionPattern{
        .first_op_idx = chain[0],
        .last_op_idx = chain[chain.len - 1],
        .fused_op = determineChainFusedOp(nodes, chain),
        .speedup = 1.0,
        .bandwidth_saved = 0,
        .chain_len = @intCast(chain.len),
    };

    for (chain, 0..) |idx, i| {
        if (i < 8) pattern.chain[i] = idx;
    }

    var total_bytes: u64 = 0;
    var intermediate_bytes: usize = 0;

    for (chain, 0..) |idx, i| {
        const node = &nodes[idx];
        total_bytes += @as(u64, node.op.memoryBytesPerElement()) * node.element_count;

        if (i > 0) {
            intermediate_bytes += node.element_count * 4;
        }
        if (i < chain.len - 1) {
            intermediate_bytes += node.element_count * 4;
        }
    }

    pattern.bandwidth_saved = intermediate_bytes;

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

    if (first_op == .add and last_op == .relu) return .fused_add_relu;
    if (first_op == .add and last_op == .gelu) return .fused_add_gelu;
    if (first_op == .mul and last_op == .add) return .fused_mul_add;
    if (first_op == .layer_norm and last_op == .gelu) return .fused_layer_norm_gelu;

    return last_op;
}

test {
    std.testing.refAllDecls(@This());
}
