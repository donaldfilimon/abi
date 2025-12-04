const std = @import("std");
const activations = @import("activations/mod.zig");

/// Abstraction for tensor operations shared across AI modules.
pub const TensorOps = struct {
    pub const Interface = struct {
        apply_activation: fn (activations.ActivationType, []f32) anyerror!void,
        add: fn ([]const f32, []const f32, []f32) anyerror!void,
        scale: fn ([]f32, f32) anyerror!void,
    };

    interface: Interface,

    pub fn applyActivation(self: TensorOps, activation: activations.ActivationType, data: []f32) !void {
        try self.interface.apply_activation(activation, data);
    }

    pub fn add(self: TensorOps, lhs: []const f32, rhs: []const f32, out: []f32) !void {
        try self.interface.add(lhs, rhs, out);
    }

    pub fn scale(self: TensorOps, data: []f32, scalar: f32) !void {
        try self.interface.scale(data, scalar);
    }
};

fn defaultApplyActivation(activation: activations.ActivationType, data: []f32) anyerror!void {
    var processor = activations.ActivationProcessor.init(.{ .activation_type = activation });
    processor.activateBatch(data, data);
    return;
}

fn defaultAdd(lhs: []const f32, rhs: []const f32, out: []f32) anyerror!void {
    std.debug.assert(lhs.len == rhs.len and lhs.len == out.len);
    for (lhs, rhs, 0..) |l, r, i| {
        out[i] = l + r;
    }
}

fn defaultScale(data: []f32, scalar: f32) anyerror!void {
    for (data) |*value| {
        value.* *= scalar;
    }
}

/// Construct tensor operations backed by default CPU implementations.
pub fn createBasicTensorOps() TensorOps {
    return TensorOps{
        .interface = .{
            .apply_activation = defaultApplyActivation,
            .add = defaultAdd,
            .scale = defaultScale,
        },
    };
}

test "tensor ops apply activation" {
    var tensor_ops = createBasicTensorOps();
    var data = [_]f32{ -1.0, 0.0, 1.0 };
    try tensor_ops.applyActivation(.relu, data[0..]);
    try std.testing.expect(data[0] == 0.0);
    try std.testing.expect(data[1] == 0.0);
    try std.testing.expect(data[2] == 1.0);
}

test "tensor ops add and scale" {
    var tensor_ops = createBasicTensorOps();
    var out = [_]f32{ 0.0, 0.0 };
    try tensor_ops.add(&[_]f32{ 1.0, 2.0 }, &[_]f32{ 3.0, 4.0 }, out[0..]);
    try tensor_ops.scale(out[0..], 0.5);
    try std.testing.expectApproxEqAbs(2.0, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(3.0, out[1], 1e-6);
}
