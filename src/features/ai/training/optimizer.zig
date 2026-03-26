//! Training Optimizers
//!
//! SGD, Adam, and AdamW optimizer implementations for neural network training.

const std = @import("std");
const mod = @import("mod.zig");

pub const OptimizerType = mod.OptimizerType;
pub const TrainingConfig = mod.TrainingConfig;
pub const ModelState = mod.ModelState;

pub const Optimizer = union(OptimizerType) {
    sgd: SgdOptimizer,
    adam: AdamOptimizer,
    adamw: AdamWOptimizer,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *ModelState,
        config: TrainingConfig,
    ) !Optimizer {
        return switch (config.optimizer) {
            .sgd => .{ .sgd = try SgdOptimizer.init(allocator, model, config) },
            .adam => .{ .adam = try AdamOptimizer.init(allocator, model, config) },
            .adamw => .{ .adamw = try AdamWOptimizer.init(allocator, model, config) },
        };
    }

    pub fn deinit(self: *Optimizer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .sgd => |*o| o.deinit(allocator),
            .adam => |*o| o.deinit(allocator),
            .adamw => |*o| o.deinit(allocator),
        }
    }

    pub fn step(self: *Optimizer, model: *ModelState, lr: f32, current_step: u64) void {
        switch (self.*) {
            .sgd => |*o| o.step(model, lr, current_step),
            .adam => |*o| o.step(model, lr, current_step),
            .adamw => |*o| o.step(model, lr, current_step),
        }
    }

    pub fn setLearningRate(self: *Optimizer, lr: f32) void {
        switch (self.*) {
            .sgd => |*o| o.learning_rate = lr,
            .adam => |*o| o.learning_rate = lr,
            .adamw => |*o| o.learning_rate = lr,
        }
    }
};

pub const SgdOptimizer = struct {
    learning_rate: f32,
    momentum: f32 = 0.9,
    nesterov: bool = false,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !SgdOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .momentum = 0.9,
            .nesterov = false,
        };
    }

    pub fn deinit(self: *SgdOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *SgdOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        _ = current_step;
        const momentum_val = self.momentum;

        if (model.momentum) |mom| {
            for (model.weights, model.gradients, mom) |*w, *g, *m| {
                const m_old = m.*;
                m.* = momentum_val * m.* + g.*;
                if (self.nesterov) {
                    // Nesterov: use old momentum for look-ahead
                    w.* -= lr * (g.* + momentum_val * m_old);
                } else {
                    w.* -= lr * m.*;
                }
            }
        } else {
            for (model.weights, model.gradients) |*w, *g| {
                w.* -= lr * g.*;
            }
        }
    }
};

pub const AdamOptimizer = struct {
    learning_rate: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    t: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !AdamOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .t = 0,
        };
    }

    pub fn deinit(self: *AdamOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *AdamOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        if (current_step == 0) return;
        self.t = current_step;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const epsilon = self.epsilon;
        const step_f = @as(f32, @floatFromInt(current_step));
        const lr_adjusted = lr * @sqrt(1 - std.math.pow(f32, beta2, step_f)) /
            (1 - std.math.pow(f32, beta1, step_f));

        if (model.momentum) |m| {
            if (model.velocity) |v| {
                for (0..model.weights.len) |i| {
                    m[i] = beta1 * m[i] + (1 - beta1) * model.gradients[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * model.gradients[i] * model.gradients[i];
                    // lr_adjusted already has bias correction; use raw m/v
                    model.weights[i] -= lr_adjusted * m[i] / (@sqrt(v[i]) + epsilon);
                }
            }
        }
    }
};

pub const AdamWOptimizer = struct {
    learning_rate: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    t: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !AdamWOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .weight_decay = config.weight_decay,
            .t = 0,
        };
    }

    pub fn deinit(self: *AdamWOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *AdamWOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        if (current_step == 0) return;
        self.t = current_step;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const epsilon = self.epsilon;
        const wd = self.weight_decay;
        const step_f = @as(f32, @floatFromInt(current_step));

        for (model.weights) |*w| {
            w.* = w.* - lr * wd * w.*;
        }

        if (model.momentum) |*m| {
            if (model.velocity) |*v| {
                const bc1 = 1 - std.math.pow(f32, beta1, step_f);
                const bc2 = 1 - std.math.pow(f32, beta2, step_f);
                const lr_adjusted = lr * @sqrt(bc2) / bc1;
                for (0..model.weights.len) |i| {
                    const g = model.gradients[i];
                    m.*[i] = beta1 * m.*[i] + (1 - beta1) * g;
                    v.*[i] = beta2 * v.*[i] + (1 - beta2) * g * g;
                    // Use lr_adjusted with raw m/v (bias correction in lr)
                    model.weights[i] -= lr_adjusted * m.*[i] / (@sqrt(v.*[i]) + epsilon);
                }
            }
        }
    }
};


test {
    std.testing.refAllDecls(@This());
}
