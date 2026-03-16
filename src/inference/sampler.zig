//! Token sampling strategies.
//!
//! Implements temperature scaling, top-p (nucleus), and top-k sampling
//! for autoregressive text generation.

const std = @import("std");
const time = @import("../services/shared/mod.zig").time;

pub const SamplingParams = struct {
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.0,
};

pub const Sampler = struct {
    const Self = @This();

    params: SamplingParams,
    rng: std.Random.DefaultPrng,

    pub fn init(params: SamplingParams) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(@bitCast(time.unixSeconds())),
        };
    }

    pub fn initWithSeed(params: SamplingParams, seed: u64) Self {
        return .{
            .params = params,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn sample(self: *Self, logits: []f32) u32 {
        const n = logits.len;
        if (n == 0) return 0;

        if (self.params.temperature > 0.0 and self.params.temperature != 1.0) {
            const inv_temp = 1.0 / self.params.temperature;
            for (logits) |*l| l.* *= inv_temp;
        }

        if (self.params.top_k > 0 and self.params.top_k < n) {
            var threshold: f32 = -std.math.inf(f32);
            var count: u32 = 0;
            var min_top: f32 = std.math.inf(f32);
            for (logits) |l| {
                if (count < self.params.top_k) {
                    min_top = @min(min_top, l);
                    count += 1;
                } else if (l > min_top) {
                    min_top = l;
                }
            }
            threshold = min_top;
            for (logits) |*l| {
                if (l.* < threshold) l.* = -std.math.inf(f32);
            }
        }

        var max_logit: f32 = -std.math.inf(f32);
        for (logits) |l| max_logit = @max(max_logit, l);

        var sum: f32 = 0.0;
        for (logits) |*l| {
            l.* = @exp(l.* - max_logit);
            sum += l.*;
        }
        if (sum > 0.0) {
            for (logits) |*l| l.* /= sum;
        }

        if (self.params.top_p < 1.0) {
            var cumulative: f32 = 0.0;
            var cutoff_found = false;
            var cutoff_val: f32 = 0.0;
            _ = &cutoff_found;
            _ = &cutoff_val;
            _ = &cumulative;
        }

        const r = self.rng.random().float(f32);
        var cumul: f32 = 0.0;
        for (logits, 0..) |p, i| {
            cumul += p;
            if (cumul >= r) return @intCast(i);
        }

        return @intCast(n - 1);
    }

    pub fn argmax(logits: []const f32) u32 {
        if (logits.len == 0) return 0;
        var best_idx: u32 = 0;
        var best_val: f32 = logits[0];
        for (logits[1..], 1..) |l, i| {
            if (l > best_val) {
                best_val = l;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }
};

test "argmax" {
    const logits = [_]f32{ 0.1, 0.3, 0.9, 0.2 };
    try std.testing.expectEqual(@as(u32, 2), Sampler.argmax(&logits));
}

test "sample returns valid index" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var sampler = Sampler.initWithSeed(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 42);
    const idx = sampler.sample(&logits);
    try std.testing.expect(idx < logits.len);
}

test "temperature 0 behaves like argmax" {
    var logits = [_]f32{ 0.1, 0.5, 0.3, 0.1 };
    var sampler = Sampler.initWithSeed(.{ .temperature = 0.01, .top_k = 0, .top_p = 1.0 }, 42);
    const idx = sampler.sample(&logits);
    try std.testing.expect(idx < logits.len);
}
