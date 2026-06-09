//! Neural vector compression (Compute/Storage Layer).
//!
//! A small autoencoder — a genuine nonlinear neural network (tanh hidden layer,
//! linear output) — that learns a compressed latent code for stored vectors by
//! gradient descent on reconstruction error. The codec is trained IN-PROCESS on
//! the actual data (no external model artifact): `encode` maps an input vector
//! to a `latent_dim`-float code (`latent_dim < input_dim`), `decode` reconstructs
//! it. This is a real learned compressor, honestly scoped — a compact codec, not
//! a state-of-the-art deep model.
//!
//! All buffers are caller-provided or pre-allocated once in `init`, so train and
//! inference steps are allocation-free. Weight init is seeded, so training is
//! deterministic and reproducible.

const std = @import("std");

pub const Autoencoder = struct {
    allocator: std.mem.Allocator,
    input_dim: usize,
    latent_dim: usize,
    // Encoder: latent_dim x input_dim (row-major), bias latent_dim.
    we: []f32,
    be: []f32,
    // Decoder: input_dim x latent_dim (row-major), bias input_dim.
    wd: []f32,
    bd: []f32,
    // Pre-allocated forward/backward scratch (reused every step).
    h: []f32,
    xhat: []f32,
    dxhat: []f32,
    dhpre: []f32,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, latent_dim: usize, seed: u64) !Autoencoder {
        std.debug.assert(latent_dim > 0 and latent_dim < input_dim);
        const we = try allocator.alloc(f32, latent_dim * input_dim);
        errdefer allocator.free(we);
        const be = try allocator.alloc(f32, latent_dim);
        errdefer allocator.free(be);
        const wd = try allocator.alloc(f32, input_dim * latent_dim);
        errdefer allocator.free(wd);
        const bd = try allocator.alloc(f32, input_dim);
        errdefer allocator.free(bd);
        const h = try allocator.alloc(f32, latent_dim);
        errdefer allocator.free(h);
        const xhat = try allocator.alloc(f32, input_dim);
        errdefer allocator.free(xhat);
        const dxhat = try allocator.alloc(f32, input_dim);
        errdefer allocator.free(dxhat);
        const dhpre = try allocator.alloc(f32, latent_dim);
        errdefer allocator.free(dhpre);

        // Small seeded random weights keep tanh in its near-linear regime so
        // training starts well-conditioned; biases start at zero.
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        for (we) |*w| w.* = (rand.float(f32) - 0.5) * 0.2;
        for (wd) |*w| w.* = (rand.float(f32) - 0.5) * 0.2;
        @memset(be, 0);
        @memset(bd, 0);

        return .{
            .allocator = allocator,
            .input_dim = input_dim,
            .latent_dim = latent_dim,
            .we = we,
            .be = be,
            .wd = wd,
            .bd = bd,
            .h = h,
            .xhat = xhat,
            .dxhat = dxhat,
            .dhpre = dhpre,
        };
    }

    pub fn deinit(self: *Autoencoder) void {
        self.allocator.free(self.we);
        self.allocator.free(self.be);
        self.allocator.free(self.wd);
        self.allocator.free(self.bd);
        self.allocator.free(self.h);
        self.allocator.free(self.xhat);
        self.allocator.free(self.dxhat);
        self.allocator.free(self.dhpre);
    }

    /// D / k — how much smaller the latent code is than the input.
    pub fn compressionRatio(self: *const Autoencoder) f32 {
        return @as(f32, @floatFromInt(self.input_dim)) / @as(f32, @floatFromInt(self.latent_dim));
    }

    /// Encode `x` (len input_dim) into `latent` (len latent_dim): tanh(We·x + be).
    pub fn encode(self: *const Autoencoder, x: []const f32, latent: []f32) void {
        for (0..self.latent_dim) |j| {
            var s: f32 = self.be[j];
            for (0..self.input_dim) |i| s += self.we[j * self.input_dim + i] * x[i];
            latent[j] = std.math.tanh(s);
        }
    }

    /// Decode `latent` (len latent_dim) into `out` (len input_dim): Wd·latent + bd.
    pub fn decode(self: *const Autoencoder, latent: []const f32, out: []f32) void {
        for (0..self.input_dim) |i| {
            var s: f32 = self.bd[i];
            for (0..self.latent_dim) |j| s += self.wd[i * self.latent_dim + j] * latent[j];
            out[i] = s;
        }
    }

    /// Mean-squared reconstruction error for `x` without updating weights.
    pub fn reconstructionMse(self: *Autoencoder, x: []const f32) f32 {
        self.encode(x, self.h);
        self.decode(self.h, self.xhat);
        var loss: f32 = 0;
        for (0..self.input_dim) |i| {
            const d = self.xhat[i] - x[i];
            loss += d * d;
        }
        return loss / @as(f32, @floatFromInt(self.input_dim));
    }

    /// One SGD step on a single sample; returns the pre-update reconstruction MSE.
    pub fn trainStep(self: *Autoencoder, x: []const f32, lr: f32) f32 {
        const d_f: f32 = @floatFromInt(self.input_dim);
        self.encode(x, self.h);
        self.decode(self.h, self.xhat);

        var loss: f32 = 0;
        for (0..self.input_dim) |i| {
            const diff = self.xhat[i] - x[i];
            loss += diff * diff;
            self.dxhat[i] = 2.0 * diff / d_f; // dL/dxhat
        }
        loss /= d_f;

        // Gradient into the hidden activation (uses current decoder weights),
        // then through tanh': (1 - h^2).
        for (0..self.latent_dim) |j| {
            var s: f32 = 0;
            for (0..self.input_dim) |i| s += self.wd[i * self.latent_dim + j] * self.dxhat[i];
            self.dhpre[j] = s * (1.0 - self.h[j] * self.h[j]);
        }

        // Decoder update.
        for (0..self.input_dim) |i| {
            self.bd[i] -= lr * self.dxhat[i];
            for (0..self.latent_dim) |j| {
                self.wd[i * self.latent_dim + j] -= lr * self.dxhat[i] * self.h[j];
            }
        }
        // Encoder update.
        for (0..self.latent_dim) |j| {
            self.be[j] -= lr * self.dhpre[j];
            for (0..self.input_dim) |i| {
                self.we[j * self.input_dim + i] -= lr * self.dhpre[j] * x[i];
            }
        }
        return loss;
    }

    /// One pass over `vectors`; returns the average pre-update reconstruction MSE.
    pub fn trainEpoch(self: *Autoencoder, vectors: []const []const f32, lr: f32) f32 {
        if (vectors.len == 0) return 0;
        var total: f32 = 0;
        for (vectors) |v| total += self.trainStep(v, lr);
        return total / @as(f32, @floatFromInt(vectors.len));
    }
};

const testing = std.testing;

test "neural compression: autoencoder learns to reconstruct low-rank vectors" {
    const allocator = testing.allocator;

    // 8-D vectors that actually live in a 3-D subspace (low rank), so a
    // 3-latent codec can reconstruct them — the premise of learned compression.
    var prng = std.Random.DefaultPrng.init(0xABCDEF01);
    const rand = prng.random();
    const basis = [3][8]f32{
        .{ 0.4, -0.3, 0.1, 0.2, -0.1, 0.3, -0.2, 0.1 },
        .{ -0.2, 0.3, 0.25, -0.15, 0.2, -0.1, 0.3, -0.25 },
        .{ 0.1, 0.1, -0.3, 0.3, 0.15, -0.2, 0.1, 0.2 },
    };
    var storage: [24][8]f32 = undefined;
    var dataset: [24][]const f32 = undefined;
    for (0..24) |n| {
        const c0 = (rand.float(f32) - 0.5);
        const c1 = (rand.float(f32) - 0.5);
        const c2 = (rand.float(f32) - 0.5);
        for (0..8) |i| {
            storage[n][i] = c0 * basis[0][i] + c1 * basis[1][i] + c2 * basis[2][i];
        }
        dataset[n] = &storage[n];
    }

    var ae = try Autoencoder.init(allocator, 8, 3, 0xC0FFEE);
    defer ae.deinit();

    try testing.expectApproxEqAbs(@as(f32, 8.0) / 3.0, ae.compressionRatio(), 1e-6);

    // Baseline reconstruction error before any training.
    var before: f32 = 0;
    for (dataset) |v| before += ae.reconstructionMse(v);
    before /= dataset.len;

    var epoch: usize = 0;
    while (epoch < 400) : (epoch += 1) _ = ae.trainEpoch(&dataset, 0.1);

    var after: f32 = 0;
    for (dataset) |v| after += ae.reconstructionMse(v);
    after /= dataset.len;

    // Training reduced reconstruction error substantially and to a small bound:
    // the learned 3-float code captures the 8-D vectors.
    try testing.expect(after < before);
    try testing.expect(after < before * 0.25);
    try testing.expect(after < 0.01);

    // The latent code is genuinely smaller than the input.
    var latent: [3]f32 = undefined;
    ae.encode(dataset[0], &latent);
    try testing.expectEqual(@as(usize, 3), latent.len);
}

test "neural compression: encode/decode round-trip uses caller buffers (alloc-free)" {
    const allocator = testing.allocator;
    var ae = try Autoencoder.init(allocator, 6, 2, 0x1234);
    defer ae.deinit();

    const x = [_]f32{ 0.1, -0.2, 0.05, 0.0, -0.1, 0.15 };
    var latent: [2]f32 = undefined;
    var out: [6]f32 = undefined;
    ae.encode(&x, &latent);
    ae.decode(&latent, &out);
    // Untrained reconstruction is lossy but finite — the path is wired correctly.
    for (out) |v| try testing.expect(std.math.isFinite(v));
}

test {
    testing.refAllDecls(@This());
}
