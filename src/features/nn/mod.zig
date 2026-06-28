//! Pure-Zig miniature character-level language-model trainer.
//!
//! A genuinely trainable next-character model: per-character embeddings feed a
//! single hidden linear layer (tanh/relu), then an output linear layer and a
//! softmax over the corpus vocabulary. Cross-entropy loss is minimized by
//! manual backpropagation and full-batch gradient descent (SGD or Adam). No
//! placeholders: the gradients are derived by hand and validated against a
//! finite-difference check in the inline tests, and the loss-decrease gate
//! asserts the average cross-entropy strictly drops over training.
//!
//! The matrix-vector products reuse the `@Vector(4, f32)` + `@reduce(.Add, ...)`
//! SIMD idiom from `src/features/gpu/vector_ops.zig`.

pub const types = @import("types.zig");

const std = @import("std");

pub const NnError = types.NnError;
pub const Error = types.Error;
pub const TrainConfig = types.TrainConfig;
pub const TrainReport = types.TrainReport;

const Activation = types.Activation;
const Optimizer = types.Optimizer;

pub fn isEnabled() bool {
    return true;
}

// ── SIMD primitives ────────────────────────────────────────────────────────

/// SIMD dot product over `[]f32`, mirroring the vectorized CPU fallback in
/// `gpu/vector_ops.zig`: a 4-wide vector loop with a scalar tail.
fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + 4 <= a.len) : (i += 4) {
        const av: @Vector(4, f32) = a[i..][0..4].*;
        const bv: @Vector(4, f32) = b[i..][0..4].*;
        sum += @reduce(.Add, av * bv);
    }
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// Tiny dense matrix stored row-major over a flat `[]f32`.
const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    fn alloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .cols = cols };
    }

    fn free(self: Matrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    fn row(self: Matrix, r: usize) []f32 {
        return self.data[r * self.cols ..][0..self.cols];
    }

    /// `out[r] = dot(row(r), x)` for every row. `out.len == rows`, `x.len == cols`.
    fn matVec(self: Matrix, x: []const f32, out: []f32) void {
        std.debug.assert(x.len == self.cols and out.len == self.rows);
        for (0..self.rows) |r| out[r] = dot(self.row(r), x);
    }
};

fn activate(act: Activation, v: f32) f32 {
    return switch (act) {
        .tanh => std.math.tanh(v),
        .relu => @max(v, 0),
    };
}

/// Derivative of the activation expressed in terms of its post-activation value.
fn activateDeriv(act: Activation, post: f32) f32 {
    return switch (act) {
        .tanh => 1 - post * post,
        .relu => if (post > 0) @as(f32, 1) else 0,
    };
}

fn softmax(logits: []const f32, out: []f32) void {
    std.debug.assert(logits.len == out.len);
    var max: f32 = logits[0];
    for (logits[1..]) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (logits, out) |v, *o| {
        const e = @exp(v - max);
        o.* = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (out) |*o| o.* *= inv;
}

// ── Model ──────────────────────────────────────────────────────────────────

/// A trained (or freshly initialized) character-level model. Owns all weight
/// storage, the id↔byte vocabulary, and a copy of the training corpus used to
/// seed sampling. Free with `deinit`.
pub const Model = struct {
    allocator: std.mem.Allocator,

    /// byte → vocab id, or -1 when the byte never appeared in the corpus.
    vocab: [256]i16,
    /// vocab id → byte.
    id_to_byte: []u8,
    vocab_size: usize,

    seq_len: usize,
    embed_dim: usize,
    hidden: usize,
    activation: Activation,

    /// vocab_size × embed_dim
    embed: Matrix,
    /// hidden × (seq_len * embed_dim)
    w1: Matrix,
    b1: []f32,
    /// vocab_size × hidden
    w2: Matrix,
    b2: []f32,

    /// Owned copy of the training corpus (drives sampling seeds).
    corpus: []u8,
    report: TrainReport,

    fn inputDim(self: *const Model) usize {
        return self.seq_len * self.embed_dim;
    }

    pub fn deinit(self: *Model) void {
        const a = self.allocator;
        self.embed.free(a);
        self.w1.free(a);
        self.w2.free(a);
        a.free(self.b1);
        a.free(self.b2);
        a.free(self.id_to_byte);
        a.free(self.corpus);
        self.* = undefined;
    }

    /// Fill `ctx` (len `seq_len`) and return the target id for the cyclic sample
    /// anchored at corpus position `p`.
    fn sampleAt(self: *const Model, p: usize, ctx: []usize) usize {
        const n = self.corpus.len;
        for (0..self.seq_len) |s| {
            ctx[s] = @intCast(self.vocab[self.corpus[(p + s) % n]]);
        }
        return @intCast(self.vocab[self.corpus[(p + self.seq_len) % n]]);
    }
};

// ── Forward / backward scratch ─────────────────────────────────────────────

/// Per-sample activation + gradient-signal buffers, sized to a model.
const Scratch = struct {
    ctx: []usize,
    x: []f32,
    h: []f32,
    logits: []f32,
    probs: []f32,
    dlogits: []f32,
    dh: []f32,
    dhpre: []f32,
    dx: []f32,

    fn alloc(allocator: std.mem.Allocator, model: *const Model) !Scratch {
        const in_dim = model.inputDim();
        return .{
            .ctx = try allocator.alloc(usize, model.seq_len),
            .x = try allocator.alloc(f32, in_dim),
            .h = try allocator.alloc(f32, model.hidden),
            .logits = try allocator.alloc(f32, model.vocab_size),
            .probs = try allocator.alloc(f32, model.vocab_size),
            .dlogits = try allocator.alloc(f32, model.vocab_size),
            .dh = try allocator.alloc(f32, model.hidden),
            .dhpre = try allocator.alloc(f32, model.hidden),
            .dx = try allocator.alloc(f32, in_dim),
        };
    }
};

/// Gradient accumulators mirroring the model parameter shapes.
const Grads = struct {
    embed: Matrix,
    w1: Matrix,
    b1: []f32,
    w2: Matrix,
    b2: []f32,

    fn alloc(allocator: std.mem.Allocator, model: *const Model) !Grads {
        return .{
            .embed = try Matrix.alloc(allocator, model.vocab_size, model.embed_dim),
            .w1 = try Matrix.alloc(allocator, model.hidden, model.inputDim()),
            .b1 = blk: {
                const b = try allocator.alloc(f32, model.hidden);
                @memset(b, 0);
                break :blk b;
            },
            .w2 = try Matrix.alloc(allocator, model.vocab_size, model.hidden),
            .b2 = blk: {
                const b = try allocator.alloc(f32, model.vocab_size);
                @memset(b, 0);
                break :blk b;
            },
        };
    }

    fn zero(self: *Grads) void {
        @memset(self.embed.data, 0);
        @memset(self.w1.data, 0);
        @memset(self.b1, 0);
        @memset(self.w2.data, 0);
        @memset(self.b2, 0);
    }
};

/// Forward pass for the sample whose context already sits in `sc.ctx`. Writes
/// `x`, `h`, `logits`, `probs` and returns the cross-entropy loss for `target`.
fn forwardLoss(model: *const Model, sc: *Scratch, target: usize) f32 {
    for (0..model.seq_len) |s| {
        const src = model.embed.row(sc.ctx[s]);
        @memcpy(sc.x[s * model.embed_dim ..][0..model.embed_dim], src);
    }
    model.w1.matVec(sc.x, sc.h);
    for (sc.h, model.b1) |*hv, bv| hv.* = activate(model.activation, hv.* + bv);
    model.w2.matVec(sc.h, sc.logits);
    for (sc.logits, model.b2) |*lv, bv| lv.* += bv;
    softmax(sc.logits, sc.probs);
    return -@log(@max(sc.probs[target], 1e-9));
}

/// Backprop a single sample, accumulating into `g`. Assumes `forwardLoss` was
/// just run with the same `sc`/`target` so the activations are current.
fn backward(model: *const Model, g: *Grads, sc: *Scratch, target: usize) void {
    // dL/dlogits = softmax - onehot(target)
    @memcpy(sc.dlogits, sc.probs);
    sc.dlogits[target] -= 1;

    // Output layer: dW2 += outer(dlogits, h); db2 += dlogits.
    for (0..model.vocab_size) |r| {
        const dl = sc.dlogits[r];
        g.b2[r] += dl;
        const gw2_row = g.w2.row(r);
        for (gw2_row, sc.h) |*gv, hv| gv.* += dl * hv;
    }

    // dh = W2^T · dlogits  (column access, no SIMD).
    for (0..model.hidden) |c| {
        var s: f32 = 0;
        for (0..model.vocab_size) |r| s += model.w2.row(r)[c] * sc.dlogits[r];
        sc.dh[c] = s;
    }
    // Through the activation.
    for (sc.dhpre, sc.dh, sc.h) |*dp, dhv, hv| dp.* = dhv * activateDeriv(model.activation, hv);

    // Hidden layer: dW1 += outer(dhpre, x); db1 += dhpre.
    for (0..model.hidden) |r| {
        const dp = sc.dhpre[r];
        g.b1[r] += dp;
        const gw1_row = g.w1.row(r);
        for (gw1_row, sc.x) |*gv, xv| gv.* += dp * xv;
    }

    // dx = W1^T · dhpre.
    for (0..model.inputDim()) |c| {
        var s: f32 = 0;
        for (0..model.hidden) |r| s += model.w1.row(r)[c] * sc.dhpre[r];
        sc.dx[c] = s;
    }

    // Scatter dx back into the embedding rows for each context position.
    for (0..model.seq_len) |s| {
        const grow = g.embed.row(sc.ctx[s]);
        const dx_slice = sc.dx[s * model.embed_dim ..][0..model.embed_dim];
        for (grow, dx_slice) |*gv, dv| gv.* += dv;
    }
}

/// Average cross-entropy over the whole cyclic corpus at the current weights.
fn evalLoss(model: *const Model, sc: *Scratch) f32 {
    var total: f32 = 0;
    for (0..model.corpus.len) |p| {
        const target = model.sampleAt(p, sc.ctx);
        total += forwardLoss(model, sc, target);
    }
    return total / @as(f32, @floatFromInt(model.corpus.len));
}

// ── Construction & training ────────────────────────────────────────────────

fn initModel(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !Model {
    if (text.len == 0) return error.EmptyCorpus;
    if (config.seq_len == 0 or config.hidden == 0 or config.embed_dim == 0) return error.InvalidConfig;
    if (text.len <= config.seq_len) return error.EmptyCorpus;

    // Build a deterministic, byte-ordered vocabulary.
    var present = std.mem.zeroes([256]bool);
    for (text) |b| present[b] = true;
    var vocab_size: usize = 0;
    for (present) |p| {
        if (p) vocab_size += 1;
    }
    if (vocab_size < 2) return error.InvalidConfig;

    const id_to_byte = try allocator.alloc(u8, vocab_size);
    errdefer allocator.free(id_to_byte);
    var vocab: [256]i16 = undefined;
    @memset(&vocab, -1);
    {
        var idx: usize = 0;
        for (0..256) |b| {
            if (present[b]) {
                vocab[b] = @intCast(idx);
                id_to_byte[idx] = @intCast(b);
                idx += 1;
            }
        }
    }

    const in_dim = config.seq_len * config.embed_dim;

    var prng = std.Random.DefaultPrng.init(config.seed);
    const rng = prng.random();

    var embed = try Matrix.alloc(allocator, vocab_size, config.embed_dim);
    errdefer embed.free(allocator);
    var w1 = try Matrix.alloc(allocator, config.hidden, in_dim);
    errdefer w1.free(allocator);
    var w2 = try Matrix.alloc(allocator, vocab_size, config.hidden);
    errdefer w2.free(allocator);

    fillUniform(rng, embed.data, 0.1);
    fillUniform(rng, w1.data, 1.0 / @sqrt(@as(f32, @floatFromInt(in_dim))));
    fillUniform(rng, w2.data, 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden))));

    const b1 = try allocator.alloc(f32, config.hidden);
    errdefer allocator.free(b1);
    @memset(b1, 0);
    const b2 = try allocator.alloc(f32, vocab_size);
    errdefer allocator.free(b2);
    @memset(b2, 0);

    const corpus = try allocator.dupe(u8, text);
    errdefer allocator.free(corpus);

    return .{
        .allocator = allocator,
        .vocab = vocab,
        .id_to_byte = id_to_byte,
        .vocab_size = vocab_size,
        .seq_len = config.seq_len,
        .embed_dim = config.embed_dim,
        .hidden = config.hidden,
        .activation = config.activation,
        .embed = embed,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .corpus = corpus,
        .report = .{ .initial_loss = 0, .final_loss = 0, .steps = 0, .improved = false },
    };
}

fn fillUniform(rng: std.Random, slice: []f32, scale: f32) void {
    for (slice) |*v| v.* = (rng.float(f32) * 2 - 1) * scale;
}

/// Train a model on `text` and return it (caller owns; call `deinit`). The
/// returned model carries its own `TrainReport` in `model.report`.
pub fn trainModel(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !Model {
    var model = try initModel(allocator, text, config);
    errdefer model.deinit();

    // All transient training buffers live in an arena freed on return; only the
    // model parameters (allocated above) survive.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch_alloc = arena.allocator();

    var sc = try Scratch.alloc(scratch_alloc, &model);
    var grads = try Grads.alloc(scratch_alloc, &model);

    const params = [_][]f32{ model.embed.data, model.w1.data, model.w2.data, model.b1, model.b2 };
    const grad_slices = [_][]f32{ grads.embed.data, grads.w1.data, grads.w2.data, grads.b1, grads.b2 };

    // Optional Adam state, allocated only when selected.
    var adam_m: [params.len][]f32 = undefined;
    var adam_v: [params.len][]f32 = undefined;
    if (config.optimizer == .adam) {
        for (params, 0..) |p, i| {
            adam_m[i] = try scratch_alloc.alloc(f32, p.len);
            adam_v[i] = try scratch_alloc.alloc(f32, p.len);
            @memset(adam_m[i], 0);
            @memset(adam_v[i], 0);
        }
    }

    const n: f32 = @floatFromInt(model.corpus.len);
    const initial = evalLoss(&model, &sc);

    var t: f32 = 0;
    for (0..config.epochs) |_| {
        grads.zero();
        for (0..model.corpus.len) |p| {
            const target = model.sampleAt(p, sc.ctx);
            _ = forwardLoss(&model, &sc, target);
            backward(&model, &grads, &sc, target);
        }

        switch (config.optimizer) {
            .sgd => {
                for (params, grad_slices) |param, grad| {
                    for (param, grad) |*pv, gv| pv.* -= config.lr * (gv / n);
                }
            },
            .adam => {
                t += 1;
                const beta1: f32 = 0.9;
                const beta2: f32 = 0.999;
                const eps: f32 = 1e-8;
                const bc1 = 1 - std.math.pow(f32, beta1, t);
                const bc2 = 1 - std.math.pow(f32, beta2, t);
                for (params, grad_slices, 0..) |param, grad, i| {
                    const m = adam_m[i];
                    const v = adam_v[i];
                    for (param, grad, m, v) |*pv, gv_raw, *mv, *vv| {
                        const gv = gv_raw / n;
                        mv.* = beta1 * mv.* + (1 - beta1) * gv;
                        vv.* = beta2 * vv.* + (1 - beta2) * gv * gv;
                        const mhat = mv.* / bc1;
                        const vhat = vv.* / bc2;
                        pv.* -= config.lr * mhat / (@sqrt(vhat) + eps);
                    }
                }
            },
        }
    }

    const final = evalLoss(&model, &sc);
    model.report = .{
        .initial_loss = initial,
        .final_loss = final,
        .steps = config.epochs * model.corpus.len,
        .improved = final < initial,
    };
    return model;
}

/// Train on `text` and return just the report (model is trained then freed).
pub fn trainOnText(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !TrainReport {
    var model = try trainModel(allocator, text, config);
    defer model.deinit();
    return model.report;
}

/// Greedily (argmax) generate `n` characters from `model`, seeding the context
/// from the first occurrence of `seed_char` in the training corpus so that
/// generation immediately tracks a real corpus context. Caller owns the result.
pub fn sample(allocator: std.mem.Allocator, model: *const Model, seed_char: u8, n: usize) ![]u8 {
    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);
    if (n == 0) return out;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var sc = try Scratch.alloc(arena.allocator(), model);

    // Anchor the initial context at the first occurrence of seed_char.
    const start = std.mem.indexOfScalar(u8, model.corpus, seed_char) orelse 0;
    _ = model.sampleAt(start, sc.ctx);

    for (out) |*slot| {
        _ = forwardLossNoTarget(model, &sc);
        var best: usize = 0;
        var best_p: f32 = sc.probs[0];
        for (sc.probs[1..], 1..) |pv, idx| {
            if (pv > best_p) {
                best_p = pv;
                best = idx;
            }
        }
        slot.* = model.id_to_byte[best];
        // Shift context left, append the predicted id.
        if (model.seq_len > 1) {
            std.mem.copyForwards(usize, sc.ctx[0 .. model.seq_len - 1], sc.ctx[1..model.seq_len]);
        }
        sc.ctx[model.seq_len - 1] = best;
    }
    return out;
}

/// Forward pass that fills `probs` without needing a target (for sampling).
fn forwardLossNoTarget(model: *const Model, sc: *Scratch) void {
    for (0..model.seq_len) |s| {
        const src = model.embed.row(sc.ctx[s]);
        @memcpy(sc.x[s * model.embed_dim ..][0..model.embed_dim], src);
    }
    model.w1.matVec(sc.x, sc.h);
    for (sc.h, model.b1) |*hv, bv| hv.* = activate(model.activation, hv.* + bv);
    model.w2.matVec(sc.h, sc.logits);
    for (sc.logits, model.b2) |*lv, bv| lv.* += bv;
    softmax(sc.logits, sc.probs);
}

// ── Tests ──────────────────────────────────────────────────────────────────

test {
    std.testing.refAllDecls(@This());
}

test "training strictly decreases cross-entropy loss (hard gate)" {
    const a = std.testing.allocator;
    const report = try trainOnText(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 16,
        .embed_dim = 8,
        .epochs = 300,
        .lr = 0.5,
        .seed = 42,
    });
    try std.testing.expect(report.steps > 0);
    try std.testing.expect(report.improved);
    try std.testing.expect(report.final_loss < report.initial_loss);
}

test "greedy sampling reproduces the learned repeating pattern" {
    const a = std.testing.allocator;
    var model = try trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 16,
        .embed_dim = 8,
        .epochs = 400,
        .lr = 0.5,
        .seed = 7,
    });
    defer model.deinit();
    try std.testing.expect(model.report.improved);

    const out = try sample(a, &model, 'h', 24);
    defer a.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "hello world") != null);
}

test "backprop gradient matches finite difference" {
    const a = std.testing.allocator;
    // epochs = 0: random init, no updates — we just want a fixed weight set.
    var model = try trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 6,
        .embed_dim = 4,
        .epochs = 0,
        .seed = 123,
    });
    defer model.deinit();

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var sc = try Scratch.alloc(arena.allocator(), &model);
    var grads = try Grads.alloc(arena.allocator(), &model);

    // Single sample, one analytic backward pass populates grads for every
    // parameter group; the central-difference check below then validates one
    // element from each (output W2/b2, hidden W1/b1, embedding) so a backprop
    // regression in any path is caught, not just W2.
    const target = model.sampleAt(0, sc.ctx);
    grads.zero();
    _ = forwardLoss(&model, &sc, target);
    backward(&model, &grads, &sc, target);

    // The embedding index must target a token actually present in this sample's
    // context (ctx is filled by sampleAt), or its gradient is trivially zero.
    const embed_idx = sc.ctx[0] * model.embed_dim; // embed.row(ctx[0])[0], on an active path

    const Case = struct {
        name: []const u8,
        params: []f32,
        analytic: []const f32,
        idx: usize,
    };
    const cases = [_]Case{
        .{ .name = "w2", .params = model.w2.data, .analytic = grads.w2.data, .idx = model.hidden },
        .{ .name = "embed", .params = model.embed.data, .analytic = grads.embed.data, .idx = embed_idx },
        .{ .name = "w1", .params = model.w1.data, .analytic = grads.w1.data, .idx = 0 },
        .{ .name = "b1", .params = model.b1, .analytic = grads.b1, .idx = 0 },
        .{ .name = "b2", .params = model.b2, .analytic = grads.b2, .idx = 0 },
    };

    const eps: f32 = 1e-2;
    for (cases) |c| {
        const orig = c.params[c.idx];
        c.params[c.idx] = orig + eps;
        const l_plus = forwardLoss(&model, &sc, target);
        c.params[c.idx] = orig - eps;
        const l_minus = forwardLoss(&model, &sc, target);
        c.params[c.idx] = orig; // restore before the next parameter
        const numeric = (l_plus - l_minus) / (2 * eps);
        std.testing.expectApproxEqAbs(c.analytic[c.idx], numeric, 5e-2) catch |err| {
            std.debug.print(
                "gradient mismatch for {s}[{d}]: analytic={d} numeric={d}\n",
                .{ c.name, c.idx, c.analytic[c.idx], numeric },
            );
            return err;
        };
    }
}

test "empty and degenerate corpora are rejected" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.EmptyCorpus, trainOnText(a, "", .{}));
    try std.testing.expectError(error.EmptyCorpus, trainOnText(a, "ab", .{ .seq_len = 5 }));
    try std.testing.expectError(error.InvalidConfig, trainOnText(a, "aaaa", .{ .seq_len = 1 }));
}
