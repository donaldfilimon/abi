const std = @import("std");
const types = @import("types.zig");
const model_mod = @import("model.zig");

const TrainConfig = types.TrainConfig;
const TrainReport = types.TrainReport;
const Model = model_mod.Model;
const Scratch = model_mod.Scratch;
const Grads = model_mod.Grads;

fn initModel(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !Model {
    if (text.len == 0) return error.EmptyCorpus;
    if (config.seq_len == 0 or config.hidden == 0 or config.embed_dim == 0) return error.InvalidConfig;
    if (text.len <= config.seq_len) return error.EmptyCorpus;

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

    var embed = try model_mod_Matrix_alloc(allocator, vocab_size, config.embed_dim);
    errdefer embed.free(allocator);
    var w1 = try model_mod_Matrix_alloc(allocator, config.hidden, in_dim);
    errdefer w1.free(allocator);
    var w2 = try model_mod_Matrix_alloc(allocator, vocab_size, config.hidden);
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

const Matrix = model_mod.Matrix;

fn model_mod_Matrix_alloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
    return Matrix.alloc(allocator, rows, cols);
}

fn fillUniform(rng: std.Random, slice: []f32, scale: f32) void {
    for (slice) |*v| v.* = (rng.float(f32) * 2 - 1) * scale;
}

pub fn trainModel(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !Model {
    var model = try initModel(allocator, text, config);
    errdefer model.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch_alloc = arena.allocator();

    var sc = try Scratch.alloc(scratch_alloc, &model);
    var grads = try Grads.alloc(scratch_alloc, &model);

    const params = [_][]f32{ model.embed.data, model.w1.data, model.w2.data, model.b1, model.b2 };
    const grad_slices = [_][]f32{ grads.embed.data, grads.w1.data, grads.w2.data, grads.b1, grads.b2 };

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
    const initial = model_mod.evalLoss(&model, &sc);

    var t: f32 = 0;
    for (0..config.epochs) |_| {
        grads.zero();
        for (0..model.corpus.len) |p| {
            const target = model.sampleAt(p, sc.ctx);
            _ = model_mod.forwardLoss(&model, &sc, target);
            model_mod.backward(&model, &grads, &sc, target);
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

    const final = model_mod.evalLoss(&model, &sc);
    model.report = .{
        .initial_loss = initial,
        .final_loss = final,
        .steps = config.epochs * model.corpus.len,
        .improved = final < initial,
    };
    return model;
}

pub fn trainOnText(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !TrainReport {
    var model = try trainModel(allocator, text, config);
    defer model.deinit();
    return model.report;
}

pub fn extractCorpusFromJsonl(allocator: std.mem.Allocator, bytes: []const u8, field: []const u8) ![]u8 {
    var corpus: std.ArrayListUnmanaged(u8) = .empty;
    errdefer corpus.deinit(allocator);

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => continue,
        };
        const value = obj.get(field) orelse continue;
        const text = switch (value) {
            .string => |s| s,
            else => continue,
        };
        if (text.len == 0) continue;

        if (corpus.items.len > 0) try corpus.append(allocator, '\n');
        try corpus.appendSlice(allocator, text);
    }

    if (corpus.items.len == 0) return error.NoCorpusData;
    return corpus.toOwnedSlice(allocator);
}

pub fn trainOnJsonl(allocator: std.mem.Allocator, path: []const u8, field: []const u8, config: TrainConfig) !TrainReport {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(bytes);

    const corpus = try extractCorpusFromJsonl(allocator, bytes, field);
    defer allocator.free(corpus);

    return trainOnText(allocator, corpus, config);
}

test {
    std.testing.refAllDecls(@This());
}
