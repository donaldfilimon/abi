const std = @import("std");
pub const types = @import("types.zig");
const model_mod = @import("model.zig");
const train_mod = @import("train.zig");
const persist = @import("persist.zig");

pub const NnError = types.NnError;
pub const Error = types.Error;
pub const TrainConfig = types.TrainConfig;
pub const TrainReport = types.TrainReport;
pub const Model = model_mod.Model;
pub const Scratch = model_mod.Scratch;
pub const Grads = model_mod.Grads;

pub const trainModel = train_mod.trainModel;
pub const trainOnText = train_mod.trainOnText;
pub const trainOnJsonl = train_mod.trainOnJsonl;
pub const extractCorpusFromJsonl = train_mod.extractCorpusFromJsonl;
pub const saveModelAlloc = persist.saveModelAlloc;
pub const loadModelBytes = persist.loadModelBytes;
pub const saveModelPath = persist.saveModelPath;
pub const loadModelPath = persist.loadModelPath;
pub const sampleStreaming = persist.sampleStreaming;
pub const trainBundled = persist.trainBundled;
pub const DEFAULT_CHECKPOINT_PATH = persist.DEFAULT_CHECKPOINT_PATH;
pub const BUNDLED_CORPUS = persist.BUNDLED_CORPUS;
pub const MAX_OUTPUT_CHARS = persist.MAX_OUTPUT_CHARS;

pub fn isEnabled() bool {
    return true;
}

pub fn sample(allocator: std.mem.Allocator, model: *const Model, seed_char: u8, n: usize) ![]u8 {
    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);
    if (n == 0) return out;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var sc = try Scratch.alloc(arena.allocator(), model);

    const start = std.mem.indexOfScalar(u8, model.corpus, seed_char) orelse 0;
    _ = model.sampleAt(start, sc.ctx);

    for (out) |*slot| {
        model_mod.forwardLossNoTarget(model, &sc);
        var best: usize = 0;
        var best_p: f32 = sc.probs[0];
        for (sc.probs[1..], 1..) |pv, idx| {
            if (pv > best_p) {
                best_p = pv;
                best = idx;
            }
        }
        slot.* = model.id_to_byte[best];
        if (model.seq_len > 1) {
            std.mem.copyForwards(usize, sc.ctx[0 .. model.seq_len - 1], sc.ctx[1..model.seq_len]);
        }
        sc.ctx[model.seq_len - 1] = best;
    }
    return out;
}

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

    const target = model.sampleAt(0, sc.ctx);
    grads.zero();
    _ = model_mod.forwardLoss(&model, &sc, target);
    model_mod.backward(&model, &grads, &sc, target);

    const embed_idx = sc.ctx[0] * model.embed_dim;

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
        const l_plus = model_mod.forwardLoss(&model, &sc, target);
        c.params[c.idx] = orig - eps;
        const l_minus = model_mod.forwardLoss(&model, &sc, target);
        c.params[c.idx] = orig;
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

test "extractCorpusFromJsonl concatenates the chosen field across nonblank lines" {
    const a = std.testing.allocator;
    const jsonl =
        \\{"text":"hello","n":1}
        \\
        \\{"text":"world"}
        \\{"other":"ignored"}
        \\
    ;
    const corpus = try extractCorpusFromJsonl(a, jsonl, "text");
    defer a.free(corpus);
    try std.testing.expectEqualStrings("hello\nworld", corpus);
}

test "extractCorpusFromJsonl errors when the field is absent everywhere" {
    const a = std.testing.allocator;
    const jsonl =
        \\{"other":"x"}
        \\{"other":"y"}
    ;
    try std.testing.expectError(error.NoCorpusData, extractCorpusFromJsonl(a, jsonl, "text"));
}

test "empty and degenerate corpora are rejected" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.EmptyCorpus, trainOnText(a, "", .{}));
    try std.testing.expectError(error.EmptyCorpus, trainOnText(a, "ab", .{ .seq_len = 5 }));
    try std.testing.expectError(error.InvalidConfig, trainOnText(a, "aaaa", .{ .seq_len = 1 }));
}
